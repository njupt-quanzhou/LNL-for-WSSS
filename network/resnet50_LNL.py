import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
from tool import attention

class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()
        self.num_cls = num_cls
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))
        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.attention = attention.SpatialAttention(2048, 8)
        self.classifier = nn.Conv2d(2048, self.num_cls - 1, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4, self.attention])
        self.newly_added = nn.ModuleList([self.classifier])

        self.queue_len = 500
        self.queue_dim = 2048
        self.momentum = 0.9
        self.temperature = 0.1

        for i in range(0, self.num_cls):
            self.register_buffer("queue" + str(i), torch.randn(self.queue_len, self.queue_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))


    def FSCM(self, norm_cam, label, feature, threshold=0.7):
        n, c, h, w = norm_cam.shape
        # iou evalution
        seeds = torch.zeros((n, h, w, c)).cuda()
        feature_s = feature.view(n, -1, h * w)
        feature_s = feature_s / (torch.norm(feature_s, dim=1, keepdim=True) + 1e-5)
        correlation = torch.sigmoid(torch.matmul(feature_s.transpose(2, 1), feature_s)).unsqueeze(
            1)  # [n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n, -1, h * w).unsqueeze(2)  # [n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter / union).view(n, self.num_class, h, w)  # [n,21,h,w]
        # probs = F.softmax(miou, dim=1)
        max_miou_values, belonging = torch.max(miou, dim=1)
        valid_mask = (max_miou_values > threshold).unsqueeze(1)
        seeds = seeds.scatter_(-1, belonging.view(n, h, w, 1), 1).permute(0, 3, 1, 2).contiguous()
        seeds = seeds * label * valid_mask
        return seeds

    def spatial_weights_compute(self, feature):
        n, c, h, w = feature.size()
        x = torch.arange(w).cuda()
        x = x.view(1, -1).repeat(h, 1).float()
        y = torch.arange(h).cuda()
        y = y.view(-1, 1).repeat(1, w).float()
        coordinates = torch.stack([x, y], dim=0).view(2, h, w)
        # (2,h,w)
        coordinates_reshape = coordinates.view(2, h * w)
        # (hw,hw)
        distances = torch.norm(coordinates_reshape[:, None, :] - coordinates_reshape[:, :, None], dim=0)
        point1 = torch.tensor([.0, .0])
        point2 = torch.tensor([h - 1, w - 1])
        distances = distances / torch.norm(point1 - point2)
        weights = torch.exp(-distances).to(torch.float32)
        weights = (weights-torch.min(weights))/(torch.max(weights)-torch.min(weights)+1e-5)
        weights = weights.repeat(n, 1, 1)
        return weights

    def erosion(self, mask, threshold=8):
        # print(mask.size())
        n, c, h, w = mask.size()
        weights = torch.ones(c, 1, 3, 3).cuda()
        # 使用pad进行膨胀操作
        new_mask = torch.nn.functional.pad(mask, (1, 1, 1, 1), mode='constant', value=1)
        # 计算3×3领域内属于该类的点数
        new_mask = torch.nn.functional.conv2d(new_mask, weight=weights, padding=0, groups=c)
        # 将大于等于阈值的且自身为1的值设为1，其余为0
        new_mask = (mask == 1)*(new_mask >= threshold).to(torch.float32)
        return new_mask

    def DFSM(self, cam, affinity_map, distance_map):
        n, _, h, w = cam.shape
        sim = (affinity_map >= 0.5).to(torch.float32) * affinity_map
        sim = sim * distance_map
        sim = sim / (torch.sum(sim, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam.view(n, -1, h * w), sim).view(n, -1, h, w)
        return cam_rv

    def forward(self, x, valid_mask, label):
        N, C, H, W = x.size()
        bg_label = torch.ones(N, 1, 1, 1, dtype=label.dtype, device=label.device)
        label = torch.cat([bg_label, label], dim=1)
        label = label.squeeze(-1).squeeze(-1)
        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        sem_feature = x4
        cam = self.classifier(x4)
        score_1 = F.adaptive_avg_pool2d(cam, 1)
        # probs = torch.sigmoid(score_1.detach())

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = (1-torch.max(norm_cam,dim=1)[0].unsqueeze(1))*0.5
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, x3.shape[2:], mode='bilinear', align_corners=True)*valid_mask
        seeds = self.FSCM(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())
        seeds = self.erosion(seeds, 8)

        feature_expansion, attn = self.attention(sem_feature)

        spatial_weight = self.spatial_weights_compute(feature_expansion)
        attn = torch.mean(attn, dim=1)
        cam_expansion = self.classifier(feature_expansion)
        score_2 = F.adaptive_avg_pool2d(cam_expansion, 1)
        cam_expansion = self.DFSM(cam_expansion, attn, spatial_weight)
        cam_expansion_norm = F.relu(cam_expansion)
        cam_expansion_norm = cam_expansion_norm / (F.adaptive_max_pool2d(cam_expansion_norm, (1, 1)) + 1e-5)
        cam_expansion_bkg = (1 - torch.max(cam_expansion_norm, dim=1)[0].unsqueeze(1))*0.5
        cam_expansion_norm = torch.cat([cam_expansion_bkg, cam_expansion_norm], dim=1)
        cam_expansion_norm = F.interpolate(cam_expansion_norm, x3.shape[2:], mode='bilinear',
                                           align_corners=True) * valid_mask
        seeds_expansion = self.FSCM(cam_expansion_norm.clone(), valid_mask.clone(), feature_expansion.clone())

        feat_memory = getattr(self, "queue0")
        for k in range(1, self.num_cls):
            feat = getattr(self, "queue" + str(k))
            feat_memory = torch.cat((feat_memory, feat), 0)

        # ---------------------------compute loss------------------------------------]
        feature = sem_feature.clone()
        n, c, h, w = feature.shape
        features_flatten = feature.permute(0, 2, 3, 1).reshape(n * h * w, c)
        features_flatten = F.normalize(features_flatten, dim=1)
        feat_neg = F.normalize(feat_memory, dim=1)
        mask = seeds_expansion.permute(0, 2, 3, 1).reshape(n * h * w, -1).detach()

        loss = torch.zeros(1).cuda()
        for i in range(1, self.num_cls):
            if (torch.sum(mask[:, i]) > 0):
                similarity_neg = features_flatten * (mask[:, i]).unsqueeze(-1) @ (feat_neg.detach().permute(1, 0))
                logit_neg = torch.div(similarity_neg, self.temperature)
                logit_neg_max, _ = torch.max(logit_neg, dim=1, keepdim=True)
                exp_logit_neg = torch.exp(logit_neg - logit_neg_max.detach())
                # feat_pos = F.normalize(torch.mean(getattr(self, "queue" + str(i)),dim=0, keepdim=True), dim=1)
                feat_pos = F.normalize(getattr(self, "queue" + str(i)), dim=1)
                similarity_pos = (features_flatten * (mask[:, i]).unsqueeze(-1)) @ (feat_pos.detach().permute(1, 0))
                logit_pos = torch.div(similarity_pos, self.temperature)
                logit_pos = logit_pos - logit_neg_max.detach()
                exp_logit_pos = torch.exp(logit_pos)
                loss_temp = -(
                            torch.log(torch.sum(exp_logit_pos, dim=1)) - torch.log(torch.sum(exp_logit_neg, dim=1)))
                loss += torch.mean(loss_temp)
        loss = loss / (self.num_cls - 1)

        # ---------------------------enqueue&dequeue------------------------------------
        for j in range(0, n):
            self._dequeue_and_enqueue(sem_feature[j], seeds[j], label[j])


        return {"score_1": score_1, "score_2": score_2, "contrast_loss": loss}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, x, map, label):
        for ind, cla in enumerate(label):
            if cla == 1:
                mask = map[ind]
                x = x * mask.float()
                index = torch.nonzero(mask)
                queue_i = getattr(self, "queue" + str(ind))
                for j in range(index.shape[0]):
                    feature = x[:, index[j, 0], index[j, 1]]
                    similarities = torch.matmul(feature, queue_i)
                    ptr = int(torch.argmax(similarities))
                    queue_i[ptr] = queue_i[ptr] * self.momentum + x[:, index[j, 0], index[j, 1]] * (1 - self.momentum)

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label):

        bg_label = torch.ones(1, 1, 1, dtype=label.dtype, device=label.device)
        label = torch.cat([bg_label, label], dim=0)
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)

        cam = self.classifier(x4)
        norm_cam = F.relu(cam)

        feature_expansion, attn = self.attention(x4)
        spatial_weight = self.spatial_weights_compute(feature_expansion)
        attn = torch.mean(attn, dim=1)
        cam_expansion = self.classifier(feature_expansion)
        cam_expansion = self.DFSM(cam_expansion, attn, spatial_weight)
        cam_expansion_norm = F.relu(cam_expansion)


        return norm_cam[0], cam_expansion_norm[0]
