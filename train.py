import torch, os
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import importlib
import numpy as np
from tensorboardX import SummaryWriter
from data import data_coco, data_voc
from tool import pyutils, torchutils, visualization, imutils
import random

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1')
    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            label_all = F.pad(label, (0, 0, 0, 0, 1, 0), 'constant', 1.0)
            output = model(img, label_all, label)
            loss1 = F.multilabel_soft_margin_loss(output['score_1'], label)
            val_loss_meter.add({'loss1': loss1.item()})

    model.train()
    loss_final = val_loss_meter.pop('loss1')

    print('loss: %.4f' % loss_final)

    return loss_final


def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoches", default=5, type=int)
    parser.add_argument("--network", default="network.resnet50_LNL", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--val_freq", default=500, type=int)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--dataset_root", default="./VOC2012", type=str)
    parser.add_argument("--seed", default=15, type=int)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'ckpt'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = args.dataset_root
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=21)
        train_dataset = data_voc.VOC12ClsDataset('data/trainaug_' + args.dataset + '.txt', voc12_root=dataset_root,
                                                                    resize_long=(320, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_voc.VOC12ClsDataset('data/val_' + args.dataset + '.txt', voc12_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    elif args.dataset == 'coco':
        args.tf_freq = 5000
        dataset_root = args.dataset_root
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=81)
        train_dataset = data_coco.COCOClsDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_coco.COCOClsDataset('data/val_' + args.dataset + '.txt', coco_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()
    
    val_loss = 100000
    for ep in range(args.max_epoches):

        print('Epoch %d/%d' % (ep + 1, args.max_epoches))

        for step, pack in enumerate(train_data_loader):
            img = pack['img'].cuda()
            n, c, h, w = img.shape
            label = pack['label'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)

            valid_mask = pack['valid_mask'].cuda()
            valid_mask[:, 1:] = valid_mask[:, 1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(h // 16, w // 16), mode='nearest')
            outputs = model.forward(img, valid_mask_lowres, label)
            score_1 = outputs['score_1']
            score_2 = outputs['score_2']
            loss_con = torch.mean(outputs['contrast_loss'])

            loss_cls1 = F.multilabel_soft_margin_loss(score_1, label)
            loss_cls2 = F.multilabel_soft_margin_loss(score_2, label)

            avg_meter.add({'loss1': loss_cls1.item()})
            avg_meter.add({'loss2': loss_cls2.item()})
            avg_meter.add({'loss3': loss_con.item()})

            loss = 0.5 * loss_cls1 + 0.5 * loss_cls2
            if (ep >= 2):
                loss += (0.1 * loss_con)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_freq == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'losscls1:%.4f' % (avg_meter.pop('loss1')),
                      'losscls2:%.4f' % (avg_meter.pop('loss2')),
                      'losscon:%.4f' % (avg_meter.pop('loss3')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)

            if (optimizer.global_step - 1) % args.val_freq == 0 and optimizer.global_step > 10:
                val_cls_now = validate(model, val_data_loader)
                torch.save({'net': model.module.state_dict()},
                           os.path.join(args.session_name, 'ckpt', 'iter_' + str(optimizer.global_step) + '.pth'))
                if (val_cls_now < val_loss):
                    val_loss = val_cls_now
                    torch.save({'net': model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'best.pth'))

        else:
            timer.reset_stage()

    torch.save({'net': model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'final.pth'))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    train()
