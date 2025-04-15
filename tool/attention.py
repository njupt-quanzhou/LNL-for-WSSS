#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # 每个头的通道维度
        # 1x1卷积用于计算Q, K, V
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  # 缩放因子

    def forward(self, x, mask=None):
        batch_size, C, H, W = x.size()
        # 计算Q, K, V
        query = self.query_conv(x)  # [N, C, H, W]
        key = self.key_conv(x)  # [N, C, H, W]
        value = self.value_conv(x)  # [N, C, H, W]
        query = query.view(batch_size, self.num_heads, self.head_dim, H * W)
        key = key.view(batch_size, self.num_heads, self.head_dim, H * W)
        value = value.view(batch_size, self.num_heads, self.head_dim, H * W)
        query = query.permute(0, 1, 3, 2)  # [N, num_heads, H*W, head_dim]
        key = key.permute(0, 1, 2, 3)  # [N, num_heads, head_dim, H*W]
        value = value.permute(0, 1, 3, 2)  # [N, num_heads, H*W, head_dim]
        # 计算注意力分数: [N, num_heads, H*W, H*W]
        attn_scores = torch.matmul(query, key) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, value)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, self.in_channels, H, W)
        out = self.output_conv(out)

        return out, attn_probs