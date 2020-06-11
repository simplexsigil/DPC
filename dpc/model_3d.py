import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../backbone')
from select_backbone import select_resnet


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class SkeleContrast(nn.Module):
    '''Module which performs contrastive learning by matching extracted feature
    vectors from the skele-motion skeleton representation and features extracted from RGB videos with a Resnet 18.'''

    def __init__(self, img_dim, seq_len=30, network='resnet18', representation_size=1024):
        super(SkeleContrast, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using SkeleContrast model.')
        self.sample_size = img_dim
        self.seq_len = seq_len
        self.last_duration = int(math.ceil(seq_len / 4))  # This is the sequence length after using the backbone
        self.last_size = int(math.ceil(img_dim / 32))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.crossm_vector_length = representation_size

        self.final_bn = nn.BatchNorm1d(4096)
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.dpc_feature_conversion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, self.crossm_vector_length),
        )

        self.skele_motion_backbone = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1536, out_features=self.crossm_vector_length),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),
        )

        self.mask = None
        self._initialize_weights(self.dpc_feature_conversion)
        self._initialize_weights(self.skele_motion_backbone)

    def _forward_sk(self, block_sk):
        (B, N, C, T, J) = block_sk.shape

        block_sk = block_sk.view(B * N, C, T, J)  # Forward each block individually like a batch input.

        features = self.skele_motion_backbone(block_sk)

        return features

    def _forward_rgb(self, block_rgb):
        # block: [B, N, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, N, C, SL, H, W) = block_rgb.shape
        block_rgb = block_rgb.view(B * N, C, SL, H, W)

        # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        feature = self.backbone(block_rgb)  # (B * N, 256, 2, 4, 4)
        del block_rgb

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feature = feature.view(B*N, self.param['feature_size'], self.last_size, self.last_size)

        feature = torch.flatten(feature, start_dim=1, end_dim=-1)

        feature = self.dpc_feature_conversion(feature)

        return feature

    def forward(self, block_rgb, block_sk):
        B = block_rgb.shape[0]

        pred_sk = self._forward_sk(block_sk)  # (B, N, D)
        pred_rgb = self._forward_rgb(block_rgb)  # (B, N, D)

        # The score is now calculated according to the other modality. for this we calculate the dot product of the feature vectors:
        pred_rgb = pred_rgb.contiguous() # .view(B*N, D)
        pred_sk = pred_sk.contiguous() # .view(B*N, D)

        score = torch.matmul(pred_sk, pred_rgb.transpose(0, 1))

        return score

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and len(param.shape) > 1:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


