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
from convrnn import ConvGRU

import torchvision.models as tm


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class DPC_RNN(nn.Module):
    '''DPC with RNN'''

    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network='resnet50'):
        super(DPC_RNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))  # This is the sequence length after using the backbone
        self.last_size = int(math.ceil(sample_size / 32))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.crossm_vector_length = 256

        # self.agg = ConvGRU(input_size=self.param['feature_size'],
        #                   hidden_size=self.param['hidden_size'],
        #                   kernel_size=1,
        #                   num_layers=self.param['num_layers'])

        # self.rgb_agg = torch.nn.GRU(input_size=self.crossm_vector_length, hidden_size=self.crossm_vector_length, num_layers=self.param['num_layers'])

        # self.rgb_network_pred = nn.Sequential(
        #     nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        # )

        # self.rgb_network_pred = nn.Sequential(
        #     nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),  # Make sure, output has correct size
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),
        # )

        self.dpc_feature_conversion = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(4096, self.crossm_vector_length),
        )

        # TODO: add second stream network which is applied to skelemotion data.
        # Reasoning:
        # Resnet 18 to interpret the block with information about the joint movements.
        # Result: A vector with 10000 features.

        # Then a linear layer to predict a representation (Or Conv1d for locality?)
        # Then Gru to predict future representations (Also account for locality?)

        # DPC result: 265 feature maps of size 6x6 (or 4x4)
        # Use a linear layer to get a vector of size 10000 (alternative:flatten and make sure sk pred length fits.)

        # DPC then uses a very simple prediction network to get a representation
        # Then ConvGru to predict future representations.

        # We then compare the future representation of one network with the representation of the other network.

        # TODO: Revise
        # A standard Resnet uses 3 input channels.
        # self.sk_backbone_mag = tm.ResNet(tm.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)  # Resnet 18
        # self.sk_backbone_ori = tm.ResNet(tm.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)

        self.skele_motion_backbone = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1),
            #nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            #Print(),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=self.crossm_vector_length),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),
            # nn.Softmax(dim=1)
        )

        # self.sk_network_pred = nn.Sequential(
        #     nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),  # Make sure, output has correct size
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),
        #    # Make sure, output has correct size (vector length 10000)
        # )

        # Output of backbone are 2 vectors each length 1000
        # Concatenate and forward with GRU
        # Split vectors in two single vectors and use prediction network.
        # Alternativ: another conv GRU
        # self.sk_agg = torch.nn.GRU(input_size=self.crossm_vector_length, hidden_size=self.crossm_vector_length, num_layers=self.param['num_layers'])

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        # self._initialize_weights(self.rgb_agg)
        # self._initialize_weights(self.sk_agg)
        # self._initialize_weights(self.sk_network_pred)
        # self._initialize_weights(self.rgb_network_pred)
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

        feature = self.dpc_feature_conversion(feature)

        return feature

    def forward(self, block_rgb, block_sk):
        # TODO: forward skelemotion data seperately on second stream.
        B = block_rgb.shape[0]

        pred_sk = self._forward_sk(block_sk)  # (B, N, D)
        pred_rgb = self._forward_rgb(block_rgb)  # (B, N, D)

        # (B, N, D) = pred_sk.shape

        # The score is now calculated according to the other modality. for this we calculate the dot product of the feature vectors:
        pred_rgb = pred_rgb.contiguous() # .view(B*N, D)
        pred_sk = pred_sk.contiguous() # .view(B*N, D)

        score = torch.matmul(pred_sk, pred_rgb.transpose(0, 1))

        return score

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
