import math
import sys

from resnet_2d3d import neq_load_customized

sys.path.append('../backbone')
from select_backbone import select_resnet
from convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LC(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len,
                 network='resnet18', dropout=0.5, num_class=101):
        super(LC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class
        print('=> Using RNN + FC model ')

        print('=> Use 2D-3D %s!' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True

        self.backbone, self.param = select_resnet(network, track_running_stats=track_running_stats)
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        print('=> using ConvRNN, kernel_size = 1')
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self._initialize_weights(self.agg)

        self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(self.param['feature_size'], self.num_class))
        self._initialize_weights(self.final_fc)

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size,
                               self.last_size)  # [B*N,D,last_size,last_size]
        context, _ = self.agg(feature)
        context = context[:, -1, :].unsqueeze(1)
        context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
        del feature

        context = self.final_bn(context.transpose(-1, -2)).transpose(-1,
                                                                     -2)  # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        output = self.final_fc(context).view(B, -1, self.num_class)

        return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
                # other resnet weights have been initialized in resnet_3d.py


class Resnet18Classifier(nn.Module):
    def __init__(self, sample_size, seq_len, network='resnet18', dropout=0.5, num_class=101,
                 crossm_vector_length=512):
        super(Resnet18Classifier, self).__init__()

        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.seq_len = seq_len
        self.num_class = num_class
        self.crossm_vector_length = crossm_vector_length
        print('=> Feature vector of backbone + Linear Layer')

        print('=> Use 2D-3D %s!' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True

        self.backbone, self.param = select_resnet(network, track_running_stats=track_running_stats)

        self.dpc_feature_conversion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, self.crossm_vector_length),
            )

        self.final_bn_ev = nn.BatchNorm1d(self.crossm_vector_length)
        self.final_bn_ev.weight.data.fill_(1)
        self.final_bn_ev.bias.data.zero_()

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.crossm_vector_length, self.crossm_vector_length),
            nn.ReLU(),
            nn.Linear(self.crossm_vector_length, self.num_class)
            )

        self._initialize_weights(self.dpc_feature_conversion)
        self._initialize_weights(self.final_fc)

    def forward(self, block):
        # block: [B, N, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, C, SL, H, W) = block.shape

        # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        feature = self.backbone(block)  # (B * N, 256, 2, 4, 4)
        del block

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feature = feature.view(B, self.param['feature_size'], self.last_size, self.last_size)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)

        feature = self.dpc_feature_conversion(feature)

        # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        feature = self.final_bn_ev(feature)
        output = self.final_fc(feature).view(B, self.num_class)

        return output

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
                # other resnet weights have been initialized in resnet_3d.py

    def load_weights_state_dict(self, state_dict, model=None):
        neq_load_customized((self if model is None else model), state_dict, ignore_layer=".*module.final_bn.*")


class R2plus1DClassifier(nn.Module):
    def __init__(self, sample_size, seq_len, backbone='r2+1d18', dropout=0.5, num_class=101, representation_size=512):
        super(R2plus1DClassifier, self).__init__()

        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size  # TODO: Remove if not required.
        self.seq_len = seq_len
        self.num_class = num_class
        self.representation_size = representation_size

        if "r2+1d" in backbone:
            if backbone == "r2+1d18":
                self.backbone = torchvision.models.video.r2plus1d_18(pretrained=False, num_classes=representation_size)
            else:
                raise ValueError

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.representation_size, self.num_class)
            )

        R2plus1DClassifier._initialize_weights(self.final_fc)

    def forward(self, block):
        (B, C, SL, H, W) = block.shape  # block: [B, C, SL, W, H] Batch, Channels, Seq Len, Width Height

        feature = self.backbone(block)
        del block

        output = self.final_fc(feature).view(B, self.num_class)

        return output

    @staticmethod
    def _initialize_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_weights_state_dict(self, state_dict, model=None):
        neq_load_customized((self if model is None else model), state_dict, ignore_layer=".*module.final_bn.*")
