import math
import sys

from resnet_2d3d import neq_load_customized

sys.path.append('../backbone')
from select_backbone import select_resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DPCResnetClassifier(nn.Module):
    def __init__(self,
                 img_dim,
                 seq_len=30,
                 downsampling=1,
                 vid_backbone='resnet18',
                 representation_size=512,
                 hidden_width=512,
                 num_class=101,
                 dropout=0.5,
                 classification_tapping=0,
                 backbone_naming="vid_backbone"):
        super(DPCResnetClassifier, self).__init__()

        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(666)

        print("============Model================")
        print('Using DPCResnet Classifier model.')

        self.img_dim = img_dim
        self.seq_len = seq_len // downsampling
        self.representation_size = representation_size
        self.vid_backbone_name = vid_backbone
        self.hidden_width = hidden_width
        self.classification_tapping = classification_tapping
        self.backbone_naming = backbone_naming

        self.num_class = num_class

        self.last_duration = int(math.ceil(self.seq_len / 4))  # This is the sequence length after using the backbone
        self.last_size = int(math.ceil(self.img_dim / 32))  # Final feature map has size (last_size, last_size)

        track_running_stats = True

        if self.backbone_naming == "vid_backbone":
            self.vid_backbone, self.param = select_resnet(self.vid_backbone_name,
                                                          track_running_stats=track_running_stats)
        elif self.backbone_naming == "backbone":
            self.backbone, self.param = select_resnet(self.vid_backbone_name, track_running_stats=track_running_stats)

        if classification_tapping > -3:
            self.vid_fc1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(4096, self.hidden_width),
                )

            _initialize_weights(self.vid_fc1)

        if classification_tapping > -2:
            self.vid_fc2 = nn.Sequential(
                nn.BatchNorm1d(self.hidden_width),
                nn.ReLU(),
                nn.Linear(self.hidden_width, self.hidden_width),
                )

            _initialize_weights(self.vid_fc2)

        if classification_tapping > -1:
            self.vid_fc_rep = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_width, self.representation_size),
                )

            _initialize_weights(self.vid_fc_rep)

        if -3 < classification_tapping < 0:
            in_size_classifier = self.hidden_width
        elif classification_tapping < -3:
            in_size_classifier = 4096
        else:
            in_size_classifier = self.representation_size

        self.final_fc_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_size_classifier, self.num_class)
            )

        _initialize_weights(self.final_fc_classifier)

        print("=================================")

    def forward(self, block):
        # block: [B, N, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, C, SL, H, W) = block.shape

        # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        if self.backbone_naming == "vid_backbone":
            feature = self.vid_backbone(block)  # (B * N, 256, 2, 4, 4)
        elif self.backbone_naming == "backbone":
            feature = self.backbone(block)  # (B * N, 256, 2, 4, 4)

        del block

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feature = feature.view(B, self.param['feature_size'], self.last_size, self.last_size)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)

        if self.classification_tapping > -3:
            feature = self.vid_fc1(feature)
        if self.classification_tapping > -2:
            feature = self.vid_fc2(feature)
        if self.classification_tapping > -1:
            feature = self.vid_fc_rep(feature)

        output = self.final_fc_classifier(feature).view(B, self.num_class)

        return output

    def load_weights_state_dict(self, state_dict, model=None):
        neq_load_customized((self if model is None else model), state_dict)


class R2plus1DClassifier(nn.Module):
    def __init__(self,
                 backbone='r2+1d18',
                 dropout=0.5,
                 num_class=101,
                 representation_size=512,
                 hidden_fc_width=512,
                 classification_tapping=0):
        super(R2plus1DClassifier, self).__init__()

        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(666)
        self.num_class = num_class
        self.representation_size = representation_size
        self.hidden_width = hidden_fc_width
        self.classification_tapping = classification_tapping

        if "r2+1d" in backbone:
            if backbone == "r2+1d18":
                self.vid_backbone = torchvision.models.video.r2plus1d_18(pretrained=False, num_classes=hidden_fc_width)
            else:
                raise ValueError

        # The first linear layer is part of the R(2+1)D architecture
        if classification_tapping > -2:
            self.vid_fc2 = nn.Sequential(
                nn.BatchNorm1d(self.hidden_width),
                nn.ReLU(),
                nn.Linear(self.hidden_width, self.hidden_width),
                )

            R2plus1DClassifier._initialize_weights(self.vid_fc2)

        if classification_tapping > -1:
            self.vid_fc_rep = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_width, self.representation_size),
                )

            R2plus1DClassifier._initialize_weights(self.vid_fc_rep)

        if -3 < classification_tapping < 0:
            in_size_classifier = self.hidden_width
        elif classification_tapping < -3:
            in_size_classifier = 4096
        else:
            in_size_classifier = self.representation_size

        self.final_fc_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_size_classifier, self.num_class)
            )

        R2plus1DClassifier._initialize_weights(self.final_fc_classifier)

    def forward(self, block):
        (B, C, SL, H, W) = block.shape  # block: [B, C, SL, W, H] Batch, Channels, Seq Len, Width Height

        feature = self.vid_backbone(block)
        del block

        if self.classification_tapping > -2:
            feature = self.vid_fc2(feature)
        if self.classification_tapping > -1:
            feature = self.vid_fc_rep(feature)

        output = self.final_fc_classifier(feature).view(B, self.num_class)

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
        neq_load_customized((self if model is None else model), state_dict)


def _initialize_weights(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name and len(param.shape) > 1:
            nn.init.orthogonal_(param, 1)
