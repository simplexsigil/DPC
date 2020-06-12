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
    '''This module computes contrastive scores between a skeleton representation using the skele-motion representation and the rgb video clip as handled
    in DPC. First, both, the information blocks for the skeleton representation and the information blocks for the rgb representation are forwarded in two separate streams.
    DPC uses a 2D3D resnet as backbone, the skeleton network only a small CNN. Both networks then use a ConvGru to predict the future.
    In the end, two small fully connected networks are used to generate representation vectors for contrastive learning.'''

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

        self.crossm_vector_length = 100

        self.rgb_agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])

        self.rgb_network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        )

        self.rgb_to_common = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.param['feature_size'] * self.last_size**2,
                      out_features=self.crossm_vector_length),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length)
        )

        # TODO: Would it benefit using a resnet here? Or just more layers?
        self.skele_motion_backbone = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 5), stride=1),
            # nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 5)),
            # nn.MaxPool2d(3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 5)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
        )

        self.sk_cnn_out_shape = (256, 10, 11)

        self.sk_agg = ConvGRU(input_size=self.param['feature_size'],
                              hidden_size=self.param['hidden_size'],
                              kernel_size=1,
                              num_layers=self.param['num_layers'])

        self.sk_network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        )

        self.sk_to_common = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.sk_cnn_out_shape[1] * self.sk_cnn_out_shape[1] * self.sk_cnn_out_shape[2],
                      out_features=self.crossm_vector_length),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length)
        )

        self.mask = None
        self.relu = nn.ReLU(inplace=False)

        self._initialize_weights(self.rgb_agg)
        self._initialize_weights(self.sk_agg)
        self._initialize_weights(self.sk_network_pred)
        self._initialize_weights(self.rgb_network_pred)
        self._initialize_weights(self.dpc_feature_conversion)
        self._initialize_weights(self.skele_motion_backbone)

    def _forward_sk(self, block_sk):
        (B, N, C, T, J) = block_sk.shape

        block_sk = block_sk.view(B * N, C, T, J)  # Forward each block individually like a batch input.

        feature = self.skele_motion_backbone(block_sk)

        # The individually forwarded blocks are aggregated into batches again. self.param['feature_size'] are feature maps?
        feature_inf_all = feature.view(B, N, self.skele_motion_last_height,
                                       self.skele_motion_last_width)  # before ReLU, (-inf, +inf)

        feature_inf = feature_inf_all[:, N - self.pred_step::,
                      :].contiguous()  # The input features after the backbone to be predicted.

        feature = self.relu(feature)  # [0, +inf)

        feature = feature.view(B, N, self.skele_motion_last_height,
                               self.skele_motion_last_width)  # [B,N,D,6,6], [0, +inf)

        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(
            feature[:, 0:N - self.pred_step, :].contiguous())  # Apply GRU on the features to predict future.
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        # Predict next time step from hidden vector, then predict next hidden vector from time step and hidden vector.
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        del hidden

        return pred, feature_inf

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

        # The individually forwarded blocks are aggregated into batches again. self.param['feature_size'] are feature maps?
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size,
                                       self.last_size)  # before ReLU, (-inf, +inf)

        feature_inf = feature_inf_all[:, N - self.pred_step::,
                      :].contiguous()  # The input features after the backbone to be predicted.

        feature = self.relu(feature)  # [0, +inf)

        feature = feature.view(B, N, self.param['feature_size'], self.last_size,
                               self.last_size)  # [B,N,D,6,6], [0, +inf)

        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(
            feature[:, 0:N - self.pred_step, :].contiguous())  # Apply GRU on the features to predict future.
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        # Predict next time step from hidden vector, then predict next hidden vector from time step and hidden vector.
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        del hidden

        return pred, feature_inf

    def forward(self, block_sk, block_rgb):
        pred_rgb, feature_inf_rgb = self._forward_sk(block_sk)  # (B, N, D)
        pred_sk, feature_inf_sk = self._forward_rgb(block_rgb)  # (B, N, D)

        (B, N, D) = pred_sk.shape

        # Project everything to a single feature vector format, so similarity / contrast can be calculated.
        pred_rgb, feature_inf_rgb = self.rgb_to_common(pred_rgb), self.rgb_to_common(feature_inf_rgb)
        pred_sk, feature_inf_sk = self.sk_to_common(pred_sk), self.sk_to_common(feature_inf_sk)

        # TODO: multiply pred with GT of other modality and adapt mask.

        # The score is now calculated according to the other modality. for this we calculate the dot product of the feature vectors:
        # pred_rgb, feature_inf_rgb = pred_rgb.contiguous().view(B*N, D)
        # pred_sk, feature_inf_sk = pred_sk.contiguous().view(B*N, D)

        # score = torch.matmul(pred_sk, pred_rgb.transpose(0, 1)).view(B, N, B, N)

        # feature_inf_rgb = feature_inf_rgb.contiguous().view(B*N,D).transpose(0, 1)
        # feature_inf_sk = feature_inf_sk.contiguous().view(B*N,D).transpose(0, 1)

        # score_rgb_sk = torch.matmul(pred_rgb, feature_inf_sk).view(B, N, B, N)
        # score_sk_rgb = torch.matmul(pred_sk, feature_inf_rgb).view(B, N, B, N)

        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # Ground Truth: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        # permute: get dimensionality to end.
        # view: 2D tensor: only make sure that each output feature has its own vector, now matter which seq step, batch idx or location.
        pred_rgb = pred_rgb.permute(0, 1, 3, 4, 2).contiguous().view(B * self.pred_step * self.last_size ** 2,
                                                                     self.param['feature_size'])

        # Same thing for feature_inf = ground truth: only: transpose matrix to calculate dot product of predicted and ground truth features.
        feature_inf_rgb = feature_inf_rgb.permute(0, 1, 3, 4, 2).contiguous().view(B * N * self.last_size ** 2,
                                                                                   self.param[
                                                                                       'feature_size']).transpose(0, 1)

        # First: make scoring by mat mul.
        # Second: Each feature vector of length D has been multiplied with its ground truth and all others.
        # Input had size B*N*LS*LS x D and D X B*N*LS*LS. Output has Size B*N*LS*LS x B*N*LS*LS
        # Third: Output now has shape (B, N, LS*LS, B, N, LS*LS)
        score = torch.matmul(pred_rgb, feature_inf_rgb).view(B, self.pred_step, self.last_size ** 2, B, N,
                                                             self.last_size ** 2)
        del feature_inf_rgb, pred_rgb

        # Given scoring Matrix of shape B*N*LS*LS x B*N*LS*LS
        # Positive: 3D diagonal
        if self.mask is None:  # only compute mask once
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
                               requires_grad=False).detach().cuda()

            # If two indexing arrays can be broadcast to have the same shape, they are used elementwise as index pairs.
            # If it is the same batch index: All negatives are spacial negatives. This includes the temporal negatives and positives at this point.
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg

            # For each matching batch index:
            # Take all dimension features which go together (this makes it a subset of the case above) and set the temporal negative (includes positives)
            for k in range(B):
                mask[k, :, torch.arange(self.last_size ** 2), k, :,
                torch.arange(self.last_size ** 2)] = -1  # temporal neg

            # Now changing the approach by permutating the matrix around: (B, LS*LS, N,  B, LS*LS, N)
            # Then: (B*LS*LS, N,  B*LS*LS, N)
            # For each element: its accompanying future prediction is a positive. Example. [0,1] and [3,4]
            # This means that the beginning is always used to predict the end, it does not predict everything after pred step.
            # Is that really what it is supposed to do? Maybe does not matter a lot, but why not predict the whole sequence until the end?
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.pred_step,
                                                                   B * self.last_size ** 2, N)
            for j in range(B * self.last_size ** 2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos

            # (B, LS * LS, N, B, LS * LS, N)
            mask = tmp.view(B, self.last_size ** 2, self.pred_step, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)

            # (B, N, LS * LS, B, N, LS * LS)
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
