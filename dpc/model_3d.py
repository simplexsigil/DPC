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

        # self.agg = ConvGRU(input_size=self.param['feature_size'],
        #                   hidden_size=self.param['hidden_size'],
        #                   kernel_size=1,
        #                   num_layers=self.param['num_layers'])

        self.rgb_agg = torch.nn.GRU(input_size=2000, hidden_size=2000, num_layers=self.param['num_layers'])

        # self.rgb_network_pred = nn.Sequential(
        #     nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        # )

        self.rgb_network_pred = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000),  # Make sure, output has correct size
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2000, out_features=2000),
        )

        self.dpc_feature_conversion = nn.Sequential(
            nn.Linear(4096, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),
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
        self.sk_backbone_mag = tm.ResNet(tm.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)  # Resnet 18
        self.sk_backbone_ori = tm.ResNet(tm.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)

        self.sk_network_pred = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000),  # Make sure, output has correct size
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2000, out_features=2000),
            # Make sure, output has correct size (vector length 10000)
        )

        # Output of backbone are 2 vectors each length 1000
        # Concatenate and forward with GRU
        # Split vectors in two single vectors and use prediction network.
        # Alternativ: another conv GRU
        self.sk_agg = torch.nn.GRU(input_size=2000, hidden_size=2000, num_layers=self.param['num_layers'])

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.rgb_agg)
        self._initialize_weights(self.sk_agg)
        self._initialize_weights(self.sk_network_pred)
        self._initialize_weights(self.rgb_network_pred)
        self._initialize_weights(self.dpc_feature_conversion)

    def _forward_sk(self, block_sk):
        (B, N, C, T, J) = block_sk.shape

        block_sk = block_sk.view(B * N, C, T, J)  # Forward each block individually like a batch input.

        block_ori = block_sk[:, 0:3, :, :]
        block_mag = block_sk[:, 3:, :, :]

        feature_ori = self.sk_backbone_ori(block_ori)
        feature_mag = self.sk_backbone_mag(block_mag)

        # feature_ori = F.avg_pool2d(feature_ori) No average pooling, since we do not have dense features in this stream.

        feature_ori = feature_ori.view(B, N, 1000)
        feature_mag = feature_mag.view(B, N, 1000)

        feature_com = torch.cat((feature_ori, feature_mag), -1)

        feature_com = feature_com.transpose(0, 1)  # GRUs take (N, B, features)

        # feature_ori_inf = feature_ori[:, N - self.pred_step::, :].contiguous()
        # feature_mag_inf = feature_mag[:, N - self.pred_step::, :].contiguous()

        feature_com_inf = feature_com[N - self.pred_step::, :, :].contiguous()  # Shape (N, B, 2000)

        # Apply GRU on the features to predict future.
        _, hidden = self.sk_agg(feature_com[0:N - self.pred_step, :, :].contiguous())

        hidden = hidden[-1, :, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        # Predict next time step from hidden vector, then predict next hidden vector from time step and hidden vector.
        for i in range(self.pred_step):
            # sequentially pred future

            p_tmp = self.sk_network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.sk_agg(self.relu(p_tmp).unsqueeze(0), hidden.unsqueeze(0))
            hidden = hidden[-1, :, :]

        pred = torch.stack(pred, 0)  # pred_step, B, xxx
        pred = pred.transpose(0, 1)
        del hidden

        return pred, feature_com_inf

    def _forward_rgb(self, block_rgb):
        # block: [B, N, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, N, C, SL, H, W) = block_rgb.shape
        block_rgb = block_rgb.view(B * N, C, SL, H, W)

        # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        feature = self.backbone(block_rgb)  # (B * N, 256, 4, 4)
        del block_rgb

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature = self.dpc_feature_conversion(feature.flatten(1, -1))
        feature = feature.view(B, N, 2000)

        feature = feature.transpose(0, 1)

        # The individually forwarded blocks are aggregated into batches again. self.param['feature_size'] are feature maps?
        # feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size,self.last_size)  # before ReLU, (-inf, +inf)
        # feature_inf_all = feature.view(B, N, 2000)

        # The input features after the backbone to be predicted.
        feature_inf = feature[N - self.pred_step::, :, :].contiguous()

        #del feature_inf_all

        feature = self.relu(feature)  # [0, +inf)

        # [B,N,D,6,6], [0, +inf)
        # feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)
        # feature = feature.view(B, N, 2000)

        ### aggregate, predict future ###
        _, hidden = self.rgb_agg(feature[0:N - self.pred_step, :, :].contiguous())  # Apply GRU on the features to predict future.
        hidden = hidden[-1, :, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        # Predict next time step from hidden vector, then predict next hidden vector from time step and hidden vector.
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.rgb_network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.rgb_agg(self.relu(p_tmp).unsqueeze(0), hidden.unsqueeze(0))
            hidden = hidden[-1, :, :]

        pred = torch.stack(pred, 0)  # B, pred_step, xxx
        pred = pred.transpose(0, 1)

        del hidden

        return pred, feature_inf

    def forward(self, block_rgb, block_sk):
        # TODO: forward skelemotion data seperately on second stream.
        B = block_rgb.shape[0]

        pred_sk, feature_inf_sk = self._forward_sk(block_sk)  # (B, N, D)
        pred_rgb, feature_inf_rgb = self._forward_rgb(block_rgb)  # (B, N, D)

        (B, N, D) = pred_rgb.shape
        # TODO: Now there shall be a second stream of processed skelemotion data. The ground truth for scoring is no longer the own future (after backbones).

        # The score is now calculated according to the other modality. for this we calculate the dot product of the feature vectors:
        pred_rgb = pred_rgb.contiguous().view(B*N,D)
        pred_sk = pred_sk.contiguous().view(B*N,D)

        feature_inf_rgb = feature_inf_rgb.contiguous().view(B*N,D).transpose(0, 1)
        feature_inf_sk = feature_inf_sk.contiguous().view(B*N,D).transpose(0, 1)

        score_rgb_sk = torch.matmul(pred_rgb, feature_inf_sk).view(B, N, B, N)
        score_sk_rgb = torch.matmul(pred_sk, feature_inf_rgb).view(B, N, B, N)

        if self.mask is None:
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg

            # By default: everything easy negatives (0).
            self.mask = torch.zeros((B, N, B, N), dtype=torch.int8, requires_grad=False).detach().cuda()

            # There are no spatial negatives (-3), since we are not using dense features here.

            # For each matching batch index:
            # Take all dimension features which go together (this makes it a subset of the case above) and set the temporal negative (includes positives)
            for k in range(B):
                # temporal neg
                self.mask[k, :, k, :] = -1

            # Simple positive (1) is the diagonal: The equivalent skeleton representation to a video representation.
            for j in range(B):
                self.mask[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos

            # Now changing the approach by permutating the matrix around: (B, LS*LS, N,  B, LS*LS, N)
            # Then: (B*LS*LS, N,  B*LS*LS, N)
            # For each element: its accompanying future prediction is a positive. Example. [0,1] and [3,4]
            # This means that the beginning is always used to predict the end, it does not predict everything after pred step.
            # Is that really what it is supposed to do? Maybe does not matter a lot, but why not predict the whole sequence until the end?






            # Todo: create mask for scoring.

        # ### Get similarity score ###
        # # pred: [B, pred_step, D, last_size, last_size]
        # # Ground Truth: [B, N, D, last_size, last_size]
        # N = self.pred_step
        # # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        # # permute: get dimensionality to end.
        # # view: 2D tensor: only make sure that each output feature has its own vector, now matter which seq step, batch idx or location.
        # pred_rgb = pred_rgb.permute(0, 1, 3, 4, 2).contiguous().view(B * self.pred_step * self.last_size ** 2,
        #                                                              self.param['feature_size'])
        #
        # # Same thing for feature_inf = ground truth: only: transpose matrix to calculate dot product of predicted and ground truth features.
        # feature_inf_rgb = feature_inf_rgb.permute(0, 1, 3, 4, 2).contiguous().view(B * N * self.last_size ** 2,
        #                                                                            self.param[
        #                                                                                'feature_size']).transpose(0, 1)
        #
        # # First: make scoring by mat mul.
        # # Second: Each feature vector of length D has been multiplied with its ground truth and all others.
        # # Input had size B*N*LS*LS x D and D X B*N*LS*LS. Output has Size B*N*LS*LS x B*N*LS*LS
        # # Third: Output now has shape (B, N, LS*LS, B, N, LS*LS)
        # score = torch.matmul(pred_rgb, feature_inf_rgb).view(B, self.pred_step, self.last_size ** 2, B, N,
        #                                                      self.last_size ** 2)
        # del feature_inf_rgb, pred_rgb
        #
        # # Given scoring Matrix of shape B*N*LS*LS x B*N*LS*LS
        # # Positive: 3D diagonal
        # if self.mask is None:  # only compute mask once
        #     # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
        #     mask = torch.zeros((B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
        #                        requires_grad=False).detach().cuda()
        #
        #     # If two indexing arrays can be broadcast to have the same shape, they are used elementwise as index pairs.
        #     # If it is the same batch index: All negatives are spacial negatives. This includes the temporal negatives and positives at this point.
        #     mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg
        #
        #     # For each matching batch index:
        #     # Take all dimension features which go together (this makes it a subset of the case above) and set the temporal negative (includes positives)
        #     for k in range(B):
        #         mask[k, :, torch.arange(self.last_size ** 2), k, :,
        #         torch.arange(self.last_size ** 2)] = -1  # temporal neg
        #
        #     # Now changing the approach by permutating the matrix around: (B, LS*LS, N,  B, LS*LS, N)
        #     # Then: (B*LS*LS, N,  B*LS*LS, N)
        #     # For each element: its accompanying future prediction is a positive. Example. [0,1] and [3,4]
        #     # This means that the beginning is always used to predict the end, it does not predict everything after pred step.
        #     # Is that really what it is supposed to do? Maybe does not matter a lot, but why not predict the whole sequence until the end?
        #     tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.pred_step,
        #                                                            B * self.last_size ** 2, N)
        #     for j in range(B * self.last_size ** 2):
        #         tmp[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos
        #
        #     # (B, LS * LS, N, B, LS * LS, N)
        #     mask = tmp.view(B, self.last_size ** 2, self.pred_step, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)
        #
        #     # (B, N, LS * LS, B, N, LS * LS)
        #     self.mask = mask

        return [score_rgb_sk, score_sk_rgb, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
