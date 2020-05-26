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
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)  # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        del block

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        # The individually forwarded blocks are aggregated into batches again. self.param['feature_size'] are feature maps?
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)

        feature = self.relu(feature) # [0, +inf)

        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()  # The input features after the backbone to be predicted.
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())  # Apply GRU on the features to predict future.
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        
        pred = []
        # Predict next time step from hidden vector, then predict next hidden vector from time step and hidden vector.
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        del hidden

        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # Ground Truth: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        # permute: get dimensionality to end.
        # view: 2D tensor: only make sure that each output feature has its own vector, now matter which seq step, batch idx or location.
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])

        # Same thing for feature_inf = ground truth: only: transpose matrix to calculate dot product of predicted and ground truth features.
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)

        # First: make scoring by mat mul.
        # Second: Each feature vector of length D has been multiplied with its ground truth and all others.
        # Input had size B*N*LS*LS x D and D X B*N*LS*LS. Output has Size B*N*LS*LS x B*N*LS*LS
        # Third: Output now has shape (B, N, LS*LS, B, N, LS*LS)
        score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        del feature_inf, pred

        # Given scoring Matrix of shape B*N*LS*LS x B*N*LS*LS
        # Positive: 3D diagonal
        if self.mask is None: # only compute mask once
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()


            # If two indexing arrays can be broadcast to have the same shape, they are used elementwise as index pairs.
            # If it is the same batch index: All negatives are spacial negatives. This includes the temporal negatives and positives at this point.
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg

            # For each matching batch index:
            # Take all dimension features which go together (this makes it a subset of the case above) and set the temporal negative (includes positives)
            for k in range(B):
                mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg

            # Now changing the approach by permutating the matrix around: (B, LS*LS, N,  B, LS*LS, N)
            # Then: (B*LS*LS, N,  B*LS*LS, N)
            # For each element: its accompanying future prediction is a positive. Example. [0,1] and [3,4]
            # This means that the beginning is always used to predict the end, it does not predict everything after pred step.
            # Is that really what it is supposed to do? Maybe does not matter a lot, but why not predict the whole sequence until the end?
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
            for j in range(B*self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos

            # (B, LS * LS, N, B, LS * LS, N)
            mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)

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

