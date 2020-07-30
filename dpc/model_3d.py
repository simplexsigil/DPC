import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../backbone')
from select_backbone import select_resnet


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


def get_debug_hook_grad(name):
    def debug_hook_grad(grad):
        print("Debug hook grad {}\n"
              "Has NaN: {}\n"
              "Has inf: {}\n"
              "Has zero: {}\n"
              "Min: {}\n"
              "Max: {}\n".format(
            name,
            torch.any(torch.isnan(grad)),
            torch.any(torch.isinf(grad)),
            torch.any(grad == 0.0),
            torch.min(grad),
            torch.max(grad)
            ))

        return grad

    return debug_hook_grad


class SkeleMotionBackbone(nn.Module):
    def __init__(self, crossm_vector_length, debug=False):
        super(SkeleMotionBackbone, self).__init__()

        self.crossm_vector_length = crossm_vector_length

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1)
        # nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(32),
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(64),
        self.relu4 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1536, out_features=self.crossm_vector_length)
        self.relu5 = nn.ReLU()

        # nn.Dropout(p=0.5),
        self.linear2 = nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length)

        if debug:
            self.linear1.weight.register_hook(get_debug_hook_grad("Linear 1"))
            self.linear2.weight.register_hook(get_debug_hook_grad("Linear 2"))

    def forward(self, skele_motion_data):
        fe = self.conv1(skele_motion_data)
        fe = self.relu1(fe)

        fe = self.conv2(fe)
        fe = self.maxpool2(fe)
        fe = self.relu2(fe)

        fe = self.conv3(fe)
        fe = self.maxpool3(fe)
        fe = self.relu3(fe)

        fe = self.conv4(fe)
        fe = self.maxpool4(fe)
        fe = self.relu4(fe)

        fe = self.flatten(fe)

        fe = self.linear1(fe)
        fe = self.relu5(fe)

        fe = self.linear2(fe)

        return fe


class SkeleContrast(nn.Module):
    '''Module which performs contrastive learning by matching extracted feature
    vectors from the skele-motion skeleton representation and features extracted from RGB videos with a Resnet 18.'''

    def __init__(self, img_dim, seq_len=30, network='resnet18',
                 representation_size=1024, score_function="dot", contrast_type="cross", debug=False):
        super(SkeleContrast, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using SkeleContrast model.')
        self.score_function = score_function
        self.sample_size = img_dim
        self.seq_len = seq_len
        self.last_duration = int(math.ceil(seq_len / 4))  # This is the sequence length after using the backbone
        self.last_size = int(math.ceil(img_dim / 32))
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.contrast_type = contrast_type
        self.debug = debug

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        # torchvision.models.video.r2plus1d_18(pretrained=False, progress=True, **kwargs)

        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.crossm_vector_length = representation_size

        self.dpc_feature_conversion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, self.crossm_vector_length),
            )

        self.skele_motion_backbone = SkeleMotionBackbone(self.crossm_vector_length)

        # self.skele_motion_backbone = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1),
        #     # nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
        #     nn.MaxPool2d(3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=1536, out_features=self.crossm_vector_length),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.5),
        #     nn.Linear(in_features=self.crossm_vector_length, out_features=self.crossm_vector_length),
        #     )

        # for name, param in self.skele_motion_backbone.named_parameters():
        #     # if the param is from a linear and is a bias
        #     if "weight" in name:
        #         param.register_hook(alarm)

        self.mask = None
        self._initialize_weights(self.dpc_feature_conversion)
        self._initialize_weights(self.skele_motion_backbone)

    def _forward_sk(self, block_sk):
        (Ba, Bo, C, T, J) = block_sk.shape

        block_sk = block_sk.view(Ba * Bo, C, T, J)  # Forward each block individually like a batch input.

        features = self.skele_motion_backbone(block_sk)

        is_zero = features == 0.0

        zero_row = is_zero.all(dim=1)

        # TODO: this is a dirty hack, weights are nan if a row is scored which is completely zero, find better solution.
        features[zero_row] = 0.00000001  # This prevents the norm of the 0 vector to be nan.
        return features

    def _forward_rgb(self, block_rgb):
        # block: [B, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        (B, C, SL, H, W) = block_rgb.shape

        # For the backbone, first dimension is the batch size -> Blocks are calculated separately.
        feature = self.backbone(block_rgb)  # (B, 256, 2, 4, 4)
        del block_rgb

        # Performs average pooling on the sequence length after the backbone -> averaging over time.
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feature = feature.view(B, self.param['feature_size'], self.last_size, self.last_size)

        feature = torch.flatten(feature, start_dim=1, end_dim=-1)

        feature = self.dpc_feature_conversion(feature)

        return feature

    def forward(self, block_rgb, block_sk, mem_vid=None, mem_sk=None, mem_vid_cont=None, mem_sk_cont=None):
        # block_rgb: (B, C, SL, W, H) Batch, Channels, Seq Len, Height, Width
        # block_sk: (Ba, Bo, C, T, J) Batch, Bodies, Channels, Timestep, Joint

        if self.debug:
            assert not (torch.any(torch.isnan(block_rgb)) or torch.any(torch.isinf(block_rgb)))
            assert not (torch.any(torch.isnan(block_sk)) or torch.any(torch.isinf(block_sk)))

            if mem_vid is not None:
                assert not (torch.any(torch.isnan(mem_vid)) or torch.any(torch.isinf(mem_vid)))
                assert not (torch.any(torch.isnan(mem_sk)) or torch.any(torch.isinf(mem_sk)))

        bs = block_rgb.shape[0]

        pred_rgb = self._forward_rgb(block_rgb)
        pred_sk = self._forward_sk(block_sk)

        # The score is now calculated according to the other modality. for this we calculate the dot product of the feature vectors:
        pred_rgb = pred_rgb.contiguous()
        pred_sk = pred_sk.contiguous()

        pred_rgb = pred_rgb / torch.norm(pred_rgb, dim=1, keepdim=True)
        pred_sk = pred_sk / torch.norm(pred_sk, dim=1, keepdim=True)

        if (mem_vid is None):
            # This uses batchwise contrast.
            score = SkeleContrast.pairwise_scores(x=pred_sk, y=pred_rgb, matching_fn=self.score_function)

            targets = list(range(len(score)))

            return score, torch.LongTensor(targets).to(block_rgb.device), pred_rgb, pred_sk
        else:
            score_rgb_to_sk, score_sk_to_rgb = SkeleContrast.memory_contrast_scores(x=pred_rgb, y=pred_sk,
                                                                                    x_mem=mem_vid, y_mem=mem_sk,
                                                                                    x_cont=mem_vid_cont,
                                                                                    y_cont=mem_sk_cont,
                                                                                    matching_fn=self.score_function,
                                                                                    contrast_type=self.contrast_type)

            # score = torch.cat((score_rgb_to_sk, score_sk_to_rgb), dim=0)

            if mem_vid_cont.shape[0] > 0:
                targets_rgb_to_sk = torch.tensor([0] * bs)
                targets_sk_to_rgb = torch.tensor([0] * bs)
            else:
                targets_rgb_to_sk = torch.tensor(list(range(bs)))
                targets_sk_to_rgb = torch.tensor(list(range(bs)))

            targets_rgb_to_sk = torch.LongTensor(targets_rgb_to_sk).to(block_rgb.device)
            targets_sk_to_rgb = torch.LongTensor(targets_sk_to_rgb).to(block_rgb.device)

            return {"rgb_to_sk": score_sk_to_rgb, "sk_to_rgb": score_rgb_to_sk}, \
                   {"rgb_to_sk": targets_rgb_to_sk, "sk_to_rgb": targets_sk_to_rgb}, pred_rgb, pred_sk

    @staticmethod
    def memory_contrast_scores(x: torch.Tensor,
                               y: torch.Tensor,
                               x_mem: torch.Tensor,
                               y_mem: torch.Tensor,
                               x_cont: torch.Tensor,
                               y_cont: torch.Tensor,
                               matching_fn: str,
                               use_current_contrast=True,
                               temp_tao=0.1,
                               contrast_type="cross") -> (torch.Tensor, torch.Tensor):
        if matching_fn == "cos-nt-xent":
            # Implement memory contrast
            # We always set the first vector to be the ground truth.

            results_x = []
            results_y = []

            batch_size = x.shape[0]

            if use_current_contrast:
                if x_cont.shape[0] > 0:
                    # To avoid batch contrast
                    x_calc = x_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)
                    y_calc = y_cont.unsqueeze(dim=0).repeat(batch_size, 1, 1)

                    x_tp = x_mem.view((batch_size, 1, -1))
                    y_tp = y_mem.view((batch_size, 1, -1))

                    x_calc = torch.cat((x_tp, x_calc), dim=1)
                    y_calc = torch.cat((y_tp, y_calc), dim=1)
                else:
                    x_calc = x.unsqueeze(dim=0).repeat(batch_size, 1, 1)
                    y_calc = y.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            else:
                x_calc = torch.cat((x_mem, x_cont))
                y_calc = torch.cat((y_mem, y_cont))

            for i in range(batch_size):
                # The scores are calculated between the output of one modality and the output
                # of the other modality. The first vectors are the ground truth (other modality).
                if contrast_type == "cross":
                    scores_x_i = SkeleContrast.pairwise_scores(x[i].view((1, -1)), y_calc[i], matching_fn=matching_fn)
                    scores_y_i = SkeleContrast.pairwise_scores(y[i].view((1, -1)), x_calc[i], matching_fn=matching_fn)
                elif contrast_type == "self":
                    scores_x_i = SkeleContrast.pairwise_scores(x[i].reshape((1, -1)), x_mem, matching_fn=matching_fn)
                    scores_y_i = SkeleContrast.pairwise_scores(y[i].reshape((1, -1)), y_mem, matching_fn=matching_fn)
                else:
                    raise ValueError

                results_x.append(scores_x_i)
                results_y.append(scores_y_i)

            results_x = torch.cat(results_x, dim=0)
            results_y = torch.cat(results_y, dim=0)

            return results_x, results_y
        else:
            raise ValueError

    @staticmethod
    def pairwise_scores(x: torch.Tensor,
                        y: torch.Tensor,
                        matching_fn: str,
                        temp_tao=1.) -> torch.Tensor:
        """Efficiently calculate pairwise distances (or other similarity scores) between
        two sets of samples.
        # Arguments
            x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
            y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
            matching_fn: Distance metric/similarity score to compute between samples
            temp_tao: A temperature parameter as used for example with NT-Xent loss (Normalized Temp. Scaled Cross Ent.)
        """
        n_x = x.shape[0]
        n_y = y.shape[0]

        eps = 1e-7

        if matching_fn == 'l2':
            distances = (
                    x.unsqueeze(1).expand(n_x, n_y, -1) -
                    y.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)
            return distances
        elif matching_fn == 'cosine':
            normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
            normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

            expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

            cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
            return 1 - cosine_similarities
        elif matching_fn == 'dot':
            return torch.matmul(x, y.transpose(0, 1))

        elif matching_fn == "cos-nt-xent":
            x_norm = x / torch.norm(x, dim=1, keepdim=True)
            y_norm = y / torch.norm(y, dim=1, keepdim=True)

            xy_n = torch.matmul(x_norm, y_norm.transpose(0, 1))
            xy_nt = xy_n / temp_tao

            return xy_nt

        elif matching_fn == "orth-nt-xent":
            x_norm = x / torch.norm(x, dim=1, keepdim=True)
            y_norm = y / torch.norm(y, dim=1, keepdim=True)
            xy_n = torch.matmul(x_norm, y_norm.transpose(0, 1))
            xy_nt = torch.acos(xy_n) / temp_tao

            return xy_nt

        elif matching_fn == "euc-nt-xent":
            x_norm = x / torch.norm(x, dim=1, keepdim=True)
            y_norm = y / torch.norm(y, dim=1, keepdim=True)

            x_sq = torch.sum((x_norm * x_norm), dim=1, keepdim=True)
            y_sq = torch.sum((y_norm * y_norm), dim=1, keepdim=True)

            y_sq = y_sq.transpose(0, 1)

            score = torch.matmul(x_norm, y_norm.transpose(0, 1))
            dst = torch.nn.functional.relu(x_sq - 2 * score + y_sq)

            '''
            eps_t = torch.full(d.shape, 1e-12)
            is_zero = d.abs().lt(1e-12)

            dst = torch.where(is_zero, eps_t, d)
            '''

            dst = torch.sqrt(dst)

            return -dst / temp_tao

        else:
            raise (ValueError('Unsupported similarity function'))

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and len(param.shape) > 1:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


if __name__ == '__main__':
    a = torch.randn((40, 512))
    b = torch.randn((40, 512))

    print("Checking distance for same input matrix:")
    d = SkeleContrast.pairwise_euc_dist(a, a)
    for i in range(len(a)):
        for j in range(len(a)):
            dist = torch.norm(a[i] - a[j])
            print("d[{},{}]: {}".format(i, j, math.fabs(d[i, j] - dist)))

    print("Checking distance for different input matrix:")
    d = SkeleContrast.pairwise_euc_dist(a, b)

    for i in range(len(a)):
        for j in range(len(b)):
            dist = torch.norm(a[i] - b[j])
            print("d[{},{}]: {}".format(i, j, math.fabs(d[i, j] - dist)))

    print("Checking distance for different input matrix:")
    d = SkeleContrast.pairwise_euc_dist_changed(a, b)

    d0 = torch.zeros(d.shape)

    for i in range(len(a)):
        for j in range(len(b)):
            dist = torch.norm(a[i] - b[j])
            d0[i, j] = dist
            print("d[{},{}]: {}".format(i, j, math.fabs(d[i, j] - dist)))

    import time

    tic1 = time.perf_counter()
    d1 = SkeleContrast.pairwise_euc_dist_changed(a, b)
    tic2 = time.perf_counter()
    d2 = -SkeleContrast.pairwise_scores(a, b, matching_fn="nt-euclidean")
    tic3 = time.perf_counter()

    print("{:.10f}".format(tic2 - tic1))
    print("{:.10f}".format(tic3 - tic2))
    print((tic3 - tic2) / (tic2 - tic1))

    print(d1 - d0)

    print(d2 - d0)
