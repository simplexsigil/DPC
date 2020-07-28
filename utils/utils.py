import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')
from collections import deque
from torchvision import transforms
import math


def save_checkpoint(state, is_best=0, model_path='models/checkpoint.pth.tar'):
    model_last = os.path.join(model_path, 'model_last.pth.tar')

    torch.save(state, model_last + ".tmp")  # This way there is always a valid file.
    torch.save(state, model_last)

    os.remove(model_last + ".tmp")

    if is_best:
        model_best = os.path.join(model_path, 'model_best.pth.tar')

        torch.save(state, model_best + ".tmp")  # This way there is always a valid file.
        torch.save(state, model_best)

        os.remove(model_best + ".tmp")


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    total = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / total))
    return res


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, locality=5):
        self.locality = locality
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0):
        if val is None:
            self.val = None
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if self.locality > 0:
            self.local_history.append(val)
            if len(self.local_history) > self.locality:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''

    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count': 0, 'correct': 0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                  % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


class ConfusionMeter(object):
    '''compute and show confusion matrix'''

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p, t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
                   cmap=plt.cm.jet,
                   interpolation=None,
                   extent=(0.5, np.shape(self.mat)[0] + 0.5, np.shape(self.mat)[1] + 0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y + 1, x + 1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i + 1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i + 1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))


def random_image_crop_square(min_area_n=0.4, max_area_n=1, image_width=150, image_height=150):
    """
    This follows the conventions of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.Crop
    Especially considering the meaning of crop_pos_x_norm and crop_pos_y_norm.
    """
    image_shorter = min(image_height, image_width)
    image_longer = max(image_height, image_width)

    # First find square crop length.
    total_area = image_width * image_height

    min_crop_length = math.ceil(math.sqrt(min_area_n * total_area))
    max_crop_length = math.floor(math.sqrt(max_area_n * total_area))

    min_crop_length = min(max(min_crop_length, 1.), image_shorter)
    max_crop_length = min(max_crop_length, image_shorter)

    crop_length = np.random.uniform(min_crop_length, max_crop_length)

    # Second, find upper left corner position. Normal distributed around center.
    crop_pos_x_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.), 1.)  # Normal distributed between 0 and 1.
    crop_pos_y_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.), 1.)  # Normal distributed between 0 and 1.

    crop_length_x = crop_length
    crop_length_y = crop_length

    return crop_length_x, crop_length_y, crop_pos_x_norm, crop_pos_y_norm
