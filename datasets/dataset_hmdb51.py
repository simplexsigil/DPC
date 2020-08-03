import os
import sys

import pandas as pd
import torch
from torch.utils import data

sys.path.insert(0, '../datasets')

from dataset import DatasetUtils


class HMDB51Dataset(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=30,
                 downsample_vid=1,
                 epsilon=5,
                 which_split=1,
                 max_samples=None,
                 random_state=42):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample_vid = downsample_vid
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = os.path.expanduser('~/datasets/HMDB51/split/train_split%02d.csv' % self.which_split)
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = os.path.expanduser(
                '~/datasets/HMDB51/split/test_split%02d.csv' % self.which_split)  # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join(os.path.expanduser('~/datasets/HMDB51/'), 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.seq_len <= 0:
                drop_idx.append(idx)

        print(f"In mode {self.mode}, {len(drop_idx)} of {len(video_info)} videos had to be dropped (too short).\n"
              f"Remaining dataset size in mode {self.mode}: {len(video_info) - len(drop_idx)}")

        self.video_info = video_info.drop(drop_idx, axis=0)

        if max_samples is not None:
            self.video_info = self.video_info.sample(max_samples, random_state=random_state)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3)  # TODO: This makes no sense with splits.
        # shuffle not required

    def __getitem__(self, index):
        dsu = DatasetUtils
        vpath, vlen = self.video_info.iloc[index]

        if self.mode == "train" or self.mode == "val":
            frame_idxs = dsu.idx_sampler(vlen, self.seq_len, vpath)
        elif self.mode == "test":
            frame_idxs = list(range((vlen // self.seq_len) * self.seq_len))  # Test on whole video. Get rid of offcut.
        else:
            raise ValueError

        frame_idxs_vid = frame_idxs[::self.downsample_vid]

        seq = [DatasetUtils.pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in frame_idxs_vid]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(len(frame_idxs_vid), C, H, W).transpose(0, 1)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)
            print("Unexpected")

        label = torch.tensor(vid, dtype=torch.long)

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
