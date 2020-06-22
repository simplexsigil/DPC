import glob
import os
import sys

import pandas as pd
import torch
from torch.utils import data

sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
import re
from typing import List
from sklearn.model_selection import train_test_split

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label

        if big:
            print('Using Kinetics400 full data (256x256)')
        else:
            print('Using Kinetics400 full data (150x150)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/kinetics400', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=',', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        if big:
            if mode == 'train':
                split = '../process_data/data/kinetics400_256/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400_256/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else:
                raise ValueError('wrong mode')
        else:  # small
            if mode == 'train':
                split = '../process_data/data/kinetics400/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else:
                raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # Why is this sampling only 30% of the validation set?
        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3, random_state=666)

        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '../process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '../process_data/data/ucf101/test_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class NTURGBD_3D(data.Dataset):  # Todo: introduce csv selection into parse args list.
    def __init__(self,
                 split='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=3,
                 big=False,
                 return_label=False,
                 nturgbd_video_info=None,
                 skele_motion_root=None,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None):
        self.split = split
        self.split_mode = split_mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample = downsample
        self.return_label = return_label

        self.use_skeleton = skele_motion_root is not None

        print('Using nturgbd data (150x150)')
        ndu = NTURGBDDatasetUtils

        self.video_info_skeletons = {}

        self.sample_info = ndu.read_video_info(nturgbd_video_info, max_samples=sample_limit)

        v_file_count = len(self.sample_info)

        min_frame_count = self.seq_len * self.downsample

        self.sample_info = ndu.filter_too_short(self.sample_info, min_frame_count)

        sample_count = len(self.sample_info)

        print("Dropped {} of {} samples due to insufficient rgb video length ({} frames needed).".format(
            v_file_count - sample_count, v_file_count, min_frame_count))

        self.sample_info = ndu.filter_nturgbd_by_split_mode(self.sample_info, self.split, self.split_mode, split_frac)

        self.sk_info = ndu.get_skeleton_info(skele_motion_root)

        self.sample_info = ndu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)

        print("Dropped {} of {} samples due to missing skeleton information.".format(
            sample_count - len(self.sample_info), sample_count))

        print("Remaining videos in mode {}: {}".format(self.split, len(self.sample_info)))

        # The original approach always used a subset of the test set for validation. Doing the same for comparability.
        if self.split == "val":
            if len(self.sample_info) > 500:
                print(
                    "Limited the validation sample to 500 to speed up training. This does not alter the structure of the train/test/val splits, " +
                    "it only reduces the samples used for validation in training among the val split.")
                self.sample_info = self.sample_info.sample(n=500, random_state=666)
        # shuffle not necessary because use RandomSampler

    def __getitem__(self, index):
        sample = self.sample_info.iloc[index]

        frame_indices = DatasetUtils.idx_sampler(sample["frame_count"], self.seq_len, self.downsample, sample["path"])

        seq = [pil_loader(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1))) for i in frame_indices]

        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()

        # One Tensor of shape (self.num_seq * self.seq_len, C, H, W)
        t_seq = torch.stack(t_seq, 0)

        # One Tensor of shape (C, self.seq_len, H, W)
        t_seq = t_seq.view(self.seq_len, C, H, W).transpose(0, 1)

        if self.use_skeleton:
            sk_seq = NTURGBDDatasetUtils.load_skeleton_seqs(self.sk_info, sample["id"], frame_indices)
            sk_seq = torch.tensor(sk_seq, dtype=torch.float)

            # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
            (sk_Bo, sk_J, sk_T, sk_C) = sk_seq.shape

            # This is transposed, so we can split the image into blocks during training.
            sk_seq = sk_seq.transpose(1, 2)
            sk_seq = sk_seq.transpose(2, 3).transpose(1, 2)  # (sk_Bo, C, T, J)

            if self.return_label:
                label = torch.tensor([sample["action"]], dtype=torch.long)
                return t_seq, sk_seq, label

            else:
                return t_seq, sk_seq

        if self.return_label:
            label = torch.tensor([sample["action"]], dtype=torch.long)
            return t_seq, label
        else:
            return t_seq

    def __len__(self):
        return len(self.sample_info)


class NTURGB3DInputIterator(object):
    """
    This one loads the images from disk.
    """

    def __init__(self,
                 nturgbd_video_info=None,
                 skele_motion_root=None,
                 batch_size=10,
                 split='train',
                 seq_len=30,
                 downsample=1,
                 return_label=False,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None):
        print('Using the NVIDIA DALI pipeline for data loading and preparation via GPU.')

        self.batch_size = batch_size
        self.split = split
        self.seq_len = seq_len
        self.downsample = downsample
        self.return_label = return_label
        self.split_mode = split_mode
        self.use_skeleton = skele_motion_root is not None

        self.image_min_height = 150
        self.image_min_width = 266

        print('Using nturgbd data (150x150)')
        ndu = NTURGBDDatasetUtils

        self.video_info_skeletons = {}

        self.sample_info = ndu.read_video_info(nturgbd_video_info, max_samples=sample_limit)

        v_file_count = len(self.sample_info)

        min_frame_count = self.seq_len * self.downsample

        self.sample_info = ndu.filter_too_short(self.sample_info, min_frame_count)

        sample_count = len(self.sample_info)

        print("Dropped {} of {} samples due to insufficient rgb video length ({} frames needed).".format(
            v_file_count - sample_count, v_file_count, min_frame_count))

        self.sample_info = ndu.filter_nturgbd_by_split_mode(self.sample_info, self.split, self.split_mode, split_frac)

        self.sk_info = ndu.get_skeleton_info(skele_motion_root)

        self.sample_info = ndu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)

        print("Dropped {} of {} samples due to missing skeleton information.".format(
            sample_count - len(self.sample_info), sample_count))

        print("Remaining videos in mode {}: {}".format(self.split, len(self.sample_info)))

        # The original approach always used a subset of the test set for validation. Doing the same for comparability.
        if self.split == "val":
            if len(self.sample_info) > 500:
                print(
                    "Limited the validation sample to 500 to speed up training. This does not alter the structure of the train/test/val splits, " +
                    "it only reduces the samples used for validation in training among the val split.")
                self.sample_info = self.sample_info.sample(n=500, random_state=666)

        self.indices = self.sample_info.index.values
        self.n = len(self.sample_info)

    def __iter__(self):
        self.i = 0
        self.indices = np.random.permutation(self.indices)
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration

        img_seqs, sk_seqs = [], []
        seq_rotations, seq_hues, seq_saturations, seq_values = [], [], [], []
        seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys = [], [], [], []

        for _ in range(self.batch_size):
            index = self.indices[self.i % self.n]  # Fill missing batch samples by wrapping around.

            img_seq, random_transforms, sk_seq = self[index]
            img_seqs.append(img_seq)
            sk_seqs.append(sk_seq)

            seq_rotations.append(random_transforms[0])
            seq_hues.append(random_transforms[1])
            seq_saturations.append(random_transforms[2])
            seq_values.append(random_transforms[3])

            seq_crop_ws.append(random_transforms[4])
            seq_crop_hs.append(random_transforms[5])
            seq_crop_xs.append(random_transforms[6])
            seq_crop_ys.append(random_transforms[7])

            self.i = self.i + 1  # Preparing next iteration.

        # sk_seqs = sk_seqs[0]
        # img_seqs = np.stack(img_seqs, 0)
        img_seqs_flat = [img for subl in img_seqs for img in subl]
        seq_rotations = np.stack(seq_rotations, 0)
        seq_hues = np.stack(seq_hues, 0)
        seq_saturations = np.stack(seq_saturations, 0)
        seq_values = np.stack(seq_values, 0)

        seq_crop_ws = np.stack(seq_crop_ws, 0)
        seq_crop_hs = np.stack(seq_crop_hs, 0)
        seq_crop_xs = np.stack(seq_crop_xs, 0)
        seq_crop_ys = np.stack(seq_crop_ys, 0)

        # B, F = img_seqs.shape[0], img_seqs.shape[1]
        # img_seqs = img_seqs.reshape(self.batch_size * F, -1)  # , H, W, C)
        F = self.seq_len
        seq_rotations = seq_rotations.reshape(self.batch_size * F, 1)
        seq_hues = seq_hues.reshape(self.batch_size * F, 1)
        seq_saturations = seq_saturations.reshape(self.batch_size * F, 1)
        seq_values = seq_values.reshape(self.batch_size * F, 1)

        seq_crop_ws = seq_crop_ws.reshape(self.batch_size * F, 1)
        seq_crop_hs = seq_crop_hs.reshape(self.batch_size * F, 1)
        seq_crop_xs = seq_crop_xs.reshape(self.batch_size * F, 1)
        seq_crop_ys = seq_crop_ys.reshape(self.batch_size * F, 1)

        sk_seqs = np.repeat(sk_seqs, repeats=F, axis=0)

        return img_seqs_flat, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, sk_seqs

    def __len__(self):
        return self.n

    def _load_img_buffer(self, sample, i):
        with open(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1)), 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    def random_image_crop_square(self, min_area_n=0.4, max_area_n=1, image_min_width=None, image_min_height=None):
        """
        This follows the conventions of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.Crop
        Especially considering the meaning of crop_pos_x_norm and crop_pos_y_norm.
        """
        if image_min_width is None:
            image_min_width = self.image_min_width
        if image_min_height is None:
            image_min_height = self.image_min_height

        image_shorter = min(image_min_height, image_min_width)
        image_longer = max(image_min_height, image_min_width)

        # First find square crop length.
        total_area = image_min_width * image_min_height

        min_crop_length = math.ceil(math.sqrt(min_area_n * total_area))
        max_crop_length = math.floor(math.sqrt(max_area_n * total_area))

        min_crop_length = max(min_crop_length, 1.)
        max_crop_length = min(max_crop_length, image_shorter)

        crop_length = np.random.uniform(min_crop_length, max_crop_length)

        # Second, find upper left corner position. Normal distributed around center.
        crop_pos_x_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.),
                              1.)  # Normal distributed between 0 and 1.
        crop_pos_y_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.),
                              1.)  # Normal distributed between 0 and 1.

        crop_length_x = crop_length
        crop_length_y = crop_length

        return crop_length_x, crop_length_y, crop_pos_x_norm, crop_pos_y_norm

    def random_image_transforms(self, frame_count,
                                rotation_range=(-30., 30.),
                                hue_range=(-10, 10),
                                saturation_range=(0., 2.),
                                value_range=(0.5, 2.),
                                hue_change_prop=0.5,
                                crop_area_range=(0.05, 1.)):
        # The same rotation for all frames.
        rotations = np.repeat(np.random.uniform(low=rotation_range[0], high=rotation_range[1]), repeats=frame_count)
        rotations = rotations.astype(np.float32)

        if np.random.random() > hue_change_prop:
            # Different hue for each frame.
            hues = np.repeat(np.random.uniform(low=hue_range[0], high=hue_range[1]), repeats=frame_count)
        else:
            hues = np.repeat(0., repeats=frame_count)

        hues = hues.astype(np.float32)

        saturations = np.repeat(np.random.uniform(low=saturation_range[0], high=saturation_range[1]),
                                repeats=frame_count)
        saturations = saturations.astype(np.float32)

        values = np.repeat(np.random.uniform(low=value_range[0], high=value_range[1]), repeats=frame_count)
        values = values.astype(np.float32)

        crop_w, crop_h, crop_x, crop_y = self.random_image_crop_square(min_area_n=min(crop_area_range),
                                                                       max_area_n=max(crop_area_range),
                                                                       image_min_width=self.image_min_width,
                                                                       image_min_height=self.image_min_height)

        crop_ws = np.repeat(crop_w, repeats=frame_count)
        crop_ws = crop_ws.astype(np.float32)
        crop_hs = np.repeat(crop_h, repeats=frame_count)
        crop_hs = crop_hs.astype(np.float32)
        crop_xs = np.repeat(crop_x, repeats=frame_count)
        crop_xs = crop_xs.astype(np.float32)
        crop_ys = np.repeat(crop_y, repeats=frame_count)
        crop_ys = crop_ys.astype(np.float32)

        return rotations, hues, saturations, values, crop_ws, crop_hs, crop_xs, crop_ys

    def __getitem__(self, index):
        sample = self.sample_info.loc[index]

        frame_indices = DatasetUtils.idx_sampler(sample["frame_count"], self.seq_len, self.downsample, sample["path"])

        img_seq = []

        for i in frame_indices:
            img_seq.append(self._load_img_buffer(sample, i))  # I did not find a way to make DALI work with this so far.
            # img_seq.append(pil_loader(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1))))

        #img_seq = np.stack(img_seq, axis=0)

        if self.use_skeleton:
            sk_seq = NTURGBDDatasetUtils.load_skeleton_seqs(self.sk_info, sample["id"], frame_indices)

            # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
            # (sk_Bo, sk_J, sk_T, sk_C) = sk_seq.shape

            # This is transposed, so we can split the image into blocks during training.
            sk_seq = np.transpose(sk_seq, axes=(0, 3, 2, 1))
            # (sk_Bo, sk_C, sk_T, sk_J) = sk_seq.shape

            if self.return_label:
                label = sample["action"]
                return img_seq, self.random_image_transforms(len(frame_indices)), sk_seq, label

            else:
                return img_seq, self.random_image_transforms(len(frame_indices)), sk_seq

        if self.return_label:
            label = sample["action"]
            return img_seq, self.random_image_transforms(len(frame_indices)), label
        else:
            return img_seq, self.random_image_transforms(len(frame_indices))

import time

class NTURGBD3DPipeline(Pipeline):
    """
    This one performs data augmentation on GPU.
    """

    def __init__(self, batch_size, seq_length, num_threads, device_id, nturgbd_input_data):
        super(NTURGBD3DPipeline, self).__init__(batch_size * seq_length,
                                                32,
                                                device_id,
                                                seed=12 + device_id,
                                                prefetch_queue_depth=4)
        self.external_data = nturgbd_input_data
        self.iterator = iter(self.external_data)

        self.loading_times = []

        self.input_imgs = ops.ExternalSource(device="cpu")
        self.input_angles = ops.ExternalSource(
            device="cpu")  # Somehow the parameter for the operations has to be on CPU.
        self.input_hues = ops.ExternalSource(device="cpu")
        self.input_saturations = ops.ExternalSource(device="cpu")
        self.input_values = ops.ExternalSource(device="cpu")

        self.input_crop_ws = ops.ExternalSource(device="cpu")
        self.input_crop_hs = ops.ExternalSource(device="cpu")
        self.input_crop_xs = ops.ExternalSource(device="cpu")
        self.input_crop_ys = ops.ExternalSource(device="cpu")

        self.input_sk_seq = ops.ExternalSource(device="cpu")

        self.img_dec = ops.ImageDecoder(device="mixed", bytes_per_sample_hint=360000)

        self.rrot = ops.Rotate(interp_type=types.INTERP_LINEAR, device="gpu")  # angle=10,
        self.rcrop = ops.Crop(device="gpu")
        self.rrsize = ops.Resize(resize_x=128., resize_y=128., interp_type=types.INTERP_TRIANGULAR, device="gpu")
        self.rhsv = ops.Hsv(device="gpu")
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_layout=types.NCHW)

    def define_graph(self):
        self.img_seq = self.input_imgs()

        self.in_angle = self.input_angles()
        self.in_hue = self.input_hues()
        self.in_saturation = self.input_saturations()
        self.in_value = self.input_values()

        self.in_crop_w = self.input_crop_ws()
        self.in_crop_h = self.input_crop_hs()
        self.in_crop_x = self.input_crop_xs()
        self.in_crop_y = self.input_crop_ys()

        self.sk_seq = self.input_sk_seq()

        image = self.img_dec(self.img_seq)
        image = self.rrot(image, angle=self.in_angle)
        image = self.rcrop(image,
                           crop_pos_x=self.in_crop_x, crop_pos_y=self.in_crop_y,
                           crop_w=self.in_crop_w, crop_h=self.in_crop_h)

        image = self.rrsize(image)
        image = self.rhsv(image, hue=self.in_hue, saturation=self.in_saturation, value=self.in_value)

        image = self.normalize(image)

        return image.gpu(), self.sk_seq.gpu()

    def iter_setup(self):
        try:
            start_loading = time.perf_counter()
            img_seqs, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, sk_seqs = next(
                self.iterator)
            end_loading = time.perf_counter()

            self.loading_times.append(end_loading - start_loading)

            if len(self.loading_times) == 100:
                print("Image loading average time: {}".format(np.mean(self.loading_times)))

            # The parameters for the transformations are also provided.
            self.feed_input(self.img_seq, img_seqs)
            self.feed_input(self.in_angle, seq_rotations)
            self.feed_input(self.in_hue, seq_hues)
            self.feed_input(self.in_saturation, seq_saturations)
            self.feed_input(self.in_value, seq_values)

            self.feed_input(self.in_crop_w, seq_crop_ws)
            self.feed_input(self.in_crop_h, seq_crop_hs)
            self.feed_input(self.in_crop_x, seq_crop_xs)
            self.feed_input(self.in_crop_y, seq_crop_ys)

            self.feed_input(self.sk_seq, sk_seqs, layout="FCHW")  # F is actually the body dimension.

        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class DatasetUtils:

    @staticmethod
    def filter_too_short(video_info: pd.DataFrame, min_frame_count: int):
        # Filtering videos based on min length.

        old_count = len(video_info)

        print("Dropping videos with less than {} frames".format(min_frame_count))
        video_info = video_info.loc[video_info["frame_count"] >= min_frame_count]

        new_count = len(video_info)

        print("Discarded {} of {} videos since they were shorter than the necessary {} frames.".format(
            old_count - new_count,
            old_count,
            min_frame_count))

        return video_info

    @staticmethod
    def idx_sampler(vlen, seq_len, downsample, vpath):
        '''sample index from a video'''
        if vlen - seq_len * downsample <= 0:
            print("Tried to sample a video which is too short. This should not happen after filtering short videos."
                  "\nVideo path: {}".format(vpath))
            return [None]

        # Randomly start anywhere within the video (as long as the remainder is long enough).
        start_idx = np.random.choice(range(vlen - seq_len * downsample))

        seq_idxs = start_idx + np.arange(seq_len) * downsample

        return seq_idxs


class NTURGBDDatasetUtils(DatasetUtils):
    # Each file/folder name in both datasets is in the format of SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013),
    # in which sss is the setup number, ccc is the camera ID, ppp is the performer (subject) ID,
    # rrr is the replication number (1 or 2), and aaa is the action class label.
    nturgbd_id_pattern = re.compile(r".*(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}).*")

    # A pattern object to recognize nturgbd sample descriptors.
    setup_pattern = re.compile(r".*S(\d{3})C\d{3}P\d{3}R\d{3}A\d{3}.*")
    camera_pattern = re.compile(r".*S\d{3}C(\d{3})P\d{3}R\d{3}A\d{3}.*")
    subject_pattern = re.compile(r".*S\d{3}C\d{3}P(\d{3})R\d{3}A\d{3}.*")
    replication_pattern = re.compile(r".*S\d{3}C\d{3}P\d{3}R(\d{3})A\d{3}.*")
    action_code_pattern = re.compile(r".*S\d{3}C\d{3}P\d{3}R\d{3}A(\d{3}).*")

    sk_magnitude_pattern = re.compile(r".*(CaetanoMagnitude).*")
    sk_orientation_pattern = re.compile(r".*(CaetanoOrientation).*")
    sk_body_pattern = re.compile(r".*S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}_([\d]+)_*")

    nturgbd_action_labels = [
        "drink water",
        "eat meal/snack",
        "brushing teeth",
        "brushing hair",
        "drop",
        "pickup",
        "throw",
        "sitting down",
        "standing up (from sitting position)",
        "clapping",
        "reading",
        "writing",
        "tear up paper",
        "wear jacket",
        "take off jacket",
        "wear a shoe",
        "take off a shoe",
        "wear on glasses",
        "take off glasses",
        "put on a hat/cap",
        "take off a hat/cap",
        "cheer up",
        "hand waving",
        "kicking something",
        "reach into pocket",
        "hopping (one foot jumping)",
        "jump up",
        "make a phone call/answer phone",
        "playing with phone/tablet",
        "typing on a keyboard",
        "pointing to something with finger",
        "taking a selfie",
        "check time (from watch)",
        "rub two hands together",
        "nod head/bow",
        "shake head",
        "wipe face",
        "salute",
        "put the palms together",
        "cross hands in front (say stop)",
        "sneeze/cough",
        "staggering",
        "falling",
        "touch head (headache)",
        "touch chest (stomachache/heart pain)",
        "touch back (backache)",
        "touch neck (neckache)",
        "nausea or vomiting condition",
        "use a fan (with hand or paper)/feeling warm",
        "punching/slapping other person",
        "kicking other person",
        "pushing other person",
        "pat on back of other person",
        "point finger at the other person",
        "hugging other person",
        "giving something to other person",
        "touch other person's pocket",
        "handshaking",
        "walking towards each other",
        "walking apart from each other",
        "put on headphone",
        "take off headphone",
        "shoot at the basket",
        "bounce ball",
        "tennis bat swing",
        "juggling table tennis balls",
        "hush (quite)",
        "flick hair",
        "thumb up",
        "thumb down",
        "make ok sign",
        "make victory sign",
        "staple book",
        "counting money",
        "cutting nails",
        "cutting paper (using scissors)",
        "snapping fingers",
        "open bottle",
        "sniff (smell)",
        "squat down",
        "toss a coin",
        "fold paper",
        "ball up paper",
        "play magic cube",
        "apply cream on face",
        "apply cream on hand back",
        "put on bag",
        "take off bag",
        "put something into a bag",
        "take something out of a bag",
        "open a box",
        "move heavy objects",
        "shake fist",
        "throw up cap/hat",
        "hands up (both hands)",
        "cross arms",
        "arm circles",
        "arm swings",
        "running on the spot",
        "butt kicks (kick backward)",
        "cross toe touch",
        "side kick",
        "yawn",
        "stretch oneself",
        "blow nose",
        "hit other person with something",
        "wield knife towards other person",
        "knock over other person (hit with body)",
        "grab other person’s stuff",
        "shoot at other person with a gun",
        "step on foot",
        "high-five",
        "cheers and drink",
        "carry something with other person",
        "take a photo of other person",
        "follow other person",
        "whisper in other person’s ear",
        "exchange things with other person",
        "support somebody with hand",
        "finger-guessing game (playing rock-paper-scissors)"
        ]

    nturgbd_cross_subject_training = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                                      38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                                      80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

    # Careful: action ids start from 0 here, but the ids in the file names start from 1.
    action_dict_encode = {label: act_id for act_id, label in enumerate(nturgbd_action_labels)}
    action_dict_decode = {act_id: label for act_id, label in enumerate(nturgbd_action_labels)}

    @staticmethod
    def read_video_info(video_info_csv, extract_infos=True, max_samples=None) -> pd.DataFrame:
        NDU = NTURGBDDatasetUtils
        sample_infos = pd.read_csv(video_info_csv, header=0, names=["path", "frame_count"])

        if max_samples is not None:
            sample_infos = sample_infos.iloc[:max_samples]

        if extract_infos:
            sample_infos = NDU.extract_infos(sample_infos)
            sample_infos = sample_infos.set_index(["id"], drop=False)
            return sample_infos
        else:
            return sample_infos

    @staticmethod
    def extract_infos(sample_infos: pd.DataFrame):
        ndu = NTURGBDDatasetUtils

        sample_infos["id"] = 0
        sample_infos = sample_infos.astype(dtype={"id": np.dtype("uint64")})

        for col in ["setup", "camera", "subject", "replication", "action"]:
            sample_infos[col] = 0
            sample_infos = sample_infos.astype(dtype={col: np.dtype("uint8")})

        # If necessary, speedup could be made with the dask framework.
        for row in tqdm(sample_infos.itertuples(), total=len(sample_infos)):
            idx = row.Index
            path = row.path
            file_name = os.path.split(path)[1]

            sample_id = ndu.nturgbd_id_pattern.match(file_name).group(1)
            sample_id = ndu._id_to_int(sample_id)  # Saving all ids as integers for efficiency.

            # setup = ndu.setup_pattern.match(file_name).group(1)
            # setup = int(setup)
            # assert setup == (sample_id // 1000 ** 4) % 1000
            setup = (sample_id // 1000 ** 4) % 1000

            # camera = ndu.camera_pattern.match(file_name).group(1)
            # camera = int(camera)
            # assert camera == (sample_id // 1000 ** 3) % 1000
            camera = (sample_id // 1000 ** 3) % 1000

            # subject = ndu.subject_pattern.match(file_name).group(1)
            # subject = int(subject)
            # assert subject == (sample_id // 1000 ** 2) % 1000
            subject = (sample_id // 1000 ** 2) % 1000

            # replication = ndu.replication_pattern.match(file_name).group(1)
            # replication = int(replication)
            # assert replication == (sample_id // 1000) % 1000
            replication = (sample_id // 1000) % 1000

            # action = ndu.action_code_pattern.match(file_name).group(1)
            # action = int(action)
            # assert action == sample_id % 1000
            action = sample_id % 1000

            sample_infos.loc[idx, ["id", "setup", "camera", "subject", "replication",
                                   "action"]] = sample_id, setup, camera, subject, replication, action

        return sample_infos

    @staticmethod
    def get_skeleton_info(skele_motion_root):
        ndu = NTURGBDDatasetUtils

        skeleton_paths = glob.glob(os.path.join(skele_motion_root, "*.npz"))
        sk_info = pd.DataFrame(skeleton_paths, columns=["sk_path"])

        sk_info["sk_file"] = sk_info["sk_path"].apply(lambda p: os.path.split(p)[1])
        sk_info["id"] = sk_info["sk_file"].apply(lambda fl: ndu._id_to_int(ndu.nturgbd_id_pattern.match(fl).group(1)))

        sk_info["body"] = sk_info["sk_file"].apply(lambda fl: ndu._id_to_int(ndu.sk_body_pattern.match(fl).group(1)))

        def _extract_type(file):
            mag_match = ndu.sk_magnitude_pattern.match(file)
            if mag_match:
                return mag_match.group(1)
            else:
                ori_match = ndu.sk_orientation_pattern.match(file)

                if ori_match:
                    return ori_match.group(1)
                else:
                    return np.NaN

        sk_info["skeleton_info_type"] = sk_info["sk_file"].apply(_extract_type)
        sk_info = sk_info.drop(columns=["sk_file"])

        sk_info_magnitude = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoMagnitude"]
        sk_info_magnitude = sk_info_magnitude.rename(columns={"sk_path": "caetano_magnitude_path"})
        sk_info_magnitude = sk_info_magnitude.drop(columns=["skeleton_info_type"])
        sk_info_magnitude = sk_info_magnitude.set_index(["id", "body"], verify_integrity=True, drop=False)

        sk_info_orientation = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoOrientation"]
        sk_info_orientation = sk_info_orientation.rename(columns={"sk_path": "caetano_orientation_path"})
        sk_info_orientation = sk_info_orientation.drop(columns=["skeleton_info_type"])
        sk_info_orientation = sk_info_orientation.set_index(["id", "body"], verify_integrity=True, drop=False)

        sk_info = sk_info_magnitude.join(sk_info_orientation, rsuffix="_right")

        # Apparently pandas can not join on index if index columns are not dropped (Column overlap not ignored).
        sk_info = sk_info.drop(columns=["id_right", "body_right"])

        count = len(sk_info)
        sk_info = sk_info.dropna()

        if count > len(sk_info):
            print("Dropped {} of {} skeleton samples due to missing information.".format(count - len(sk_info), count))

        return sk_info

    @staticmethod
    def filter_by_missing_skeleton_info(sample_info: pd.DataFrame, skeleton_info: pd.DataFrame):
        sk_ids = set(skeleton_info.index.get_level_values("id"))

        return sample_info.loc[sample_info.index.isin(sk_ids)]

    @staticmethod
    def filter_nturgbd_by_split_mode(sample_info, split, split_mode, split_frac=0.2, random_state=42):
        ndu = NTURGBDDatasetUtils

        # Splits
        # Cross setup mode
        if split_mode == "cross-setup":
            if split == "train":
                sample_info = sample_info.iloc[sample_info["setup"] % 2 == 0]  # All even setups are used for training.

            elif split == "val":
                sample_info = sample_info.iloc[sample_info["setup"] % 2 == 1]  # All odd setups are used for evaluation.
            else:
                raise ValueError()
        # Cross subject mode
        elif split_mode == "cross-subject":
            if split == "train":
                sample_info = sample_info.iloc[sample_info["setup"].isin(ndu.nturgbd_cross_subject_training)]

            elif split == "val":
                sample_info = sample_info.iloc[~sample_info["setup"].isin(ndu.nturgbd_cross_subject_training)]
            else:
                raise ValueError()
        elif split_mode == "perc":
            train_inf, test_inf = train_test_split(sample_info, test_size=split_frac, random_state=random_state)
            if split == "train":
                sample_info = train_inf
            elif split == "val":
                sample_info = test_inf
            else:
                raise ValueError()
        elif split_mode == "all":
            pass
        else:
            raise ValueError()

        return sample_info

    @staticmethod
    def _id_to_int(nturgbd_id):
        return int("".join([n for n in nturgbd_id if n.isdigit()]))

    @staticmethod
    def encode_action(action_name, zero_indexed=True):
        '''give action name, return category'''
        ndu = NTURGBDDatasetUtils
        return ndu.action_dict_encode[action_name] if zero_indexed else ndu.action_dict_encode[action_name] + 1

    @staticmethod
    def decode_action(action_code, zero_indexed=True):
        '''give action code, return action name'''
        ndu = NTURGBDDatasetUtils
        return ndu.action_dict_decode[action_code] if zero_indexed else ndu.action_dict_decode[action_code - 1]

    @staticmethod
    def load_skeleton_seqs(sk_info: pd.DataFrame, sample_id, frame_indices, only_first_body=True) -> np.ndarray:
        """
        Loads a skele-motion representation and selects the columns which are indexed by idx_block.
        Returns a tensor of shape (Joints, Length, Channels).
        The length describes the number of time steps (frame count when downsampling is 1).
        First 3 channels are orientation, last channel is magnitude.
        """
        sk_body_infos = sk_info.xs(sample_id, level="id")

        if only_first_body is False:
            # It has to be checked what happens, when bodies are not there for the whole time. Untested version.
            raise NotImplementedError

        sk_seqs_mag = []
        sk_seqs_ori = []

        for body_id in [1] if only_first_body else sorted(sk_body_infos.index.values):
            sk_seq_mag_path = sk_body_infos.loc[body_id]["caetano_magnitude_path"]
            sk_seq_ori_path = sk_body_infos.loc[body_id]["caetano_orientation_path"]

            sk_mag = np.load(sk_seq_mag_path)
            sk_ori = np.load(sk_seq_ori_path)

            sk_mag = sk_mag['arr_0']
            sk_ori = sk_ori['arr_0']

            (J_m, T_m, C_m) = sk_ori.shape
            (J_o, T_o, C_o) = sk_ori.shape

            assert J_m == J_o and T_m == T_o

            sk_seqs_mag.append(sk_mag)
            sk_seqs_ori.append(sk_ori)

        sk_seqs_mag = np.stack(sk_seqs_mag)
        sk_seqs_ori = np.stack(sk_seqs_ori)

        sk_seq = np.concatenate((sk_seqs_ori, sk_seqs_mag), axis=-1)  # Concatenating on channel dimension.

        (Bo, J, T, C_o) = sk_seqs_ori.shape

        mask = [False] * T
        for i in frame_indices:
            mask[i] = True

        sk_seq = sk_seq[:, :, mask, :]

        (Bo, J, T, C) = sk_seq.shape

        assert T == len(frame_indices)

        return sk_seq


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(image_batch, batch_size, seq_len):
    columns = seq_len
    rows = batch_size
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(rows, columns)

    image_batch = image_batch.as_cpu().as_array()

    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img = image_batch[j]
        img = np.transpose(img, (1,2,0))
        plt.imshow(img)

    plt.show()


def test():
    batch_size = 5
    seq_len = 5
    video_info_csv = os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/video_info.csv")
    skele_motion_root = os.path.expanduser("~/datasets/nturgbd/skele-motion")

    nii = NTURGB3DInputIterator(nturgbd_video_info=video_info_csv, skele_motion_root=skele_motion_root,
                                batch_size=batch_size, seq_len=seq_len, unit_test=True)

    pipeline = NTURGBD3DPipeline(batch_size=batch_size, seq_length=seq_len, num_threads=1, device_id=0,
                                 nturgbd_input_data=nii)

    pipeline.build()

    images, sk_seqs = pipeline.run()

    show_images(images, batch_size, seq_len)
    pass


if __name__ == "__main__":
    test()
