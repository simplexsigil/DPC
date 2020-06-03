import os
import sys

import pandas as pd
import torch
from torch.utils import data

sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
import re


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

        if big: print('Using Kinetics400 full data (256x256)')
        else: print('Using Kinetics400 full data (150x150)')

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
            else: raise ValueError('wrong mode')
        else:  # small
            if mode == 'train':
                split = '../process_data/data/kinetics400/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')

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
        else: raise ValueError('wrong mode')

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
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False,
                 train_csv=None,
                 val_csv=None,
                 train_skeleton_csv=None,
                 val_skeleton_csv=None):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label
        # A pattern object to recognize nturgbd action ids.
        self.action_code_pattern = re.compile(r"S\d{3}C\d{3}P\d{3}R\d{3}A(\d{3})")

        print('Using nturgbd data (150x150)')

        # Careful: action ids start from 0 here, but the ids in the file names start from 1.
        self.action_dict_encode = {label: act_id for act_id, label in enumerate(nturgbd_action_labels)}
        self.action_dict_decode = {act_id: label for act_id, label in enumerate(nturgbd_action_labels)}

        # splits
        if mode == 'train':
            video_info = pd.read_csv(train_csv, header=None)
        elif (mode == 'val') or (mode == 'test'):
            video_info = pd.read_csv(val_csv, header=None)
        else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)

        print("Discarded {} of {} videos since they were shorter than the necessary {} frames.".format(len(drop_idx), len(video_info), self.num_seq * self.seq_len * self.downsample))
        self.video_info = video_info.drop(drop_idx, axis=0)

        # TODO: load skeleton info.
        # Filter according to video info
        # Make sure only videos and skeletons with same length are available.

        # For some reason, the original approach also samples only 30 % of the validation set.
        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            print("Tried to sample a video which is too short. This should not happen after filtering short videos."
                  "\nVideo path: {}".format(vpath))
            return [None]

        # Randomly start anywhere within the video (as long as the remainder is long enough).
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)

        # This calculates the start index of each sequence block which we want to use in the folder. Result has shape (num_seq,1)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # This calculates the index of each frame which we want to use in the blocks. Result has shape (num_seq, len_seq)
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        # TODO: Read skeleton data according to seq and pass it on as well.

        # TODO: convert skeleton data into skelemotion representation (joint dynamic orientation and magnitude).

        # TODO: Pass on skelemotion representation per seq.

        (C, H, W) = t_seq[0].size()

        # One Tensor of shape (self.num_seq * self.seq_len, C, H, W)
        t_seq = torch.stack(t_seq, 0)

        # One Tensor of shape (self.num_seq, C, self.seq_len, H, W)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                action_id = int(self.action_code_pattern.match(vpath).group())
            except:
                print("Could not extract action id from video path: {}".format(vpath))

            label = torch.LongTensor([action_id])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name, zero_indexed=True):
        '''give action name, return category'''
        return self.action_dict_encode[action_name] if zero_indexed else self.action_dict_encode[action_name] + 1

    def decode_action(self, action_code, zero_indexed=True):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code] if zero_indexed else self.action_dict_decode[action_code - 1]


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
