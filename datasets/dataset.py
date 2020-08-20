"""
------------
dataset_3d
------------

What?
--------
A module which provides the basic logic for working with the datasets in this project.
In the end, everything here is used in PyTorch dataloaders.

How?
--------
This module contains utility classes with logic which can be generalized for the datasets.

Author: David Schneider david.schneider2@student.kit.edu
Some code has been used from Tengda Han: https://github.com/TengdaHan/DPC
"""

import os
import re

import pandas as pd
from tqdm import tqdm

from augmentation import *


class DatasetUtils:
    sk_magnitude_pattern = re.compile(r".*(CaetanoMagnitude).*")
    sk_orientation_pattern = re.compile(r".*(CaetanoOrientation).*")

    @staticmethod
    def filter_too_short(video_info: pd.DataFrame, min_frame_count: int) -> pd.DataFrame:
        # Filtering videos based on min length.
        video_info = video_info.loc[video_info["frame_count"] >= min_frame_count]

        return video_info

    @staticmethod
    def idx_sampler(vlen, seq_len, vpath, sample_discretization=None, start_frame=None, multi_time_shifts=None):
        shift_span = 0 if multi_time_shifts is None else max(multi_time_shifts) - min(multi_time_shifts)

        '''sample index from a video'''
        if vlen - (seq_len + shift_span) < 0:
            print("Tried to sample a video which is too short. This should not happen after filtering short videos."
                  "\nVideo path: {}".format(vpath))
            return [None]

        # Randomly start anywhere within the video (as long as the remainder is long enough).
        first_possible_start = 0 if multi_time_shifts is None or min(multi_time_shifts) > 0 else 0 - min(
            multi_time_shifts)
        last_possible_start = vlen - (seq_len + shift_span)
        if sample_discretization is None:
            starts = list(range(first_possible_start, last_possible_start))
            start_idx = np.random.choice(starts) if len(starts) > 0 else (
                0 if min(multi_time_shifts) > 0 else - min(multi_time_shifts))
        else:
            if start_frame is not None:
                if vlen < start_frame + seq_len + shift_span:
                    print(
                        f"Not all frames were available at position {start_frame}, for limited vlen {vlen} of {vpath}  "
                        f"with discretization {sample_discretization}. Sampling in the middle.")
                    start_idx = (vlen - last_possible_start) // 2
                else:
                    start_idx = start_frame
            else:
                start_points = (vlen - last_possible_start) // sample_discretization
                start_idx = np.random.choice(range(start_points) * sample_discretization)

        if multi_time_shifts is not None:
            seq_idxs = [start_idx + np.arange(seq_len) + time_shift for time_shift in multi_time_shifts]
        else:
            seq_idxs = [start_idx + np.arange(seq_len)]

        return seq_idxs

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    @staticmethod
    def load_img_buffer(sample, i):
        with open(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1)), 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    @staticmethod
    def get_skeleton_info(skele_motion_root):  # TODO
        raise NotImplementedError

    @staticmethod
    def filter_by_missing_skeleton_info(sample_info: pd.DataFrame, skeleton_info: pd.DataFrame):
        sk_ids = skeleton_info.index.get_level_values("id")
        sk_ids = set(sk_ids)

        vid_ids = list(sample_info.index.get_level_values("id"))
        ids = [vid_id for vid_id in vid_ids if vid_id in sk_ids]
        return sample_info.loc[ids]

    @staticmethod
    def load_skeleton_seqs(sk_info: pd.DataFrame, sample_id, only_first_body=True) -> (np.ndarray, int):
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

        for body_id in set(sk_body_infos.index.values):
            sk_seq_mag_path = sk_body_infos.loc[body_id]["caetano_magnitude_path"]
            sk_seq_ori_path = sk_body_infos.loc[body_id]["caetano_orientation_path"]

            sk_mag = np.load(sk_seq_mag_path)
            sk_ori = np.load(sk_seq_ori_path)

            sk_mag = sk_mag['arr_0']
            sk_ori = sk_ori['arr_0']

            (J_m, T_m, C_m) = sk_mag.shape
            (J_o, T_o, C_o) = sk_ori.shape

            assert J_m == J_o and T_m == T_o

            sk_seqs_mag.append(sk_mag)
            sk_seqs_ori.append(sk_ori)

        sk_seqs = [np.concatenate((sk_s_ori, sk_s_mag), axis=-1) for sk_s_ori, sk_s_mag in
                   zip(sk_seqs_ori, sk_seqs_mag)]  # Concatenating on channel dimension.

        sk_seqs = np.stack(sk_seqs, axis=0)  # (Bo, J, T, C)
        is_zero_seqs = [np.all(sk_s_mag == 0) for sk_s_mag in sk_seqs_mag]

        if not all(is_zero_seqs):
            sk_seqs = sk_seqs[np.logical_not(is_zero_seqs)]
        else:
            sk_seqs = sk_seqs[:1]

        # sk_seqs_mag = np.stack(sk_seqs_mag)
        # sk_seqs_ori = np.stack(sk_seqs_ori)

        T = sk_seqs.shape[2]

        sk_seqs = DatasetUtils.check_sk_seq_nan_inf(sk_seqs)

        return sk_seqs, T

    @staticmethod
    def check_sk_seq_nan_inf(sk_seq):
        if np.isnan(sk_seq).any() or np.isinf(sk_seq).any():
            print("Skeleton sequence for {} contained nan or inf. Converting to 0.".format(sample_id))
        sk_seq = np.nan_to_num(sk_seq)

        return sk_seq

    @staticmethod
    def select_skeleton_seqs(sk_seq, frame_indices):
        (Bo, J, T, C) = sk_seq.shape

        mask = [False] * T
        for i in frame_indices:
            mask[i] = True

        sk_seq = sk_seq[:, :, mask, :]

        (Bo, J, T, C) = sk_seq.shape

        assert T == len(frame_indices)

        return sk_seq

    @staticmethod
    def subsample_discretely(sample_info: pd.DataFrame, sample_discretion: int, seq_len: int):
        subs_sample_info = {col: [] for col in sample_info.columns}
        subs_sample_info["start_frame"] = []

        for idx, row in tqdm(sample_info.iterrows(), total=len(sample_info)):
            sub_count = (row["frame_count"] - seq_len) // sample_discretion

            if row["frame_count"] > 305:
                print(f'Frame count is unusually high: {row["frame_count"]}, frame count {row["frame_count"]} '
                      f'for path {row["path"]}. Skipping video.')

                continue

            for i in range(sub_count):
                row["start_frame"] = i * sample_discretion

                for key, val in row.to_dict().items():
                    subs_sample_info[key].append(val)

        subs_sample_info = pd.DataFrame.from_dict(subs_sample_info)
        subs_sample_info = subs_sample_info.set_index(["id", "start_frame"], drop=False)

        return subs_sample_info

    @staticmethod
    def _extract_skeleton_info_type(file):
        mag_match = DatasetUtils.sk_magnitude_pattern.match(file)
        if mag_match:
            return mag_match.group(1)
        else:
            ori_match = DatasetUtils.sk_orientation_pattern.match(file)

            if ori_match:
                return ori_match.group(1)
            else:
                return np.NaN

    @staticmethod
    def read_video_info(video_info, extract_infos=True, max_samples=None) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def extract_infos(sample_infos: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def encode_action(action_name, zero_indexed=True):
        '''give action name, return category'''
        raise NotImplementedError

    @staticmethod
    def decode_action(action_code, zero_indexed=True):
        '''give action code, return action name'''
        raise NotImplementedError
