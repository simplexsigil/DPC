import glob
import os
import sys

sys.path.append('../utils')
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils import data
from datasets.dataset import DatasetUtils
import re
from sklearn.model_selection import train_test_split


class NTURGBDDataset(data.Dataset):  # Todo: introduce csv selection into parse args list.
    def __init__(self,
                 split='train',
                 transform=None,
                 seq_len=30,
                 downsample_vid=3,
                 return_label=False,
                 nturgbd_video_info=None,
                 skele_motion_root=None,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None,
                 sub_sample_shift=None,
                 random_state=42,
                 sub_sample_limit=None,
                 validation_max_subsamples=500):
        self.split = split
        self.split_mode = split_mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample_vid = downsample_vid
        self.return_label = return_label
        self.sub_sample_shift = sub_sample_shift
        self.sub_sample_limit = sub_sample_limit

        self.use_skeleton = skele_motion_root is not None

        ndu = NTURGBDDatasetUtils

        print("=================================")
        print(f'Dataset NTURGBD {split} split (Split method: {split_mode})')
        if split_mode == "perc":
            print(f"Train/Val ratio: {1 - split_frac}/{split_frac}")

        print(f'Total number of frames: {seq_len}\n'
              f'Downsampling on video frames: {downsample_vid}')

        self.sample_info = ndu.read_video_info(nturgbd_video_info, max_samples=sample_limit, random_state=random_state)

        sample_count = len(self.sample_info)
        self.sample_info = ndu.filter_too_short(self.sample_info, self.seq_len)
        print(f"Dropped {sample_count - len(self.sample_info)} of {sample_count} samples "
              f"due to insufficient rgb video length ({self.seq_len} frames needed).")

        sample_count = len(self.sample_info)
        self.sample_info = ndu.filter_nturgbd_by_split_mode(self.sample_info, self.split, self.split_mode, split_frac)
        print(f"Selected {len(self.sample_info)} of {sample_count} video samples for the {self.split} split.")
        sample_count = len(self.sample_info)
        self.sk_info = ndu.get_skeleton_info(skele_motion_root)
        self.sample_info = ndu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
        print(f"Dropped {sample_count - len(self.sample_info)} of "
              f"{sample_count} samples due to missing skeleton information.")

        print(f"Remaining videos in mode {self.split}: {len(self.sample_info)}")

        if sub_sample_shift is not None:
            print(f"Subsampling videos to clips of length {seq_len}. \n"
                  f"Shifting by {sub_sample_shift} frames on each subsample. This may take a while.")
            self.sample_info = ndu.subsample_discretely(self.sample_info, sub_sample_shift, seq_len=seq_len)

            print(f"Generated {len(self.sample_info)} subsamples.")

        if self.sub_sample_limit is not None:
            self.sample_info = self.sample_info.sample(min(self.sub_sample_limit, len(self.sample_info)),
                                                       random_state=random_state)
            print(f"Limited training to {len(self.sample_info)} randomly selected subsamples.")

        # For faster validation.
        if self.split == "val":
            if len(self.sample_info) > validation_max_subsamples:
                print("Limited the validation sample to 500 to speed up training. "
                      "This does not alter the structure of the train/test/val splits, " +
                      "it only reduces the samples used for validation in training among the val split.")
                self.sample_info = self.sample_info.sample(n=validation_max_subsamples, random_state=random_state)
        print("=================================")

    def __getitem__(self, index):
        ndu = NTURGBDDatasetUtils
        sample = self.sample_info.iloc[index]

        st_frame = sample["start_frame"] if "start_frame" in sample.index else None

        frame_idxs = ndu.idx_sampler(sample["frame_count"], self.seq_len, sample["path"], self.sub_sample_shift,
                                     st_frame)

        frame_idxs_vid = frame_idxs[::self.downsample_vid]

        seq = [ndu.pil_loader(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1))) for i in frame_idxs_vid]

        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()

        # One Tensor of shape (self.num_seq * self.seq_len, C, H, W)
        t_seq = torch.stack(t_seq, 0)

        # One Tensor of shape (C, self.seq_len, H, W)
        t_seq = t_seq.view(len(frame_idxs_vid), C, H, W).transpose(0, 1)

        if self.use_skeleton:
            sk_seq, skeleton_frame_count = ndu.load_skeleton_seqs(self.sk_info, sample["id"])
            sk_seq = ndu.select_skeleton_seqs(sk_seq, frame_idxs)
            sk_seq = torch.tensor(sk_seq, dtype=torch.float)

            # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
            (sk_Bo, sk_J, sk_T, sk_C) = sk_seq.shape

            # This is transposed, so we can split the image into blocks during training.
            sk_seq = sk_seq.transpose(1, 2)
            sk_seq = sk_seq.transpose(2, 3).transpose(1, 2)  # (sk_Bo, C, T, J)

            if self.return_label:
                label = torch.tensor([sample["action"]], dtype=torch.long)
                return index, t_seq, sk_seq, label

            else:
                return index, t_seq, sk_seq

        if self.return_label:
            label = torch.tensor([sample["action"]], dtype=torch.long)
            return index, t_seq, label
        else:
            return index, t_seq

    def __len__(self):
        return len(self.sample_info)


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
    def read_video_info(video_info_csv, extract_infos=True, max_samples=None, random_state=42) -> pd.DataFrame:
        ndu = NTURGBDDatasetUtils
        sample_infos = pd.read_csv(video_info_csv, header=0, names=["path", "frame_count"])

        if max_samples is not None:
            sample_infos = sample_infos.sample(max_samples, random_state=random_state)

        if extract_infos:
            sample_infos = ndu.extract_infos(sample_infos)
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

        sk_info["skeleton_info_type"] = sk_info["sk_file"].apply(ndu._extract_skeleton_info_type)
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
    def load_img_buffer(sample, i):
        with open(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1)), 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8)
