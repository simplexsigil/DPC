import glob
import multiprocessing as mp
import os
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from tqdm import tqdm

from augmentation import *
from dataset import DatasetUtils


class Kinetics400Dataset(data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 seq_len=30,
                 downsample_vid=1,
                 return_label=False,
                 video_info=None,
                 skele_motion_root=None,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None,
                 sub_sample_limit=None,
                 use_cache=False,
                 cache_folder="cache",
                 sampling_shift=60,
                 random_state=42):
        self.split = split
        self.split_mode = split_mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample_vid = downsample_vid
        self.return_label = return_label

        self.use_skeleton = skele_motion_root is not None

        self.sampling_shift = sampling_shift

        tqdm.pandas(mininterval=0.5)

        print("=================================")
        print('Dataset Kinetics 400 {} split (Split method: {})'.format(split, split_mode))
        if split_mode == "perc":
            print("Train/Val ratio: {}/{}".format(1 - split_frac, split_frac))

        kdu = KineticsDatasetUtils

        start_time = time.perf_counter()
        print("Loading and preparing video info. This might take time.")
        self.sample_info = kdu.read_video_info(video_info, max_samples=sample_limit, random_state=random_state)
        stop_time = time.perf_counter()

        print("Loaded video info ({} s)".format(stop_time - start_time))
        start_time = time.perf_counter()

        sample_count = len(self.sample_info)
        self.sample_info = kdu.filter_too_short(self.sample_info, self.seq_len)
        stop_time = time.perf_counter()

        print("Dropped {} of {} video samples due to insufficient rgb video length ({} frames needed) ({} s).".format(
            sample_count - len(self.sample_info), sample_count, self.seq_len, stop_time - start_time))

        sample_count = len(self.sample_info)
        start_time = time.perf_counter()
        self.sample_info = kdu.filter_kinetics_by_split_mode(self.sample_info, self.split, self.split_mode, split_frac)
        stop_time = time.perf_counter()
        print("Selected {} of {} video samples for the {} split ({} s).".format(len(self.sample_info), sample_count,
                                                                                self.split, stop_time - start_time))

        sample_count = len(self.sample_info)

        sk_cache_name = "sk_info_cache_mfc-{}.csv".format(seq_len)

        if use_cache and os.path.exists(os.path.join(cache_folder, sk_cache_name)):
            self.sk_info = pd.read_csv(os.path.join(cache_folder, sk_cache_name), index_col=False)
            self.sk_info = self.sk_info.set_index(["id", "body"], verify_integrity=True, drop=False)
            print("Loaded skeleton info from cache.")
        else:
            print("Loading and preparing skeleton info. This might take time.")
            start_time = time.perf_counter()
            self.sk_info = kdu.get_skeleton_info(skele_motion_root)
            sk_count = len(self.sk_info)
            self.sk_info = kdu.filter_too_short(self.sk_info, 290)  # In theory there should be 300 frames.
            stop_time = time.perf_counter()
            print("Loaded skeleton info and dropped {} of {} skeleton samples due to insufficient sequence length "
                  "({} frames needed) ({} s).".format(
                sk_count - len(self.sk_info), sk_count, self.seq_len, stop_time - start_time))

            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                self.sk_info.to_csv(os.path.join(cache_folder, sk_cache_name))

        print("Found {} skeleton sequences of sufficient length.".format(len(self.sk_info)))

        start_time = time.perf_counter()
        self.sample_info = kdu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
        drop_count = sample_count - len(self.sample_info)
        stop_time = time.perf_counter()
        print("Dropped {} of {} samples due to missing skeleton information. ({} s)".format(drop_count, sample_count,
                                                                                            stop_time - start_time))

        print("Remaining videos in mode {}: {}".format(self.split, len(self.sample_info)))

        if sampling_shift is not None:
            print(f"Subsampling videos to clips of length {seq_len}. \n"
                  f"Shifting by {sampling_shift} frames on each subsample. This may take a while.")
            self.sample_info = kdu.subsample_discretely(self.sample_info, sampling_shift, seq_len=seq_len)

            print(f"Generated {len(self.sample_info)} subsamples.")

        if sub_sample_limit is not None:
            self.sample_info = self.sample_info.sample(min(sub_sample_limit, len(self.sample_info)))
            print(f"Limited training to {len(self.sample_info)} randomly selected subsamples.")

        # The original approach always used a subset of the test set for validation. Doing the same for comparability.
        if self.split == "val":
            if len(self.sample_info) > 500:
                print(
                    "Limited the validation sample to 500 to speed up training. This does not alter the structure of the train/test/val splits, " +
                    "it only reduces the samples used for validation in training among the val split.")
                self.sample_info = self.sample_info.sample(n=500, random_state=666)
        # shuffle not necessary because use RandomSampler

        print("=================================")

    def __getitem__(self, index):
        kdu = KineticsDatasetUtils

        sample = self.sample_info.iloc[index]

        t_seq = None
        sk_seq = None

        v_len = sample["frame_count"]

        if self.use_skeleton:
            sk_seq, skeleton_frame_count = kdu.load_skeleton_seqs(self.sk_info, sample["id"])

            # This is because with kinetics, it is not certain to have all sk data.
            v_len = min(v_len, skeleton_frame_count)

        st_frame = sample["start_frame"] if "start_frame" in sample.index else None

        frame_indices = kdu.idx_sampler(v_len, self.seq_len, sample["path"],
                                        self.sampling_shift, st_frame)

        frame_indices_vid = frame_indices[::self.downsample_vid]

        file_name_template = sample["file_name"] + "_{:04}.jpg"  # example: '9MHv2sl-gxs_000007_000017'

        seq = [kdu.pil_loader(os.path.join(sample["path"], file_name_template.format(i + 1))) for i in
               frame_indices_vid]

        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()

        # One Tensor of shape (self.num_seq * self.seq_len, C, H, W)
        t_seq = torch.stack(t_seq, 0)

        # One Tensor of shape (C, self.seq_len, H, W)
        t_seq = t_seq.transpose(0, 1)

        if self.use_skeleton:
            sk_seq = kdu.select_skeleton_seqs(sk_seq, frame_indices)
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


class KineticsDatasetUtils(DatasetUtils):
    kinetics_action_labels = ['abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream',
                              'archery',
                              'arm wrestling', 'arranging flowers', 'assembling computer', 'auctioning',
                              'baby waking up',
                              'baking cookies', 'balloon blowing', 'bandaging', 'barbequing', 'bartending',
                              'beatboxing',
                              'bee keeping', 'belly dancing', 'bench pressing', 'bending back', 'bending metal',
                              'biking through snow', 'blasting sand', 'blowing glass', 'blowing leaves', 'blowing nose',
                              'blowing out candles', 'bobsledding', 'bookbinding', 'bouncing on trampoline', 'bowling',
                              'braiding hair', 'breading or breadcrumbing', 'breakdancing', 'brush painting',
                              'brushing hair', 'brushing teeth', 'building cabinet', 'building shed', 'bungee jumping',
                              'busking', 'canoeing or kayaking', 'capoeira', 'carrying baby', 'cartwheeling',
                              'carving pumpkin', 'catching fish', 'catching or throwing baseball',
                              'catching or throwing frisbee', 'catching or throwing softball', 'celebrating',
                              'changing oil', 'changing wheel', 'checking tires', 'cheerleading', 'chopping wood',
                              'clapping', 'clay pottery making', 'clean and jerk', 'cleaning floor', 'cleaning gutters',
                              'cleaning pool', 'cleaning shoes', 'cleaning toilet', 'cleaning windows',
                              'climbing a rope',
                              'climbing ladder', 'climbing tree', 'contact juggling', 'cooking chicken', 'cooking egg',
                              'cooking on campfire', 'cooking sausages', 'counting money', 'country line dancing',
                              'cracking neck', 'crawling baby', 'crossing river', 'crying', 'curling hair',
                              'cutting nails',
                              'cutting pineapple', 'cutting watermelon', 'dancing ballet', 'dancing charleston',
                              'dancing gangnam style', 'dancing macarena', 'deadlifting',
                              'decorating the christmas tree',
                              'digging', 'dining', 'disc golfing', 'diving cliff', 'dodgeball', 'doing aerobics',
                              'doing laundry', 'doing nails', 'drawing', 'dribbling basketball', 'drinking',
                              'drinking beer', 'drinking shots', 'driving car', 'driving tractor', 'drop kicking',
                              'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger', 'eating cake',
                              'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream',
                              'eating spaghetti', 'eating watermelon', 'egg hunting', 'exercising arm',
                              'exercising with an exercise ball', 'extinguishing fire', 'faceplanting', 'feeding birds',
                              'feeding fish', 'feeding goats', 'filling eyebrows', 'finger snapping', 'fixing hair',
                              'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins', 'folding paper',
                              'front raises', 'frying vegetables', 'garbage collecting', 'gargling',
                              'getting a haircut',
                              'getting a tattoo', 'giving or receiving award', 'golf chipping', 'golf driving',
                              'golf putting', 'grinding meat', 'grooming dog', 'grooming horse', 'gymnastics tumbling',
                              'hammer throw', 'headbanging', 'headbutting', 'high jump', 'high kick',
                              'hitting baseball',
                              'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging', 'hula hooping',
                              'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing', 'ice skating', 'ironing',
                              'javelin throw', 'jetskiing', 'jogging', 'juggling balls', 'juggling fire',
                              'juggling soccer ball', 'jumping into pool', 'jumpstyle dancing', 'kicking field goal',
                              'kicking soccer ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing',
                              'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich', 'making bed',
                              'making jewelry', 'making pizza', 'making snowman', 'making sushi', 'making tea',
                              'marching',
                              'massaging back', 'massaging feet', 'massaging legs', "massaging person's head",
                              'milking cow',
                              'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn', 'news anchoring',
                              'opening bottle', 'opening present', 'paragliding', 'parasailing', 'parkour',
                              'passing American football (in game)', 'passing American football (not in game)',
                              'peeling apples', 'peeling potatoes', 'petting animal (not cat)', 'petting cat',
                              'picking fruit', 'planting trees', 'plastering', 'playing accordion', 'playing badminton',
                              'playing bagpipes', 'playing basketball', 'playing bass guitar', 'playing cards',
                              'playing cello', 'playing chess', 'playing clarinet', 'playing controller',
                              'playing cricket',
                              'playing cymbals', 'playing didgeridoo', 'playing drums', 'playing flute',
                              'playing guitar',
                              'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard',
                              'playing kickball', 'playing monopoly', 'playing organ', 'playing paintball',
                              'playing piano',
                              'playing poker', 'playing recorder', 'playing saxophone', 'playing squash or racquetball',
                              'playing tennis', 'playing trombone', 'playing trumpet', 'playing ukulele',
                              'playing violin',
                              'playing volleyball', 'playing xylophone', 'pole vault', 'presenting weather forecast',
                              'pull ups', 'pumping fist', 'pumping gas', 'punching bag', 'punching person (boxing)',
                              'push up', 'pushing car', 'pushing cart', 'pushing wheelchair', 'reading book',
                              'reading newspaper', 'recording music', 'riding a bike', 'riding camel',
                              'riding elephant',
                              'riding mechanical bull', 'riding mountain bike', 'riding mule',
                              'riding or walking with horse', 'riding scooter', 'riding unicycle', 'ripping paper',
                              'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating',
                              'running on treadmill', 'sailing', 'salsa dancing', 'sanding floor', 'scrambling eggs',
                              'scuba diving', 'setting table', 'shaking hands', 'shaking head', 'sharpening knives',
                              'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes',
                              'shooting basketball', 'shooting goal (soccer)', 'shot put', 'shoveling snow',
                              'shredding paper', 'shuffling cards', 'side kick', 'sign language interpreting',
                              'singing',
                              'situp', 'skateboarding', 'ski jumping', 'skiing (not slalom or crosscountry)',
                              'skiing crosscountry', 'skiing slalom', 'skipping rope', 'skydiving', 'slacklining',
                              'slapping', 'sled dog racing', 'smoking', 'smoking hookah', 'snatch weight lifting',
                              'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling',
                              'somersaulting', 'spinning poi', 'spray painting', 'spraying', 'springboard diving',
                              'squat',
                              'sticking tongue out', 'stomping grapes', 'stretching arm', 'stretching leg',
                              'strumming guitar', 'surfing crowd', 'surfing water', 'sweeping floor',
                              'swimming backstroke',
                              'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs',
                              'swinging on something', 'sword fighting', 'tai chi', 'taking a shower', 'tango dancing',
                              'tap dancing', 'tapping guitar', 'tapping pen', 'tasting beer', 'tasting food',
                              'testifying',
                              'texting', 'throwing axe', 'throwing ball', 'throwing discus', 'tickling', 'tobogganing',
                              'tossing coin', 'tossing salad', 'training dog', 'trapezing', 'trimming or shaving beard',
                              'trimming trees', 'triple jump', 'tying bow tie', 'tying knot (not on a tie)',
                              'tying tie',
                              'unboxing', 'unloading truck', 'using computer', 'using remote controller (not gaming)',
                              'using segway', 'vault', 'waiting in line', 'walking the dog', 'washing dishes',
                              'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding',
                              'watering plants', 'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs',
                              'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present', 'wrestling',
                              'writing', 'yawning', 'yoga', 'zumba']

    action_dict_encode = {label: act_id for act_id, label in enumerate(kinetics_action_labels)}
    action_dict_decode = {act_id: label for act_id, label in enumerate(kinetics_action_labels)}

    @staticmethod
    def encode_action(action_name, zero_indexed=True):
        '''give action name, return category'''
        kdu = KineticsDatasetUtils
        return kdu.action_dict_encode[action_name] if zero_indexed else kdu.action_dict_encode[action_name] + 1

    @staticmethod
    def decode_action(action_code, zero_indexed=True):
        '''give action code, return action name'''
        kdu = KineticsDatasetUtils
        return kdu.action_dict_decode[action_code] if zero_indexed else kdu.action_dict_decode[action_code - 1]

    @staticmethod
    def read_video_info(video_info_csv, extract_infos=True, max_samples=None, worker_count=None,
                        random_state=42) -> pd.DataFrame:
        kdu = KineticsDatasetUtils

        sample_infos = pd.read_csv(video_info_csv, header=0, names=["path", "frame_count"])

        if max_samples is not None:
            sample_infos = sample_infos.sample(min(max_samples, len(sample_infos)), random_state=random_state)

        if extract_infos:
            if worker_count == 0:
                sample_infos = kdu.extract_infos(sample_infos)
            else:
                procs = mp.cpu_count() if worker_count is None else worker_count
                print("Using multiprocessing with {} processes.".format(procs))
                df_splits = np.array_split(sample_infos, procs)
                pool = mp.Pool(procs)

                sample_infos = pd.concat(pool.map(kdu.extract_infos, df_splits))

                pool.close()
                pool.join()

            sample_infos = sample_infos.set_index(["id"], drop=False)
            return sample_infos
        else:
            return sample_infos

    @staticmethod
    def extract_infos(sample_infos: pd.DataFrame):
        kdu = KineticsDatasetUtils

        sample_infos["id"] = ""
        sample_infos["action"] = 0
        sample_infos["time_start"] = 0
        sample_infos["time_end"] = 0
        sample_infos["file_name"] = ""

        sample_infos = sample_infos.astype(dtype={"id":         'string',
                                                  "file_name":  'string',
                                                  "action":     np.dtype("uint16"),
                                                  "time_start": np.dtype("uint32"),
                                                  "time_end":   np.dtype("uint32")})

        for row in tqdm(sample_infos.itertuples(), total=len(sample_infos)):
            idx = row.Index
            path = row.path
            base_path, file_name = os.path.split(path)

            _, action = os.path.split(base_path)
            action = kdu.encode_action(action)

            sample_id = file_name[0:11]
            time_start = int(file_name[12:18])
            time_end = int(file_name[19:25])

            sample_infos.loc[idx, ["id", "action", "time_start", "time_end",
                                   "file_name"]] = sample_id, action, time_start, time_end, file_name

        return sample_infos

    @staticmethod
    def filter_kinetics_by_split_mode(sample_info, split, split_mode, split_frac=0.2, random_state=42):
        kdu = KineticsDatasetUtils

        # Splits
        # Cross setup mode
        if split_mode == "perc":
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
    def get_skeleton_length(path):
        sk_seq = np.load(path)

        sk_seq = sk_seq['arr_0']

        (J, T, C) = sk_seq.shape

        return T

    @staticmethod
    def df_get_skeleton_length(df: pd.DataFrame):
        kdu = KineticsDatasetUtils
        df["frame_count"] = df["sk_path"].progress_apply(kdu.get_skeleton_length)
        return df

    @staticmethod
    def get_skeleton_info(skele_motion_root, worker_count=None) -> pd.DataFrame:
        kdu = KineticsDatasetUtils

        skeleton_paths = glob.glob(os.path.join(skele_motion_root, "*.npz"))
        sk_info = pd.DataFrame(skeleton_paths, columns=["sk_path"])

        sk_info["sk_file"] = sk_info["sk_path"].apply(lambda p: os.path.split(p)[1])
        sk_info["id"] = sk_info["sk_file"].apply(lambda fl: fl[:11])
        sk_info = sk_info.astype(dtype={"id": 'string'})

        sk_info["body"] = sk_info["sk_file"].apply(lambda fl: int(fl[12:].split("_")[0]))

        def _extract_type(file):
            mag_match = kdu.sk_magnitude_pattern.match(file)
            if mag_match:
                return mag_match.group(1)
            else:
                ori_match = kdu.sk_orientation_pattern.match(file)

                if ori_match:
                    return ori_match.group(1)
                else:
                    return np.NaN

        sk_info["skeleton_info_type"] = sk_info["sk_file"].apply(_extract_type)

        if worker_count != 0:
            procs = mp.cpu_count() if worker_count is None else worker_count
            print("Using multiprocessing with {} processes.".format(procs))
            df_split = np.array_split(sk_info, procs)
            pool = mp.Pool(procs)

            sk_info = pd.concat(pool.map(kdu.df_get_skeleton_length, df_split))

            pool.close()
            pool.join()
        else:
            sk_info["frame_count"] = sk_info["sk_path"].apply(kdu.get_skeleton_length)

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
