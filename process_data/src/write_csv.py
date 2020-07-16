import argparse
import csv
import glob
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)


def main_UCF101(f_root, splits_root, csv_root='../data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1, 2, 3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.split(' ')[0][0:-4]) + '/'
                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.rstrip()[0:-4]) + '/'
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


def main_HMDB51(f_root, splits_root, csv_root='../data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1, 2, 3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1]
                    vpath = os.path.join(f_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


### For Kinetics ###
def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []
    split_content = pd.read_csv(split_path).iloc[:, 0:4]
    split_list = Parallel(n_jobs=64) \
        (delayed(check_exists)(row, root) \
         for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list


def check_exists(row, root):
    dirname = '_'.join([row['youtube_id'], '%06d' % row['time_start'], '%06d' % row['time_end']])
    full_dirname = os.path.join(root, row['label'], dirname)
    if os.path.exists(full_dirname):
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        return None


def main_Kinetics400(f_root, csv_root):
    """
        This function prepares two files for the dataset: The list of video files for training and the list of video files for testing.
        Since NTURGBD does not define a fixed train test split, we just use a random 80 20 split.
        """
    if not os.path.exists(csv_root): os.makedirs(csv_root)

    # fid_p = re.compile(r"(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3})")  # A pattern object to recognize nturgbd file ids.
    # action_p = re.compile(r"S\d{3}C\d{3}P\d{3}R\d{3}A(\d{3})")  # A pattern object to recognize nturgbd action ids.

    fr_root, act_dirs, _ = next(os.walk(f_root))

    input_set = []
    for action_dir in tqdm(act_dirs, total=len(act_dirs)):
        act_root, vid_dirs, _ = next(os.walk(os.path.join(fr_root, action_dir)))

        for vid_dir in tqdm(vid_dirs, total=len(vid_dirs)):
            # action_id = action_p.match(video_folder).group()  # This extracts the action id (e.g. 001) as a string

            _, _, vid_frames = next(os.walk(os.path.join(act_root, vid_dir)))
            frame_count = len(vid_frames)

            input_set.append([os.path.join(act_root, vid_dir), frame_count])

    write_list(input_set, os.path.join(csv_root, 'video_info.csv'))


#  Making code comments immensely helps people to understand what this is supposed to do
def main_NTURGBD(f_root, csv_root):
    """
    This function prepares two files for the dataset: The list of video files for training and the list of video files for testing.
    Since NTURGBD does not define a fixed train test split, we just use a random 80 20 split.
    """
    if not os.path.exists(csv_root): os.makedirs(csv_root)

    # fid_p = re.compile(r"(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3})")  # A pattern object to recognize nturgbd file ids.
    # action_p = re.compile(r"S\d{3}C\d{3}P\d{3}R\d{3}A(\d{3})")  # A pattern object to recognize nturgbd action ids.

    root, dirs, files = next(os.walk(f_root))

    input_set = []
    for video_folder in tqdm(dirs, total=len(dirs)):
        # action_id = action_p.match(video_folder).group()  # This extracts the action id (e.g. 001) as a string

        vid_root, vid_dirs, vid_frames = next(os.walk(os.path.join(root, video_folder)))
        frame_count = len(vid_frames)

        input_set.append([os.path.join(root, video_folder), frame_count])

    write_list(input_set, os.path.join(csv_root, 'video_info.csv'))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='kinetics', choices=["kinetics", "nturgbd", "ucf101", "hmdb51"], type=str)
parser.add_argument('--f_root', default='', type=str, required=True)
parser.add_argument('--csv_root', default='', type=str, required=True)
parser.add_argument('--splits_root', default='', type=str)

if __name__ == '__main__':
    # f_root is the frame path
    # edit 'your_path' here:

    args = parser.parse_args()

    if args.dataset == "nturgbd":
        main_NTURGBD(f_root=args.f_root,
                     csv_root=args.csv_root)

    elif args.dataset == "ucf101":
        if args.splits_root == "":
            raise ValueError

        main_UCF101(f_root=args.f_root,
                    csv_root=args.csv_root,
                    splits_root=os.path.expanduser('~/datasets/UCF101/split/ucfTrainTestlist'),
                    )
    elif args.dataset == "hmdb51":
        if args.splits_root == "":
            raise ValueError

        main_HMDB51(f_root=args.f_root,
                    csv_root=args.csv_root,
                    splits_root=os.path.expanduser('~/datasets/HMDB51/split/testTrainMulti_7030_splits'),
                    )
    elif args.dataset == "kinetics":
        main_Kinetics400(f_root=args.f_root,
                         csv_root=args.csv_root,
                         )
    else:
        raise ValueError
