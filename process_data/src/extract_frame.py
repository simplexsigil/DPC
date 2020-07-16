import argparse
import glob
import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

plt.switch_backend('agg')


def extract_video_ffmpeg(v_path, f_root, dim=240, force=False, log_level="error"):
    '''v_path: single video path;
        f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        preexisted = False
    else:
        preexisted = True
        imgs = glob.glob(os.path.join(out_dir, '*.jpg'))

        if (not force) and len(imgs) > 30:
            print("SKIPPING {}: preexisting with {:3} images".format(v_name, len(imgs)))
            return

    ####
    # Construct command to trim the videos (ffmpeg required).

    command = [
        'ffmpeg',
        '-i', '"{v_path}"'.format(v_path=v_path),
        '-vf',
        'scale="if(gt(ih\,iw)\,{min_dim}\,-2)":"if(gt(ih\,iw)\,-2\,{min_dim})"'.format(min_dim=dim),
        '-r', '30',
        '-threads', '0',
        '-loglevel', str(log_level),
        '"{out_dir}/{out_name}_%04d.jpg"'.format(out_dir=out_dir, out_name=v_name)
        ]

    command = " ".join(command)

    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if len(output) > 0:
            print("RESULT " + v_path + "\n" + output.decode("utf-8", "replace"))
        else:
            print("SUCCESS " + v_path)


    except subprocess.CalledProcessError as err:
        print()
        output = err.output.decode("utf-8", "replace")
        output = output.split("\n")
        if len(output) > 20:
            output = output[:10] + ["...", "...", "..."] + output[-10:]

        output = "\n".join(output)

        print(str(err) + "\n" + output)


def extract_video_opencv(v_path, f_root, dim=240, force=False, log_level=None):
    '''v_path: single video path;
       f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)

    preexisted = True
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        preexisted = False

    vidcap = cv2.VideoCapture(v_path)

    if not vidcap.isOpened():
        print("Vidcap had to be manually opened for {}".format(v_name))
        vidcap.open(v_path)

    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if preexisted is True and force is False:
        imgs = glob.glob(os.path.join(out_dir, '*.jpg'))
        if abs(len(imgs) - nb_frames) < 2:
            print("Skipping extracted video {}".format(v_name))
            vidcap.release()
            return

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    if (width == 0) or (height == 0):
        print(v_path, 'not successfully loaded, drop ..');
        return

    new_dim = resize_dim(width, height, dim)

    count = 0
    success, image = vidcap.read()

    while success:
        count += 1
        image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, 'image_%05d.jpg' % count), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])  # quality from 0-100, 95 is default, high is good
        success, image = vidcap.read()

    # It is expected behaviour that cv2.CAP_PROP_FRAME_COUNT can be an estimation based on FPS and be off by a frame.
    if abs(nb_frames - count) > 1:
        print('/'.join(out_dir.split('/')[-2::]), ' The number of extracted frames differs from the expected number: '
                                                  '%df/%df' % (count, nb_frames))

    vidcap.release()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w))


def main_UCF101(v_root, f_root, dim=256, force=False, n_jobs=32, use_ocv=False, log_level="error"):
    print('extracting UCF101 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)

    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv if use_ocv else extract_video_ffmpeg)(p, f_root, force=force, dim=dim,
                                                                               log_level=log_level) for p in
            tqdm(v_paths, total=len(v_paths)))


def main_HMDB51(v_root, f_root, dim=256, force=False, n_jobs=32, use_ocv=False, log_level="error"):
    print('extracting HMDB51 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)

    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv if use_ocv else extract_video_ffmpeg)(p, f_root, force=force, dim=dim,
                                                                               log_level=log_level) for p in
            tqdm(v_paths, total=len(v_paths)))


def main_kinetics400(v_root, f_root, dim=256, force=False, n_jobs=32, use_ocv=False, log_level="error"):
    # Video root for reading videos, frame root for writing
    print('extracting Kinetics400 ... ')
    if not os.path.exists(f_root): os.makedirs(f_root)

    if not os.path.exists(v_root):
        print('Wrong v_root')
        sys.exit()
    print('Extract to: \nframe: %s' % f_root)

    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    v_act_root = sorted(v_act_root)

    # if resume, remember to delete the last video folder
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.mp4'))
        v_paths = sorted(v_paths)
        # for resume:
        v_class = j.split('/')[-2]
        out_dir = os.path.join(f_root, v_class)
        print('\nextracting: %s' % v_class)
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv if use_ocv else extract_video_ffmpeg)(p, f_root, force=force, dim=dim,
                                                                               log_level=log_level) for p in
            tqdm(v_paths, total=len(v_paths)))

    sys.exit(0)


def main_nturgbd(v_root, f_root, dim=256, force=False, n_jobs=32, use_ocv=False, log_level="error"):
    # Create universally usable pattern matchers.
    # fid_p = re.compile(r"(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3})")  # A pattern object to recognize nturgbd file ids.

    print('extracting NTURGBD ... ')
    if not os.path.exists(v_root):
        print("Video folder not found:\n{}".format(v_root))
        sys.exit()

    if not os.path.exists(f_root):
        print("Creating output folder:\n{}".format(f_root))
        os.makedirs(f_root)

    print("Reading from: {}\nWriting to: {}".format(v_root, f_root))

    v_files = next(os.walk(v_root))[2]
    v_paths = [os.path.join(v_root, v_file) for v_file in v_files]

    # The file identifiers are tupled with their respective complete file names.
    # files = list(zip([fid_p.match(file).group() for file in files], files))  # [(file_id, filename),...]

    v_count = len(v_paths)
    v_pathss = [v_paths[int(v_count / 100) * i:int(v_count / 100) * (i + 1)] for i in range(100)]

    for i, v_paths in tqdm(enumerate(v_pathss), total=len(v_pathss)):
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv if use_ocv else extract_video_ffmpeg)(v_path, f_root, force=force, dim=dim,
                                                                               log_level=log_level) for v_path in
            tqdm(v_paths, total=len(v_paths)))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='kinetics', choices=["kinetics", "nturgbd", "ucf101", "hmdb51"], type=str)
parser.add_argument('--v_root', default='', type=str, required=True)
parser.add_argument('--f_root', default='', type=str, required=True)
parser.add_argument('--img_dim', default=256, type=int)
parser.add_argument('--n_jobs', default=32, type=int)
parser.add_argument('--force', action='store_true', default=False, help="Overwrites existing files.")
parser.add_argument('--open_cv', action='store_true', default=False,
                    help="Uses open cv instead of ffmpeg for extracting images.")
parser.add_argument('--log_level', default="level+error",
                    help="Depending on the version of ffmpeg, the level+error notation might not be accepted.")

if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    # edit 'your_path' here:

    args = parser.parse_args()

    if args.n_jobs > 1 and not args.open_cv:
        print("WARNING: For unknown reasons, ffmpeg can fail horribly when executed in parallel with multiple jobs.")

    if args.dataset == "nturgbd":
        main_nturgbd(v_root=args.v_root,
                     f_root=args.f_root,
                     dim=args.img_dim,
                     force=args.force,
                     n_jobs=args.n_jobs,
                     use_ocv=args.open_cv,
                     log_level=args.log_level)
    elif args.dataset == "ucf101":
        main_UCF101(v_root=args.v_root,
                    f_root=args.f_root,
                    dim=args.img_dim,
                    force=args.force,
                    n_jobs=args.n_jobs,
                    use_ocv=args.open_cv,
                    log_level=args.log_level)
    elif args.dataset == "hmdb51":
        main_HMDB51(v_root=args.v_root,
                    f_root=args.f_root,
                    dim=args.img_dim,
                    force=args.force,
                    n_jobs=args.n_jobs,
                    use_ocv=args.open_cv,
                    log_level=args.log_level)
    elif args.dataset == "kinetics":
        main_kinetics400(v_root=args.v_root,
                         f_root=args.f_root,
                         dim=args.img_dim,
                         force=args.force,
                         n_jobs=args.n_jobs,
                         use_ocv=args.open_cv,
                         log_level=args.log_level)
    else:
        raise ValueError
