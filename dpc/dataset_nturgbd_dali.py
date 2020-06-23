"""
------------
dataset_nturgbd_dali
------------

What?
--------
A Dali implementation to prefetch training data directly to the GPU and perform augmentations there.
This is not always better (needs extra GPU memory and not always faster) but in situations where
a machine has lots of GPU power and limited CPU power (shared server with mutliple GPUs) this performs better.

How?
--------
The NTURGB3DInputIterator is an iterator which provides dataset samples.

Author: David Schneider david.schneider2@student.kit.edu
"""

import os
import sys
import time

from torch.utils import data

from dataset_3d import DatasetUtils

sys.path.append('../utils')
from augmentation import *

import numpy as np
from dataset_nturgbd import NTURGBDDatasetUtils

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class NTURGB3DInputIterator(data.Dataset):
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

    # def __iter__(self):
    #     self.i = 0
    #     self.indices = np.random.permutation(self.indices)
    #     return self
    #
    # def __next__(self):
    #     if self.i >= self.n:
    #         raise StopIteration
    #
    #     img_seqs, sk_seqs = [], []
    #     seq_rotations, seq_hues, seq_saturations, seq_values = [], [], [], []
    #     seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys = [], [], [], []
    #
    #     for _ in range(self.batch_size):
    #         index = self.indices[self.i % self.n]  # Fill missing batch samples by wrapping around.
    #
    #         img_seq, random_transforms, sk_seq = self[index]
    #         img_seqs.append(img_seq)
    #         sk_seqs.append(sk_seq)
    #
    #         seq_rotations.append(random_transforms[0])
    #         seq_hues.append(random_transforms[1])
    #         seq_saturations.append(random_transforms[2])
    #         seq_values.append(random_transforms[3])
    #
    #         seq_crop_ws.append(random_transforms[4])
    #         seq_crop_hs.append(random_transforms[5])
    #         seq_crop_xs.append(random_transforms[6])
    #         seq_crop_ys.append(random_transforms[7])
    #
    #         self.i = self.i + 1  # Preparing next iteration.
    #
    #     # sk_seqs = sk_seqs[0]
    #     # img_seqs = np.stack(img_seqs, 0)
    #     img_seqs_flat = [img for subl in img_seqs for img in subl]
    #     seq_rotations = np.stack(seq_rotations, 0)
    #     seq_hues = np.stack(seq_hues, 0)
    #     seq_saturations = np.stack(seq_saturations, 0)
    #     seq_values = np.stack(seq_values, 0)
    #
    #     seq_crop_ws = np.stack(seq_crop_ws, 0)
    #     seq_crop_hs = np.stack(seq_crop_hs, 0)
    #     seq_crop_xs = np.stack(seq_crop_xs, 0)
    #     seq_crop_ys = np.stack(seq_crop_ys, 0)
    #
    #     # B, F = img_seqs.shape[0], img_seqs.shape[1]
    #     # img_seqs = img_seqs.reshape(self.batch_size * F, -1)  # , H, W, C)
    #     F = self.seq_len
    #     seq_rotations = seq_rotations.reshape(self.batch_size * F, 1)
    #     seq_hues = seq_hues.reshape(self.batch_size * F, 1)
    #     seq_saturations = seq_saturations.reshape(self.batch_size * F, 1)
    #     seq_values = seq_values.reshape(self.batch_size * F, 1)
    #
    #     seq_crop_ws = seq_crop_ws.reshape(self.batch_size * F, 1)
    #     seq_crop_hs = seq_crop_hs.reshape(self.batch_size * F, 1)
    #     seq_crop_xs = seq_crop_xs.reshape(self.batch_size * F, 1)
    #     seq_crop_ys = seq_crop_ys.reshape(self.batch_size * F, 1)
    #
    #     sk_seqs = np.repeat(sk_seqs, repeats=F, axis=0)
    #
    #     return img_seqs_flat, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, sk_seqs

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # The caller key will always be in range(0, len(self))

        sample = self.sample_info.loc[self.indices[index]]

        frame_indices = DatasetUtils.idx_sampler(sample["frame_count"], self.seq_len, self.downsample, sample["path"])

        img_seq = []

        for i in frame_indices:
            img_seq.append(NTURGBDDatasetUtils.load_img_buffer(sample, i))

        (seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs,
         seq_crop_ys) = self.random_image_transforms(len(frame_indices))

        if self.use_skeleton:
            sk_seq = NTURGBDDatasetUtils.load_skeleton_seqs(self.sk_info, sample["id"], frame_indices)

            # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
            # (sk_Bo, sk_J, sk_T, sk_C) = sk_seq.shape

            # This is transposed, so we can split the image into blocks during training.
            sk_seq = np.transpose(sk_seq, axes=(0, 3, 2, 1))
            # (sk_Bo, sk_C, sk_T, sk_J) = sk_seq.shape

            if self.return_label:
                label = sample["action"]

                return img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, sk_seq, label

            else:
                return img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, sk_seq

        if self.return_label:
            label = sample["action"]
            return img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys, label
        else:
            return img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs, seq_crop_ys

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

            img_seqs = [img for seq in img_seqs for img in seq]
            sk_seqs = np.stack(sk_seqs, 0)

            (B, Bo, C, T, J) = sk_seqs.shape
            sk_seqs = sk_seqs.repeat(T, axis=0)

            seq_rotations = np.array(seq_rotations).reshape(B*T)
            seq_hues = np.array(seq_hues).reshape(B * T, 1)
            seq_saturations = np.array(seq_saturations).reshape(B * T, 1)
            seq_values = np.array(seq_values).reshape(B * T, 1)
            seq_crop_ws = np.array(seq_crop_ws).reshape(B * T, 1)
            seq_crop_hs = np.array(seq_crop_hs).reshape(B * T, 1)
            seq_crop_xs = np.array(seq_crop_xs).reshape(B * T, 1)
            seq_crop_ys = np.array(seq_crop_ys).reshape(B * T, 1)

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
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)

    plt.show()


def list_collate(batch):
    """Batch is a list of shape (N,D) where D stands for the return values in a single sample
    and N stands for the batch size.
    This collation functions converts that to a list of shape (D,N) which is the same behaviour as the default
    collation function without converting to a torch tensor first.
    We do not use the default collate function, since it can not handle variable size inputs
    (As they exist in undecoded images)."""
    assert len(batch) > 0

    val_count = len(batch[0])
    elems = [[] for i in range(val_count)]

    for sample in batch:
        for i in range(val_count):
            elems[i].append(sample[i])

    return elems


def test():
    batch_size = 2
    seq_len = 5
    video_info_csv = os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/video_info.csv")
    skele_motion_root = os.path.expanduser("~/datasets/nturgbd/skele-motion")

    nii = NTURGB3DInputIterator(nturgbd_video_info=video_info_csv, skele_motion_root=skele_motion_root,
                                batch_size=batch_size, seq_len=seq_len, sample_limit=100)

    pipeline = NTURGBD3DPipeline(batch_size=batch_size, seq_length=seq_len, num_threads=1, device_id=0,
                                 nturgbd_input_data=data.DataLoader(dataset=nii, batch_size=2, shuffle=False,
                                                                    num_workers=1, pin_memory=False,
                                                                    collate_fn=list_collate))

    pipeline.build()

    images, sk_seqs = pipeline.run()

    show_images(images, batch_size, seq_len)
    pass


if __name__ == "__main__":
    test()
