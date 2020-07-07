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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data

from dataset_3d import DatasetUtils
from dataset_nturgbd import NTURGBDDatasetUtils

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.plugin.pytorch as ndpt
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI.")

sys.path.append('../utils')
import utils as utl


class NTURGB3DInputReader(data.Dataset):
    """
    This one loads the images from disk.
    """

    def __init__(self,
                 nturgbd_video_info=None,
                 skele_motion_root=None,
                 split='train',
                 seq_len=30,
                 downsample=1,
                 return_label=False,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None,
                 image_min_height=150,
                 image_min_width=266,
                 aug_settings=None):
        self.split = split
        self.seq_len = seq_len
        self.downsample = downsample
        self.return_label = return_label
        self.split_mode = split_mode
        self.use_skeleton = skele_motion_root is not None

        self.image_min_height = image_min_height
        self.image_min_width = image_min_width

        self.sample_info = None
        self.sk_info = None

        self.aug_rotation_range = (-10., 10.) if aug_settings is None else aug_settings["rot_range"]
        self.aug_hue_range = (-3, 3) if aug_settings is None else aug_settings["hue_range"]
        self.aug_saturation_range = (0., 1.3) if aug_settings is None else aug_settings["sat_range"]
        self.aug_value_range = (0.5, 1.5) if aug_settings is None else aug_settings["val_range"]
        self.aug_hue_change_prop = 0.5 if aug_settings is None else aug_settings["hue_prob"]
        self.aug_crop_area_range = (0.15, 1.) if aug_settings is None else aug_settings["crop_arr_range"]

        if self.split == "test":
            self.aug_rotation_range = (0., 0)
            self.aug_hue_range = (0., 0.)
            self.aug_saturation_range = (1., 1.)
            self.aug_value_range = (1., 1.)
            self.aug_hue_change_prop = 0.
            self.aug_crop_area_range = (1., 1.)

        ndu = NTURGBDDatasetUtils

        print("=================================")
        print('Dataset NTURGBD {} split (Split method: {})'.format(split, split_mode))
        if split_mode == "perc":
            print("Train/Val ratio: {}/{}".format(1 - split_frac, split_frac))
        print('Using the NVIDIA DALI pipeline for data loading and preparation via GPU.')
        print("Assuming min image height {} and min image width {}".format(self.image_min_height, self.image_min_width))

        self.sample_info = ndu.read_video_info(nturgbd_video_info, max_samples=sample_limit)

        min_frame_count = self.seq_len * self.downsample

        sample_count = len(self.sample_info)
        self.sample_info = ndu.filter_too_short(self.sample_info, min_frame_count)
        print("Dropped {} of {} samples due to insufficient rgb video length ({} frames needed).".format(
            sample_count - len(self.sample_info), sample_count, min_frame_count))

        sample_count = len(self.sample_info)
        self.sample_info = ndu.filter_nturgbd_by_split_mode(self.sample_info, self.split, self.split_mode, split_frac)
        print("Selected {} of {} video samples for the {} split.".format(len(self.sample_info), sample_count,
                                                                         self.split))
        sample_count = len(self.sample_info)
        self.sk_info = ndu.get_skeleton_info(skele_motion_root)
        self.sample_info = ndu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
        print(
            "Dropped {} of {} samples due to missing skeleton information.".format(sample_count - len(self.sample_info),
                                                                                   sample_count))

        print("Remaining videos in mode {}: {}".format(self.split, len(self.sample_info)))

        # The original approach always used a subset of the test set for validation. Doing the same for comparability.
        if self.split == "val":
            if len(self.sample_info) > 1000:
                print(
                    "Limited the validation sample to 500 to speed up training. "
                    "This does not alter the structure of the train/test/val splits, " +
                    "it only reduces the samples used for validation in training among the val split.")
                self.sample_info = self.sample_info.sample(n=1000, random_state=666)

        print("=================================")

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, index):
        # The caller key will always be in range(0, len(self))

        sample = self.sample_info.iloc[index]

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

                return (img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs,
                        seq_crop_xs, seq_crop_ys, sk_seq, label)

            else:
                return (img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs,
                        seq_crop_xs, seq_crop_ys, sk_seq)

        if self.return_label:
            label = sample["action"]
            return (img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs,
                    seq_crop_xs, seq_crop_ys, label)
        else:
            return (img_seq, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs,
                    seq_crop_xs, seq_crop_ys)

    def random_image_transforms(self, frame_count,
                                rotation_range=None,
                                hue_range=None,
                                saturation_range=None,
                                value_range=None,
                                hue_change_prop=None,
                                crop_area_range=None):
        rotation_range = self.aug_rotation_range if rotation_range is None else rotation_range
        hue_range = self.aug_hue_range if hue_range is None else hue_range
        saturation_range = self.aug_saturation_range if saturation_range is None else saturation_range
        value_range = self.aug_value_range if value_range is None else value_range
        hue_change_prop = self.aug_hue_change_prop if hue_change_prop is None else hue_change_prop
        crop_area_range = self.aug_crop_area_range if crop_area_range is None else crop_area_range

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

        crop_w, crop_h, crop_x, crop_y = utl.random_image_crop_square(min_area_n=min(crop_area_range),
                                                                      max_area_n=max(crop_area_range),
                                                                      image_width=self.image_min_width,
                                                                      image_height=self.image_min_height)

        crop_ws = np.repeat(crop_w, repeats=frame_count)
        crop_ws = crop_ws.astype(np.float32)
        crop_hs = np.repeat(crop_h, repeats=frame_count)
        crop_hs = crop_hs.astype(np.float32)
        crop_xs = np.repeat(crop_x, repeats=frame_count)
        crop_xs = crop_xs.astype(np.float32)
        crop_ys = np.repeat(crop_y, repeats=frame_count)
        crop_ys = crop_ys.astype(np.float32)

        return rotations, hues, saturations, values, crop_ws, crop_hs, crop_xs, crop_ys


class NTURGBD3DPipeline(Pipeline):
    """
    This one performs data augmentation on GPU.
    """

    def __init__(self, batch_size, num_threads, nturgbd_input_loader, device_id=0, seed=42, prefetch_queue_depth=2,
                 normalize=True, output_layout="CHW"):
        super(NTURGBD3DPipeline, self).__init__(batch_size=batch_size,
                                                num_threads=num_threads,
                                                device_id=device_id,
                                                seed=seed,
                                                prefetch_queue_depth=prefetch_queue_depth)
        self.external_data = nturgbd_input_loader
        self.iterator = iter(self.external_data)

        self.output_normalized = normalize
        self.output_layout = output_layout

        self.loading_times = []

        self.batch_order = ops.ExternalSource(device="cpu")

        self.input_imgs = ops.ExternalSource(device="cpu")

        # Somehow the parameters for the operations have to be on CPU.
        self.input_angles = ops.ExternalSource(device="cpu")
        self.input_hues = ops.ExternalSource(device="cpu")
        self.input_saturations = ops.ExternalSource(device="cpu")
        self.input_values = ops.ExternalSource(device="cpu")

        self.input_crop_ws = ops.ExternalSource(device="cpu")
        self.input_crop_hs = ops.ExternalSource(device="cpu")
        self.input_crop_xs = ops.ExternalSource(device="cpu")
        self.input_crop_ys = ops.ExternalSource(device="cpu")

        self.input_sk_seq = ops.ExternalSource(device="cpu")

        self.img_dec = ops.ImageDecoder(device="mixed", bytes_per_sample_hint=500000, device_memory_padding=500000)

        self.rrot = ops.Rotate(interp_type=types.INTERP_LINEAR, device="gpu")  # angle=10,
        self.rcrop = ops.Crop(device="gpu")
        self.rrsize = ops.Resize(resize_x=128., resize_y=128., interp_type=types.INTERP_TRIANGULAR, device="gpu")
        self.rhsv = ops.Hsv(device="gpu")

        if self.output_normalized:
            self.normalize = ops.CropMirrorNormalize(
                device="gpu",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=0,
                output_layout=output_layout)
        else:
            self.transpose = ops.Transpose(perm=[0, 1, 2], transpose_layout=False, output_layout=output_layout,
                                           device="gpu")

        self.batch_order_id = None
        self.img_seq = None
        self.in_angle = None
        self.in_hue = None
        self.in_saturation = None
        self.in_value = None

        self.in_crop_w = None
        self.in_crop_h = None
        self.in_crop_x = None
        self.in_crop_y = None

        self.sk_seq = None

    def define_graph(self):
        self.batch_order_id = self.batch_order()

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

        if self.output_normalized:
            image = self.normalize(image)
        else:
            image = self.transpose(image)

        return image.gpu(), self.sk_seq.gpu(), self.batch_order_id

    def iter_setup(self):
        try:
            start_loading = time.perf_counter()
            (img_seqs, seq_rotations, seq_hues, seq_saturations, seq_values, seq_crop_ws, seq_crop_hs, seq_crop_xs,
             seq_crop_ys, sk_seqs) = next(self.iterator)

            img_seqs = [img for seq in img_seqs for img in seq]
            sk_seqs = np.stack(sk_seqs, 0)

            if not sk_seqs.dtype == np.float32:
                sk_seqs = sk_seqs.astype(np.float32)

            (B, Bo, C, T, J) = sk_seqs.shape
            sk_seqs = sk_seqs.repeat(T, axis=0)

            seq_rotations = np.array(seq_rotations).reshape(B * T)
            seq_hues = np.array(seq_hues).reshape(B * T, 1)
            seq_saturations = np.array(seq_saturations).reshape(B * T, 1)
            seq_values = np.array(seq_values).reshape(B * T, 1)
            seq_crop_ws = np.array(seq_crop_ws).reshape(B * T, 1)
            seq_crop_hs = np.array(seq_crop_hs).reshape(B * T, 1)
            seq_crop_xs = np.array(seq_crop_xs).reshape(B * T, 1)
            seq_crop_ys = np.array(seq_crop_ys).reshape(B * T, 1)

            end_loading = time.perf_counter()

            self.loading_times.append(end_loading - start_loading)

            if len(self.loading_times) % len(self.external_data) == 0:
                # This will be printed for every prefetch, so actually this might happen multiple times.
                print("Image loading average time: {}".format(np.mean(self.loading_times[-len(self.external_data):])))

            self.feed_input(self.batch_order_id, np.arange(start=0, stop=len(img_seqs), dtype=np.int32))

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


class NTURGBD3DDali:

    def __init__(self,
                 split='train',
                 nturgbd_video_info=None,
                 skele_motion_root=None,
                 batch_size=5,
                 seq_len=30,
                 downsample=1,
                 return_label=False,
                 split_mode="perc",
                 split_frac=0.1,
                 sample_limit=None,
                 num_workers_loader=0,
                 num_workers_dali=0,
                 dali_prefetch_queue_depth=2,
                 dali_devices=(0,),
                 aug_settings=None):
        self.split = split
        self.split_mode = split_mode
        self.seq_len = seq_len
        self.downsample = downsample
        self.return_label = return_label

        self.use_skeleton = skele_motion_root is not None
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dali_devices = dali_devices

        # The input reader handles loading video samples from disk. Decoding is done via DALI, buffer loading only.
        self.nir = NTURGB3DInputReader(nturgbd_video_info=nturgbd_video_info,
                                       skele_motion_root=skele_motion_root,
                                       seq_len=seq_len,
                                       downsample=downsample,
                                       return_label=return_label,
                                       split=split,
                                       split_mode=split_mode,
                                       split_frac=split_frac,
                                       sample_limit=sample_limit,
                                       aug_settings=aug_settings)

        # Using a random samples is extremely important, otherwise the network learns the dataset by heart.
        # TODO: How can it learn, which videos go together with seq sampler?
        #       Videos are sampled on different positions each time.
        sampler = torch.utils.data.RandomSampler(self.nir)

        # The nir loader provides batches with data shape (D,B,F,...) where D is the number of value types
        # (image buffers, skeleton data, augmentation settings etc.) The dimension D describes a list.
        self.nir_loader = data.DataLoader(dataset=self.nir, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers_loader, pin_memory=True,
                                          collate_fn=NTURGBD3DDali.list_collate, sampler=sampler, drop_last=True)

        # The DALI pipeline is really not good with handling video sequences, so actually the frames are handled like
        # individual batch samples. The pipeline completely encapsulates this behaviour for the input, but we
        # have to reshape the output accordingly.
        self.pipeline = NTURGBD3DPipeline(batch_size=batch_size * seq_len,
                                          num_threads=num_workers_dali,
                                          nturgbd_input_loader=self.nir_loader,
                                          device_id=dali_devices[0],
                                          prefetch_queue_depth=dali_prefetch_queue_depth,
                                          seed=42,
                                          output_layout="CHW")

        self.pipeline.build()
        self.first_run = True
        self.pipeline.schedule_run()

    def __iter__(self):
        if not self.first_run:
            self.pipeline.reset()
            self.pipeline.schedule_run()
        else:
            self.first_run = False

        return self

    def __next__(self):
        input_seq_dali, sk_seq_dali, batch_order = self.pipeline.share_outputs()  # (B*T, ...)
        input_seq_dali = input_seq_dali.as_tensor()
        sk_seq_dali = sk_seq_dali.as_tensor()
        batch_order = batch_order.as_array()

        in_order = True
        for idx, i in enumerate(batch_order):
            if idx != int(i):
                in_order = False

        assert in_order, "The batch returned by the Dali Pipeline was found not to be in the same order " \
                         "as it was provided. This is a problem because we flatten the image sequences to be handled" \
                         "like individual frames and now we want to reshape them back into sequence samples."

        cuda_device = torch.cuda.current_device()
        cuda_stream = torch.cuda.current_stream()

        # noinspection PyTypeChecker
        input_seq = torch.zeros(input_seq_dali.shape(), dtype=torch.float32, device=cuda_device)
        ndpt.feed_ndarray(input_seq_dali, input_seq, cuda_stream=cuda_stream)

        # noinspection PyTypeChecker
        sk_seq = torch.zeros(sk_seq_dali.shape(), dtype=torch.float32, device=cuda_device)
        ndpt.feed_ndarray(sk_seq_dali, sk_seq, cuda_stream=cuda_stream)

        self.pipeline.release_outputs()
        self.pipeline.schedule_run()

        (BF, C, H, W) = input_seq.shape
        input_seq = input_seq.view(self.batch_size, self.seq_len, C, H, W)

        input_seq = input_seq.transpose(1, 2).float()
        # [B, C, SL, H, W]

        (BF, sk_Bo, sk_C, sk_T, sk_J) = sk_seq.shape
        sk_seq = sk_seq.view(self.batch_size, self.seq_len, sk_Bo, sk_C, sk_T, sk_J)
        sk_seq = sk_seq[:, 0, :]
        sk_seq = sk_seq.view(self.batch_size, sk_Bo, sk_C, sk_T, sk_J).float()
        # (Ba, Bo, C, T, J)

        # print("Tensor size image seqs in bytes: {}".format(input_seq.element_size() * input_seq.nelement()))
        # print("Tensor size sk seqs in bytes: {}".format(sk_seq.element_size() * sk_seq.nelement()))

        return input_seq, sk_seq

    def __len__(self):
        return len(self.nir_loader)

    @staticmethod
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

    def reset(self):
        self.pipeline.reset()


def show_images(image_batch, batch_size, seq_len):
    columns = seq_len
    rows = batch_size
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = gridspec.GridSpec(rows, columns, wspace=0.2, hspace=0.2)

    image_batch = image_batch.as_cpu().as_array()

    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img = image_batch[j]
        # img = np.transpose(img, (1, 2, 0))
        plt.imshow(img, interpolation="bicubic")

    plt.show()


def test():
    batch_size = 10
    seq_len = 30
    video_info_csv = os.path.expanduser("~/datasets/nturgbd/project_specific/dpc_converted/video_info.csv")
    skele_motion_root = os.path.expanduser("~/datasets/nturgbd/skele-motion")

    nii = NTURGB3DInputReader(nturgbd_video_info=video_info_csv, skele_motion_root=skele_motion_root,
                              seq_len=seq_len, sample_limit=100)

    pipeline = NTURGBD3DPipeline(batch_size=batch_size * seq_len, num_threads=6, device_id=0,
                                 normalize=True, output_layout="HWC",
                                 nturgbd_input_loader=data.DataLoader(dataset=nii, batch_size=batch_size, shuffle=False,
                                                                      num_workers=3, pin_memory=False,
                                                                      collate_fn=NTURGBD3DDali.list_collate))

    pipeline.build()

    images, sk_seqs, batch_order = pipeline.run()

    show_images(images, batch_size, seq_len)
    pass


if __name__ == "__main__":
    test()
