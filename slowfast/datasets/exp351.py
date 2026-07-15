#!/usr/bin/env python3

import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)

# Which agent's gaze/view to load. Defaults to "child"; set env EXP351_AGENT=parent
# to evaluate/train on the parent stream (cam08) instead of the child (cam07).
_AGENT = os.environ.get("EXP351_AGENT", "child").lower()
assert _AGENT in ("child", "parent"), \
    "EXP351_AGENT must be 'child' or 'parent', got {!r}".format(_AGENT)

_SPLIT_TO_JSON = {
    "train": "data/chen/exp351/{}_eye_data_train.json".format(_AGENT),
    "val":   "data/chen/exp351/{}_eye_data_val.json".format(_AGENT),
    "test":  "data/chen/exp351/{}_eye_data_test.json".format(_AGENT),
}

_FPS = 30.0


@DATASET_REGISTRY.register()
class Exp351(torch.utils.data.Dataset):
    """
    Experiment 351 egocentric gaze dataset.
    5-second video chunks at 30fps with pre-extracted JPEG frames.
    Gaze annotations are inline per frame in the JSON.
    Follows the same preprocessing pipeline as Egteagaze.
    """

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], \
            "Split '{}' not supported for Exp351".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        else:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing Exp351 {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        path_to_file = _SPLIT_TO_JSON[self.mode]
        assert pathmgr.exists(path_to_file), "{} not found".format(path_to_file)

        with pathmgr.open(path_to_file, "r") as f:
            chunks = json.load(f)

        # Probe one image per unique subject_dir to get (W, H) for gaze normalization.
        # Subjects have multiple camera resolutions: 1280x720 and 1600x1200.
        self._subject_res = {}
        for chunk_data in chunks:
            sdir = chunk_data["subject_dir"]
            if sdir not in self._subject_res:
                fpath = chunk_data["frames"][0]["frame_path"]
                img = Image.open(fpath)
                self._subject_res[sdir] = img.size  # (W, H)
                img.close()

        self._chunks = []
        self._spatial_temporal_idx = []

        for chunk_data in chunks:
            for idx in range(self._num_clips):
                self._chunks.append(chunk_data)
                self._spatial_temporal_idx.append(idx)

        if self.mode == "train":
            random.shuffle(self._chunks)
            # _spatial_temporal_idx is all zeros in train (num_clips=1), no need to co-shuffle

        logger.info("Constructing exp351 dataloader (size: {}) from {}".format(
            len(self._chunks), path_to_file))

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode == "train":
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(
                    self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                    * self.cfg.MULTIGRID.DEFAULT_S
                ))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        #Here is where most of the logic is implemented
        for i_try in range(self._num_retries):
            chunk = self._chunks[index]
            frames_list = chunk["frames"]
            N = len(frames_list)

            # Temporal window: same formula as EGTEA/PyAV decode
            clip_size = sampling_rate * self.cfg.DATA.NUM_FRAMES / self.cfg.DATA.TARGET_FPS * _FPS
            start_idx, end_idx = decoder.get_start_end_idx(
                video_size=N,
                clip_size=clip_size,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            )

            # Select NUM_FRAMES indices evenly from [start_idx, end_idx]
            frame_indices = torch.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES)
            frame_indices = torch.clamp(frame_indices, 0, N - 1).long()

            # Load frames from disk
            try:
                imgs = []
                for fi in frame_indices:
                    fpath = frames_list[int(fi)]["frame_path"]
                    img = np.array(Image.open(fpath).convert("RGB"))
                    imgs.append(img)
                frames = torch.as_tensor(np.stack(imgs))  # T H W C, uint8
            except Exception as e:
                logger.warning("Failed to load frames for chunk idx {} trial {}: {}".format(
                    index, i_try, e))
                if self.mode != "test" and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._chunks) - 1)
                continue

            # Build gaze label: (T, 4) with [px, py, gaze_type, 0]
            # gaze_type: 0=untracked (None), 1=tracked in-bounds, 4=truncated out-of-bounds
            gaze_w, gaze_h = self._subject_res[chunk["subject_dir"]]
            label = self._parse_gaze(frames_list, frame_indices, gaze_w, gaze_h)

            # Set untracked frames to image center (placeholder; heatmap will use uniform)
            label[:, 0][label[:, 2] == 0] = 0.5
            label[:, 1][label[:, 2] == 0] = 0.5

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:
                    frame_list, label_list, index_list = [], [], []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames, spatial_sample_index, min_scale, max_scale, crop_size
                        )
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}
                else:
                    frames = self._aug_frame(
                        frames, spatial_sample_index, min_scale, max_scale, crop_size
                    )
            else:
                frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                # T H W C -> C T H W
                frames = frames.permute(3, 0, 1, 2)
                frames, label = utils.spatial_sampling(
                    frames,
                    gaze_loc=label,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            frames = utils.pack_pathway_output(self.cfg, frames)

            # Build Gaussian heatmap at 1/4 spatial resolution: (T, H//4, W//4)
            T = frames[0].size(1)
            hm_h = frames[0].size(2) // 4
            hm_w = frames[0].size(3) // 4
            label_hm = np.zeros((T, hm_h, hm_w), dtype=np.float32)
            for i in range(T):
                if label[i, 2] == 0:  # untracked -> uniform
                    label_hm[i] += 1.0 / (hm_h * hm_w)
                else:
                    self._get_gaussian_map(
                        label_hm[i],
                        center=(label[i, 0] * hm_w, label[i, 1] * hm_h),
                        kernel_size=self.cfg.DATA.GAUSSIAN_KERNEL,
                        sigma=-1,
                    )
                    d_sum = label_hm[i].sum()
                    if d_sum == 0:  # gaze outside heatmap bounds -> uniform
                        label_hm[i] += 1.0 / (hm_h * hm_w)
                    elif d_sum != 1:
                        label_hm[i] /= d_sum

            label_hm = torch.as_tensor(label_hm).float()
            meta = {
                "chunk_idx": chunk.get("idx"),
                "frame_indices": frame_indices.numpy(),
                # Original (pre-crop) subject camera resolution in pixels, same
                # convention tools/dino_probe.py's GazeDataset uses for its "dims"
                # -- lets downstream code (e.g. AAE) key a per-resolution FOV table.
                "dims": np.array([gaze_w, gaze_h], dtype=np.float32),
            }
            # Per-frame gaze-target ROI bbox for the bbox-accuracy eval metric.
            # Only meaningful on the deterministic val/test crop; skipped in the
            # train aug path. Rows for frames without a 'roi_bbox' annotation are
            # NaN (dropped downstream).
            if self.mode in ("val", "test"):
                meta["roi_bbox"] = self._parse_roi_bbox(
                    frames_list, frame_indices, gaze_w, gaze_h,
                    spatial_sample_index, crop_size)
            return frames, label, label_hm, index, meta

        raise RuntimeError("Failed to fetch chunk after {} retries.".format(self._num_retries))

    def _parse_gaze(self, frames_list, frame_indices, gaze_w, gaze_h):
        """
        Build a (num_frames, 4) label array.
        Columns: [px_norm, py_norm, gaze_type, 0]
        gaze_type: 0=untracked (None), 1=tracked in-bounds, 4=truncated out-of-bounds.
        gaze_w/gaze_h are the frame pixel dimensions for this subject.
        """
        label = np.zeros((len(frame_indices), 4), dtype=np.float32)
        for row, fi in enumerate(frame_indices):
            frame = frames_list[int(fi)]
            gx = frame.get("gaze_x")
            gy = frame.get("gaze_y")
            if gx is None or gy is None:
                label[row, 2] = 0  # untracked
            elif gx < 0 or gx > gaze_w - 1 or gy < 0 or gy > gaze_h - 1:
                label[row, 2] = 4  # truncated out-of-bounds
                label[row, 0] = np.clip(gx, 0, gaze_w - 1) / gaze_w
                label[row, 1] = np.clip(gy, 0, gaze_h - 1) / gaze_h
            else:
                label[row, 2] = 1  # tracked
                label[row, 0] = gx / gaze_w
                label[row, 1] = gy / gaze_h
        return label

    def _parse_roi_bbox(self, frames_list, frame_indices, gaze_w, gaze_h,
                        spatial_sample_index, crop_size):
        """Per-sampled-frame gaze-target ROI bbox as (num_frames, 4) corner
        coords [x1, y1, x2, y2], normalized to the *cropped* frame space so it
        lines up with the predicted / target gaze point (which the val/test
        pipeline expresses in that same space). Frames without a 'roi_bbox'
        annotation become NaN rows.

        Stored roi_bbox is normalized [x_center, y_center, w, h] in the original
        camera frame (see gaze/get_data_351_fullfps.m and
        gaze/draw_bbox_examples.py). Here we convert to corners and replay the
        deterministic val/test transform (short-side scale to crop_size +
        uniform crop, matching transform.random_short_side_scale_jitter and
        transform.uniform_crop_gaze).
        """
        out = np.full((len(frame_indices), 4), np.nan, dtype=np.float32)

        # 1) short-side scale to `crop_size` (min_scale == max_scale here).
        W0, H0, S = float(gaze_w), float(gaze_h), float(crop_size)
        if W0 < H0:
            sw, sh = S, math.floor(H0 / W0 * S)
        else:
            sh, sw = S, math.floor(W0 / H0 * S)

        # 2) uniform crop offsets (spatial_sample_index: 0/1/2 -> start/center/end
        #    along the longer axis), matching transform.uniform_crop_gaze.
        x_off = math.ceil((sw - S) / 2)
        y_off = math.ceil((sh - S) / 2)
        if sh > sw:
            if spatial_sample_index == 0:
                y_off = 0
            elif spatial_sample_index == 2:
                y_off = sh - S
        else:
            if spatial_sample_index == 0:
                x_off = 0
            elif spatial_sample_index == 2:
                x_off = sw - S

        def _map(nx, ny):
            cx = min(max((nx * sw - x_off) / S, 0.0), 1.0)
            cy = min(max((ny * sh - y_off) / S, 0.0), 1.0)
            return cx, cy

        for row, fi in enumerate(frame_indices):
            bb = frames_list[int(fi)].get("roi_bbox")
            if not bb:
                continue
            xc, yc, w, h = bb
            x1, y1 = _map(xc - w / 2.0, yc - h / 2.0)
            x2, y2 = _map(xc + w / 2.0, yc + h / 2.0)
            out[row] = [x1, y1, x2, y2]
        return out

    def _aug_frame(self, frames, spatial_sample_index, min_scale, max_scale, crop_size):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W
        frames = frames.permute(0, 3, 1, 2)
        list_img = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        list_img = aug_transform(list_img)
        list_img = [transforms.ToTensor()(img) for img in list_img]
        frames = torch.stack(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W
        frames = frames.permute(3, 0, 1, 2)
        scl = self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE
        asp = self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
        relative_scales = None if (self.mode != "train" or len(scl) == 0) else scl
        relative_aspect = None if (self.mode != "train" or len(asp) == 0) else asp
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode == "train" else False,
        )
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)
        return frames

    @staticmethod
    def _get_gaussian_map(heatmap, center, kernel_size, sigma):
        h, w = heatmap.shape
        mu_x, mu_y = round(center[0]), round(center[1])
        left   = max(mu_x - (kernel_size - 1) // 2, 0)
        right  = min(mu_x + (kernel_size - 1) // 2, w - 1)
        top    = max(mu_y - (kernel_size - 1) // 2, 0)
        bottom = min(mu_y + (kernel_size - 1) // 2, h - 1)
        if left >= right or top >= bottom:
            return
        kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
        kernel_2d = kernel_1d * kernel_1d.T
        k_left   = (kernel_size - 1) // 2 - mu_x + left
        k_right  = (kernel_size - 1) // 2 + right - mu_x
        k_top    = (kernel_size - 1) // 2 - mu_y + top
        k_bottom = (kernel_size - 1) // 2 + bottom - mu_y
        heatmap[top:bottom + 1, left:right + 1] = kernel_2d[k_top:k_bottom + 1, k_left:k_right + 1]

    def __len__(self):
        return len(self._chunks)

    @property
    def num_videos(self):
        return len(self._chunks)
