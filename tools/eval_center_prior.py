#!/usr/bin/env python3
"""
Evaluate a Gaussian center-prior baseline on a gaze val set.
Mirrors the exact adaptive_f1 + aggregation pipeline used during training
(eval_epoch + ValGazeMeter) so numbers are directly comparable.

Usage:
    python tools/eval_center_prior.py --dataset parent   # exp351 parent val
    python tools/eval_center_prior.py --dataset child    # exp351 child val
    python tools/eval_center_prior.py --dataset egtea    # EGTEA Gaze val
"""

import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, ".")

import slowfast.utils.metrics as metrics
import slowfast.datasets.exp351 as exp351_module
from slowfast.config.defaults import get_cfg
from slowfast.datasets import loader


def make_center_prior(B, T, hm_h, hm_w, kernel_size):
    """
    Returns a (B, 1, T, hm_h, hm_w) center-prior heatmap built with the same
    cv2.getGaussianKernel + stamping logic as the dataset's _get_gaussian_map,
    placed at the image center and L1-normalized to sum to 1.
    """
    heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)
    mu_x, mu_y = round((hm_w - 1) / 2), round((hm_h - 1) / 2)
    left   = max(mu_x - (kernel_size - 1) // 2, 0)
    right  = min(mu_x + (kernel_size - 1) // 2, hm_w - 1)
    top    = max(mu_y - (kernel_size - 1) // 2, 0)
    bottom = min(mu_y + (kernel_size - 1) // 2, hm_h - 1)
    kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=-1, ktype=cv2.CV_32F)
    kernel_2d = kernel_1d * kernel_1d.T
    k_left   = (kernel_size - 1) // 2 - mu_x + left
    k_right  = (kernel_size - 1) // 2 + right - mu_x
    k_top    = (kernel_size - 1) // 2 - mu_y + top
    k_bottom = (kernel_size - 1) // 2 + bottom - mu_y
    heatmap[top:bottom + 1, left:right + 1] = kernel_2d[k_top:k_bottom + 1, k_left:k_right + 1]
    heatmap /= heatmap.sum()

    frame = torch.as_tensor(heatmap).float()  # (hm_h, hm_w)
    # (H, W) -> (1, 1, 1, H, W) -> (B, 1, T, H, W)
    return frame.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, 1, T, hm_h, hm_w).contiguous()


def minmax_normalize(preds):
    """Same normalization applied in eval_epoch before adaptive_f1."""
    flat = preds.view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
    mn = flat.min(dim=-1, keepdim=True)[0]
    mx = flat.max(dim=-1, keepdim=True)[0]
    flat = (flat - mn) / (mx - mn + 1e-6)
    return flat.view(preds.size())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["parent", "child", "egtea"], default="parent")
    args = parser.parse_args()

    if args.dataset == "child":
        exp351_module._SPLIT_TO_JSON["val"] = "data/chen/exp351/child_eye_data_val.json"
        cfg_path = "configs/Exp351/MVIT_B_16x4_CONV.yaml"
        dataset_name = "exp351"
    elif args.dataset == "parent":
        exp351_module._SPLIT_TO_JSON["val"] = "data/chen/exp351/parent_eye_data_val.json"
        cfg_path = "configs/Exp351/MVIT_B_16x4_CONV.yaml"
        dataset_name = "exp351"
    else:  # egtea
        cfg_path = "configs/Egtea/MVIT_B_16x4_CONV.yaml"
        dataset_name = "egteagaze"

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.NUM_GPUS = 0  # CPU only — no model, no CUDA needed
    if args.dataset == "egtea":
        cfg.DATA.PATH_PREFIX = "data/egtea"

    hm_h = cfg.DATA.TEST_CROP_SIZE // 4   # 64
    hm_w = cfg.DATA.TEST_CROP_SIZE // 4   # 64
    kernel_size = cfg.DATA.GAUSSIAN_KERNEL  # 19

    print(f"Dataset: {args.dataset}")
    print(f"Heatmap size: {hm_h}x{hm_w},  Gaussian kernel: {kernel_size} (sigma=-1, OpenCV default)")

    val_loader = loader.construct_loader(cfg, "val")
    print(f"Val set: {len(val_loader)} batches")

    recall_total = 0.0
    precision_total = 0.0
    num_samples = 0

    for cur_iter, (inputs, labels, labels_hm, _, meta) in enumerate(val_loader):
        B = labels.size(0)
        T = labels.size(1)

        preds = make_center_prior(B, T, hm_h, hm_w, kernel_size)
        preds_rescale = minmax_normalize(preds)

        f1, recall, precision, threshold = metrics.adaptive_f1(
            preds_rescale, labels_hm, labels, dataset=dataset_name
        )

        labels_flat = labels.view(B * T, -1)
        mb_size = torch.where(labels_flat[:, 2] == 1)[0].size(0)

        if mb_size == 0:
            continue

        num_samples += mb_size
        recall_total += recall * mb_size
        precision_total += precision * mb_size

        if (cur_iter + 1) % 20 == 0:
            print(f"  iter {cur_iter+1}/{len(val_loader)}  "
                  f"f1={f1:.4f}  recall={recall:.4f}  precision={precision:.4f}  "
                  f"thr={threshold:.4f}  fixation_frames={mb_size}")

    # Aggregate the same way as ValGazeMeter.log_epoch_stats
    recall_agg    = recall_total / num_samples
    precision_agg = precision_total / num_samples
    f1_agg        = 2 * recall_agg * precision_agg / (recall_agg + precision_agg + 1e-6)

    print(f"\n=== Center prior baseline — {args.dataset} (kernel={kernel_size}, sigma=-1 at {hm_h}x{hm_w}) ===")
    print(f"  F1        : {f1_agg:.4f}")
    print(f"  Recall    : {recall_agg:.4f}")
    print(f"  Precision : {precision_agg:.4f}")
    print(f"  Fixation frames evaluated: {num_samples}")


if __name__ == "__main__":
    main()
