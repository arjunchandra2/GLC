#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import math
import numpy as np
import os
import pickle
import torch
from scipy import ndimage

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, TestGazeMeter
from slowfast.utils.utils import frame_softmax

logger = logging.get_logger(__name__)

# Camera FOV keyed by (width, height) in pixels, mirrored from tools/dino_probe.py's
# FOV_BY_RES table so AAE here matches that script's angular-error convention. All
# values are HORIZONTAL FOV except the 1600x1200 cam, which is anisotropic and also
# gives vertical FOV explicitly; other cameras are treated as having square pixels
# (fy == fx).
_FOV_BY_RES = {
    (1600, 1200): {"h": 103.0, "v": 77.0},
    (1280, 720):  {"h": 122.0},
    (720, 480):   {"h": 70.0},
    (640, 480):   {"h": 70.0},
    (320, 240):   {"h": 70.0},
}


def _focal_px_xy(dims):
    """Per-row (fx, fy) in pixels, from _FOV_BY_RES keyed by each row's (W, H)."""
    W, H = dims[:, 0], dims[:, 1]
    wi, hi = torch.round(W).long(), torch.round(H).long()
    fx = torch.full_like(W, float("nan"))
    fy = torch.full_like(H, float("nan"))
    for (rw, rh), fov in _FOV_BY_RES.items():
        m = (wi == rw) & (hi == rh)
        if not torch.any(m):
            continue
        fx_m = (W[m] / 2) / math.tan(math.radians(fov["h"]) / 2)
        fx[m] = fx_m
        fy[m] = (H[m] / 2) / math.tan(math.radians(fov["v"]) / 2) if "v" in fov else fx_m
    if torch.isnan(fx).any():
        bad = sorted({(int(wi[i]), int(hi[i]))
                      for i in torch.nonzero(torch.isnan(fx)).flatten().tolist()})
        raise ValueError(
            f"No FOV configured for resolution(s) {bad}. Add them to _FOV_BY_RES.")
    return fx, fy


def _pixel_error(pred_xy_norm, target_xy, dims):
    """Euclidean pixel distance between predicted and target gaze point, in the
    original (pre-crop) subject camera's pixel space. dims: (N, 2) of (W, H)."""
    dx = (pred_xy_norm[:, 0] - target_xy[:, 0]) * dims[:, 0]
    dy = (pred_xy_norm[:, 1] - target_xy[:, 1]) * dims[:, 1]
    return torch.sqrt(dx * dx + dy * dy)


def _angular_error_deg(pred_xy_norm, target_xy, dims):
    dx = (pred_xy_norm[:, 0] - target_xy[:, 0]) * dims[:, 0]
    dy = (pred_xy_norm[:, 1] - target_xy[:, 1]) * dims[:, 1]
    fx, fy = _focal_px_xy(dims)
    tan_err = torch.sqrt((dx / fx) ** 2 + (dy / fy) ** 2)
    return torch.atan(tan_err) * (180.0 / math.pi)


def _gaze_point_preds(preds, dataset_grid_shape):
    """Soft-argmax (center of mass) of each predicted heatmap, as normalized
    (x, y) in [0, 1]. preds: (N, hm_h, hm_w)."""
    hm_h, hm_w = dataset_grid_shape
    preds_np = preds.cpu().numpy()
    pred_xy = torch.zeros((preds_np.shape[0], 2))
    for i in range(preds_np.shape[0]):
        row, col = ndimage.center_of_mass(preds_np[i])
        if math.isnan(row) or math.isnan(col):
            row, col = hm_h / 2.0, hm_w / 2.0
        pred_xy[i, 0] = col / hm_w
        pred_xy[i, 1] = row / hm_h
    return pred_xy


def _tracked_gaze_errors(preds, labels, dims, dataset):
    """Mean pixel error and AAE (mean angular error, degrees) between each
    predicted heatmap's peak and the ground-truth gaze point, over tracked
    frames only. Mirrors slowfast.utils.metrics.auc's fixation_idx / tracked_idx
    masking, and tools/dino_probe.py's _pixel_error / _angular_error_deg formulas
    (dims is each frame's original, pre-crop subject camera (W, H), matching
    dino_probe.py's convention of computing pixel/angular error in that space
    regardless of any subsequent crop).
    """
    if dataset in ("egteagaze", "exp351"):
        fixation_idx = 1
    elif dataset in ("ego4dgaze", "ego4d_av_gaze"):
        fixation_idx = 0
    else:
        raise NotImplementedError(f"Metrics of {dataset} is not implemented.")

    B, T = labels.size(0), labels.size(1)
    labels_flat = labels.view(B * T, labels.size(2))
    dims_flat = dims.unsqueeze(1).expand(B, T, 2).reshape(B * T, 2).float()
    tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
    if tracked_idx.numel() == 0:
        return None

    target_xy = labels_flat.index_select(0, tracked_idx)[:, :2].cpu()
    dims_sel = dims_flat.index_select(0, tracked_idx).cpu()

    preds = preds.squeeze(1)
    preds_flat = preds.reshape(B * T, preds.size(-2), preds.size(-1))
    preds_sel = preds_flat.index_select(0, tracked_idx.to(preds_flat.device))

    pred_xy = _gaze_point_preds(preds_sel, preds_sel.shape[-2:])

    return {
        "px_err": _pixel_error(pred_xy, target_xy, dims_sel),
        "aae_deg": _angular_error_deg(pred_xy, target_xy, dims_sel),
    }


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    gaze_error_chunks = []  # per-iteration dicts from _tracked_gaze_errors, concatenated below

    for cur_iter, (inputs, labels, labels_hm, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()
            video_idx = video_idx.cuda()

        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs)
            # preds, glc = model(inputs, return_glc=True)  # used to visualization glc correlation

            preds = frame_softmax(preds, temperature=2)  # KLDiv
            # Per-frame original (pre-crop) subject camera (W, H), in pixels -- only
            # Exp351's meta carries this (see slowfast/datasets/exp351.py); other gaze
            # datasets (egteagaze, ego4dgaze) don't, so AAE/pixel-error are skipped there.
            dims = meta.get("dims")
            if dims is not None and cfg.NUM_GPUS:
                dims = dims.cuda()

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                if dims is not None:
                    preds, labels, labels_hm, video_idx, dims = du.all_gather(
                        [preds, labels, labels_hm, video_idx, dims])
                else:
                    preds, labels, labels_hm, video_idx = du.all_gather([preds, labels, labels_hm, video_idx])

            # PyTorch
            if cfg.NUM_GPUS:  # compute on cpu
                preds = preds.cpu()
                labels = labels.cpu()
                labels_hm = labels_hm.cpu()
                video_idx = video_idx.cpu()
                if dims is not None:
                    dims = dims.cpu()

            preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
            preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
            preds_rescale = preds_rescale.view(preds.size())
            f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)
            auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)

            if dims is not None:
                chunk_errors = _tracked_gaze_errors(preds_rescale, labels, dims, dataset=cfg.TEST.DATASET)
                if chunk_errors is not None:
                    gaze_error_chunks.append(chunk_errors)

            test_meter.iter_toc()

            # Update and log stats.
            test_meter.update_stats(f1, recall, precision, auc, preds=preds_rescale, labels_hm=labels_hm, labels=labels)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    test_meter.finalize_metrics()

    if not cfg.DETECTION.ENABLE and gaze_error_chunks:
        mean_px_err = torch.cat([c["px_err"] for c in gaze_error_chunks]).mean().item()
        mean_aae_deg = torch.cat([c["aae_deg"] for c in gaze_error_chunks]).mean().item()
        logging.log_json_stats({
            "split": "test_final_gaze_error",
            "mean_px_err": mean_px_err,
            "aae_deg": mean_aae_deg,
        })

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
        # Create meters for multi-view testing.
        test_meter = TestGazeMeter(
            num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            num_cls=cfg.MODEL.NUM_CLASSES,
            overall_iters=len(test_loader),
            dataset=cfg.TEST.DATASET
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    logger.info("Testing finished!")
