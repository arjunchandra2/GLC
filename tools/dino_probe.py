"""Linear (and optionally end-to-end) gaze probe on a DINOv2 ViT-B/14 backbone.

Two prediction modes:
  --predict-mode xy    : 2-dim regression of normalized (x, y).      (default)
  --predict-mode grid  : NxN cell classification, N from --grid-size.
                         GT label is the cell the GT gaze falls into.
                         Loss = KL divergence (default).

At eval time we always report grid accuracy at the sizes in EVAL_GRID_SIZES
(16, 64), regardless of which size the model was trained on. The predicted
cell at the training resolution is mapped back to its center xy, then
re-binned into each eval grid for accuracy.

Data is read from this repo's chunked exp351 JSON format (see
slowfast/datasets/exp351.py and glc_eval/agent_datasets.py):
    data/chen/exp{--exp}/{--agent}_eye_data_{train,val,test}.json
each a list of {"subject_dir": ..., "frames": [...]}, where each frame has
"frame_path" and raw-pixel "gaze_x"/"gaze_y" (optionally None for an
untracked frame). See _load_rows() for how this is flattened into one
training row per usable frame, including the optional per-frame ROI bbox.

Usage:
    python dino_probe.py --frozen   --predict-mode xy
    python dino_probe.py --frozen   --predict-mode grid --grid-size 64
    python dino_probe.py --baseline subject_mean
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# wandb derives its "root" from os.path.realpath, which on this network mount
# resolves Z:\... to \\172.16.6.71\space\..., while os.getcwd() still reports
# the Z: drive-letter form. wandb then calls os.path.relpath(cwd, root) and
# crashes with "path is on mount 'Z:', start on mount '\\\\172.16.6.71\\space'".
# Align them by chdir'ing to the realpath form BEFORE wandb is imported.
_realcwd = os.path.realpath(os.getcwd())
if _realcwd != os.getcwd():
    os.chdir(_realcwd)

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Hardcoded so this script can run from any machine without a netrc / env var.
os.environ.setdefault(
    "WANDB_API_KEY",
    "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",
)
# Skip wandb's auto code-path detection (it crashes on this network mount,
# where os.getcwd() reports the Z: drive-letter but wandb walks to the
# \\172.16.6.71\space\ UNC form and os.path.relpath rejects the mismatch).
os.environ.setdefault("WANDB_PROGRAM", "dino_probe.py")
os.environ.setdefault("WANDB_PROGRAM_RELPATH", "dino_probe.py")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")

_wandb_step = 0

# Eye-tracker reports gaze in each camera's native pixel coordinates. Angular
# error is
#   atan( sqrt( (dx/fx)^2 + (dy/fy)^2 ) ),
# with per-row focal lengths fx,fy derived from the camera FOV, looked up
# by the row's resolution (image_w x image_h), which is probed per
# subject_dir from the frame images themselves (see _load_rows()).
FOV_H_DEG = 103.0   # 1600x1200 scene cam, horizontal (kept for reference/config)
FOV_V_DEG = 77.0    # 1600x1200 scene cam, vertical

# Camera FOV keyed by (width, height) in pixels. All FOV values are
# HORIZONTAL (not diagonal). The 1600x1200 cam is anisotropic, so its
# vertical FOV is given std ("v"); the other cameras are treated as
# having square pixels, so the vertical focal length equals the horizontal
# one (fy == fx) and only horizontal FOV ("h") is needed.
FOV_BY_RES = {
    (1600, 1200): {"h": FOV_H_DEG, "v": FOV_V_DEG},
    (1280, 720):  {"h": 122.0},
    (720, 480):   {"h": 70.0},
    (640, 480):   {"h": 70.0},
    (320, 240):   {"h": 70.0}
}
# Gaussian smoothing applied to the KL-grid target during TRAINING ONLY.
# Stds are in pixels and applied independently per axis. These values are
# calibrated for the 1600x1200 camera; for lower-resolution frames they are
# scaled down by image_h / GAUSSIAN_STD_REF_H (a frame half as tall uses half
# the std). The smoothing also auto-scales per grid size since cell sizes
# change with grid_size.
GAUSSIAN_STD_PX = (150.0, 180.0)  # ~(12.25, 13.42) at 1600x1200
GAUSSIAN_STD_REF_H = 1200.0       # reference frame height the stds are tuned for
IMG_SIZE = 224
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 768  # ViT-B/14
EVAL_GRID_SIZES = (16, 64)  # accuracies always reported at these sizes
# Default padding applied to the ROI bbox in the eval hit-rate: the box is
# scaled about its center by this factor (1.5 => 1.5x wider/taller).
BBOX_EVAL_PAD = 1.5
# --random-crop floor: the random training crop is at least this fraction of
# the frame on each side (raised if the gaze+box need more room to fit).
RANDOM_CROP_MIN_FRAC = 0.5


# ----------------------------- data ----------------------------------- #

def _bbox_corners_raw(bbox_str):
    """Normalized [x1,y1,x2,y2] corners from a stored [xc,yc,w,h] box (no pad),
    or None for a missing/empty/degenerate box. Used as the bbox-KL loss
    target; accepts any real box (object roi 1..27 or face roi 28)."""
    s = (bbox_str or "").strip()
    if s in ("[]", ""):
        return None
    try:
        v = [float(t) for t in s.strip("[]").split(",")]
    except ValueError:
        return None
    if len(v) != 4 or all(abs(t) < 1e-9 for t in v):
        return None
    xc, yc, w, h = v
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def _normalize_roi_bbox(raw):
    """Convert a frame's optional "roi_bbox" field to this script's stored
    [0,1]-normalized "[xc,yc,w,h]" string convention (consumed by
    _bbox_corners_raw / _parse_roi_bbox_corners below).

    The chunked exp351 data stores roi_bbox ALREADY normalized to [0,1] as
    [x_center, y_center, w, h] in the original camera frame -- unlike
    gaze_x/gaze_y, which are raw pixels. See
    slowfast.datasets.exp351.Exp351._parse_roi_bbox ("Stored roi_bbox is
    normalized [x_center, y_center, w, h]") and gaze/get_data_351_fullfps.m.
    So we pass the values through unchanged (do NOT divide by W/H -- doing so
    double-normalizes the box to a sliver at the top-left corner and wrecks the
    bbox_kl / sum loss). Not every frame has this field; accepts a 4-element
    list/tuple or an already-stringified "[xc,yc,w,h]" normalized box. Returns
    None if the field is absent, empty, or degenerate (all-zero).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s in ("", "[]"):
            return None
        try:
            raw = [float(t) for t in s.strip("[]").split(",")]
        except ValueError:
            return None
    if len(raw) != 4:
        return None
    xc, yc, w, h = (float(v) for v in raw)
    if all(abs(v) < 1e-9 for v in (xc, yc, w, h)):
        return None
    return f"[{xc},{yc},{w},{h}]"


def _load_chunks(json_path):
    """Load this repo's chunked exp351 JSON grouped BY CHUNK: a list (one entry
    per chunk with >=1 usable frame) of lists of row dicts. Each row has the
    same shape _load_rows yields (normalized x/y, pixel w/h, roi, normalized
    roi_bbox string). Untracked frames (gaze None) are dropped; (W, H) is probed
    once per subject_dir. This per-chunk grouping is what lets the train dataset
    draw a fresh random frame from each chunk every epoch (see GazeDataset)."""
    with open(json_path) as f:
        chunks = json.load(f)

    subject_res = {}
    grouped = []
    for chunk in chunks:
        sdir = chunk["subject_dir"]
        chunk_rows = []
        for frame in chunk["frames"]:
            gx_px, gy_px = frame.get("gaze_x"), frame.get("gaze_y")
            if gx_px is None or gy_px is None:
                continue
            if sdir not in subject_res:
                with Image.open(frame["frame_path"]) as probe:
                    subject_res[sdir] = probe.size  # (W, H)
            W, H = subject_res[sdir]
            chunk_rows.append({
                "frame_path": frame["frame_path"],
                "subject_dir": sdir,
                "x": min(max(float(gx_px) / W, 0.0), 1.0),
                "y": min(max(float(gy_px) / H, 0.0), 1.0),
                "w": float(W), "h": float(H),
                "roi": frame.get("roi"),
                "roi_bbox": _normalize_roi_bbox(frame.get("roi_bbox")),
            })
        if chunk_rows:
            grouped.append(chunk_rows)
    return grouped


def _load_rows(json_path, frames_per_chunk=None):
    """Flatten this repo's chunked exp351 JSON -- a list of
    {"subject_dir": ..., "frames": [...]} -- into one row per usable frame.

    Each frame dict is expected to look like:
      {"frame_path": ..., "gaze_x": <px or null>, "gaze_y": <px or null>,
       "roi": <int, optional>, "roi_bbox": <[xc,yc,w,h] px, optional>}

    Frames with gaze_x/gaze_y == null (untracked, matching Exp351's
    gaze_type 0) are dropped -- there is no supervision target for them.
    Out-of-bounds ("truncated", gaze_type 4) gaze is kept and clamped to the
    frame edge, matching slowfast.datasets.exp351.Exp351._parse_gaze.

    Frame (W, H) is probed once per subject_dir from its first frame image
    (all frames under one subject_dir share a camera/resolution), the same
    approach slowfast.datasets.exp351.Exp351._construct_loader uses.

    Chunk sampling: if frames_per_chunk is set, each chunk is reduced to that
    many evenly-spaced usable frames via torch.linspace(0, n-1, K) -- mirroring
    how slowfast.datasets.exp351.Exp351.__getitem__ downsamples a chunk to
    DATA.NUM_FRAMES (8) frames -- so every chunk contributes equally regardless
    of length, instead of long chunks dominating the epoch. A chunk with fewer
    than frames_per_chunk usable frames contributes all of them. If None (the
    default, used for val/test/baselines) every usable frame is kept, i.e. the
    original flatten-all behavior. This selection is DETERMINISTIC; for the
    train-time random-per-epoch variant see GazeDataset(random_frames=True).

    Returned rows carry normalized x/y in [0,1], pixel w/h, and (if present)
    a normalized "[xc,yc,w,h]" roi_bbox string -- the same shape the rest of
    this script's loss/metric code already expects.
    """
    rows = []
    for chunk_rows in _load_chunks(json_path):
        if frames_per_chunk is not None and len(chunk_rows) > frames_per_chunk:
            sel = torch.linspace(0, len(chunk_rows) - 1, frames_per_chunk)
            sel = torch.unique(sel.round().long())  # dedupe if K > n-gaps
            chunk_rows = [chunk_rows[int(i)] for i in sel]
        rows.extend(chunk_rows)
    return rows


class GazeDataset(Dataset):
    """Returns (image, target_xy, dims, bbox) for one usable frame of this
    repo's chunked exp351 JSON (see _load_rows).
    target_xy is normalized to [0,1] using the frame's own probed size.
    dims is [W, H] in pixels, used downstream for pixel/angular error.
    bbox is [x1,y1,x2,y2] normalized ROI-box corners (NaN-filled if the row
    has no usable box) -- consumed only by the bbox-KL loss."""

    def __init__(self, json_path, transform, random_crop=False,
                 frames_per_chunk=None, random_frames=False):
        self.transform = transform
        self.random_crop = random_crop
        # random_frames (train only): index the dataset by chunk "slots" and draw
        # a FRESH random usable frame from the chunk on every __getitem__, so the
        # frames seen vary each epoch (like exp351's random temporal sampling)
        # rather than the fixed linspace subset _load_rows picks. Epoch size is
        # preserved -- each chunk gets min(n_usable, frames_per_chunk) slots.
        self.random_frames = random_frames and frames_per_chunk is not None
        if self.random_frames:
            self.chunks = _load_chunks(json_path)
            self._slot_chunk = [ci for ci, cr in enumerate(self.chunks)
                                for _ in range(min(len(cr), frames_per_chunk))]
            self.rows = None
        else:
            self.chunks = None
            self._slot_chunk = None
            self.rows = _load_rows(json_path, frames_per_chunk=frames_per_chunk)

    def __len__(self):
        return len(self._slot_chunk) if self.random_frames else len(self.rows)

    def _random_crop(self, img, gx, gy, bb):
        """Aspect-preserving random crop that still contains the gaze point and
        (if present) the ROI box, with the gaze placed at a random location in
        the crop -- removing the dataset's center bias. Because pixel aspect
        (s*W)/(s*H) == W/H, an aspect-preserving crop is a SQUARE of side s in
        normalized space. Returns (img, gx, gy, bb) re-expressed in the crop's
        normalized coords (dims are left unchanged on purpose, so the loss
        smoothing stays a constant fraction of the frame and the FOV/angular
        lookup still keys on the original resolution)."""
        W, H = img.size
        c = lambda v: 0.0 if v < 0 else (1.0 if v > 1 else v)
        gxc, gyc = c(gx), c(gy)
        if bb is not None:
            xlo, xhi = min(gxc, c(bb[0])), max(gxc, c(bb[2]))
            ylo, yhi = min(gyc, c(bb[1])), max(gyc, c(bb[3]))
        else:
            xlo = xhi = gxc
            ylo = yhi = gyc

        need = max(xhi - xlo, yhi - ylo)
        s = random.uniform(min(max(need, RANDOM_CROP_MIN_FRAC), 1.0), 1.0)

        def pick(lo_must, hi_must):
            lo = max(0.0, hi_must - s)        # crop must reach hi_must, stay >=0
            hi = min(lo_must, 1.0 - s)        # crop start <= lo_must, end <= 1
            return lo if lo >= hi else random.uniform(lo, hi)

        x0, y0 = pick(xlo, xhi), pick(ylo, yhi)
        px1, py1 = int(round(x0 * W)), int(round(y0 * H))
        px2 = min(max(int(round((x0 + s) * W)), px1 + 1), W)
        py2 = min(max(int(round((y0 + s) * H)), py1 + 1), H)
        img = img.crop((px1, py1, px2, py2))

        ax0, ay0 = px1 / W, py1 / H
        sx, sy = (px2 - px1) / W, (py2 - py1) / H
        gx, gy = (gx - ax0) / sx, (gy - ay0) / sy
        if bb is not None:
            bb = [(bb[0] - ax0) / sx, (bb[1] - ay0) / sy,
                  (bb[2] - ax0) / sx, (bb[3] - ay0) / sy]
        return img, gx, gy, bb

    def __getitem__(self, i):
        # In random_frames mode the slot maps to a chunk; draw a random usable
        # frame from it (varies every epoch). Otherwise use the fixed row.
        r = (random.choice(self.chunks[self._slot_chunk[i]])
             if self.random_frames else self.rows[i])
        img = Image.open(r["frame_path"]).convert("RGB")
        gx, gy = r["x"], r["y"]
        bb = _bbox_corners_raw(r["roi_bbox"])
        if self.random_crop:
            img, gx, gy, bb = self._random_crop(img, gx, gy, bb)
        x = self.transform(img)
        tgt = torch.tensor([gx, gy], dtype=torch.float32)
        dims = torch.tensor([r["w"], r["h"]], dtype=torch.float32)
        bbox = torch.tensor(bb if bb is not None else [float("nan")] * 4,
                            dtype=torch.float32)
        return x, tgt, dims, bbox


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(DINOV2_MEAN, DINOV2_STD),
    ])


# ----------------------------- model ---------------------------------- #

def load_dinov2(checkpoint_path, device):
    hub_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main"
    )
    if os.path.isdir(hub_dir):
        model = torch.hub.load(hub_dir, "dinov2_vitb14", source="local", pretrained=False)
    else:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False
        )
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(device)


class GazeProbe(nn.Module):
    def __init__(self, backbone, frozen, output_dim):
        super().__init__()
        self.backbone = backbone
        self.frozen = frozen
        self.head = nn.Linear(FEATURE_DIM, output_dim)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats)


# ------------------------ grid <-> xy helpers ------------------------- #

def xy_to_grid_idx(target_xy, grid_size):
    """[B,2] normalized xy -> [B] flat cell index in 0..grid_size^2-1.
    Out-of-image gaze is clamped to the nearest edge cell."""
    eps = 1e-6
    col = (target_xy[:, 0].clamp(0.0, 1.0 - eps) * grid_size).long()
    row = (target_xy[:, 1].clamp(0.0, 1.0 - eps) * grid_size).long()
    return row * grid_size + col


def grid_idx_to_center_xy(idx, grid_size):
    """[B] flat cell index -> [B,2] normalized cell-center xy."""
    row = (idx // grid_size).float()
    col = (idx %  grid_size).float()
    return torch.stack([(col + 0.5) / grid_size,
                        (row + 0.5) / grid_size], dim=1)


# ----------------------------- losses --------------------------------- #
# xy losses have signature   (pred, target_xy, dims, bbox)
# grid losses have signature (pred, target_xy, dims, bbox, grid_size)
# so run_epoch can call them uniformly. Each loss ignores the args it does
# not need (xy losses ignore dims/bbox; the gaussian-KL loss ignores bbox;
# the bbox-KL loss ignores target_xy). bbox is [B,4] normalized corners
# [x1,y1,x2,y2] per row, NaN-filled where the row has no usable ROI box.

def loss_mse_xy(pred, target_xy, dims, bbox=None):
    return F.mse_loss(pred[:, 0], target_xy[:, 0]) + \
           F.mse_loss(pred[:, 1], target_xy[:, 1])


def loss_l1_xy(pred, target_xy, dims, bbox=None):
    return F.l1_loss(pred[:, 0], target_xy[:, 0]) + \
           F.l1_loss(pred[:, 1], target_xy[:, 1])


def loss_huber_xy(pred, target_xy, dims, bbox=None):
    return F.smooth_l1_loss(pred[:, 0], target_xy[:, 0]) + \
           F.smooth_l1_loss(pred[:, 1], target_xy[:, 1])


def _gaussian_grid_target(target_xy, dims, grid_size):
    """Build a [B, grid_size^2] target probability distribution. Cell
    centers are evaluated in pixel space, Gaussian density is computed
    with anisotropic stds, then softmax-normalized across cells."""
    B = target_xy.shape[0]
    device = target_xy.device

    target_px = target_xy * dims                                   # [B, 2]

    frac = (torch.arange(grid_size, device=device, dtype=dims.dtype)
            + 0.5) / grid_size                                     # [G]

    cx = frac[None, :] * dims[:, 0:1]                              # [B, G]
    cy = frac[None, :] * dims[:, 1:2]                              # [B, G]

    # Scale the (1600x1200-calibrated) stds down for lower-res frames by the
    # per-row height ratio: a frame half as tall uses half the std.
    res_scale = dims[:, 1:2] / GAUSSIAN_STD_REF_H                  # [B, 1]
    std_x = GAUSSIAN_STD_PX[0] * res_scale                         # [B, 1]
    std_y = GAUSSIAN_STD_PX[1] * res_scale                         # [B, 1]
    dx2 = (cx - target_px[:, 0:1]) ** 2 / (2.0 * std_x ** 2)       # [B, G]
    dy2 = (cy - target_px[:, 1:2]) ** 2 / (2.0 * std_y ** 2)       # [B, G]

    log_p = -(dy2[:, :, None] + dx2[:, None, :])                   # [B,G,G]
    log_p = log_p.reshape(B, -1)                                   # [B, G*G]
    return F.softmax(log_p, dim=-1)


def loss_kl_grid(pred, target_xy, dims, bbox, grid_size):
    target_p = _gaussian_grid_target(target_xy, dims, grid_size)
    log_pred = F.log_softmax(pred, dim=-1)
    return F.kl_div(log_pred, target_p, reduction="batchmean")


def _bbox_grid_target(bbox_corners, grid_size):
    """[Bv, grid_size^2] target: uniform over grid cells whose CENTER falls
    inside the (already-valid) bbox corners [x1,y1,x2,y2] (normalized), 0
    elsewhere. Rows whose box is too small to contain any cell center fall
    back to the single cell containing the box center, so every row sums to 1.
    Flat cell index = row*grid_size + col (matches xy_to_grid_idx)."""
    Bv = bbox_corners.shape[0]
    G = grid_size
    dev = bbox_corners.device
    frac = (torch.arange(G, device=dev, dtype=bbox_corners.dtype) + 0.5) / G  # [G]
    x1 = bbox_corners[:, 0:1]; y1 = bbox_corners[:, 1:2]
    x2 = bbox_corners[:, 2:3]; y2 = bbox_corners[:, 3:4]
    in_x = (frac[None, :] >= x1) & (frac[None, :] <= x2)                # [Bv,G] (cols)
    in_y = (frac[None, :] >= y1) & (frac[None, :] <= y2)                # [Bv,G] (rows)
    mask = (in_y[:, :, None] & in_x[:, None, :]).reshape(Bv, G * G).to(bbox_corners.dtype)
    counts = mask.sum(dim=1)                                            # [Bv]
    empty = counts == 0
    if empty.any():
        cxc = ((x1 + x2) / 2).squeeze(1)
        cyc = ((y1 + y2) / 2).squeeze(1)
        col = (cxc * G).clamp(0, G - 1).long()
        row = (cyc * G).clamp(0, G - 1).long()
        idx = row * G + col
        em = torch.nonzero(empty).flatten()
        mask[em] = 0.0
        mask[em, idx[em]] = 1.0
        counts = mask.sum(dim=1)
    return mask / counts[:, None]


def loss_kl_bbox(pred, target_xy, dims, bbox, grid_size):
    """KL( uniform-over-ROI-bbox || predicted heatmap ). Only rows with a
    usable bbox (non-NaN) contribute; the loss is the mean over those rows.
    Falls back to zero (no gradient) if a batch has no valid box.

    Computed as cross-entropy minus target entropy (KL = CE - H), which is
    numerically stable: the target is zero outside the box, so F.kl_div would
    hit 0*log(0)=NaN, but target*log_pred is 0 there since log_pred is finite."""
    valid = ~torch.isnan(bbox).any(dim=1)
    if not valid.any():
        return pred.sum() * 0.0
    target = _bbox_grid_target(bbox[valid], grid_size)                 # [Bv, G*G]
    log_pred = F.log_softmax(pred[valid], dim=-1)
    ce = -(target * log_pred).sum(dim=1)                               # [Bv]
    ent = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=1)    # [Bv]
    return (ce - ent).mean()


def loss_sum(pred, target_xy, dims, bbox, grid_size):
    """Sum of the gaussian-on-GT-point KL and the uniform-over-ROI-bbox KL.
    The bbox-KL term is weighted twice as much as the gaussian-KL term."""
    return loss_kl_grid(pred, target_xy, dims, bbox, grid_size) + \
           2.0 * loss_kl_bbox(pred, target_xy, dims, bbox, grid_size)


LOSSES_XY   = {"mse_xy": loss_mse_xy,
               "l1_xy":  loss_l1_xy,
               "huber_xy": loss_huber_xy}
LOSSES_GRID = {"kl": loss_kl_grid,
               "bbox_kl": loss_kl_bbox,
               "sum": loss_sum}
LOSSES_BY_MODE = {"xy": LOSSES_XY, "grid": LOSSES_GRID}


# ----------------------------- metrics -------------------------------- #

def _pixel_error(pred_xy_norm, target_xy, dims):
    """dims: [B, 2] of (W, H) per row, in pixels."""
    dx = (pred_xy_norm[:, 0] - target_xy[:, 0]) * dims[:, 0]
    dy = (pred_xy_norm[:, 1] - target_xy[:, 1]) * dims[:, 1]
    return torch.sqrt(dx * dx + dy * dy)


def _focal_px_xy(dims):
    """Per-row (fx, fy) in pixels, from a per-resolution FOV table.

    Each row's (W, H) selects an entry in FOV_BY_RES. fx uses the horizontal
    FOV; fy uses the explicit vertical FOV if given, else equals fx (square
    pixels). Raises if any row's resolution is not configured, so AA is never
    silently computed with the wrong FOV."""
    W = dims[:, 0]
    H = dims[:, 1]
    fx = torch.full_like(W, float("nan"))
    fy = torch.full_like(H, float("nan"))
    wi = torch.round(W).long()
    hi = torch.round(H).long()
    for (rw, rh), fov in FOV_BY_RES.items():
        m = (wi == rw) & (hi == rh)
        if not torch.any(m):
            continue
        fx_m = (W[m] / 2) / math.tan(math.radians(fov["h"]) / 2)
        fx[m] = fx_m
        if "v" in fov:
            fy[m] = (H[m] / 2) / math.tan(math.radians(fov["v"]) / 2)
        else:
            fy[m] = fx_m  # square pixels
    if torch.isnan(fx).any():
        bad = sorted({(int(wi[i]), int(hi[i]))
                      for i in torch.nonzero(torch.isnan(fx)).flatten().tolist()})
        raise ValueError(
            f"No FOV configured for resolution(s) {bad}. "
            f"Add them to FOV_BY_RES in dino_probe.py.")
    return fx, fy


def _angular_error_deg(pred_xy_norm, target_xy, dims):
    dx = (pred_xy_norm[:, 0] - target_xy[:, 0]) * dims[:, 0]
    dy = (pred_xy_norm[:, 1] - target_xy[:, 1]) * dims[:, 1]
    fx, fy = _focal_px_xy(dims)
    tan_err = torch.sqrt((dx / fx) ** 2 + (dy / fy) ** 2)
    return torch.atan(tan_err) * (180.0 / math.pi)


def _grid_accuracies(pred_xy_norm, target_xy):
    """Cell-match accuracy at each size in EVAL_GRID_SIZES. Grid math is in
    [0,1] normalized space so it does not need dims."""
    out = {}
    for g in EVAL_GRID_SIZES:
        gt_cell   = xy_to_grid_idx(target_xy,    g)
        pred_cell = xy_to_grid_idx(pred_xy_norm, g)
        out[f"accuracy_{g}"] = (pred_cell == gt_cell).float().mean().item()
    return out


def metrics_xy(pred, target_xy, dims):
    return {
        "aa_deg": _angular_error_deg(pred, target_xy, dims).mean().item(),
        "px_err": _pixel_error(pred, target_xy, dims).mean().item(),
        **_grid_accuracies(pred, target_xy),
    }


def metrics_grid(pred, target_xy, dims, train_grid_size):
    pred_idx = pred.argmax(dim=-1)
    pred_center_xy = grid_idx_to_center_xy(pred_idx, train_grid_size)
    return {
        "aae_deg": _angular_error_deg(pred_center_xy, target_xy, dims).mean().item(),
        **_grid_accuracies(pred_center_xy, target_xy),
    }


# Lower is better for these; used to track best-val checkpoint.
PRIMARY_METRIC = {"xy": "aa_deg", "grid": "aae_deg"}


# --------------------- test-only bbox hit-rate ------------------------ #

def _parse_roi_bbox_corners(roi, bbox_str, pad=BBOX_EVAL_PAD,
                            roi_lo=1, roi_hi=27):
    """[x1,y1,x2,y2] normalized corners for a gaze-target bbox, or None when it
    should be excluded: roi outside [roi_lo, roi_hi], or a missing/empty/
    degenerate box. Stored boxes are normalized [x_center, y_center, w, h]. The
    box is scaled about its center by `pad` (default BBOX_EVAL_PAD).
    Default range 1..27 = objects; pass roi_lo=roi_hi=28 for the face box."""
    try:
        roi_i = int(roi)
    except (TypeError, ValueError):
        return None
    if not (roi_lo <= roi_i <= roi_hi):
        return None
    s = (bbox_str or "").strip()
    if s in ("[]", ""):
        return None
    try:
        vals = [float(t) for t in s.strip("[]").split(",")]
    except ValueError:
        return None
    if len(vals) != 4 or all(abs(v) < 1e-9 for v in vals):
        return None
    xc, yc, w, h = vals
    hw, hh = w * pad / 2, h * pad / 2
    return [xc - hw, yc - hh, xc + hw, yc + hh]


def _bbox_hit_counts(preds_xy, rows, roi_lo, roi_hi):
    """(n_hit, n_gt_hit, n_valid) over rows whose roi is in [roi_lo, roi_hi] and
    has a real box: predicted point (and GT point) inside the padded box."""
    n = n_hit = n_gt = 0
    for (px, py), r in zip(preds_xy, rows):
        bb = _parse_roi_bbox_corners(r.get("roi"), r.get("roi_bbox"),
                                     roi_lo=roi_lo, roi_hi=roi_hi)
        if bb is None:
            continue
        n += 1
        if bb[0] <= px <= bb[2] and bb[1] <= py <= bb[3]:
            n_hit += 1
        gx, gy = r["x"], r["y"]
        if bb[0] <= gx <= bb[2] and bb[1] <= gy <= bb[3]:
            n_gt += 1
    return n_hit, n_gt, n


def _bbox_hit_dict(preds_xy, rows):
    """Hit-rate metrics over all gaze-target boxes (roi 1..28, 'bbox_hit_rate'),
    the object subset (roi 1..27, 'bbox_hit_rate_object'), and the face subset
    (roi 28, 'bbox_hit_rate_face'), each with its GT ceiling and example count."""
    def rate(h, n):
        return (h / n) if n else float("nan")
    out = {}
    for suffix, lo, hi in (("", 1, 28), ("_object", 1, 27), ("_face", 28, 28)):
        h, g, n = _bbox_hit_counts(preds_xy, rows, lo, hi)
        out[f"bbox_hit_rate{suffix}"] = rate(h, n)
        out[f"gt_bbox_hit_rate{suffix}"] = rate(g, n)
        out[f"bbox_hit_n{suffix}"] = n
    return out


def evaluate_bbox_hit_rate(model, dataset, device, predict_mode, grid_size,
                           batch_size, workers):
    """TEST-ONLY metric. Fraction of examples whose predicted gaze point lands
    inside the ground-truth ROI bbox, for the object subset (roi 1..27,
    'bbox_hit_rate') and the face subset (roi 28, 'bbox_hit_rate_face'), each
    with its GT ceiling and example count. The loader is unshuffled, so
    predictions line up with dataset.rows by index."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _t, _d, _b in loader:
            out = model(imgs.to(device, non_blocking=True))
            if predict_mode == "grid":
                xy = grid_idx_to_center_xy(out.argmax(dim=-1).cpu(), grid_size)
            else:
                xy = out.cpu()
            preds.append(xy)
    preds = torch.cat(preds, dim=0).tolist()
    return _bbox_hit_dict(preds, dataset.rows)


# ----------------------------- train ---------------------------------- #

def run_epoch(model, loader, loss_fn, metrics_fn, optimizer, device,
              train, use_wandb=False, loss_components_fn=None):
    """loss_components_fn (optional): (pred, target_xy, dims, bbox) -> dict of
    named scalar sub-losses. Used to log the individual terms behind a composite
    loss (e.g. the gaussian-KL and bbox-KL parts of --loss sum) alongside the
    total. Averaged per-batch like the total loss and returned as a third value
    (an empty dict when loss_components_fn is None)."""
    global _wandb_step
    model.train(train)
    if model.frozen:
        model.backbone.eval()

    total_loss = 0.0
    metric_sums = None
    comp_sums = None
    n_batches = 0
    n_samples = 0

    for imgs, targets, dims, bboxes in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        dims = dims.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets, dims, bboxes)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(imgs)
                loss = loss_fn(preds, targets, dims, bboxes)

        batch_metrics = metrics_fn(preds.detach(), targets, dims, bboxes)
        if metric_sums is None:
            metric_sums = {k: 0.0 for k in batch_metrics}

        batch_comps = ({} if loss_components_fn is None
                       else loss_components_fn(preds.detach(), targets, dims, bboxes))
        if comp_sums is None:
            comp_sums = {k: 0.0 for k in batch_comps}

        total_loss += loss.item()
        for k, v in batch_metrics.items():
            metric_sums[k] += v * imgs.size(0)
        for k, v in batch_comps.items():
            comp_sums[k] += v
        n_batches += 1
        n_samples += imgs.size(0)

        if train and use_wandb:
            wandb.log({"train/loss_step": loss.item(),
                       **{f"train/{k}_step": v for k, v in batch_metrics.items()},
                       **{f"train/{k}_step": v for k, v in batch_comps.items()}},
                      step=_wandb_step)
            _wandb_step += 1

    avg_loss = total_loss / n_batches
    avg_metrics = {k: s / n_samples for k, s in metric_sums.items()}
    avg_comps = {k: s / n_batches for k, s in comp_sums.items()}
    return avg_loss, avg_metrics, avg_comps


def _fmt_metrics(d):
    return " ".join(f"{k} {v:.3f}" for k, v in d.items())


# ---------------------------- baselines ------------------------------- #

def evaluate_subject_mean(json_path):
    """Subject-mean baseline. Predicted xy = per-subject mean gaze.

    Unlike the original repo's flat JSON, this format has no precomputed
    subject-average field, so the mean is derived here from the rows of
    THIS split only (grouped by subject_dir). Note this differs from a
    precomputed field that might have been fit over a different scope (e.g.
    train+val+test combined) -- that scope can't be reconstructed from this
    JSON format. Reports xy metrics, grid accuracies at EVAL_GRID_SIZES, and
    KL-grid loss at each."""
    rows = _load_rows(json_path)

    dims = torch.tensor([[r["w"], r["h"]] for r in rows], dtype=torch.float32)
    targets = torch.tensor([[r["x"], r["y"]] for r in rows], dtype=torch.float32)

    sums = {}
    counts = {}
    for r in rows:
        sx, sy = sums.get(r["subject_dir"], (0.0, 0.0))
        sums[r["subject_dir"]] = (sx + r["x"], sy + r["y"])
        counts[r["subject_dir"]] = counts.get(r["subject_dir"], 0) + 1
    subject_mean = {s: (sx / counts[s], sy / counts[s]) for s, (sx, sy) in sums.items()}
    means_norm = torch.tensor(
        [list(subject_mean[r["subject_dir"]]) for r in rows], dtype=torch.float32)

    out = {
        "n": len(rows),
        "loss_mse_xy": loss_mse_xy(means_norm, targets, dims).item(),
        "aa_deg": _angular_error_deg(means_norm, targets, dims).mean().item(),
        "px_err": _pixel_error(means_norm, targets, dims).mean().item(),
        **_grid_accuracies(means_norm, targets),
    }
    for g in EVAL_GRID_SIZES:
        cell = xy_to_grid_idx(means_norm, g)
        preds_g = torch.zeros(len(rows), g * g)
        preds_g.scatter_(1, cell.unsqueeze(1), 20.0)
        out[f"loss_kl_grid_{g}"] = loss_kl_grid(
            preds_g, targets, dims, None, g).item()

    # bbox hit-rate: does the subject-mean guess land in the ROI bbox?
    # Object subset (roi 1..27) and face subset (roi 28) -- comparable to the
    # probe metric.
    out.update(_bbox_hit_dict(means_norm.tolist(), rows))
    return out


def _split_json_path(exp, agent, split):
    return f"Z:\max\GLC\data/chen/exp{exp}/{agent}_eye_data_{split}.json"


def run_baseline_subject_mean(args, use_wandb):
    run_name = f"baseline_subjectmean_{int(time.time())}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_name}.jsonl"

    if use_wandb:
        wandb_config = {
            **vars(args),
            "baseline": "subject_mean",
            "trainable_params": 0,
            "reports": "both xy and grid metrics",
            "eval_grid_sizes": list(EVAL_GRID_SIZES),
            "fov_h_deg": FOV_H_DEG, "fov_v_deg": FOV_V_DEG,
            "gaussian_std_px_x": GAUSSIAN_STD_PX[0],
            "gaussian_std_px_y": GAUSSIAN_STD_PX[1],
            "img_size": IMG_SIZE,
            "feature_dim": FEATURE_DIM,
            "backbone": "none",
            "frame_dims": "probed per subject_dir from frame_path",
        }
        wandb.init(project=args.wandb_project,
                   config=wandb_config, name=run_name)

    splits = [("train", _split_json_path(args.exp, args.agent, "train")),
              ("val",   _split_json_path(args.exp, args.agent, "val")),
              ("test",  _split_json_path(args.exp, args.agent, "test"))]
    all_results = {}

    with log_path.open("w") as logf:
        for name, path in splits:
            r = evaluate_subject_mean(path)
            all_results[name] = r
            kl_parts = "  ".join(
                f"kl{g} {r[f'loss_kl_grid_{g}']:.4f}" for g in EVAL_GRID_SIZES)
            acc_parts = "  ".join(
                f"acc{g} {r[f'accuracy_{g}']:.3f}" for g in EVAL_GRID_SIZES)
            print(f"{name:5s}: n={r['n']:6d} | "
                  f"mse_xy {r['loss_mse_xy']:.4f}  {kl_parts} | "
                  f"aa_deg {r['aa_deg']:.3f}  px_err {r['px_err']:.2f}  "
                  f"{acc_parts} | "
                  f"bbox_hit({BBOX_EVAL_PAD}x) all {r['bbox_hit_rate']:.3f} "
                  f"(n={r['bbox_hit_n']}) obj {r['bbox_hit_rate_object']:.3f} "
                  f"(n={r['bbox_hit_n_object']}) face {r['bbox_hit_rate_face']:.3f} "
                  f"(n={r['bbox_hit_n_face']})")
            logf.write(json.dumps({"split": name, **r}) + "\n")
            if use_wandb:
                wandb.log({f"{name}/{k}": v for k, v in r.items()})

    if use_wandb:
        wandb.finish()
    return all_results


# ------------------------------- main --------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="351",
                   help="Selects data/chen/exp{exp}/{agent}_eye_data_{split}.json.")
    p.add_argument("--checkpoint",
                   default="models/dinov2_vitb14_pretrain.pth")

    p.add_argument("--predict-mode", choices=["xy", "grid"], default="xy",
                   help="xy: 2-dim regression. grid: NxN cell classification.")
    p.add_argument("--grid-size", type=int, default=16,
                   help="Grid resolution N for --predict-mode grid "
                        "(model output dim = N*N). Eval-time accuracy is "
                        f"always reported at sizes {EVAL_GRID_SIZES}.")
    p.add_argument("--baseline", action="store_true")

    backbone_mode = p.add_mutually_exclusive_group()
    backbone_mode.add_argument("--frozen",    action="store_true",
                               help="Freeze the ViT backbone (default).")
    backbone_mode.add_argument("--trainable", action="store_true",
                               help="Fine-tune the ViT backbone end-to-end.")

    p.add_argument("--loss", default=None,
                   help="Loss name. xy: mse_xy (default), l1_xy, huber_xy. "
                        "grid: kl (default, gaussian-on-GT-point), bbox_kl "
                        "(KL to a uniform-over-ROI-bbox heatmap), or sum "
                        "(kl + bbox_kl).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="Only used when --trainable.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="gaze_dinov2_probe")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Path to a *_best.pt checkpoint to load into the "
                        "model before training. Backbone+head weights "
                        "are restored; optimizer state is not.")
    p.add_argument("--agent", default="parent",
                   help="Selects data/chen/exp{exp}/{agent}_eye_data_{split}.json "
                        "(e.g. 'parent' or 'child').")
    p.add_argument("--random-crop", action="store_true",
                   help="Train-time augmentation: aspect-preserving random "
                        "crop that still contains the gaze point and ROI box, "
                        "placing the gaze at a random location to remove the "
                        "dataset's center bias. Eval is never cropped.")
    args = p.parse_args()

    losses = LOSSES_BY_MODE[args.predict_mode]
    if args.loss is None:
        args.loss = "mse_xy" if args.predict_mode == "xy" else "kl"
    if args.loss not in losses:
        raise SystemExit(
            f"--loss {args.loss!r} not valid for --predict-mode {args.predict_mode}. "
            f"Choices: {list(losses)}"
        )

    use_wandb = not args.no_wandb
    frozen = not args.trainable
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.baseline:
        print(f"BASELINE: subject_mean | eval grids {EVAL_GRID_SIZES}")
        run_baseline_subject_mean(args, use_wandb)
        return

    # Build mode-specific output_dim, loss_fn, metrics_fn.
    # loss_fn   signature: (pred, target_xy, dims, bbox) -> scalar
    # metrics_fn signature: (pred, target_xy, dims, bbox) -> dict
    # loss_components_fn (optional): logs the individual terms behind a
    # composite loss. For --loss sum (grid) that's the gaussian-on-GT-point KL
    # and the (unweighted) uniform-over-ROI-bbox KL -- the two "original terms"
    # summed (with the bbox term weighted 2x) into the total. None otherwise.
    loss_components_fn = None
    if args.predict_mode == "grid":
        output_dim = args.grid_size ** 2
        _gl = LOSSES_GRID[args.loss]
        loss_fn    = lambda p, t, d, b: _gl(p, t, d, b, args.grid_size)
        metrics_fn = lambda p, t, d, b: metrics_grid(p, t, d, args.grid_size)
        if args.loss == "sum":
            loss_components_fn = lambda p, t, d, b: {
                "loss_kl": loss_kl_grid(p, t, d, b, args.grid_size).item(),
                "loss_bbox_kl": loss_kl_bbox(p, t, d, b, args.grid_size).item(),
            }
    else:
        output_dim = 2
        _xl = LOSSES_XY[args.loss]
        loss_fn    = lambda p, t, d, b: _xl(p, t, d, b)
        metrics_fn = lambda p, t, d, b: metrics_xy(p, t, d)

    primary_metric = PRIMARY_METRIC[args.predict_mode]

    tfm = build_transform()
    # Train + val: sample 8 frames per chunk (mirrors exp351.py's
    # DATA.NUM_FRAMES=8) so every chunk is weighted equally and a val epoch
    # isn't larger than a train one. Train draws its 8 RANDOMLY per chunk, fresh
    # each epoch (random_frames=True), like exp351's random temporal sampling;
    # val uses the deterministic linspace subset so its metric is stable across
    # epochs. Test is intentionally NOT evaluated here -- a separate script owns
    # the final test metric / bbox hit-rate.
    train_ds = GazeDataset(_split_json_path(args.exp, args.agent, "train"), tfm,
                           random_crop=args.random_crop, frames_per_chunk=8,
                           random_frames=True)
    val_ds   = GazeDataset(_split_json_path(args.exp, args.agent, "val"),   tfm,
                           frames_per_chunk=8)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    print(f"train: {len(train_ds)} | val: {len(val_ds)}")
    print(f"mode: {args.predict_mode}"
          f"{(' grid=' + str(args.grid_size)) if args.predict_mode == 'grid' else ''}"
          f" | output_dim: {output_dim} | "
          f"backbone: {'FROZEN' if frozen else 'TRAINABLE'} | "
          f"loss: {args.loss} | device: {device}")

    backbone = load_dinov2(args.checkpoint, device)
    model = GazeProbe(backbone, frozen=frozen, output_dim=output_dim).to(device)

    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"resumed weights from {args.resume} "
              f"(saved at epoch {sd.get('epoch', '?')})")

    if frozen:
        optimizer = torch.optim.AdamW(model.head.parameters(),
                                      lr=args.head_lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.head.parameters(),     "lr": args.head_lr},
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        ], weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_tag = f"_g{args.grid_size}" if args.predict_mode == "grid" else ""
    run_name = (f"{args.predict_mode}{grid_tag}"
                f"_{'frozen' if frozen else 'finetune'}"
                f"_{args.loss}_{int(time.time())}")
    log_path = out_dir / f"{run_name}.jsonl"
    ckpt_path = out_dir / f"{run_name}_best.pt"

    if use_wandb:
        wandb_config = {
            **vars(args),
            "frozen": frozen,
            "output_dim": output_dim,
            "primary_metric": primary_metric,
            "eval_grid_sizes": list(EVAL_GRID_SIZES),
            "fov_h_deg": FOV_H_DEG, "fov_v_deg": FOV_V_DEG,
            "gaussian_std_px_x": GAUSSIAN_STD_PX[0],
            "gaussian_std_px_y": GAUSSIAN_STD_PX[1],
            "img_size": IMG_SIZE,
            "feature_dim": FEATURE_DIM,
            "backbone": "dinov2_vitb14",
            "frame_dims": "probed per subject_dir from frame_path",
        }
        wandb.init(project=args.wandb_project,
                   config=wandb_config, name=run_name)

    best_primary = float("inf")

    with log_path.open("w") as logf:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_m, tr_comp = run_epoch(
                model, train_loader, loss_fn, metrics_fn, optimizer, device,
                train=True, use_wandb=use_wandb,
                loss_components_fn=loss_components_fn)
            vl_loss, vl_m, vl_comp = run_epoch(
                model, val_loader, loss_fn, metrics_fn, None, device,
                train=False, loss_components_fn=loss_components_fn)
            dt = time.time() - t0

            rec = {"epoch": epoch,
                   "train_loss": tr_loss, **{f"train_{k}": v for k, v in tr_m.items()},
                   **{f"train_{k}": v for k, v in tr_comp.items()},
                   "val_loss":   vl_loss, **{f"val_{k}":   v for k, v in vl_m.items()},
                   **{f"val_{k}": v for k, v in vl_comp.items()},
                   "seconds":    dt}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            print(f"epoch {epoch:3d} | "
                  f"train loss {tr_loss:.4f} {_fmt_metrics({**tr_comp, **tr_m})} | "
                  f"val loss {vl_loss:.4f} {_fmt_metrics({**vl_comp, **vl_m})} | "
                  f"{dt:.1f}s")

            if use_wandb:
                wandb.log({"train/loss_epoch": tr_loss,
                           **{f"train/{k}_epoch": v for k, v in tr_m.items()},
                           **{f"train/{k}_epoch": v for k, v in tr_comp.items()},
                           "val/loss": vl_loss,
                           **{f"val/{k}":   v for k, v in vl_m.items()},
                           **{f"val/{k}": v for k, v in vl_comp.items()},
                           "epoch": epoch,
                           "seconds": dt},
                          step=_wandb_step)

            if vl_m[primary_metric] < best_primary:
                best_primary = vl_m[primary_metric]
                torch.save({"model": model.state_dict(),
                            "args": vars(args),
                            "val_metrics": vl_m,
                            "epoch": epoch}, ckpt_path)

    # Test is intentionally not evaluated here -- a separate script owns the
    # final test metric / bbox hit-rate. The best-val checkpoint written above
    # (ckpt_path) is what that script consumes.
    print(f"\nDone. Best-val checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
