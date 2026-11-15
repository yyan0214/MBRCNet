#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breast 3D reconstruction training and evaluation pipeline.

This file includes:
1. Data loading for multi-view images and 3D voxel labels.
2. Model building and training.
3. Validation metrics and mesh export.
"""

import io  # In-memory byte stream for binvox parsing.
import os  # File and folder operations.
import random  # Random sampling and dropout decisions.
import argparse  # Command-line argument parser.
import csv  # Write per-sample evaluation tables.
import math  # Basic math helpers.
from typing import List, Dict, Tuple, Optional  # Type hints for readability.

import numpy as np  # Array and voxel processing.
from PIL import Image  # Read and resize input images.

import torch  # Core deep learning framework.
import torch.nn as nn  # Neural network layers.
import torch.nn.functional as F  # Functional tensor operations.
from torch.utils.data import Dataset, DataLoader  # Dataset and mini-batch loading.
import torchvision.models as models  # Pretrained ResNet backbone.

import matplotlib.pyplot as plt  # Plot training and validation curves.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Matplotlib mesh drawing.
from skimage import measure  # Marching cubes for mesh extraction.
from scipy import ndimage as ndi  # Morphology and connected components.


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training and evaluation.

    Args:
        seed: Integer seed value.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    Args:
        worker_id: Worker index from PyTorch DataLoader.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_binvox(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Read a .binvox file into a boolean tensor.

    Args:
        path: Path to a .binvox file.
        device: Optional target device (cpu/cuda).

    Returns:
        Tensor with shape (D, H, W), dtype bool.
    """
    with open(path, "rb") as f:
        data = f.read()

    stream = io.BytesIO(data)
    line = stream.readline().strip()
    if not line.startswith(b"#binvox"):
        raise ValueError(f"Not a binvox file: {path}")

    depth = height = width = None
    header_ok = False
    while True:
        line = stream.readline()
        if line == b"":
            break
        line = line.strip()
        if line.startswith(b"dim"):
            _, dx, dy, dz = line.split()
            depth, height, width = int(dx), int(dy), int(dz)
        elif line.startswith(b"data"):
            header_ok = True
            break

    if not header_ok or depth is None or height is None or width is None:
        raise ValueError(f"Malformed binvox header: {path}")

    raw = stream.read()
    total = depth * height * width
    values = []
    i = 0
    count_sum = 0
    while i + 1 < len(raw):
        value = raw[i]
        count = raw[i + 1]
        i += 2
        values.extend([value] * count)
        count_sum += count
        if count_sum >= total:
            break

    if len(values) < total:
        raise ValueError(f"RLE decode shorter than expected in {path}")

    vox = torch.tensor(values[:total], dtype=torch.bool).view(depth, height, width)
    if device is not None:
        vox = vox.to(device)
    return vox


def save_voxel_as_mesh(voxel, out_path, level=0.5):
    """
    voxel: torch.Tensor or np.ndarray, shape (D, D, D), values in [0,1]
    out_path: *.obj
    """
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.detach().cpu().numpy()

    if voxel.max() <= 0:
        print(f"[WARN] Empty voxel, skip mesh export: {out_path}")
        return

    verts, faces, normals, _ = measure.marching_cubes(voxel, level=level)

    with open(out_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        for face in faces:
            # OBJ index starts from 1
            f.write(
                f"f {face[0]+1}//{face[0]+1} "
                f"{face[1]+1}//{face[1]+1} "
                f"{face[2]+1}//{face[2]+1}\n"
            )

# ------------------------------
# Training curve helper
# ------------------------------
def plot_epoch_curves(epochs: List[int], losses: List[float], ious: List[float], out_path: str) -> None:
    """
    Save one curve image with loss and IoU over epochs.

    Args:
        epochs: Epoch numbers.
        losses: Mean training loss for each epoch.
        ious: Mean validation IoU for each epoch.
        out_path: Output PNG path.

    Returns:
        None.
    """
    if not epochs:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(epochs, losses, label="TotalLoss")
    plt.plot(epochs, ious, label="IoU3D")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



# ------------------------------
# Dataset config (naming logic)
# ------------------------------

SIDE_SUFFIX = {"left": "LS", "right": "RS"}

VIEW_PREFIX = {
    "left": {"front": "F", "top": "T", "side": "SL"},
    "right": {"front": "F", "top": "T", "side": "SR"},
}

NIPPLE_VIEW_PREFIX = {"front": "FV", "side": "SV", "top": "TV"}
NIPPLE_SIDE_CODE = {"left": "LB", "right": "RB"}

VIEW_ORDER = ["front", "side", "top"]


def _safe_int(x: str):
    """
    Convert a string to int when possible.

    Args:
        x: Input text.

    Returns:
        int if 'x' is numeric, otherwise the original string.
    """
    try:
        return int(x)
    except ValueError:
        return x


def detect_input_layout(input_root: str, layout_hint: str = "auto") -> str:
    """
    Detect the image folder layout.

    Args:
        input_root: Root folder for input images.
        layout_hint: "auto", "single_breast", or "nipple_only".

    Returns:
        Detected layout name.
    """
    if layout_hint != "auto":
        return layout_hint
    if os.path.isdir(os.path.join(input_root, "raw_data")):
        return "single_breast"
    if (
        os.path.isdir(os.path.join(input_root, "Left", "Raw")) and
        os.path.isdir(os.path.join(input_root, "Right", "Raw"))
    ):
        return "nipple_only"
    return "single_breast"


def detect_gt_layout(gt_root: str, gt_layout: str = "auto") -> str:
    """
    Detect the 3D label folder layout.

    Args:
        gt_root: Root folder for voxel labels.
        gt_layout: "auto", "single_breast", or "nipple_only".

    Returns:
        Detected layout name.
    """
    if gt_layout != "auto":
        return gt_layout
    if os.path.isdir(os.path.join(gt_root, "gt_data", "3D")):
        return "single_breast"
    if (
        os.path.isdir(os.path.join(gt_root, "Left", "3D_gt")) and
        os.path.isdir(os.path.join(gt_root, "Right", "3D_gt"))
    ):
        return "nipple_only"
    return "single_breast"


def _voxel_path_candidates(gt_root: str, split: str, sid: str, side: str, gt_layout: str) -> List[str]:
    """
    Build possible voxel file paths for one sample.

    Args:
        gt_root: Label root folder.
        split: Dataset split name.
        sid: Sample id.
        side: "left" or "right".
        gt_layout: Layout mode.

    Returns:
        A list of candidate file paths.
    """
    side_cap = side.capitalize()
    candidates: List[str] = []
    if gt_layout in ("auto", "single_breast"):
        candidates.append(os.path.join(gt_root, "gt_data", "3D", split, side, f"{sid}.binvox"))
    if gt_layout in ("auto", "nipple_only"):
        candidates.extend([
            os.path.join(gt_root, side_cap, "3D_gt", f"{sid}.binvox"),
            os.path.join(gt_root, side_cap, "3D_gt", f"{side}_{sid}.binvox"),
            os.path.join(gt_root, side_cap, "3D_gt", f"{sid}_{side}.binvox"),
            os.path.join(gt_root, side_cap, "3D_gt", f"{NIPPLE_SIDE_CODE[side]}_{sid}.binvox"),
        ])
    return candidates


def resolve_voxel_path(gt_root: str, split: str, sid: str, side: str, gt_layout: str) -> Optional[str]:
    """
    Find the first existing voxel file path for one sample.

    Args:
        gt_root: Label root folder.
        split: Dataset split name.
        sid: Sample id.
        side: "left" or "right".
        gt_layout: Layout mode.

    Returns:
        The found file path, or None if not found.
    """
    for p in _voxel_path_candidates(gt_root, split, sid, side, gt_layout):
        if os.path.isfile(p):
            return p
    return None


def resolve_split_ids(split_ref_root: str, split: str) -> Optional[set]:
    """
    Resolve split IDs from a reference root.
    Priority:
      1) raw_data/<split>/<id> directories
      2) gt_data/3D/<split>/{left,right}/*.binvox
      3) splits/<split>.txt  (first token per line is ID)
    """
    if not split_ref_root:
        return None

    # 1) raw_data split folders
    raw_dir = os.path.join(split_ref_root, "raw_data", split)
    if os.path.isdir(raw_dir):
        ids = {
            d for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        }
        if ids:
            return ids

    # 2) gt_data/3D split folders
    g3d_root = os.path.join(split_ref_root, "gt_data", "3D", split)
    if os.path.isdir(g3d_root):
        ids = set()
        for side in ["left", "right"]:
            side_dir = os.path.join(g3d_root, side)
            if not os.path.isdir(side_dir):
                continue
            for fname in os.listdir(side_dir):
                if fname.endswith(".binvox"):
                    ids.add(os.path.splitext(fname)[0])
        if ids:
            return ids

    # 3) optional text file
    split_txt = os.path.join(split_ref_root, "splits", f"{split}.txt")
    if os.path.isfile(split_txt):
        ids = set()
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                tok = s.replace(",", " ").split()[0]
                if tok:
                    ids.add(tok)
        if ids:
            return ids

    return None


def scan_samples_for_splits(
    gt_root: str,
    splits: List[str],
    gt_layout: str = "auto",
    split_ref_root: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Collect all `(sample_id, side)` pairs for one or more splits.

    Args:
        gt_root: Label root folder.
        splits: List of split names (for example: train, valid, test).
        gt_layout: Layout mode.
        split_ref_root: Optional root used only to read split ids.

    Returns:
        Sorted unique list of `(id, side)` pairs.
    """
    out = []
    layout = detect_gt_layout(gt_root, gt_layout)
    if layout == "single_breast":
        for split in splits:
            split_root = os.path.join(gt_root, "gt_data", "3D", split)
            for side in ["left", "right"]:
                side_dir = os.path.join(split_root, side)
                if not os.path.isdir(side_dir):
                    continue
                for fname in os.listdir(side_dir):
                    if fname.endswith(".binvox"):
                        sid = os.path.splitext(fname)[0]
                        out.append((sid, side))
    else:
        ref_root = split_ref_root if split_ref_root else gt_root
        for split in splits:
            allowed_ids = resolve_split_ids(ref_root, split)
            if not allowed_ids:
                raise RuntimeError(
                    f"[SplitGuard] gt_layout=nipple_only but cannot resolve split IDs for split='{split}' "
                    f"from split_ref_root='{ref_root}'. "
                    "This would mix train/val/test. "
                    "Provide a valid --split-ref-root (usually Single Breast root)."
                )
            for side in ["left", "right"]:
                side_dir = os.path.join(gt_root, side.capitalize(), "3D_gt")
                if not os.path.isdir(side_dir):
                    continue
                for fname in os.listdir(side_dir):
                    if fname.endswith(".binvox"):
                        sid = os.path.splitext(fname)[0]
                        if sid in allowed_ids:
                            out.append((sid, side))
    out = sorted(list(set(out)), key=lambda x: (x[1], _safe_int(x[0])))
    return out


def build_global_index_map(
    gt_root: str,
    splits: List[str],
    gt_layout: str = "auto",
    split_ref_root: Optional[str] = None,
) -> Dict[Tuple[str, str], int]:
    """Only train split samples get an embedding index. Val/test samples are NOT included."""
    pairs = scan_samples_for_splits(
        gt_root, splits, gt_layout=gt_layout, split_ref_root=split_ref_root
    )
    return {(sid, side): i for i, (sid, side) in enumerate(pairs)}


# ------------------------------
# Attribute extraction from GT voxel (kept as optional supervision/analysis)
# ------------------------------

def _axis_to_int(axis: str) -> int:
    """
    Convert axis label to voxel dimension index.

    Args:
        axis: "x", "y", or "z".

    Returns:
        0 for z, 1 for y, 2 for x.
    """
    axis = axis.lower().strip()
    if axis == "z":
        return 0  # our voxel arrays use (z,y,x)
    if axis == "y":
        return 1
    if axis == "x":
        return 2
    raise ValueError(f"Invalid axis: {axis} (use x/y/z)")


def compute_profile_curvature_proxy(vox: np.ndarray, protrude_axis_zyx: int, sign: int) -> float:
    """
    Estimate surface profile curvature with a simple 2D projection.

    Args:
        vox: Binary voxel mask.
        protrude_axis_zyx: Axis index in `(z, y, x)` order.
        sign: +1 or -1 direction on that axis.

    Returns:
        A normalized curvature-like score in [0, 1].
    """
    v = vox.astype(bool)
    if protrude_axis_zyx == 0:
        proj = v.max(axis=2)  # (z,y)
        proj2 = proj
    elif protrude_axis_zyx == 1:
        proj = v.max(axis=2)  # (z,y)
        proj2 = proj.T        # (y,z)
    else:
        proj2 = v.max(axis=0)  # (y,x)

    if proj2.sum() == 0:
        return 0.0

    P0, P1 = proj2.shape
    if sign >= 0:
        far = np.full((P1,), -1, dtype=np.int32)
        for j in range(P1):
            idxs = np.where(proj2[:, j])[0]
            far[j] = int(idxs.max()) if idxs.size > 0 else -1
    else:
        far = np.full((P1,), P0, dtype=np.int32)
        for j in range(P1):
            idxs = np.where(proj2[:, j])[0]
            far[j] = int(idxs.min()) if idxs.size > 0 else P0

    valid = (far >= 0) & (far < P0)
    if valid.sum() < 8:
        return 0.0

    curve = far[valid].astype(np.float32)
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-6)
    d2 = np.abs(curve[2:] - 2.0 * curve[1:-1] + curve[:-2])
    return float(np.clip(d2.mean(), 0.0, 1.0))


def compute_nipple_length_radius(
    vox: np.ndarray,
    protrude_axis_zyx: int,
    sign: int,
    area_thresh_ratio: float = 0.35,
    min_len_vox: int = 2,
) -> Tuple[float, float]:
    """
    Estimate nipple length and radius from voxel cross-section areas.

    Args:
        vox: Binary voxel mask.
        protrude_axis_zyx: Axis index in `(z, y, x)` order.
        sign: +1 or -1 direction on that axis.
        area_thresh_ratio: Ratio used to find the base region.
        min_len_vox: Minimum voxel length to keep non-zero length.

    Returns:
        `(length_norm, radius_norm)`, both normalized to [0, 1].
    """
    v = vox.astype(bool)
    G = v.shape[0]
    if protrude_axis_zyx == 0:
        areas = v.reshape(G, -1).sum(axis=1)
    elif protrude_axis_zyx == 1:
        areas = v.transpose(1, 0, 2).reshape(G, -1).sum(axis=1)
    else:
        areas = v.transpose(2, 0, 1).reshape(G, -1).sum(axis=1)

    if areas.max() <= 0:
        return 0.0, 0.0

    max_area = float(areas.max())
    thresh = area_thresh_ratio * max_area
    occ_idx = np.where(areas > 0)[0]
    if occ_idx.size == 0:
        return 0.0, 0.0

    tip = int(occ_idx.max()) if sign >= 0 else int(occ_idx.min())

    base = tip
    if sign >= 0:
        for i in range(tip, -1, -1):
            if areas[i] >= thresh:
                base = i
                break
    else:
        for i in range(tip, G):
            if areas[i] >= thresh:
                base = i
                break

    length_vox = abs(tip - base)
    length_norm = 0.0 if length_vox < min_len_vox else float(np.clip(length_vox / (G - 1), 0.0, 1.0))

    mid = int((tip + base) // 2)
    area_mid = float(areas[mid])
    r_vox = np.sqrt(area_mid / (np.pi + 1e-8))
    radius_norm = float(np.clip(r_vox / G, 0.0, 1.0))
    return length_norm, radius_norm


def compute_gt_attributes_from_vox(vox01: np.ndarray, nipple_axis: str = "z", nipple_sign: int = +1) -> np.ndarray:
    """
    Build a compact 4-value attribute vector from one voxel volume.

    Args:
        vox01: Float voxel in [0, 1].
        nipple_axis: Axis used for nipple direction.
        nipple_sign: +1 or -1 direction on that axis.

    Returns:
        NumPy array `[volume, length, radius, curvature]` in [0, 1].
    """
    v = (vox01 > 0.5).astype(bool)
    vol = float(np.clip(v.mean(), 0.0, 1.0))
    ax = _axis_to_int(nipple_axis)
    sign = +1 if nipple_sign >= 0 else -1
    nlen, nrad = compute_nipple_length_radius(v, protrude_axis_zyx=ax, sign=sign)
    curv = compute_profile_curvature_proxy(v, protrude_axis_zyx=ax, sign=sign)
    return np.array([vol, nlen, nrad, curv], dtype=np.float32)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.

    Args:
        mask: Binary mask.

    Returns:
        Binary mask with only one largest component.
    """
    lab, n = ndi.label(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (lab == keep)


def _num_cc(mask: np.ndarray) -> int:
    """
    Count connected components in a binary mask.

    Args:
        mask: Binary mask.

    Returns:
        Number of connected components.
    """
    lab, n = ndi.label(mask.astype(np.uint8))
    return int(n)


def build_nipple_areola_roi_mask(
    gt_bin: np.ndarray,
    axis: str = "z",
    sign: int = 1,
    radius_ratio: float = 0.18,
    depth_ratio: float = 0.16,
    areola_extra_ratio: float = 0.07,
) -> np.ndarray:
    """
    Build a local nipple+areola ROI from GT.
    The same mask should be applied to both pred and GT for fair comparison.
    """
    occ = _largest_cc((gt_bin > 0).astype(bool))
    if occ.sum() == 0:
        return np.zeros_like(occ, dtype=bool)

    # Frontal-view circular ROI only (NO depth slab clipping).
    a = _axis_to_int(axis)  # 0:z, 1:y, 2:x
    Gz, Gy, Gx = occ.shape
    Gm = float(max(Gz, Gy, Gx))
    r0 = max(2.0, radius_ratio * Gm)

    # Frontal projection and its center
    if a == 0:
        proj = occ.max(axis=0)  # (y,x)
        py, px = np.where(proj)
        if py.size == 0:
            return np.zeros_like(occ, dtype=bool)
        c1, c2 = float(py.mean()), float(px.mean())  # y,x center
    elif a == 1:
        proj = occ.max(axis=1)  # (z,x)
        pz, px = np.where(proj)
        if pz.size == 0:
            return np.zeros_like(occ, dtype=bool)
        c1, c2 = float(pz.mean()), float(px.mean())  # z,x center
    else:
        proj = occ.max(axis=2)  # (z,y)
        pz, py = np.where(proj)
        if pz.size == 0:
            return np.zeros_like(occ, dtype=bool)
        c1, c2 = float(pz.mean()), float(py.mean())  # z,y center

    occ_vol = int(occ.sum())
    target_min = max(150, int(0.02 * occ_vol))
    roi = np.zeros_like(occ, dtype=bool)

    zz, yy, xx = np.indices(occ.shape)
    for scale in [1.0, 1.15, 1.30, 1.50, 1.70]:
        r = r0 * scale

        if a == 0:
            radial2 = (yy - c1) ** 2 + (xx - c2) ** 2
        elif a == 1:
            radial2 = (zz - c1) ** 2 + (xx - c2) ** 2
        else:
            radial2 = (zz - c1) ** 2 + (yy - c2) ** 2
        circle = radial2 <= (1.85 * r) ** 2
        roi_try = circle & occ
        roi = roi_try
        if int(roi_try.sum()) >= target_min:
            break

    if roi.sum() < 10:
        roi = (circle & occ)
    return roi.astype(bool)


def build_nipple_areola_cylinder_mask(
    gt_bin: np.ndarray,
    axis: str = "z",
    sign: int = 1,
    radius_ratio: float = 0.18,
    depth_ratio: float = 0.16,
    areola_extra_ratio: float = 0.07,
    center_mode: str = "gt_proj",
) -> np.ndarray:
    """
    Build a fixed cylindrical ROI mask for nipple+areola evaluation.

    Difference vs build_nipple_areola_roi_mask():
      - This returns the *pure* cylinder mask (defined by a projection center + radius),
        NOT intersected with GT occupancy. This avoids leaking GT shape into the ROI.
      - The center is either estimated from GT occupancy projection (center_mode="gt_proj"),
        or taken as the grid center (center_mode="grid_center") which avoids using GT to
        place the cylinder.

    Notes:
      - depth_ratio / areola_extra_ratio are kept for API compatibility; the mask is a
        cylinder (no depth slab clipping) to match the current evaluation design.
    """
    if gt_bin.ndim != 3:
        raise ValueError(f"gt_bin must be 3D (G,G,G), got shape={getattr(gt_bin, 'shape', None)}")

    occ = _largest_cc((gt_bin > 0).astype(bool))
    Gz, Gy, Gx = gt_bin.shape
    if (Gz * Gy * Gx) == 0:
        return np.zeros_like(gt_bin, dtype=bool)

    a = _axis_to_int(axis)  # 0:z, 1:y, 2:x
    Gm = float(max(Gz, Gy, Gx))
    r0 = max(2.0, radius_ratio * Gm)

    # Cylinder center
    if str(center_mode).lower().strip() == "grid_center":
        # Purely geometric cylinder placement (no GT used).
        if a == 0:
            c1, c2 = (Gy - 1) / 2.0, (Gx - 1) / 2.0  # y,x
        elif a == 1:
            c1, c2 = (Gz - 1) / 2.0, (Gx - 1) / 2.0  # z,x
        else:
            c1, c2 = (Gz - 1) / 2.0, (Gy - 1) / 2.0  # z,y
    else:
        # Projection center (same as build_nipple_areola_roi_mask).
        # If GT is empty, fall back to grid center.
        if occ.sum() == 0:
            if a == 0:
                c1, c2 = (Gy - 1) / 2.0, (Gx - 1) / 2.0
            elif a == 1:
                c1, c2 = (Gz - 1) / 2.0, (Gx - 1) / 2.0
            else:
                c1, c2 = (Gz - 1) / 2.0, (Gy - 1) / 2.0
        elif a == 0:
            proj = occ.max(axis=0)  # (y,x)
            py, px = np.where(proj)
            if py.size == 0:
                c1, c2 = (Gy - 1) / 2.0, (Gx - 1) / 2.0
            else:
                c1, c2 = float(py.mean()), float(px.mean())
        elif a == 1:
            proj = occ.max(axis=1)  # (z,x)
            pz, px = np.where(proj)
            if pz.size == 0:
                c1, c2 = (Gz - 1) / 2.0, (Gx - 1) / 2.0
            else:
                c1, c2 = float(pz.mean()), float(px.mean())
        else:
            proj = occ.max(axis=2)  # (z,y)
            pz, py = np.where(proj)
            if pz.size == 0:
                c1, c2 = (Gz - 1) / 2.0, (Gy - 1) / 2.0
            else:
                c1, c2 = float(pz.mean()), float(py.mean())

    zz, yy, xx = np.indices(gt_bin.shape)
    if a == 0:
        radial2 = (yy - c1) ** 2 + (xx - c2) ** 2
    elif a == 1:
        radial2 = (zz - c1) ** 2 + (xx - c2) ** 2
    else:
        radial2 = (zz - c1) ** 2 + (yy - c2) ** 2

    circle = radial2 <= (1.85 * r0) ** 2
    return circle.astype(bool)


# ------------------------------
# Dataset
# ------------------------------

class SingleBreastDataset(Dataset):
    """
    Returns:
      images: (V,3,H,W)
      voxel:  (G,G,G) float {0,1}
      attrs:  (4,) float in [0,1]  (optional supervision/analysis)
      sample_idx: int (global embedding index)
      id, side
    """
    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int,
        grid_size: int,
        index_map: Dict[Tuple[str, str], int],
        nipple_axis: str,
        nipple_sign: int,
        input_root: Optional[str] = None,
        gt_root: Optional[str] = None,
        input_layout: str = "auto",
        gt_layout: str = "auto",
        split_ref_root: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.input_root = input_root if input_root else data_root
        self.gt_root = gt_root if gt_root else data_root
        self.split_ref_root = split_ref_root if split_ref_root else self.gt_root
        self.split = split
        self.img_size = img_size
        self.grid_size = grid_size
        self.index_map = index_map
        self.nipple_axis = nipple_axis
        self.nipple_sign = nipple_sign
        self.input_layout = detect_input_layout(self.input_root, input_layout)
        self.gt_layout = detect_gt_layout(self.gt_root, gt_layout)
        self.samples = self._scan_split_samples()
        print(f"[Dataset] split={split}, input_layout={self.input_layout}, gt_layout={self.gt_layout}, total samples = {len(self.samples)}")

    def _scan_split_samples(self) -> List[Dict]:
        samples = []
        if self.gt_layout == "single_breast":
            gt_root = os.path.join(self.gt_root, "gt_data", "3D", self.split)
            for side in ["left", "right"]:
                side_dir = os.path.join(gt_root, side)
                if not os.path.isdir(side_dir):
                    continue
                for fname in os.listdir(side_dir):
                    if not fname.endswith(".binvox"):
                        continue
                    sid = os.path.splitext(fname)[0]
                    key = (sid, side)
                    idx = self.index_map.get(key, -1)
                    samples.append({"id": sid, "side": side, "sample_idx": idx})
        else:
            allowed_ids = resolve_split_ids(self.split_ref_root, self.split)
            if not allowed_ids:
                raise RuntimeError(
                    f"[SplitGuard] Dataset split='{self.split}' with gt_layout=nipple_only cannot resolve split IDs "
                    f"from split_ref_root='{self.split_ref_root}'. "
                    "This would mix sets; provide --split-ref-root."
                )
            for side in ["left", "right"]:
                side_dir = os.path.join(self.gt_root, side.capitalize(), "3D_gt")
                if not os.path.isdir(side_dir):
                    continue
                for fname in os.listdir(side_dir):
                    if not fname.endswith(".binvox"):
                        continue
                    sid = os.path.splitext(fname)[0]
                    if sid not in allowed_ids:
                        continue
                    key = (sid, side)
                    idx = self.index_map.get(key, -1)
                    samples.append({"id": sid, "side": side, "sample_idx": idx})
        samples.sort(key=lambda x: (x["side"], _safe_int(x["id"])))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_rgb_view(self, sid: str, side: str, view: str) -> torch.Tensor:
        if self.input_layout == "single_breast":
            suffix = SIDE_SUFFIX[side]
            prefix = VIEW_PREFIX[side][view]
            candidates = [
                os.path.join(self.input_root, "raw_data", self.split, str(sid), f"{prefix}_{suffix}.png"),
                os.path.join(self.input_root, "raw_data", self.split, str(sid), f"{prefix}_{suffix}.jpg"),
                os.path.join(self.input_root, "raw_data", self.split, str(sid), f"{prefix}_{suffix}.jpeg"),
            ]
        else:
            side_cap = side.capitalize()
            side_code = NIPPLE_SIDE_CODE[side]
            view_code = NIPPLE_VIEW_PREFIX[view]
            # Support both nipple_only layouts:
            #  1) <root>/<Left|Right>/Raw/<id>/...
            #  2) <root>/<Left|Right>/Raw/<split>/<id>/...  (common in this repo)
            candidates = []
            for base in [
                os.path.join(self.input_root, side_cap, "Raw", self.split, str(sid)),
                os.path.join(self.input_root, side_cap, "Raw", str(sid)),
            ]:
                candidates.extend([
                    os.path.join(base, f"{side_code}_{view_code}.png"),
                    os.path.join(base, f"{side_code}_{view_code}.jpg"),
                    os.path.join(base, f"{side_code}_{view_code}.jpeg"),
                ])
        img_path = next((p for p in candidates if os.path.isfile(p)), None)
        if img_path is None:
            raise FileNotFoundError(f"Missing image for id={sid}, side={side}, view={view}. Tried: {candidates}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def _load_voxel(self, sid: str, side: str) -> torch.Tensor:
        gt_path = resolve_voxel_path(self.gt_root, self.split, sid, side, self.gt_layout)
        if gt_path is None:
            raise FileNotFoundError(
                f"Missing voxel for id={sid}, side={side}, split={self.split}. "
                f"Checked roots under gt_root={self.gt_root}"
            )
        vox = load_binvox(gt_path, device=None)  # (D,H,W) bool
        v = vox.float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        v = F.interpolate(v, size=(self.grid_size, self.grid_size, self.grid_size),
                          mode="trilinear", align_corners=False)
        v = (v > 0.5).float()
        return v[0, 0]  # (G,G,G)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        sid, side, sample_idx = meta["id"], meta["side"], meta["sample_idx"]

        views = [self._load_rgb_view(sid, side, v) for v in VIEW_ORDER]
        views = torch.stack(views, dim=0)  # (V,3,H,W)
        vox = self._load_voxel(sid, side)  # (G,G,G)

        # Data augmentation (train split only, sample_idx>=0)
        if sample_idx >= 0:
            views, vox = self._augment(views, vox)

        attrs = compute_gt_attributes_from_vox(vox.numpy(), nipple_axis=self.nipple_axis, nipple_sign=self.nipple_sign)
        attrs = torch.from_numpy(attrs)  # (4,)
        return {"id": sid, "side": side, "images": views, "voxel": vox, "attrs": attrs, "sample_idx": sample_idx}

    def _augment(self, views: torch.Tensor, vox: torch.Tensor):
        """
        Consistent augmentation applied to both 2D views and 3D voxel.
        Designed for small datasets (70 samples): aggressive but clinically plausible.
        """
        # 1. Horizontal flip (left-right mirror - anatomically valid for breast)
        if random.random() < 0.5:
            views = torch.flip(views, dims=[-1])      # flip W dimension
            vox   = torch.flip(vox,   dims=[-1])      # flip X dimension

        # 2. 2D image augmentation only (colour jitter, blur - no geometric to keep 2D-3D consistency)
        V, C, H, W = views.shape
        for v in range(V):
            img = views[v]  # (3,H,W)

            # Brightness / contrast / saturation jitter
            if random.random() < 0.7:
                factor = random.uniform(0.75, 1.25)
                img = torch.clamp(img * factor, 0, 1)
            if random.random() < 0.5:
                mean = img.mean()
                img  = torch.clamp((img - mean) * random.uniform(0.8, 1.2) + mean, 0, 1)

            # Gaussian noise (simulate imaging variability)
            if random.random() < 0.4:
                img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)

            views[v] = img

        return views, vox


# ------------------------------
# Model components
# ------------------------------

class ResNetEncoder(nn.Module):
    """
    Multi-view image encoder based on pretrained ResNet-50.

    Input:
        x: Tensor of shape (B, 3, H, W).
    Output:
        latent: Tensor of shape (B, latent_dim).
        featmap (optional): Tensor of shape (B, C, H', W').
    """
    def __init__(self, latent_dim=512, freeze_layers=True, unfreeze_layer3: bool = False):
        super().__init__()
        # Load pretrained backbone.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.out_channels = self.backbone.fc.in_features

        # Freeze early layers for small datasets.
        if freeze_layers:
            layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']
            if not unfreeze_layer3:
                layers_to_freeze.append('layer3')
            for name, child in self.backbone.named_children():
                if name in layers_to_freeze:
                    for param in child.parameters():
                        param.requires_grad = False
                    print(f"Frozen layer: {name}")
                elif name == 'layer3' and unfreeze_layer3:
                    print("Unfrozen layer: layer3")

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.Dropout(p=0.3),   # dropout before output to regularise small-dataset fine-tuning
        )

    def forward(self, x, return_featmap: bool = False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        featmap = self.backbone.layer4(x)
        pooled = self.backbone.avgpool(featmap).flatten(1)
        latent = self.backbone.fc(pooled)
        if return_featmap:
            return latent, featmap
        return latent

class ViewTransformerFusion(nn.Module):
    """
    Fuse view features with a Transformer encoder.

    Input:
        view_feats: Tensor of shape (B, V, D).
    Output:
        Tensor of shape (B, D).
    """
    def __init__(self, latent_dim: int = 256, n_heads: int = 4, n_layers: int = 2, max_views: int = 6):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_views, latent_dim))

    def forward(self, view_feats: torch.Tensor) -> torch.Tensor:
        B, V, D = view_feats.shape
        pos = self.pos_embed[:, :V, :]
        x = self.encoder(view_feats + pos)
        return x.mean(dim=1)


class SpatialFeatureLifter(nn.Module):
    """
    Lift 2D feature maps to a coarse 3D conditioning volume.

    Input:
        featmaps: Tensor of shape (B, V, C, H, W).
    Output:
        Tensor of shape (B, out_channels, G, G, G).
    """
    def __init__(self, in_channels: int, out_channels: int = 8, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.proj2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, featmaps: torch.Tensor) -> torch.Tensor:
        # featmaps: (B, V, C, H, W)
        B, V, C, H, W = featmaps.shape
        x = self.proj2d(featmaps.view(B * V, C, H, W))
        x = x.view(B, V, -1, H, W).mean(dim=1)  # multi-view average
        x = x.unsqueeze(2).repeat(1, 1, H, 1, 1)  # coarse cubic lift
        x = F.interpolate(
            x, size=(self.grid_size, self.grid_size, self.grid_size),
            mode="trilinear", align_corners=False
        )
        return self.refine3d(x)


class VoxelDecoder(nn.Module):
    """
    Decode latent vector to a 3D voxel logit volume.

    Input:
        z: Tensor of shape (B, latent_dim_total).
    Output:
        Tensor of shape (B, 1, grid_size, grid_size, grid_size).
    """
    def __init__(self, latent_dim_total: int, base_channels: int = 128, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        # 4^3 -> 8^3 -> 16^3 -> 32^3 -> upsample -> 64^3
        self.fc = nn.Linear(latent_dim_total, base_channels * 4 * 4 * 4)
        self.conv = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose3d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),

            # 8 -> 16
            nn.ConvTranspose3d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),

            # 16 -> 32  (new layer - captures mid-level shape detail)
            nn.ConvTranspose3d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels // 4),
            nn.ReLU(inplace=True),

            # refine at 32^3
            nn.Conv3d(base_channels // 4, base_channels // 4, 3, padding=1),
            nn.BatchNorm3d(base_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels // 4, 1, 3, padding=1),
        )
        
        # MOVED FROM FORWARD: upsample 32 -> 64 with trilinear (fine-grained, no checkerboard)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        feat = self.fc(z)
        C = feat.shape[1] // (4 * 4 * 4)
        feat = feat.view(B, C, 4, 4, 4)
        
        vox = self.conv(feat)     # -> (B,1,32,32,32)
        vox = self.upsample(vox)  # -> (B,1,64,64,64)
        
        return vox

class GeometryRefineBlock(nn.Module):
    """
    Refine coarse voxel logits using gradient cues and optional 3D condition.

    Input:
        coarse_logits: Tensor (B, 1, G, G, G).
        spatial_cond: Optional tensor (B, C, G, G, G).
    Output:
        Refined logits tensor (B, 1, G, G, G).
    """
    def __init__(self, channels: int = 64, cond_channels: int = 0):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(4 + cond_channels, channels, 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            # extra conv for sharper boundary detail
            nn.Conv3d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm3d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, 1, 1),
        )

    @staticmethod
    def compute_gradients(x: torch.Tensor) -> torch.Tensor:
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return torch.cat([dx, dy, dz], dim=1)

    def forward(self, coarse_logits: torch.Tensor, spatial_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        grads = self.compute_gradients(coarse_logits)
        x = torch.cat([coarse_logits, grads], dim=1)
        if spatial_cond is not None:
            x = torch.cat([x, spatial_cond], dim=1)
        return self.refine(x)


class HiresRefineHead(nn.Module):
    """Lightweight 64->128 refinement head driven by spatial condition."""
    def __init__(self, cond_channels: int, inner_channels: int = 8, out_size: int = 128):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Conv3d(4 + cond_channels, inner_channels, 3, padding=1),
            nn.BatchNorm3d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_channels, inner_channels, 3, padding=1),
            nn.BatchNorm3d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_channels, 1, 1),
        )

    @staticmethod
    def _grads(x: torch.Tensor) -> torch.Tensor:
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return torch.cat([dx, dy, dz], dim=1)

    def forward(self, refined_logits_64: torch.Tensor, spatial_cond_64: torch.Tensor) -> torch.Tensor:
        logits_hi = F.interpolate(
            refined_logits_64, size=(self.out_size, self.out_size, self.out_size),
            mode="trilinear", align_corners=False
        )
        cond_hi = F.interpolate(
            spatial_cond_64, size=(self.out_size, self.out_size, self.out_size),
            mode="trilinear", align_corners=False
        )
        x = torch.cat([logits_hi, self._grads(logits_hi), cond_hi], dim=1)
        return self.net(x)


class PositionalEncoding3D(nn.Module):
    """
    Create sinusoidal 3D positional features for point coordinates.

    Input:
        pts: Tensor (B, N, 3), normalized to [-1, 1].
    Output:
        Tensor (B, N, encoded_dim).
    """
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.register_buffer("freq_bands", (2.0 ** torch.arange(num_freqs)).float())
        self.num_freqs = num_freqs

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1) * np.pi
        sin = torch.sin(x)
        cos = torch.cos(x)
        enc = torch.cat([sin, cos], dim=-1).reshape(pts.shape[0], pts.shape[1], -1)
        return torch.cat([pts, enc], dim=-1)


class ImplicitSurfaceHead(nn.Module):
    """
    Predict point occupancy from latent code and 3D points.

    Input:
        z: Tensor (B, latent_dim_total).
        pts: Tensor (B, N, 3).
    Output:
        Tensor (B, N, 1) with occupancy logits.
    """
    def __init__(self, latent_dim_total: int, hidden_dim: int = 256, pe_freqs: int = 6):
        super().__init__()
        self.pe = PositionalEncoding3D(num_freqs=pe_freqs)
        pe_dim = 3 + 3 * 2 * pe_freqs
        in_dim = latent_dim_total + pe_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        B, N, _ = pts.shape
        z_expand = z.unsqueeze(1).expand(-1, N, -1)
        pts_enc = self.pe(pts)
        x = torch.cat([z_expand, pts_enc], dim=-1)
        return self.net(x)


class Perceptual3DNet(nn.Module):
    """
    Small 3D CNN used for feature-based reconstruction loss.

    Input:
        x: Tensor (B, 1, G, G, G).
    Output:
        Feature tensor (B, feat_dim).
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)


class Discriminator3D(nn.Module):
    """
    3D discriminator used by adversarial loss.

    Input:
        x: Tensor (B, 1, G, G, G).
    Output:
        Tensor (B,) of real/fake logits.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class AttributeHead(nn.Module):
    """
    Predict simple interpretable attributes from `z_global`.

    Input:
        z_global: Tensor (B, latent_dim).
    Output:
        Tensor (B, attr_dim) in [0, 1].
    """
    def __init__(self, latent_dim: int = 512, attr_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, attr_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_global)


class GradScale(torch.autograd.Function):
    """Scale gradients flowing back into embeddings to prevent them dominating z_global."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Forward helper for gradient scaling.

    Args:
        x: Input tensor.
        scale: Gradient multiplier used in backward pass.

    Returns:
        Tensor equal to `x` in forward pass.
    """
    return GradScale.apply(x, scale)


class HybridBreastRecon(nn.Module):
    """
    z_cond = concat(z_global[512], z_inst[32], z_attr_embed[16])
    Dimension ratio is intentional: z_global dominates (512 >> 32+16=48).

    z_global     : image-only multi-view embedding - THE primary component.
                   Supervised by recon + attr regression + voxel stats regression.
                   Gradient from embeddings is SCALED DOWN so encoder is forced to learn.
    z_inst [32]  : per-train-instance shape memory. GT-anchored via attr supervision.
    z_attr_embed [16]: per-train-instance attr memory.
                   Initialised from GT attr values (not random), then fine-tuned.

    Train: 80% SELF (inst+attr own embeddings), 20% ZERO dropout.
           -> Model must work without embeddings, so z_global can't be ignored.
    Val/test: use ZERO payload (z_inst/z_attr_embed both zeros), image-only conditioning.
    """
    def __init__(
        self,
        n_views: int,
        latent_dim: int,       # 512 - z_global
        inst_dim: int,         # 32  - z_inst
        attr_embed_dim: int,   # 16  - z_attr_embed
        grid_size: int,
        num_train_instances: int,
        pe_freqs: int = 6,
        attr_dim: int = 4,
        embed_grad_scale: float = 0.1,   # gradient scale for embeddings -> encoder sees 10x more signal
        unfreeze_layer3: bool = False,
        spatial_cond_channels: int = 8,
        enable_hires_refine: bool = False,
        hires_refine_size: int = 128,
    ):
        super().__init__()
        self.n_views = n_views
        self.latent_dim = latent_dim
        self.inst_dim = inst_dim
        self.attr_embed_dim = attr_embed_dim
        self.grid_size = grid_size
        self.embed_grad_scale = embed_grad_scale
        self.latent_dim_total = latent_dim + inst_dim + attr_embed_dim
        self.enable_hires_refine = enable_hires_refine

        self.encoder = ResNetEncoder(latent_dim=latent_dim, unfreeze_layer3=unfreeze_layer3)
        self.fusion = ViewTransformerFusion(latent_dim=latent_dim, n_heads=4, n_layers=2, max_views=n_views)
        self.spatial_lifter = SpatialFeatureLifter(
            in_channels=self.encoder.out_channels,
            out_channels=spatial_cond_channels,
            grid_size=grid_size,
        )

        # Embeddings - only train samples get indices
        self.inst_embed      = nn.Embedding(num_train_instances, inst_dim)
        self.attr_embed_table = nn.Embedding(num_train_instances, attr_embed_dim)
        nn.init.normal_(self.inst_embed.weight, mean=0.0, std=0.1)
        # attr_embed_table is initialised from GT attr in init_attr_embed_from_gt() below

        # PRIMARY global supervision: z_global -> GT measurable attributes
        self.attr_head = AttributeHead(latent_dim=latent_dim, attr_dim=attr_dim, hidden=256)

        # SECONDARY global supervision: z_global -> voxel global statistics
        # [volume_ratio, centroid_z, centroid_y, centroid_x] all in [0,1]
        self.voxel_stats_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self.voxel_dec   = VoxelDecoder(latent_dim_total=self.latent_dim_total, base_channels=128, grid_size=grid_size)
        self.geom_refine = GeometryRefineBlock(channels=64, cond_channels=spatial_cond_channels)
        self.hires_refine = HiresRefineHead(cond_channels=spatial_cond_channels, out_size=hires_refine_size) if enable_hires_refine else None
        self.implicit_head = ImplicitSurfaceHead(latent_dim_total=self.latent_dim_total, hidden_dim=256, pe_freqs=pe_freqs)

    @torch.no_grad()
    def init_attr_embed_from_gt(self, index_map: dict, gt_root: str, split: str,
                                 nipple_axis: str = "z", nipple_sign: int = 1, grid_size: int = 64,
                                 gt_layout: str = "auto"):
        """
        Initialise attr_embed_table weights from GT attr values instead of random.
        Maps each (sid, side) -> sample_idx -> compute_gt_attributes_from_vox() -> embed weight.
        attr_dim=4, attr_embed_dim=16: we tile/project GT attrs into the embed space.
        """
        print("[AttrEmbed] Initialising from GT attributes...")
        attr_dim = 4
        # Simple linear projection: repeat GT attrs to fill attr_embed_dim
        repeats = self.attr_embed_dim // attr_dim
        remainder = self.attr_embed_dim % attr_dim

        filled = 0
        for (sid, side), idx in index_map.items():
            gt_path = resolve_voxel_path(gt_root, split, sid, side, gt_layout)
            if gt_path is None:
                continue
            try:
                vox = load_binvox(gt_path, device=None).float()
                v = F.interpolate(vox.unsqueeze(0).unsqueeze(0),
                                  size=(grid_size, grid_size, grid_size),
                                  mode="trilinear", align_corners=False)[0, 0]
                attrs = compute_gt_attributes_from_vox(
                    v.numpy(), nipple_axis=nipple_axis, nipple_sign=nipple_sign
                )  # (4,) in [0,1]
                # Tile to attr_embed_dim and add tiny noise for diversity
                embed_val = np.tile(attrs, repeats)
                if remainder > 0:
                    embed_val = np.concatenate([embed_val, attrs[:remainder]])
                embed_val = embed_val.astype(np.float32)
                embed_val += np.random.randn(*embed_val.shape).astype(np.float32) * 0.01
                self.attr_embed_table.weight[idx] = torch.from_numpy(embed_val)
                filled += 1
            except Exception as e:
                pass  # leave random init for missing samples

        print(f"[AttrEmbed] Initialised {filled}/{len(index_map)} entries from GT.")

    def encode_global(self, images: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = images.shape
        imgs = images.view(B * V, C, H, W)
        view_feats = self.encoder(imgs).view(B, V, -1)
        return self.fusion(view_feats)

    def encode_global_and_spatial(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, C, H, W = images.shape
        imgs = images.view(B * V, C, H, W)
        view_feats, featmaps = self.encoder(imgs, return_featmap=True)
        z_global = self.fusion(view_feats.view(B, V, -1))
        fmap_BV = featmaps.view(B, V, featmaps.shape[1], featmaps.shape[2], featmaps.shape[3])
        spatial_cond = self.spatial_lifter(fmap_BV)
        return z_global, spatial_cond

    def _get_embeddings(self, sample_idx: torch.Tensor):
        """Fetch embeddings and scale their gradients so encoder is not overshadowed."""
        safe_idx = sample_idx.clamp(min=0)
        z_inst      = grad_scale(self.inst_embed(sample_idx.long()),       self.embed_grad_scale)
        z_attr_embed = grad_scale(self.attr_embed_table(sample_idx.long()), self.embed_grad_scale)
        return z_inst, z_attr_embed

    def forward(self, images: torch.Tensor, sample_idx: torch.Tensor,
                dropout_embed: bool = False):
        """
        Train-time forward.
        dropout_embed=True: zero out z_inst and z_attr_embed (20% of steps).
        """
        z_global, spatial_cond = self.encode_global_and_spatial(images)
        z_inst, z_attr_embed = self._get_embeddings(sample_idx)

        if dropout_embed:
            z_inst       = torch.zeros_like(z_inst)
            z_attr_embed = torch.zeros_like(z_attr_embed)

        z_cond = torch.cat([z_global, z_inst, z_attr_embed], dim=1)
        coarse_logits  = self.voxel_dec(z_cond)
        refined_logits = self.geom_refine(coarse_logits, spatial_cond)
        hires_logits = self.hires_refine(refined_logits, spatial_cond) if self.hires_refine is not None else None

        attrs_pred       = self.attr_head(z_global)
        voxel_stats_pred = self.voxel_stats_head(z_global)
        return coarse_logits, refined_logits, hires_logits, z_cond, z_global, z_inst, z_attr_embed, attrs_pred, voxel_stats_pred

    def forward_with_payload(self, images: torch.Tensor,
                              z_inst: torch.Tensor, z_attr_embed: torch.Tensor):
        """
        Val/test-time forward with externally provided payload.
        No sample_idx needed.
        """
        z_global, spatial_cond = self.encode_global_and_spatial(images)
        z_cond = torch.cat([z_global, z_inst, z_attr_embed], dim=1)
        coarse_logits  = self.voxel_dec(z_cond)
        refined_logits = self.geom_refine(coarse_logits, spatial_cond)
        hires_logits = self.hires_refine(refined_logits, spatial_cond) if self.hires_refine is not None else None

        attrs_pred       = self.attr_head(z_global)
        voxel_stats_pred = self.voxel_stats_head(z_global)
        return coarse_logits, refined_logits, hires_logits, z_cond, z_global, attrs_pred, voxel_stats_pred

    def implicit_query(self, z_cond: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        return self.implicit_head(z_cond, pts)


# ------------------------------
# Geometry / losses
# ------------------------------

def voxel_iou(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """
    Compute mean voxel IoU for a batch.

    Args:
        pred: Predicted occupancy/probability tensor.
        target: Ground-truth occupancy tensor.
        thresh: Threshold used to binarize both tensors.

    Returns:
        Scalar tensor with mean IoU over the batch.
    """
    pred_bin = (pred > thresh).float()
    tgt_bin = (target > thresh).float()
    inter = (pred_bin * tgt_bin).sum(dim=[1, 2, 3, 4])
    union = (pred_bin + tgt_bin - pred_bin * tgt_bin).sum(dim=[1, 2, 3, 4]) + 1e-6
    return (inter / union).mean()


def voxel_to_pointcloud(vox: torch.Tensor, thresh: float = 0.5, max_points: int = 4096) -> torch.Tensor:
    """
    Convert voxel occupancy to padded point clouds in [-1, 1] coordinates.

    Args:
        vox: Tensor (B,1,D,H,W) or (B,D,H,W).
        thresh: Occupancy threshold.
        max_points: Max sampled points per sample.

    Returns:
        Tensor (B, N, 3), where N is max points in this batch after padding.
    """
    if vox.dim() == 5:
        vox = vox[:, 0]
    B, D, H, W = vox.shape
    pcs = []
    for b in range(B):
        v = vox[b]
        idxs = (v > thresh).nonzero(as_tuple=False)  # (K,3) z,y,x
        if idxs.numel() == 0:
            pcs.append(torch.zeros(1, 3, device=vox.device))
            continue
        if idxs.shape[0] > max_points:
            perm = torch.randperm(idxs.shape[0], device=vox.device)[:max_points]
            idxs = idxs[perm]
        coords = idxs.float() / (torch.tensor([D, H, W], device=vox.device).float() - 1.0) * 2.0 - 1.0
        pcs.append(coords)
    maxN = max(pc.shape[0] for pc in pcs)
    out = []
    for pc in pcs:
        if pc.shape[0] < maxN:
            out.append(torch.cat([pc, pc.new_zeros(maxN - pc.shape[0], 3)], dim=0))
        else:
            out.append(pc)
    return torch.stack(out, dim=0)


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric Chamfer distance between two batched point clouds.

    Args:
        pc1: Tensor (B, N, 3).
        pc2: Tensor (B, M, 3).

    Returns:
        Scalar tensor: mean Chamfer distance.
    """
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # (B,N,M,3)
    dist2 = (diff ** 2).sum(dim=-1)
    cd1 = dist2.min(dim=2)[0].mean(dim=1)
    cd2 = dist2.min(dim=1)[0].mean(dim=1)
    return (cd1 + cd2).mean()


def surface_normals_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Estimate 3D normal vectors from voxel logits by finite differences.

    Args:
        logits: Tensor (B,1,G,G,G).

    Returns:
        Tensor (B,3,G,G,G) of unit normals.
    """
    dz = F.pad(logits[:, :, 1:, :, :] - logits[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
    dy = F.pad(logits[:, :, :, 1:, :] - logits[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
    dx = F.pad(logits[:, :, :, :, 1:] - logits[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    n = torch.cat([dx, dy, dz], dim=1)
    return F.normalize(n, dim=1, eps=1e-6)


def normal_consistency_loss(pred_logits: torch.Tensor, gt_vox: torch.Tensor) -> torch.Tensor:
    """
    Penalize normal direction mismatch between prediction and target.

    Args:
        pred_logits: Predicted logits (B,1,G,G,G).
        gt_vox: Ground-truth occupancy (B,G,G,G).

    Returns:
        Scalar tensor loss.
    """
    with torch.no_grad():
        gt_logits = gt_vox.unsqueeze(1) * 2.0 - 1.0
        gt_normals = surface_normals_from_logits(gt_logits)
    pred_normals = surface_normals_from_logits(pred_logits)
    cos = (pred_normals * gt_normals).sum(dim=1)
    return (1.0 - cos).mean()


# ------------------------------
# SOLIDIFY
# ------------------------------

def solidify_voxel(
    vox_bin: np.ndarray,
    keep_largest: bool = True,
    min_component: int = 64,
    do_closing: bool = True,
    closing_iters: int = 1,
    fill_holes: bool = False,   # DEFAULT OFF: fill_holes destroys GT concavities (nipple base,
                                # chest wall indentations). Only enable for final export/viz.
) -> np.ndarray:
    """
    Clean a binary voxel with simple morphology and component filtering.

    Args:
        vox_bin: Input binary voxel.
        keep_largest: Keep only the largest connected component.
        min_component: Remove components smaller than this size.
        do_closing: Apply binary closing before filtering.
        closing_iters: Number of closing iterations.
        fill_holes: Fill inner holes (use only for export visualization).

    Returns:
        Cleaned binary voxel.
    """
    v = vox_bin.astype(bool)

    if do_closing and closing_iters > 0:
        struct = ndi.generate_binary_structure(3, 1)
        v = ndi.binary_closing(v, structure=struct, iterations=closing_iters)

    struct = ndi.generate_binary_structure(3, 1)
    lab, num = ndi.label(v, structure=struct)
    if num > 0:
        sizes = ndi.sum(v, lab, index=np.arange(1, num + 1)).astype(np.int64)
        for i, sz in enumerate(sizes, start=1):
            if sz < min_component:
                v[lab == i] = False

        lab, num = ndi.label(v, structure=struct)
        if keep_largest and num > 1:
            sizes = ndi.sum(v, lab, index=np.arange(1, num + 1)).astype(np.int64)
            k = int(np.argmax(sizes)) + 1
            v = (lab == k)

    if fill_holes:
        v = ndi.binary_fill_holes(v)

    return v.astype(bool)


def solidify_voxel_for_export(
    vox_bin: np.ndarray,
    min_component: int = 64,
    closing_iters: int = 1,
) -> np.ndarray:
    """Solidify WITH fill_holes=True - only for mesh export/viz, never for metrics."""
    return solidify_voxel(
        vox_bin,
        keep_largest=True,
        min_component=min_component,
        do_closing=True,
        closing_iters=closing_iters,
        fill_holes=True,
    )



# ------------------------------
# Point-cloud F-score (F1@tau) and mesh comparison visualization
# ------------------------------
def _voxel_to_pointcloud(voxel: np.ndarray, thresh: float = 0.5, max_points: int = 4096) -> np.ndarray:
    """Return Nx3 points in normalized [0,1] coordinates from a (D,D,D) voxel."""
    pts = np.argwhere(voxel > thresh)
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    D = voxel.shape[0]
    pts = pts.astype(np.float32) / float(max(D - 1, 1))
    return pts

def fscore_points(pred_pts: torch.Tensor, gt_pts: torch.Tensor, tau: float) -> float:
    """F-score between two point sets with distance threshold tau (in normalized coords)."""
    if pred_pts.numel() == 0 or gt_pts.numel() == 0:
        return 0.0
    # pred->gt and gt->pred nearest neighbor distances
    diff = pred_pts.unsqueeze(1) - gt_pts.unsqueeze(0)  # (Np, Ng, 3)
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (Np, Ng)
    min_p = dist.min(dim=1)[0]  # (Np,)
    min_g = dist.min(dim=0)[0]  # (Ng,)
    precision = (min_p < tau).float().mean()
    recall = (min_g < tau).float().mean()
    denom = (precision + recall).clamp_min(1e-8)
    f1 = (2.0 * precision * recall / denom).item()
    return float(f1)

def f1_from_voxels(pred_vox: np.ndarray, gt_vox: np.ndarray, tau: float, max_points: int = 4096, device: str = "cpu") -> float:
    """
    Compute point-cloud F1 score from two voxel volumes.

    Args:
        pred_vox: Predicted binary voxel.
        gt_vox: Ground-truth binary voxel.
        tau: Distance threshold for F1.
        max_points: Max sampled points per volume.
        device: Torch device string for distance compute.

    Returns:
        F1 score in [0, 1].
    """
    pred_pts = _voxel_to_pointcloud(pred_vox, thresh=0.5, max_points=max_points)
    gt_pts = _voxel_to_pointcloud(gt_vox, thresh=0.5, max_points=max_points)
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return 0.0
    pred_t = torch.from_numpy(pred_pts).to(device=device)
    gt_t = torch.from_numpy(gt_pts).to(device=device)
    return fscore_points(pred_t, gt_t, tau=tau)

def plot_mesh_pair(verts_gt: np.ndarray, faces_gt: np.ndarray,
                   verts_pred: np.ndarray, faces_pred: np.ndarray,
                   path: str):
    """
    Save one side-by-side mesh image (GT on left, prediction on right).

    Args:
        verts_gt, faces_gt: GT mesh vertices/faces.
        verts_pred, faces_pred: Pred mesh vertices/faces.
        path: Output image path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    if verts_gt.shape[0] > 0 and faces_gt.shape[0] > 0:
        tri_gt = verts_gt[faces_gt]
        mesh_gt = Poly3DCollection(tri_gt, linewidths=0.05, alpha=0.9)
        mesh_gt.set_edgecolor("k")
        ax1.add_collection3d(mesh_gt)
        ax1.auto_scale_xyz(verts_gt[:, 0], verts_gt[:, 1], verts_gt[:, 2])
    ax1.set_title("GT mesh")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if verts_pred.shape[0] > 0 and faces_pred.shape[0] > 0:
        tri_pred = verts_pred[faces_pred]
        mesh_pred = Poly3DCollection(tri_pred, linewidths=0.05, alpha=0.9)
        mesh_pred.set_edgecolor("k")
        ax2.add_collection3d(mesh_pred)
        ax2.auto_scale_xyz(verts_pred[:, 0], verts_pred[:, 1], verts_pred[:, 2])
    ax2.set_title("Pred mesh")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_mesh_comparison(gt_voxel, pred_voxel, out_path, level: float = 0.5):
    """
    Marching Cubes -> (verts, faces) -> plot_mesh_pair() (matplotlib Poly3DCollection)
    Visual style matches your provided plot_mesh_pair().
    """
    if isinstance(gt_voxel, torch.Tensor):
        gt_voxel = gt_voxel.detach().cpu().numpy()
    if isinstance(pred_voxel, torch.Tensor):
        pred_voxel = pred_voxel.detach().cpu().numpy()

    def _mc(vox):
        if vox is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

        vox = vox.astype(np.float32)

        # If empty, return an empty mesh.
        if np.max(vox) <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

        # Keep level inside data range to avoid marching-cubes errors.
        vmin, vmax = float(vox.min()), float(vox.max())
        lv = level
        if not (vmin < lv < vmax):
            lv = (vmin + vmax) / 2.0

        verts, faces, _, _ = measure.marching_cubes(vox, level=lv)
        return verts.astype(np.float32), faces.astype(np.int64)

    verts_gt, faces_gt = _mc(gt_voxel)
    verts_pr, faces_pr = _mc(pred_voxel)

    # Save side-by-side mesh comparison image.
    plot_mesh_pair(verts_gt, faces_gt, verts_pr, faces_pr, out_path)


# ------------------------------
# Surface-biased sampling (implicit supervision)
# ------------------------------

def compute_boundary_mask(vox_bin: torch.Tensor) -> torch.Tensor:
    """
    Compute a simple voxel boundary mask by dilation minus erosion.

    Args:
        vox_bin: Binary voxel tensor (B,1,G,G,G).

    Returns:
        Boundary mask tensor (B,1,G,G,G), boolean.
    """
    dil = F.max_pool3d(vox_bin, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool3d(1.0 - vox_bin, kernel_size=3, stride=1, padding=1)
    return (dil - ero) > 0.0


def sample_points_surface_biased(
    vox_bin: torch.Tensor,
    n_pts: int,
    surface_ratio: float = 0.7,
    jitter: float = 0.35,
) -> torch.Tensor:
    """
    Sample 3D points with higher probability near the surface.

    Args:
        vox_bin: Binary voxel tensor (B,1,G,G,G).
        n_pts: Number of points per sample.
        surface_ratio: Fraction sampled from boundary area.
        jitter: Small random offset for boundary points.

    Returns:
        Tensor (B, n_pts, 3) in normalized [-1, 1] coordinates.
    """
    device = vox_bin.device
    B, _, G, _, _ = vox_bin.shape
    n_surf = int(n_pts * surface_ratio)
    n_uni = n_pts - n_surf

    bd_mask = compute_boundary_mask(vox_bin.float())
    pts_out = []

    for b in range(B):
        bd_idx = bd_mask[b, 0].nonzero(as_tuple=False)
        if bd_idx.numel() == 0:
            pts_s = torch.rand(n_surf, 3, device=device) * 2.0 - 1.0
        else:
            if bd_idx.shape[0] >= n_surf:
                sel = bd_idx[torch.randperm(bd_idx.shape[0], device=device)[:n_surf]]
            else:
                ridx = torch.randint(0, bd_idx.shape[0], (n_surf,), device=device)
                sel = bd_idx[ridx]

            coords = sel.float() / (G - 1) * 2.0 - 1.0
            j = (torch.rand_like(coords) * 2.0 - 1.0) * (jitter * (2.0 / (G - 1)))
            pts_s = (coords + j).clamp(-1.0, 1.0)

        pts_u = torch.rand(n_uni, 3, device=device) * 2.0 - 1.0
        pts = torch.cat([pts_s, pts_u], dim=0)
        pts = pts[torch.randperm(pts.shape[0], device=device)]
        pts_out.append(pts)

    return torch.stack(pts_out, dim=0)


def query_occupancy_from_voxel(vox_bin: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Query occupancy labels at normalized point coordinates.

    Args:
        vox_bin: Binary voxel tensor (B,1,G,G,G).
        pts: Points tensor (B,N,3) in [-1, 1].

    Returns:
        Occupancy tensor (B,N) with values in {0,1}.
    """
    device = vox_bin.device
    B, _, G, _, _ = vox_bin.shape
    idx_f = (pts + 1.0) * 0.5 * (G - 1)
    idx = idx_f.long().clamp(0, G - 1)
    b_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, pts.shape[1])
    z_idx, y_idx, x_idx = idx[..., 0], idx[..., 1], idx[..., 2]
    return vox_bin[:, 0][b_ids, z_idx, y_idx, x_idx]


# ------------------------------
# Training / evaluation
# ------------------------------
# Training / evaluation
# ------------------------------

def train_one_epoch(
    model: HybridBreastRecon,
    perc3d: Perceptual3DNet,
    disc3d: Discriminator3D,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    lambda_chamfer: float = 0.0,
    lambda_normal: float = 0.0,
    lambda_perc: float = 0.0,
    lambda_adv: float = 0.00,
    implicit_pts: int = 4096,
    implicit_surface_ratio: float = 0.75,
    lambda_implicit: float = 0.0,
    lambda_attr_sup: float = 0.5,
    lambda_stats_sup: float = 0.3,
    lambda_contrastive: float = 0.1,
    embed_dropout_prob: float = 0.2,
    pos_weight: float = 1.0,
    recon_bce_weight: float = 1.0,
    recon_dice_weight: float = 0.5,
    coarse_weight: float = 0.3,
    lambda_volume: float = 0.2,
    hires_weight: float = 0.0,
    lambda_boundary: float = 0.0,
    hires_refine_size: int = 128,
):
    """
    Four mechanisms to prevent z_global being overshadowed by embeddings:

    1. DIMENSION DOMINANCE: z_global=512 >> z_inst=32 + z_attr=16.
       Decoder structurally relies on z_global for the bulk of information.

    2. GRADIENT SCALING: gradients flowing back into inst_embed and attr_embed_table
       are scaled by embed_grad_scale (default 0.1). Encoder sees 10x stronger signal
       than embeddings for the same reconstruction loss.

    3. CONTRASTIVE LOSS on z_global: within each batch, samples from different
       instances should have dissimilar z_global. Uses a simple batch-hard contrastive:
       same-instance pairs (left/right of same ID) are pulled together,
       different-instance pairs are pushed apart.

    4. EMBED DROPOUT: 20% of steps zero both z_inst and z_attr_embed.
       Model must reconstruct from z_global alone -> encoder cannot be lazy.
    """
    model.train()
    perc3d.train()
    disc3d.train()

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    def dice_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss - directly optimises IoU approximation."""
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.view(pred.shape[0], -1)
        gt_flat   = gt.view(gt.shape[0], -1)
        inter = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
        dice  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        return dice.mean()

    def boundary_map(vol: torch.Tensor) -> torch.Tensor:
        dz = F.pad(torch.abs(vol[:, :, 1:, :, :] - vol[:, :, :-1, :, :]), (0, 0, 0, 0, 0, 1))
        dy = F.pad(torch.abs(vol[:, :, :, 1:, :] - vol[:, :, :, :-1, :]), (0, 0, 0, 1, 0, 0))
        dx = F.pad(torch.abs(vol[:, :, :, :, 1:] - vol[:, :, :, :, :-1]), (0, 1, 0, 0, 0, 0))
        return torch.clamp(dx + dy + dz, 0.0, 1.0)

    _sum_loss_g = 0.0
    _sum_iou    = 0.0
    _n_steps    = 0

    for it, batch in enumerate(loader):
        images     = batch["images"].to(device)
        vox_gt     = batch["voxel"].to(device).unsqueeze(1)
        attrs_gt   = batch["attrs"].to(device)
        sample_idx = batch["sample_idx"].to(device).long()
        ids        = batch["id"]

        # Skip non-train samples
        valid = sample_idx >= 0
        if not valid.any():
            continue
        if not valid.all():
            images, vox_gt, attrs_gt, sample_idx = (
                images[valid], vox_gt[valid], attrs_gt[valid], sample_idx[valid]
            )
            ids = [ids[i] for i in range(len(ids)) if valid[i].item()]

        # ---- Compute GT voxel statistics (for z_global supervision) ----
        with torch.no_grad():
            vox_f  = vox_gt[:, 0].float()
            B, G   = vox_f.shape[0], vox_f.shape[1]
            coords = torch.linspace(0, 1, G, device=device)
            gz = coords.view(G,1,1).expand(G,G,G)
            gy = coords.view(1,G,1).expand(G,G,G)
            gx = coords.view(1,1,G).expand(G,G,G)
            mass       = vox_f.sum(dim=[1,2,3]).clamp_min(1.0)
            vol_ratio  = vox_f.mean(dim=[1,2,3])
            cz = (vox_f * gz).sum(dim=[1,2,3]) / mass
            cy = (vox_f * gy).sum(dim=[1,2,3]) / mass
            cx = (vox_f * gx).sum(dim=[1,2,3]) / mass
            voxel_stats_gt = torch.stack([vol_ratio, cz, cy, cx], dim=1)  # (B,4)

        # ---- Mechanism 4: embed dropout 20% of steps ----
        do_dropout = (random.random() < embed_dropout_prob)

        coarse_logits, refined_logits, hires_logits, z_cond, z_global, z_inst, z_attr_embed, attrs_pred, voxel_stats_pred = model(images, sample_idx, dropout_embed=do_dropout)

        pred_occ = torch.sigmoid(refined_logits)

        # Reconstruction losses
        loss_bce_refine = bce(refined_logits, vox_gt)
        loss_dice_refine = dice_loss(refined_logits, vox_gt)
        loss_bce_coarse = bce(coarse_logits, vox_gt)
        loss_dice_coarse = dice_loss(coarse_logits, vox_gt)
        loss_recon = (
            recon_bce_weight * loss_bce_refine
            + recon_dice_weight * loss_dice_refine
            + coarse_weight * (loss_bce_coarse + loss_dice_coarse)
        )
        if hires_logits is not None and hires_weight > 0.0:
            gt_hires = F.interpolate(
                vox_gt,
                size=(hires_refine_size, hires_refine_size, hires_refine_size),
                mode="trilinear",
                align_corners=False,
            )
            loss_bce_hires = bce(hires_logits, gt_hires)
            loss_dice_hires = dice_loss(hires_logits, gt_hires)
            loss_recon = loss_recon + hires_weight * (loss_bce_hires + loss_dice_hires)
        loss_volume = F.l1_loss(
            pred_occ.mean(dim=[1, 2, 3, 4]),
            vox_gt.mean(dim=[1, 2, 3, 4]),
        )
        pred_boundary = boundary_map(pred_occ)
        gt_boundary = boundary_map(vox_gt)
        loss_boundary = F.l1_loss(pred_boundary, gt_boundary)

        loss_bce = loss_bce_refine
        loss_dice = loss_dice_refine
        iou      = voxel_iou(pred_occ, vox_gt)
        loss_ch  = chamfer_distance(
            voxel_to_pointcloud(pred_occ, thresh=0.5, max_points=2048),
            voxel_to_pointcloud(vox_gt,   thresh=0.5, max_points=2048)
        )
        loss_n   = normal_consistency_loss(refined_logits, vox_gt[:, 0])
        loss_p   = F.l1_loss(perc3d(pred_occ), perc3d(vox_gt))

        # Implicit surface
        vox_bin     = (vox_gt > 0.5).float()
        pts         = sample_points_surface_biased(vox_bin, n_pts=implicit_pts, surface_ratio=implicit_surface_ratio)
        occ_gt_impl = query_occupancy_from_voxel(vox_bin, pts)
        loss_impl   = bce(model.implicit_query(z_cond, pts).squeeze(-1), occ_gt_impl)

        # Adversarial
        loss_adv_g = F.binary_cross_entropy_with_logits(
            disc3d(pred_occ), torch.ones_like(disc3d(pred_occ))
        )

        # ---- Mechanism 3: Contrastive loss on z_global ----
        # Same-ID samples (left & right of same patient) -> pull together
        # Different-ID samples -> push apart (margin=0.5)
        loss_contrastive = torch.tensor(0.0, device=device)
        if B > 1:
            z_g_norm = F.normalize(z_global, dim=1)  # (B, D)
            sim_mat  = z_g_norm @ z_g_norm.t()        # (B, B) cosine similarity
            # Build same-ID label matrix
            same_id  = torch.zeros(B, B, dtype=torch.bool, device=device)
            for i in range(B):
                for j in range(B):
                    if i != j and ids[i] == ids[j]:
                        same_id[i, j] = True
            margin = 0.5
            pos_loss = (1.0 - sim_mat[same_id]).clamp_min(0.0)
            neg_loss = (sim_mat[~same_id & ~torch.eye(B, dtype=torch.bool, device=device)] - margin).clamp_min(0.0)
            if same_id.any():
                loss_contrastive = loss_contrastive + pos_loss.mean()
            if (~same_id & ~torch.eye(B, dtype=torch.bool, device=device)).any():
                loss_contrastive = loss_contrastive + neg_loss.mean()

        # z_global supervision losses (Mechanisms 1+2 reinforce these being primary)
        loss_attr_sup  = F.l1_loss(attrs_pred, attrs_gt)
        loss_stats_sup = F.mse_loss(voxel_stats_pred, voxel_stats_gt)

        # Curriculum weights
        t        = epoch / max_epoch
        w_ch     = lambda_chamfer     * t
        w_n      = lambda_normal      * t
        w_p      = lambda_perc        * t
        w_a      = lambda_adv         * t
        w_impl   = lambda_implicit    * t
        w_attr   = lambda_attr_sup    * (0.5 + 0.5 * t)
        w_stats  = lambda_stats_sup   * (0.5 + 0.5 * t)
        w_contra = lambda_contrastive * min(t * 2, 1.0)  # ramp up faster

        loss_g = (
            loss_recon
            + lambda_volume * loss_volume
            + lambda_boundary * loss_boundary
            + w_ch    * loss_ch
            + w_n     * loss_n
            + w_p     * loss_p
            + w_a     * loss_adv_g
            + w_impl  * loss_impl
            + w_attr  * loss_attr_sup
            + w_stats * loss_stats_sup
            + w_contra * loss_contrastive
        )

        _sum_loss_g += float(loss_g.item())
        _sum_iou    += float(iou.item())
        _n_steps    += 1

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        # Discriminator
        logits_real = disc3d(vox_gt)
        logits_fake = disc3d(pred_occ.detach())
        loss_d = 0.5 * (
            F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) +
            F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        )
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        if (it + 1) % 10 == 0:
            drop_tag = "[DROP]" if do_dropout else "      "
            print(
                f"[Epoch {epoch}] iter {it+1}/{len(loader)} {drop_tag} "
                f"loss_g={loss_g.item():.4f}  loss_d={loss_d.item():.4f}  "
                f"iou={iou.item():.4f}  bce={loss_bce.item():.4f}  dice={loss_dice.item():.4f} "
                f"vol={loss_volume.item():.4f}  "
                f"bd={loss_boundary.item():.4f}  "
                f"impl={loss_impl.item():.4f}  "
                f"attr={loss_attr_sup.item():.4f}  stats={loss_stats_sup.item():.4f}  "
                f"contra={loss_contrastive.item():.4f}"
            )

    mean_train_iou = float(_sum_iou / max(1, _n_steps))
    print(f"[Epoch {epoch}] Train mean IoU={mean_train_iou:.4f}  mean loss_g={_sum_loss_g/max(1,_n_steps):.4f}")
    return float(_sum_loss_g / max(1, _n_steps)), mean_train_iou


@torch.no_grad()
def evaluate_and_export(
    model: HybridBreastRecon,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epoch: int,
    max_visual: int = 5,
    # solidify
    solidify_min_component: int = 64,
    solidify_closing_iters: int = 1,
    solidify_gt: bool = False,
    # local ROI (nipple+areola)
    eval_local_roi: bool = False,
    local_roi_radius_ratio: float = 0.18,
    local_roi_depth_ratio: float = 0.16,
    local_areola_extra_ratio: float = 0.07,
    local_roi_axis: str = "z",
    local_roi_sign: int = 1,
    local_roi_center_mode: str = "gt_proj",
    export_thresh: float = 0.5,
):
    """
    MAIN path: always ZERO (image-only conditioning).
    SELF path: also ZERO (kept for backward-compatible logging).

    Both paths use the same metric pipeline:
      solidify -> voxel_iou + chamfer + F1@{0.01,0.02,0.05}
    Both paths cover the FULL validation set.
    Export (mesh/viz) uses the MAIN path, only for selected_indices samples.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mesh_dir    = os.path.join(out_dir, "meshes")
    viz_dir     = os.path.join(out_dir, "viz")
    details_dir = os.path.join(out_dir, "details")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    if eval_local_roi:
        print(
            f"[EvalROI] ENABLED axis={local_roi_axis} sign={local_roi_sign}(ignored) "
            f"radius={local_roi_radius_ratio:.3f} depth=IGNORED areola_extra=IGNORED"
        )
    print(f"[EvalExport] mesh export threshold={export_thresh:.2f} (aligned with IoU thresholding recommendation)")

    n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0
    selected_indices = set(random.sample(range(n_samples), min(max_visual, n_samples))) if n_samples > 0 else set()

    rows_main: List[Dict[str, object]] = []
    rows_self: List[Dict[str, object]] = []

    def _compute_row_metrics(pred_prob_np, gt_bin_np, roi_mask_np, sid, side):
        pred_prob_eval = pred_prob_np * roi_mask_np.astype(np.float32)
        gt_eval = gt_bin_np & roi_mask_np

        # pred_prob_np must be a continuous probability map in [0, 1].
        if solidify_gt:
            gt_solid = solidify_voxel(
                gt_eval.astype(bool),
                min_component=solidify_min_component,
                closing_iters=solidify_closing_iters,
                fill_holes=False,
            )
        else:
            gt_solid = gt_eval.astype(bool)
        gt_t = torch.from_numpy(gt_solid.astype(np.float32))[None, None].to(device)

        # Helper: compute IoU at one threshold.
        def get_iou_at(thresh):
            bin_np = pred_prob_eval > thresh
            solid_np = solidify_voxel(bin_np, min_component=solidify_min_component, closing_iters=solidify_closing_iters)
            pred_t = torch.from_numpy(solid_np.astype(np.float32))[None, None].to(device)
            # pred_t is already binary, so IoU threshold can stay at 0.5.
            iou = voxel_iou(pred_t, gt_t, thresh=0.5).item() 
            return iou, solid_np

        iou_03, _ = get_iou_at(0.3)
        iou_04, _ = get_iou_at(0.4)
        iou_05, pred_solid = get_iou_at(0.5)

        # Use threshold 0.5 for point-cloud metrics (Chamfer and F1).
        pred_pts_np = _voxel_to_pointcloud(pred_solid, thresh=0.5, max_points=4096)
        gt_pts_np   = _voxel_to_pointcloud(gt_solid,   thresh=0.5, max_points=4096)
        
        if pred_pts_np.shape[0] == 0 or gt_pts_np.shape[0] == 0:
            chamfer_val = float("nan")
        else:
            pred_pts = torch.from_numpy(pred_pts_np)[None].to(device)
            gt_pts   = torch.from_numpy(gt_pts_np)[None].to(device)
            chamfer_val = chamfer_distance(pred_pts, gt_pts).item()

        f1_01 = f1_from_voxels(pred_solid, gt_solid, tau=0.01, max_points=4096, device=str(device))
        f1_02 = f1_from_voxels(pred_solid, gt_solid, tau=0.02, max_points=4096, device=str(device))
        f1_05 = f1_from_voxels(pred_solid, gt_solid, tau=0.05, max_points=4096, device=str(device))

        # Report IoU at multiple thresholds.
        row = {
            "epoch": int(epoch), "id": str(sid), "side": str(side),
            "IoU@0.3": float(iou_03), "IoU@0.4": float(iou_04), "IoU@0.5": float(iou_05), 
            "chamferGood": float(chamfer_val),
            "F1@0.01": float(f1_01), "F1@0.02": float(f1_02), "F1@0.05": float(f1_05),
        }
        return row, pred_solid, gt_solid

    for idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        vox_gt = batch["voxel"].to(device).unsqueeze(1)   # (1,1,G,G,G)
        sid    = batch["id"][0]
        side   = batch["side"][0]
        gt_bin_np = (vox_gt[0, 0] > 0.5).cpu().numpy()
        if eval_local_roi:
            roi_mask_np = build_nipple_areola_cylinder_mask(
                gt_bin_np,
                axis=local_roi_axis,
                sign=local_roi_sign,
                radius_ratio=local_roi_radius_ratio,
                depth_ratio=local_roi_depth_ratio,
                areola_extra_ratio=local_areola_extra_ratio,
                center_mode=local_roi_center_mode,
            )
            # Fairness check: ROI should not introduce extra fragmentation on GT.
            cc_gt_full = _num_cc(gt_bin_np)
            cc_gt_roi = _num_cc(gt_bin_np & roi_mask_np)
            if cc_gt_roi > cc_gt_full + 1:
                print(
                    f"[EvalROI-WARN] id={sid} side={side} GT components full={cc_gt_full} roi={cc_gt_roi} "
                    f"(ROI may be too aggressive on this sample)"
                )
        else:
            roi_mask_np = np.ones_like(gt_bin_np, dtype=bool)

        B = images.shape[0]

        # ---- SELF path: zero z_inst + zero z_attr (image-only ablation baseline) ----
        z_inst_zero = torch.zeros(B, model.inst_dim,       device=device)
        z_attr_zero = torch.zeros(B, model.attr_embed_dim, device=device)
        _, refined_self, _, _, _, _, _ = model.forward_with_payload(images, z_inst_zero, z_attr_zero)
        
        # Use raw probabilities here (do not binarize early).
        pred_prob_self_np = torch.sigmoid(refined_self)[0, 0].cpu().numpy()
        row_self, _, _ = _compute_row_metrics(pred_prob_self_np, gt_bin_np, roi_mask_np, sid, side)
        rows_self.append(row_self)

        # ---- MAIN path: ZERO payload ----
        z_inst_main, z_attr_main = z_inst_zero, z_attr_zero

        _, refined_main, _, _, _, _, _ = model.forward_with_payload(images, z_inst_main, z_attr_main)
        
        # Use raw probabilities here (do not binarize early).
        pred_prob_main_np = torch.sigmoid(refined_main)[0, 0].cpu().numpy()
        row_main, pred_solid_main, gt_solid_main = _compute_row_metrics(pred_prob_main_np, gt_bin_np, roi_mask_np, sid, side)
        rows_main.append(row_main)

        # ---- Export (selected samples, MAIN voxels) ----
        if idx in selected_indices:
            tag = f"ep{epoch:03d}_{sid}_{side}"
            pred_bin_for_export = (pred_prob_main_np * roi_mask_np.astype(np.float32)) > export_thresh
            # Use fill_holes=True only for export/viz - not for metrics
            pred_export = solidify_voxel_for_export(pred_bin_for_export,
                              min_component=solidify_min_component,
                              closing_iters=solidify_closing_iters)
            gt_export   = solidify_voxel_for_export(gt_bin_np & roi_mask_np,
                              min_component=solidify_min_component,
                              closing_iters=solidify_closing_iters)
            save_voxel_as_mesh(pred_export, os.path.join(mesh_dir, f"{tag}_pred.obj"))
            save_voxel_as_mesh(gt_export,   os.path.join(mesh_dir, f"{tag}_gt.obj"))
            save_mesh_comparison(gt_export, pred_export, os.path.join(viz_dir, f"{tag}_mesh_compare.png"))

    # ---- Summary ----
    def _mean_ignore_nan(vals):
        vv = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(sum(vv) / len(vv)) if vv else 0.0

    def _means(rows):
        return {k: _mean_ignore_nan([r[k] for r in rows])
                for k in ["IoU@0.3", "IoU@0.4", "IoU@0.5", "chamferGood", "F1@0.01", "F1@0.02", "F1@0.05"]}

    m_main = _means(rows_main)

    best_thr_main = max(["IoU@0.3", "IoU@0.4", "IoU@0.5"], key=lambda k: m_main[k])
  
    print(
        f"[Eval] Epoch {epoch}: "
        f"IoU3D@0.3={m_main['IoU@0.3']:.4f},IoU3D@0.4={m_main['IoU@0.4']:.4f}, IoU3D@0.5={m_main['IoU@0.5']:.4f}, chamferGood={m_main['chamferGood']:.4f}, "
        f"F1@0.01={m_main['F1@0.01']:.4f}, F1@0.02={m_main['F1@0.02']:.4f}, F1@0.05={m_main['F1@0.05']:.4f}, "
        f"best={best_thr_main}:{m_main[best_thr_main]:.4f}"
    )


    # ---- CSV + Markdown ----
    fieldnames = ["epoch","id","side","IoU@0.3","IoU@0.4","IoU@0.5","chamferGood","F1@0.01","F1@0.02","F1@0.05"]

    for tag, rows, means in [("main", rows_main, m_main)]:
        csv_path = os.path.join(details_dir, f"epoch{epoch:03d}_{tag}_details.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            writer.writerow({"epoch": int(epoch), "id": "MEAN", "side": "",
                             **{k: means[k] for k in ["IoU@0.3", "IoU@0.4", "IoU@0.5", "chamferGood", "F1@0.01", "F1@0.02", "F1@0.05"]}})

    md_path = os.path.join(out_dir, "val_details.md")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## Epoch {epoch:03d}\n\n")
        for tag, rows, means, label in [
            ("MAIN", rows_main, m_main, "MAIN"),
            
        ]:
            best_thr = max(["IoU@0.3", "IoU@0.4", "IoU@0.5"], key=lambda k: means[k])
            f.write(f"### {label}\n")
            f.write(f"IoU3D={means['IoU@0.5']:.6f}, bestThr={best_thr}, bestIoU={means[best_thr]:.6f}, chamferGood={means['chamferGood']:.6f}, "
                    f"F1@0.01={means['F1@0.01']:.6f}, F1@0.02={means['F1@0.02']:.6f}, F1@0.05={means['F1@0.05']:.6f}\n\n")
            f.write("| id | side | IoU3D | chamferGood | F1@0.01 | F1@0.02 | F1@0.05 |\n")
            f.write("|---:|:----:|------:|------------:|------:|------:|------:|\n")
            for r in rows:
                f.write(f"| {r['id']} | {r['side']} | {float(r['IoU@0.5']):.6f} | {float(r['chamferGood']):.6f} | "
                        f"{float(r['F1@0.01']):.6f} | {float(r['F1@0.02']):.6f} | {float(r['F1@0.05']):.6f} |\n")
            f.write(f"| MEAN |  | {means['IoU@0.5']:.6f} | {means['chamferGood']:.6f} | "
                    f"{means['F1@0.01']:.6f} | {means['F1@0.02']:.6f} | {means['F1@0.05']:.6f} |\n\n")

    return m_main["IoU@0.5"], m_main


# ------------------------------
# Main
# ------------------------------

def parse_args():
    """
    Build command-line options for this script.

    Input:
        No direct function input. Values are read from the shell command.

    Output:
        argparse.Namespace with all user settings.
    """
    ap = argparse.ArgumentParser()
    # ---- Data location and split control ----
    ap.add_argument("--data-root", type=str, required=True,
                    help="Default root. If --input-root/--gt-root are not set, both use this root.")
    ap.add_argument("--input-root", type=str, default="",
                    help="Root for RGB inputs. Example: /content/drive/MyDrive/NewModel/Dataset/Nipple_only")
    ap.add_argument("--gt-root", type=str, default="",
                    help="Root for 3D GT/splits. Example: /content/drive/MyDrive/NewModel/Dataset/Single Breast")
    ap.add_argument("--split-ref-root", type=str, default="",
                    help="Reference root that contains split info (raw_data/train|test or gt_data/3D/train|test). "
                         "Required for strict split filtering when gt_layout=nipple_only.")
    ap.add_argument("--input-layout", type=str, default="auto", choices=["auto", "single_breast", "nipple_only"],
                    help="Image folder layout.")
    ap.add_argument("--gt-layout", type=str, default="auto", choices=["auto", "single_breast", "nipple_only"],
                    help="3D GT folder layout.")
    ap.add_argument("--split-train", type=str, default="train")
    ap.add_argument("--split-val", type=str, default="test")

    # ---- Basic data size settings ----
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--grid-size", type=int, default=64)

    # ---- Basic training settings ----
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed for reproducibility.")

    # ---- Model and feature settings ----
    ap.add_argument("--inst-dim",       type=int,   default=32,
                    help="Size of the per-case ID vector.")
    ap.add_argument("--attr-embed-dim", type=int,   default=16,
                    help="Size of the attribute lookup vector.")
    ap.add_argument("--embed-grad-scale", type=float, default=0.1,
                    help="Scale for embedding updates. Lower means embeddings change slower.")
    ap.add_argument("--unfreeze-layer3", action="store_true", default=False,
                    help="Allow ResNet layer3 to train.")
    ap.add_argument("--encoder-lr", type=float, default=1e-5,
                    help="Learning rate for encoder layers.")
    ap.add_argument("--enable-hires-refine", action="store_true", default=False,
                    help="Turn on extra 64->128 output refinement.")
    ap.add_argument("--hires-refine-size", type=int, default=128,
                    help="Output size used by the high-resolution refinement step.")
    ap.add_argument("--hires-weight", type=float, default=0.2,
                    help="Loss weight for the high-resolution output.")
    ap.add_argument("--lambda-boundary", type=float, default=0.1,
                    help="Loss weight for edge/boundary matching.")
    ap.add_argument("--embed-dropout-prob", type=float, default=0.3,
                    help="Chance to drop both embedding vectors in one training step.")
    ap.add_argument("--lambda-contrastive", type=float, default=0.0,
                    help="Loss weight for feature separation across samples.")
    ap.add_argument("--pe-freqs", type=int, default=6)

    # ---- Loss weights ----
    ap.add_argument("--lambda-attr-sup",  type=float, default=0.0,
                    help="Loss weight for predicted global attributes.")
    ap.add_argument("--lambda-stats-sup", type=float, default=0.0,
                    help="Loss weight for predicted voxel summary values.")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--recon-bce-weight", type=float, default=1.0)
    ap.add_argument("--recon-dice-weight", type=float, default=0.7)
    ap.add_argument("--coarse-weight", type=float, default=0.3)
    ap.add_argument("--lambda-volume", type=float, default=0.1)
    ap.add_argument("--lambda-chamfer", type=float, default=0.0)
    ap.add_argument("--lambda-normal", type=float, default=0.0)
    ap.add_argument("--lambda-perc", type=float, default=0.0)
    ap.add_argument("--lambda-implicit", type=float, default=0.0)
    ap.add_argument("--stage2-start-epoch", type=int, default=-1,
                    help="If > 0, switch to stage-2 training at this epoch.")
    ap.add_argument("--stage2-lambda-chamfer", type=float, default=0.02,
                    help="Chamfer loss weight used in stage-2.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    # ---- Nipple geometry settings for GT attribute extraction ----
    ap.add_argument("--nipple-axis", type=str, default="z", choices=["x", "y", "z"])
    ap.add_argument("--nipple-sign", type=int, default=1, choices=[-1, 1])

    # ---- Local ROI settings used in ablation evaluation/export ----
    ap.add_argument("--eval-local-roi", action="store_true", default=False,
                    help="Use local nipple+areola region for both metrics and mesh export.")
    ap.add_argument("--local-roi-radius-ratio", type=float, default=0.18,
                    help="ROI radius as fraction of grid size.")
    ap.add_argument("--local-roi-depth-ratio", type=float, default=0.16,
                    help="Deprecated: depth slab is disabled; kept only for backward compatibility.")
    ap.add_argument("--local-areola-extra-ratio", type=float, default=0.07,
                    help="Deprecated: depth slab is disabled; kept only for backward compatibility.")
    ap.add_argument("--local-roi-axis", type=str, default="z", choices=["x", "y", "z"],
                    help="Axis used to find nipple direction for ROI center.")
    ap.add_argument("--local-roi-sign", type=int, default=1, choices=[-1, 1],
                    help="Direction (+1 or -1) along the ROI axis.")
    ap.add_argument("--export-thresh", type=float, default=0.5,
                    help="Threshold for exported prediction mesh. 0.5 matches IoU@0.5.")

    # ---- Implicit head sampling settings ----
    ap.add_argument("--implicit-pts", type=int, default=4096)
    ap.add_argument("--implicit-surface-ratio", type=float, default=0.75)

    # ---- Optional post-process settings for mesh stability ----
    ap.add_argument("--solidify-min-component", type=int, default=64)
    ap.add_argument("--solidify-closing-iters", type=int, default=1)

    # ---- Output and runtime settings ----
    ap.add_argument("--out-dir", type=str, default="/content/drive/MyDrive/NewModel/outputs_hybrid_mesh_noTTO")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    return ap.parse_args()


def main():
    """
    Run the full training and validation workflow.

    Input:
        No direct function input. Uses parsed command-line options.

    Output:
        None. Writes checkpoints, plots, and evaluation files to disk.
    """
    # 1) Read user options and choose CPU/GPU.
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[Seed] {args.seed}")

    # 2) Resolve input/label roots and split reference root.
    input_root = args.input_root if args.input_root else args.data_root
    gt_root = args.gt_root if args.gt_root else args.data_root
    split_ref_root = args.split_ref_root if args.split_ref_root else gt_root

    # 3) Build one global (id, side) index map from training split only.
    index_map = build_global_index_map(
        gt_root,
        [args.split_train],
        gt_layout=args.gt_layout,
        split_ref_root=split_ref_root,
    )  # only train samples get embeddings
    num_instances = len(index_map)
    print(f"[InstanceEmbedding] total unique (id,side) across splits = {num_instances}")

    # 4) Create train/validation datasets and loaders.
    train_ds = SingleBreastDataset(
        data_root=args.data_root, split=args.split_train, img_size=args.img_size, grid_size=args.grid_size, index_map=index_map,
        nipple_axis=args.nipple_axis, nipple_sign=args.nipple_sign,
        input_root=input_root, gt_root=gt_root, input_layout=args.input_layout, gt_layout=args.gt_layout,
        split_ref_root=split_ref_root,
    )
    val_ds = SingleBreastDataset(
        data_root=args.data_root, split=args.split_val, img_size=args.img_size, grid_size=args.grid_size, index_map=index_map,
        nipple_axis=args.nipple_axis, nipple_sign=args.nipple_sign,
        input_root=input_root, gt_root=gt_root, input_layout=args.input_layout, gt_layout=args.gt_layout,
        split_ref_root=split_ref_root,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    # 5) Build model and move it to selected device.
    model = HybridBreastRecon(
        n_views=len(VIEW_ORDER),
        latent_dim=256,   # 512 overfits on 70 samples; 256 gives better generalisation
        inst_dim=args.inst_dim,
        attr_embed_dim=args.attr_embed_dim,
        grid_size=args.grid_size,
        num_train_instances=num_instances,
        pe_freqs=args.pe_freqs,
        attr_dim=4,
        embed_grad_scale=args.embed_grad_scale,
        unfreeze_layer3=args.unfreeze_layer3,
        enable_hires_refine=args.enable_hires_refine,
        hires_refine_size=args.hires_refine_size,
    ).to(device)

    # 6) Initialize attribute embedding table from GT attributes.
    model.init_attr_embed_from_gt(
        index_map=index_map,
        gt_root=gt_root,
        split=args.split_train,
        nipple_axis=args.nipple_axis,
        nipple_sign=args.nipple_sign,
        grid_size=args.grid_size,
        gt_layout=args.gt_layout,
    )

    perc3d = Perceptual3DNet(in_channels=1, feat_dim=128).to(device)
    disc3d = Discriminator3D(in_channels=1).to(device)

    # 7) Build optimizers with separate learning rates for key parameter groups.
    embed_params = list(model.inst_embed.parameters()) + list(model.attr_embed_table.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    group_param_ids = {id(p) for p in embed_params + encoder_params}
    other_params = [p for p in list(model.parameters()) + list(perc3d.parameters())
                    if id(p) not in group_param_ids]
    opt_g = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": other_params,  "lr": args.lr},
            {"params": embed_params,  "lr": args.lr * args.embed_grad_scale},
        ],
        betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    opt_d = torch.optim.Adam(
        disc3d.parameters(),
        lr=args.lr * 0.5, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )

    # 8) Learning-rate schedulers.
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=args.epochs, eta_min=args.lr * 0.025
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # 9) Load checkpoint if resume path is provided.
    best_score = -1.0
    start_epoch = 1

    if args.resume:
        print(f"[Resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        perc3d.load_state_dict(ckpt["perc3d"])
        disc3d.load_state_dict(ckpt["disc3d"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score  = ckpt.get("best_score", -1.0)
        if "scheduler_g" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if "scheduler_d" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"[Resume] Start from epoch {start_epoch}, best_score={best_score:.4f}")

    if start_epoch > args.epochs:
        print(f"[Warning] start_epoch ({start_epoch}) > total epochs ({args.epochs}), nothing to do.")
        return

    # 10) Prepare history buffers for curve plotting.
    hist_epochs: List[int] = []
    hist_total_loss: List[float] = []
    hist_iou3d: List[float] = []
    hist_train_iou: List[float] = []
    curve_path     = os.path.join(args.out_dir, "curves", "loss_vs_iou3d.png")
    iou_curve_path = os.path.join(args.out_dir, "curves", "train_vs_val_iou.png")

    # 11) Main epoch loop: train, evaluate, save best/periodic checkpoints.
    for ep in range(start_epoch, args.epochs + 1):
        in_stage2 = (args.stage2_start_epoch > 0 and ep >= args.stage2_start_epoch)
        lambda_chamfer_ep = args.stage2_lambda_chamfer if in_stage2 else args.lambda_chamfer
        if ep == start_epoch or (in_stage2 and ep == args.stage2_start_epoch):
            stage_tag = "stage2" if in_stage2 else "stage1"
            print(f"[TrainConfig] epoch={ep} mode={stage_tag} lambda_chamfer={lambda_chamfer_ep}")
        epoch_mean_loss, train_iou = train_one_epoch(
            model=model,
            perc3d=perc3d,
            disc3d=disc3d,
            loader=train_loader,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
            epoch=ep,
            max_epoch=args.epochs,
            lambda_chamfer=lambda_chamfer_ep,
            lambda_normal=args.lambda_normal,
            lambda_perc=args.lambda_perc,
            implicit_pts=args.implicit_pts,
            implicit_surface_ratio=args.implicit_surface_ratio,
            lambda_implicit=args.lambda_implicit,
            lambda_attr_sup=args.lambda_attr_sup,
            lambda_stats_sup=args.lambda_stats_sup,
            lambda_contrastive=args.lambda_contrastive,
            embed_dropout_prob=args.embed_dropout_prob,
            pos_weight=args.pos_weight,
            recon_bce_weight=args.recon_bce_weight,
            recon_dice_weight=args.recon_dice_weight,
            coarse_weight=args.coarse_weight,
            lambda_volume=args.lambda_volume,
            hires_weight=args.hires_weight if args.enable_hires_refine else 0.0,
            lambda_boundary=args.lambda_boundary,
            hires_refine_size=args.hires_refine_size,
        )

        eval_dir = os.path.join(args.out_dir, "eval")
        mean_main_iou3d, mean_details = evaluate_and_export(
            model=model,
            loader=val_loader,
            device=device,
            out_dir=eval_dir,
            epoch=ep,
            max_visual=5,
            solidify_min_component=args.solidify_min_component,
            solidify_closing_iters=args.solidify_closing_iters,
            eval_local_roi=args.eval_local_roi,
            local_roi_radius_ratio=args.local_roi_radius_ratio,
            local_roi_depth_ratio=args.local_roi_depth_ratio,
            local_areola_extra_ratio=args.local_areola_extra_ratio,
            local_roi_axis=args.local_roi_axis,
            local_roi_sign=args.local_roi_sign,
            export_thresh=args.export_thresh,
        )
        scheduler_g.step()
        scheduler_d.step()

        # 12) Update and save epoch curves.
        hist_epochs.append(int(ep))
        hist_total_loss.append(float(epoch_mean_loss))
        hist_iou3d.append(float(mean_main_iou3d))
        hist_train_iou.append(float(train_iou))
        plot_epoch_curves(hist_epochs, hist_total_loss, hist_iou3d, curve_path)

        # Save a direct train-vs-validation IoU chart to check overfitting.
        os.makedirs(os.path.dirname(iou_curve_path), exist_ok=True)
        plt.figure()
        plt.plot(hist_epochs, hist_train_iou, label="Train IoU")
        plt.plot(hist_epochs, hist_iou3d,     label="Val IoU (MAIN)")
        plt.xlabel("Epoch"); plt.ylabel("IoU3D")
        plt.title("Train vs Val IoU  (gap = overfitting)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(iou_curve_path, dpi=160); plt.close()


        # 13) Use validation MAIN IoU3D as the score for "best model".
        score_for_best = mean_main_iou3d

        if score_for_best > best_score:
            best_score = score_for_best
            ckpt_best_path = os.path.join(args.out_dir, "ckpt_best.pth")
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "perc3d": perc3d.state_dict(),
                "disc3d": disc3d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "args": vars(args),
                "best_score": best_score,
            }, ckpt_best_path)
            print(f"[Checkpoint] New best score={best_score:.4f}, saved to {ckpt_best_path}")

        # 14) Also save periodic checkpoints for recovery and later analysis.
        if ep % 5 == 0 or (ep > 20 and ep % 1 == 0):
            ckpt_ep_path = os.path.join(args.out_dir, f"ckpt_ep{ep:03d}.pth")
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "perc3d": perc3d.state_dict(),
                "disc3d": disc3d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "args": vars(args),
                "best_score": best_score,
            }, ckpt_ep_path)
            print(f"[Checkpoint] Saved periodic checkpoint to {ckpt_ep_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
