#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified runner for breast reconstruction and 3D classification.

This script provides four actions:
1) 3D pipeline reconstruction for full single-side breast.
2) 2D pipeline reconstruction for full single-side breast.
3) 3D pipeline reconstruction for nipple+areola input.
4) Classification over reconstructed 3D binvox results.

Design goals:
- One command window for non-technical users.
- No GT metrics during reconstruction steps.
- Reconstruction outputs: OBJ mesh + visualization image (+binvox for classification).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SINGLE_DATA = BASE_DIR / "Dataset" / "Single Breast"
DEFAULT_NIPPLE_DATA = BASE_DIR / "Dataset" / "Nipple_only"
DEFAULT_NIPPLE_CYL_GT_DATA = BASE_DIR / "Dataset" / "Single Breast (nipple_cyl_gt)"

DEFAULT_CKPT_3D_FULL = BASE_DIR / "3D_Pipeline_WholeBreast_Checkpoint.pth"
DEFAULT_CKPT_3D_NIPPLE = BASE_DIR / "3D_Pipeline_Nipple_Checkpoint.pth"
DEFAULT_CKPT_2D_FULL = BASE_DIR / "2D_Pipeline_Checkpoint.pth"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

# These preview constants are kept for the 2D pipeline preview only.
PREVIEW_VIEW_ELEV = 30
PREVIEW_VIEW_AZIM = 120


def _safe_int(x: str):
    try:
        return int(x)
    except ValueError:
        return x


def _load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _torch_load_checkpoint(ckpt_path: Path, device):
    """
    Load a torch checkpoint with backward compatibility for PyTorch 2.6+.

    Why this is needed:
    - In PyTorch 2.6, torch.load defaults to weights_only=True.
    - Older checkpoints may contain extra Python objects and fail to load.

    Behavior:
    1) Try normal torch.load first.
    2) If it fails with "Weights only load failed", retry with weights_only=False.

    Note:
    - weights_only=False can execute pickled code.
    - Only use this path for trusted local checkpoints.
    """
    import torch

    try:
        return torch.load(str(ckpt_path), map_location=device)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg:
            raise
        print(
            "[WARN] Checkpoint requires legacy pickle loading. "
            "Retrying with weights_only=False (trusted checkpoint assumed)."
        )
        try:
            return torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            # For older torch versions that do not expose weights_only.
            return torch.load(str(ckpt_path), map_location=device)


def _state_dict_has_module_prefix(state_dict: Dict) -> bool:
    if not state_dict:
        return False
    first_key = next(iter(state_dict.keys()))
    return str(first_key).startswith("module.")


def _strip_module_prefix(state_dict: Dict) -> OrderedDict:
    out = OrderedDict()
    for k, v in state_dict.items():
        key = str(k)
        out[key[7:] if key.startswith("module.") else key] = v
    return out


def _add_module_prefix(state_dict: Dict) -> OrderedDict:
    out = OrderedDict()
    for k, v in state_dict.items():
        key = str(k)
        out[key if key.startswith("module.") else f"module.{key}"] = v
    return out


def _load_state_dict_with_prefix_fix(model, state_dict: Dict, name: str):
    """
    Load model weights and auto-fix DataParallel prefix mismatch.

    It handles this common mismatch:
    - checkpoint keys: module.xxx
    - model keys: xxx
    (or the reverse)
    """
    model_has_module = _state_dict_has_module_prefix(model.state_dict())
    ckpt_has_module = _state_dict_has_module_prefix(state_dict)

    candidates = [state_dict]
    if ckpt_has_module:
        candidates.append(_strip_module_prefix(state_dict))
    else:
        candidates.append(_add_module_prefix(state_dict))

    # Put the most likely key format first for faster success.
    if model_has_module != ckpt_has_module:
        candidates = [candidates[1], candidates[0]]

    last_err = None
    for cand in candidates:
        try:
            model.load_state_dict(cand, strict=True)
            return
        except RuntimeError as e:
            last_err = e

    raise RuntimeError(f"Failed to load state_dict for {name}: {last_err}")


def _resolve_first_existing(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_input_path(raw_path: str) -> Path:
    """
    Resolve user-facing path arguments relative to the project folder.

    This keeps CLI usage portable across machines: users can pass paths like
    "outputs/recon3d_full" without depending on a machine-specific absolute root.
    """
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve(strict=False)


def _make_image_candidates(folder: Path, stem: str) -> List[Path]:
    return [folder / f"{stem}{ext}" for ext in IMAGE_EXTS]


def _write_manifest(rows: List[Dict], out_csv: Path):
    _ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "side", "status", "message"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _link_or_copy_dir(src: Path, dst: Path):
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _scan_single_breast_samples(data_root: Path, split: str) -> List[Dict]:
    raw_root = data_root / "raw_data" / split
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing split folder: {raw_root}")

    samples: List[Dict] = []
    for sid_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: _safe_int(p.name)):
        sid = sid_dir.name

        left_views = {
            "front": _resolve_first_existing(_make_image_candidates(sid_dir, "F_LS")),
            "side": _resolve_first_existing(_make_image_candidates(sid_dir, "SL_LS")),
            "top": _resolve_first_existing(_make_image_candidates(sid_dir, "T_LS")),
        }
        if all(v is not None for v in left_views.values()):
            samples.append({"id": sid, "side": "left", "views": left_views})

        right_views = {
            "front": _resolve_first_existing(_make_image_candidates(sid_dir, "F_RS")),
            "side": _resolve_first_existing(_make_image_candidates(sid_dir, "SR_RS")),
            "top": _resolve_first_existing(_make_image_candidates(sid_dir, "T_RS")),
        }
        if all(v is not None for v in right_views.values()):
            samples.append({"id": sid, "side": "right", "views": right_views})

    return samples


def _scan_nipple_samples(data_root: Path, split: str = "test") -> List[Dict]:
    samples: List[Dict] = []
    side_cfg = {
        "left": ("Left", "LB"),
        "right": ("Right", "RB"),
    }
    for side, (side_folder, code) in side_cfg.items():
        raw_base = data_root / side_folder / "Raw"
        split_root = raw_base / split if split else raw_base

        if split and split_root.exists():
            raw_root = split_root
        else:
            # If split folders exist but requested split is missing, fail fast.
            known_splits = ["train", "valid", "test"]
            if split and any((raw_base / s).exists() for s in known_splits):
                raise FileNotFoundError(
                    f"Missing split folder for nipple data: {split_root} "
                    f"(available under {raw_base}: {known_splits})"
                )
            raw_root = raw_base

        if not raw_root.exists():
            continue
        for sid_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: _safe_int(p.name)):
            sid = sid_dir.name
            views = {
                "front": _resolve_first_existing(_make_image_candidates(sid_dir, f"{code}_FV")),
                "side": _resolve_first_existing(
                    _make_image_candidates(sid_dir, f"{code}_SV") +
                    _make_image_candidates(sid_dir, f"{code}__SV")
                ),
                "top": _resolve_first_existing(_make_image_candidates(sid_dir, f"{code}_TV")),
            }
            if all(v is not None for v in views.values()):
                samples.append({"id": sid, "side": side, "views": views})
    return samples


def _load_image_for_3d(path: Path, img_size: int) -> torch.Tensor:
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def _load_image_for_2d(path: Path, p2_module, mask_path: Path | None = None) -> np.ndarray:

    try:
        import cv2
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OpenCV (cv2) is required for 2D reconstruction. "
            "Please install opencv-python first."
        ) from e

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        fallback_mask = (raw > 0).astype(np.uint8) * 255
    elif raw.shape[2] == 4:
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
        fallback_mask = (img[:, :, 3] > 0).astype(np.uint8) * 255
    else:
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        fallback_mask = (np.mean(img[:, :, :3], axis=2) < 250).astype(np.uint8) * 255

    # Prefer official 2D GT mask for alignment if available (faithful to original 2D pipeline).
    if mask_path is not None and mask_path.exists():
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None and gt_mask.shape[:2] == img.shape[:2]:
            mask = gt_mask
        else:
            mask = fallback_mask
    else:
        mask = fallback_mask

    # Keep this exactly aligned with Pipeline_2d.BreastDataset logic.
    aligned_img, _ = p2_module.normalize_scale_and_center(img, mask, target_max_size=160)
    return aligned_img


def _view_filename_2d(side: str, view_key: str) -> str:
    """
    Resolve raw/GT 2D file name by side and view key.
    side: 'left' or 'right'
    view_key: 'front' | 'top' | 'side'
    """
    if side == "left":
        mp = {"front": "F_LS.png", "top": "T_LS.png", "side": "SL_LS.png"}
    else:
        mp = {"front": "F_RS.png", "top": "T_RS.png", "side": "SR_RS.png"}
    return mp[view_key]


def _mask_path_2d(data_root: Path, split: str, sid: str, side: str, view_key: str) -> Path:
    return data_root / "gt_data" / "2D" / split / sid / _view_filename_2d(side, view_key)


def _save_binvox_bool(vox: np.ndarray, out_path: Path):
    vox = (vox > 0).astype(np.uint8)
    d, h, w = vox.shape
    flat = vox.reshape(-1)

    _ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        f.write(b"#binvox 1\n")
        f.write(f"dim {d} {h} {w}\n".encode("ascii"))
        f.write(b"translate 0.0 0.0 0.0\n")
        f.write(b"scale 1.0\n")
        f.write(b"data\n")

        if flat.size == 0:
            return

        val = int(flat[0])
        count = 0
        for x in flat:
            x = int(x)
            if x == val and count < 255:
                count += 1
            else:
                f.write(bytes([val, count]))
                val = x
                count = 1
        f.write(bytes([val, count]))


def _save_projection_preview(vox: np.ndarray, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_path.parent)
    v = (vox > 0).astype(np.float32)
    front = v.max(axis=0)
    top = v.max(axis=1)
    side = v.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(title)
    axes[0].imshow(front, cmap="gray")
    axes[0].set_title("Front")
    axes[1].imshow(top, cmap="gray")
    axes[1].set_title("Top")
    axes[2].imshow(side, cmap="gray")
    axes[2].set_title("Side")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_mesh_preview(
    vox: np.ndarray,
    out_path: Path,
    title: str,
    zoom_to_object: bool = False,
    zoom_margin_vox: int = 2,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    _ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    if np.sum(vox > 0) > 0:
        verts, faces, _, _ = measure.marching_cubes(vox.astype(np.float32), level=0.5)
        tri = verts[faces]
        mesh = Poly3DCollection(tri, linewidths=0.05, alpha=0.9)
        mesh.set_edgecolor("k")
        ax.add_collection3d(mesh)

        if zoom_to_object:
            occ = np.argwhere(vox > 0)
            if occ.shape[0] > 0:
                zmin, ymin, xmin = occ.min(axis=0)
                zmax, ymax, xmax = occ.max(axis=0)
                m = int(max(0, zoom_margin_vox))
                ax.set_xlim(max(0, zmin - m), min(vox.shape[0], zmax + m + 1))
                ax.set_ylim(max(0, ymin - m), min(vox.shape[1], ymax + m + 1))
                ax.set_zlim(max(0, xmin - m), min(vox.shape[2], xmax + m + 1))
            else:
                ax.set_xlim(0, vox.shape[0])
                ax.set_ylim(0, vox.shape[1])
                ax.set_zlim(0, vox.shape[2])
        else:
            ax.set_xlim(0, vox.shape[0])
            ax.set_ylim(0, vox.shape[1])
            ax.set_zlim(0, vox.shape[2])

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=PREVIEW_VIEW_ELEV, azim=PREVIEW_VIEW_AZIM)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)



def _build_3d_model_from_checkpoint(p3, ckpt_path: Path, device: torch.device):
    ckpt = _torch_load_checkpoint(ckpt_path, device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ck_args = ckpt.get("args", {})

    inst_w = state["inst_embed.weight"]
    attr_w = state["attr_embed_table.weight"]
    num_train_instances = int(inst_w.shape[0])
    inst_dim = int(inst_w.shape[1])
    attr_embed_dim = int(attr_w.shape[1])

    model = p3.HybridBreastRecon(
        n_views=len(p3.VIEW_ORDER),
        latent_dim=int(ck_args.get("latent_dim", 256)),
        inst_dim=inst_dim,
        attr_embed_dim=attr_embed_dim,
        grid_size=int(ck_args.get("grid_size", 64)),
        num_train_instances=num_train_instances,
        pe_freqs=int(ck_args.get("pe_freqs", 6)),
        attr_dim=4,
        embed_grad_scale=float(ck_args.get("embed_grad_scale", 0.1)),
        unfreeze_layer3=bool(ck_args.get("unfreeze_layer3", False)),
        enable_hires_refine=bool(ck_args.get("enable_hires_refine", False)),
        hires_refine_size=int(ck_args.get("hires_refine_size", 128)),
    ).to(device)
    _load_state_dict_with_prefix_fix(model, state, "3D model")
    model.eval()
    return model, ck_args


def _load_gt_voxel_for_roi(p3, args, sid: str, side: str):
    """
    Load the GT voxel for one sample so we can compute the ROI mask in the
    same way as evaluate_and_export() does.  Returns a (G,G,G) bool numpy
    array, or None when no GT file is found (runner then falls back to
    all-ones mask, matching evaluate_and_export eval_local_roi=False).
    """
    import torch
    import torch.nn.functional as F

    data_root = Path(getattr(args, "gt_root", args.data_root))
    split = args.split

    # Resolve GT path using the same helper as 3D_pipeline.py
    gt_layout = "single_breast" if args.layout == "single_breast" else "single_breast"
    gt_path = p3.resolve_voxel_path(str(data_root), split, sid, side, gt_layout)
    if gt_path is None:
        return None

    grid_size = int(getattr(args, "grid_size", 64))
    vox = p3.load_binvox(gt_path, device=None)          # (D,H,W) bool tensor
    v = vox.float().unsqueeze(0).unsqueeze(0)            # (1,1,D,H,W)
    v = F.interpolate(v, size=(grid_size, grid_size, grid_size),
                      mode="trilinear", align_corners=False)
    return (v[0, 0] > 0.5).numpy()                      # (G,G,G) bool numpy


def run_recon_3d(args):
    """
    3D reconstruction runner. Output logic is kept identical to
    evaluate_and_export() in 3D_pipeline.py:

    1. Inference: forward_with_payload with zero z_inst / z_attr.
    2. Thresholding: prob > export_thresh  (NO GT-derived masking).
       Rationale: do not leak GT information into inference outputs.
    3. Solidify:
       - recon3d_full: solidify_voxel(..., fill_holes=False) so saved binvox is
         aligned with metric-time solidify (no hole filling).
       - recon3d_nipple: solidify_voxel_for_export(..., fill_holes=True) for
         cleaner visualisation meshes (legacy/default behavior).
    4. Optional ROI visualisation: if --eval-local-roi is set AND GT exists,
       we also save an additional ROI-cropped mesh_compare image that applies
       the same GT-derived cylinder mask to BOTH GT and Pred.
    5. OBJ export: save_voxel_as_mesh(..., level=0.5).
    6. Visualization: save_mesh_comparison() -> plot_mesh_pair()
       — 8x4 figure, dpi=200, auto_scale_xyz, GT on left / Pred on right.
    """
    import torch

    p3 = _load_module("pipeline3d_mod", BASE_DIR / "3D_pipeline.py")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ck_args = _build_3d_model_from_checkpoint(p3, Path(args.checkpoint), device)

    img_size = int(ck_args.get("img_size", 224))
    inst_dim = model.inst_dim
    attr_dim = model.attr_embed_dim

    if args.layout == "single_breast":
        samples = _scan_single_breast_samples(Path(args.data_root), args.split)
    else:
        samples = _scan_nipple_samples(Path(args.data_root), args.split)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid samples found for reconstruction.")

    out_dir = Path(args.out_dir)
    mesh_dir = out_dir / "meshes"
    viz_dir = out_dir / "visualization"
    left_bin_dir = out_dir / "binvoxFiles" / "left_breast_recon"
    right_bin_dir = out_dir / "binvoxFiles" / "right_breast_recon"
    for p in [mesh_dir, viz_dir, left_bin_dir, right_bin_dir]:
        _ensure_dir(p)

    print(f"[3D] Device: {device}")
    print(f"[3D] Samples: {len(samples)}")
    print(f"[3D] Split: {args.split}")
    print(f"[3D] Export threshold: {args.export_thresh}")
    print(f"[3D] Solidify: min_component={args.solidify_min_component}, closing_iters={args.solidify_closing_iters}")
    if args.eval_local_roi:
        print(
            f"[3D] GT-based ROI mask: ENABLED  axis={args.local_roi_axis}  "
            f"radius={args.local_roi_radius_ratio}  depth={args.local_roi_depth_ratio}  "
            f"areola_extra={args.local_areola_extra_ratio}"
        )
    else:
        print("[3D] GT-based ROI mask: DISABLED (full-grid prediction)")

    manifest_rows: List[Dict] = []
    ok_count = 0
    skip_count = 0

    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            sid, side = s["id"], s["side"]
            tag = f"{sid}_{side}"
            try:
                # ---- Load input images (identical to evaluate_and_export DataLoader) ----
                views = [_load_image_for_3d(s["views"][v], img_size) for v in ["front", "side", "top"]]
                images = torch.stack(views, dim=0).unsqueeze(0).to(device)  # [1,3,H,W] x3 -> [1,3,3,H,W]

                # ---- Inference: zero embeddings, image-only (MAIN path) ----
                z_inst = torch.zeros((1, inst_dim), dtype=torch.float32, device=device)
                z_attr = torch.zeros((1, attr_dim), dtype=torch.float32, device=device)
                _, refined_logits, _, _, _, _, _ = model.forward_with_payload(images, z_inst, z_attr)
                pred_prob_np = torch.sigmoid(refined_logits)[0, 0].cpu().numpy()

                # ---- GT voxel (for nipple output visualisation / ROI masking) ----
                gt_bin_np = _load_gt_voxel_for_roi(p3, args, sid, side)

                # ---- Optional cylinder ROI (nipple-only): mask both pred & GT consistently ----
                roi_mask_np = None
                if getattr(args, "layout", "") == "nipple_only" and bool(getattr(args, "eval_local_roi", True)):
                    # For nipple-only outputs we always export the *ROI-only* reconstruction,
                    # aligned with 3D_pipeline.evaluate_and_export() ROI logic.
                    if gt_bin_np is None:
                        grid = int(getattr(args, "grid_size", 64))
                        gt_for_shape = np.zeros((grid, grid, grid), dtype=bool)
                    else:
                        gt_for_shape = gt_bin_np
                    roi_mask_np = p3.build_nipple_areola_cylinder_mask(
                        gt_for_shape,
                        axis=args.local_roi_axis,
                        sign=args.local_roi_sign,
                        radius_ratio=args.local_roi_radius_ratio,
                        depth_ratio=args.local_roi_depth_ratio,
                        areola_extra_ratio=args.local_areola_extra_ratio,
                        center_mode="grid_center",
                    )
                    pred_prob_np = pred_prob_np * roi_mask_np.astype(np.float32)

                # ---- Threshold (NO GT-derived masking; keep raw prediction) ----
                pred_bin_for_export = (pred_prob_np > args.export_thresh)

                # ---- Solidify ----
                # For recon3d_full we disable hole filling so downstream evaluation
                # (IoU/Chamfer/F1) matches 3D_pipeline metric defaults.
                if getattr(args, "layout", "") == "single_breast":
                    pred_export = p3.solidify_voxel(
                        pred_bin_for_export,
                        keep_largest=True,
                        min_component=int(args.solidify_min_component),
                        do_closing=True,
                        closing_iters=int(args.solidify_closing_iters),
                        fill_holes=False,
                    )
                else:
                    pred_export = p3.solidify_voxel_for_export(
                        pred_bin_for_export,
                        min_component=int(args.solidify_min_component),
                        closing_iters=int(args.solidify_closing_iters),
                    )

                # ---- OBJ export (level=0.5, identical) ----
                p3.save_voxel_as_mesh(pred_export, str(mesh_dir / f"{tag}.obj"), level=0.5)

                # ---- Visualization ----
                if getattr(args, "layout", "") == "nipple_only":
                    gt_viz = None
                    if gt_bin_np is not None:
                        gt_bin_viz = (gt_bin_np & roi_mask_np) if roi_mask_np is not None else gt_bin_np
                        gt_viz = p3.solidify_voxel_for_export(
                            gt_bin_viz,
                            min_component=int(args.solidify_min_component),
                            closing_iters=int(args.solidify_closing_iters),
                        )
                        # Optional: also export GT obj for nipple-only outputs (matches 3D_pipeline export outputs).
                        p3.save_voxel_as_mesh(gt_viz, str(mesh_dir / f"{tag}_gt.obj"), level=0.5)
                        p3.save_voxel_as_mesh(pred_export, str(mesh_dir / f"{tag}_pred.obj"), level=0.5)
                    p3.save_mesh_comparison(
                        gt_viz,
                        pred_export,
                        str(viz_dir / f"{tag}_mesh_compare.png"),
                    )
                else:
                    # full-grid GT+Pred (no masking)
                    p3.save_mesh_comparison(
                        gt_bin_np if gt_bin_np is not None else None,
                        pred_export,
                        str(viz_dir / f"{tag}_mesh_compare.png"),
                    )

                # ---- Binvox ----
                bin_dir = left_bin_dir if side == "left" else right_bin_dir
                _save_binvox_bool(pred_export, bin_dir / f"{sid}.binvox")

                ok_count += 1
                manifest_rows.append({"id": sid, "side": side, "status": "ok", "message": ""})
                print(f"[3D] {i}/{len(samples)} done: {tag}")
            except Exception as e:
                msg = str(e)
                manifest_rows.append({"id": sid, "side": side, "status": "skipped", "message": msg})
                skip_count += 1
                if args.strict:
                    raise
                print(f"[3D][WARN] skipped {tag}: {msg}")

    _write_manifest(manifest_rows, out_dir / "run_manifest.csv")
    print(f"[3D] Summary: ok={ok_count}, skipped={skip_count}")
    print(f"[3D] Finished. Output: {out_dir}")


def _build_2d_models_from_checkpoint(p2, ckpt_path: Path, device: torch.device):
    import torch

    cfg = p2.cfg
    encoder = p2.Encoder(cfg)
    decoder = p2.Decoder(cfg)
    refiner = p2.Refiner(cfg)
    merger = p2.Merger(cfg)

    if device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder).to(device)
        decoder = torch.nn.DataParallel(decoder).to(device)
        refiner = torch.nn.DataParallel(refiner).to(device)
        merger = torch.nn.DataParallel(merger).to(device)
    else:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        refiner = refiner.to(device)
        merger = merger.to(device)

    ckpt = _torch_load_checkpoint(ckpt_path, device)
    print(f"[2D] Checkpoint: {ckpt_path}")
    if isinstance(ckpt, dict):
        ep = ckpt.get("epoch_idx", None)
        biou = ckpt.get("best_iou", None)
        bep = ckpt.get("best_epoch", None)
        msg = f"[2D] Checkpoint meta: epoch={ep}"
        if biou is not None:
            msg += f", best_iou={biou:.4f}"
        if bep is not None:
            msg += f", best_epoch={bep}"
        print(msg)

    _load_state_dict_with_prefix_fix(encoder, ckpt["encoder_state_dict"], "2D encoder")
    _load_state_dict_with_prefix_fix(decoder, ckpt["decoder_state_dict"], "2D decoder")
    if "refiner_state_dict" in ckpt:
        _load_state_dict_with_prefix_fix(refiner, ckpt["refiner_state_dict"], "2D refiner")
    if "merger_state_dict" in ckpt:
        _load_state_dict_with_prefix_fix(merger, ckpt["merger_state_dict"], "2D merger")

    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()
    return cfg, encoder, decoder, refiner, merger


def run_recon_2d(args):
    import torch

    p2 = _load_module("pipeline2d_mod", BASE_DIR / "2D_pipeline.py")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg, encoder, decoder, refiner, merger = _build_2d_models_from_checkpoint(
        p2, Path(args.checkpoint), device
    )

    samples = _scan_single_breast_samples(Path(args.data_root), args.split)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid samples found for reconstruction.")

    out_dir = Path(args.out_dir)
    mesh_dir = out_dir / "meshes"
    viz_dir = out_dir / "visualization"
    left_bin_dir = out_dir / "binvoxFiles" / "left_breast_recon"
    right_bin_dir = out_dir / "binvoxFiles" / "right_breast_recon"
    for p in [mesh_dir, viz_dir, left_bin_dir, right_bin_dir]:
        _ensure_dir(p)

    print(f"[2D] Device: {device}")
    print(f"[2D] Samples: {len(samples)}")
    gt_mask_root = Path(args.data_root) / "gt_data" / "2D" / args.split
    print(f"[2D] Alignment mask source: {'GT masks' if gt_mask_root.exists() else 'alpha/brightness fallback'}")
    # Match evaluate_2d.py / original test transform order exactly.
    test_tf = p2.DataTransforms.Compose([
        p2.DataTransforms.CenterCrop(
            (cfg.CONST.IMG_H, cfg.CONST.IMG_W),
            (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W),
        ),
        p2.DataTransforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        p2.DataTransforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        p2.DataTransforms.ToTensor(),
    ])
    manifest_rows: List[Dict] = []
    ok_count = 0
    skip_count = 0

    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            sid, side = s["id"], s["side"]
            tag = f"{sid}_{side}"
            try:
                views = []
                for v in ["front", "top", "side"]:
                    mpath = _mask_path_2d(Path(args.data_root), args.split, sid, side, v)
                    views.append(_load_image_for_2d(s["views"][v], p2, mask_path=mpath))
                images = test_tf(views).unsqueeze(0).float().to(device)  # [1,3,3,H,W]

                feat = encoder(images)
                raw, gv = decoder(feat)
                gv = merger(raw, gv) if cfg.NETWORK.USE_MERGER else torch.mean(gv, dim=1)
                if cfg.NETWORK.USE_REFINER:
                    gv = refiner(gv)

                vol_prob = gv.detach().cpu().numpy()[0]
                if vol_prob.ndim == 4:
                    vol_prob = vol_prob[0]
                clean_vol = p2.post_process_voxel(
                    vol_prob,
                    threshold=args.threshold,
                    remove_outliers=(not args.disable_remove_outliers),
                    fill_holes=(not args.disable_fill_holes),
                )

                p2.save_smooth_mesh(clean_vol, str(mesh_dir / f"{tag}.obj"), level=args.mesh_level)
                _save_projection_preview(clean_vol, viz_dir / f"{tag}_proj.png", f"{tag} projections")
                _save_mesh_preview(clean_vol, viz_dir / f"{tag}_mesh.png", f"{tag} mesh")

                bin_dir = left_bin_dir if side == "left" else right_bin_dir
                _save_binvox_bool(clean_vol, bin_dir / f"{sid}.binvox")
                ok_count += 1
                manifest_rows.append({"id": sid, "side": side, "status": "ok", "message": ""})
                print(f"[2D] {i}/{len(samples)} done: {tag}")
            except Exception as e:
                msg = str(e)
                manifest_rows.append({"id": sid, "side": side, "status": "skipped", "message": msg})
                skip_count += 1
                if args.strict:
                    raise
                print(f"[2D][WARN] skipped {tag}: {msg}")

    _write_manifest(manifest_rows, out_dir / "run_manifest.csv")
    print(f"[2D] Summary: ok={ok_count}, skipped={skip_count}")

    print(f"[2D] Finished. Output: {out_dir}")


def run_classification(args):
    classifier = args.classifier
    script = BASE_DIR / ("Classify_3d_only.py" if classifier == "only" else "Classify_3d_plus_meta.py")
    if not script.exists():
        raise FileNotFoundError(f"Missing classifier script: {script}")

    recon_dir = _resolve_input_path(args.recon_dir)
    left_src = _resolve_input_path(args.left_dir) if args.left_dir else recon_dir / "binvoxFiles" / "left_breast_recon"
    right_src = _resolve_input_path(args.right_dir) if args.right_dir else recon_dir / "binvoxFiles" / "right_breast_recon"
    if not left_src.exists():
        raise FileNotFoundError(f"Missing left binvox directory: {left_src}")
    if not right_src.exists():
        raise FileNotFoundError(f"Missing right binvox directory: {right_src}")

    work_dir = _resolve_input_path(args.work_dir)
    _ensure_dir(work_dir)
    binvox_root = work_dir / "binvoxFiles"
    _ensure_dir(binvox_root)

    _link_or_copy_dir(left_src, binvox_root / "left_breast_recon")
    _link_or_copy_dir(right_src, binvox_root / "right_breast_recon")
    _link_or_copy_dir(right_src, binvox_root / "right_breast")  # needed by Classify_3d_only.py

    if classifier == "plus_meta":
        meta_src = _resolve_input_path(args.metadata)
        if not meta_src.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_src}")
        meta_dst = work_dir / "metadata.xlsx"
        if meta_dst.exists() or meta_dst.is_symlink():
            meta_dst.unlink()
        try:
            os.symlink(meta_src, meta_dst)
        except OSError:
            shutil.copy2(meta_src, meta_dst)

    cmd = [sys.executable, str(script)]
    print(f"[CLS] Running: {' '.join(cmd)}")
    print(f"[CLS] Working directory: {work_dir}")
    subprocess.run(cmd, cwd=str(work_dir), check=True)
    print(f"[CLS] Finished. Outputs saved in: {work_dir}")


def build_parser():
    """
    Build the command-line parser for all runner actions.

    Input:
        None.

    Output:
        argparse.ArgumentParser with four sub-commands:
        recon3d_full, recon2d_full, recon3d_nipple, classify.
    """
    # Create the top-level parser shown when user runs: python runner.py --help
    parser = argparse.ArgumentParser(description="Unified runner for reconstruction and classification")
    # Create sub-command container. One command must be selected.
    sub = parser.add_subparsers(dest="command", required=True)

    # --------------------------
    # Command: recon3d_full
    # --------------------------
    # Reconstruct full breast shape with 3D pipeline.
    p_3d_full = sub.add_parser("recon3d_full", help="3D pipeline: reconstruct full single-side breast")
    # Input dataset root (single breast layout).
    p_3d_full.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_SINGLE_DATA),
        help="Input folder of full-breast test data (Single Breast layout).",
    )
    # Which split folder to read: train / valid / test.
    p_3d_full.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name to run, for example: test.",
    )
    # Checkpoint for the trained 3D full-breast model.
    p_3d_full.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT_3D_FULL),
        help="Model checkpoint file (.pth) for 3D full-breast reconstruction.",
    )
    # Output folder for obj, visualization, and binvox.
    p_3d_full.add_argument(
        "--out-dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "recon3d_full"),
        help="Output folder for reconstructed meshes and preview images.",
    )
    # Threshold for binarizing voxel prediction (matches evaluate_and_export export_thresh).
    p_3d_full.add_argument(
        "--export-thresh",
        type=float,
        default=0.5,
        help="Voxel threshold (0~1) for mesh export. Matches --export-thresh in 3D_pipeline.py.",
    )
    # Voxel grid size — must match the checkpoint's training grid_size.
    p_3d_full.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Voxel grid resolution. Must match the training checkpoint (default: 64).",
    )
    # Remove very small disconnected voxel components.
    p_3d_full.add_argument(
        "--solidify-min-component",
        type=int,
        default=64,
        help="Remove isolated tiny voxel parts smaller than this size.",
    )
    # Morphology closing steps for smoother mesh.
    p_3d_full.add_argument(
        "--solidify-closing-iters",
        type=int,
        default=1,
        help="Smoothing steps to close small gaps in voxel results.",
    )
    # GT-based local ROI mask (mirrors evaluate_and_export eval_local_roi).
    p_3d_full.add_argument(
        "--eval-local-roi",
        action="store_true",
        default=False,
        help="Apply GT-derived nipple+areola ROI mask before export (mirrors evaluate_and_export).",
    )
    p_3d_full.add_argument("--local-roi-radius-ratio", type=float, default=0.18)
    p_3d_full.add_argument("--local-roi-depth-ratio",  type=float, default=0.16)
    p_3d_full.add_argument("--local-areola-extra-ratio", type=float, default=0.07)
    p_3d_full.add_argument("--local-roi-axis", type=str, default="z", choices=["x", "y", "z"])
    p_3d_full.add_argument("--local-roi-sign", type=int, default=1, choices=[-1, 1])
    # Limit sample count for quick debug. 0 means process all.
    p_3d_full.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    # If enabled, stop immediately when one sample fails.
    p_3d_full.add_argument("--strict", action="store_true",
                           help="Stop on first bad sample. Default is skip bad samples and continue.")
    # Force CPU mode even if CUDA is available.
    p_3d_full.add_argument("--cpu", action="store_true", help="Run on CPU only.")
    # Attach runtime handler for this sub-command.
    p_3d_full.set_defaults(func=lambda a: run_recon_3d(argparse.Namespace(**vars(a), layout="single_breast")))

    # --------------------------
    # Command: recon2d_full
    # --------------------------
    # Reconstruct full breast shape with 2D pipeline.
    p_2d_full = sub.add_parser("recon2d_full", help="2D pipeline: reconstruct full single-side breast")
    # Input dataset root (single breast layout).
    p_2d_full.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_SINGLE_DATA),
        help="Input folder of full-breast test data (Single Breast layout).",
    )
    # Which split folder to read: train / valid / test.
    p_2d_full.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name to run, for example: test.",
    )
    # Checkpoint for the trained 2D full-breast model.
    p_2d_full.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT_2D_FULL),
        help="Model checkpoint file (.pth) for 2D full-breast reconstruction.",
    )
    # Output folder for obj, visualization, and binvox.
    p_2d_full.add_argument(
        "--out-dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "recon2d_full"),
        help="Output folder for reconstructed meshes and preview images.",
    )
    # Probability threshold for binarizing voxel prediction.
    p_2d_full.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Voxel threshold (0~1). Larger value keeps fewer voxels.",
    )
    p_2d_full.add_argument(
        "--mesh-level",
        type=float,
        default=0.4,
        help="Marching-cubes level for OBJ surface extraction.",
    )
    p_2d_full.add_argument(
        "--disable-remove-outliers",
        action="store_true",
        help="Disable removal of small disconnected voxel parts in post-process.",
    )
    p_2d_full.add_argument(
        "--disable-fill-holes",
        action="store_true",
        help="Disable hole filling in post-process.",
    )
    # Limit sample count for quick debug. 0 means process all.
    p_2d_full.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    # If enabled, stop immediately when one sample fails.
    p_2d_full.add_argument("--strict", action="store_true",
                           help="Stop on first bad sample. Default is skip bad samples and continue.")
    # Force CPU mode even if CUDA is available.
    p_2d_full.add_argument("--cpu", action="store_true", help="Run on CPU only.")
    # Attach runtime handler for this sub-command.
    p_2d_full.set_defaults(func=run_recon_2d)

    # --------------------------
    # Command: recon3d_nipple
    # --------------------------
    # Reconstruct nipple+areola region with 3D pipeline.
    p_3d_nipple = sub.add_parser("recon3d_nipple", help="3D pipeline: reconstruct nipple+areola region")
    # Input dataset root (nipple_only layout).
    p_3d_nipple.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_NIPPLE_DATA),
        help="Input folder of nipple-only test data.",
    )
    p_3d_nipple.add_argument(
        "--gt-root",
        type=str,
        default=str(DEFAULT_NIPPLE_CYL_GT_DATA),
        help="GT folder for nipple-only evaluation/visualisation (default: Single Breast (nipple_cyl_gt)).",
    )
    # Which split folder to read: train / valid / test.
    p_3d_nipple.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name for nipple-only data, default is test.",
    )
    # Checkpoint for trained nipple-only 3D model.
    p_3d_nipple.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT_3D_NIPPLE),
        help="Model checkpoint file (.pth) for 3D nipple+areola reconstruction.",
    )
    # Output folder for obj, visualization, and binvox.
    p_3d_nipple.add_argument(
        "--out-dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "recon3d_nipple"),
        help="Output folder for reconstructed meshes and preview images.",
    )
    # Threshold for binarizing voxel prediction (matches evaluate_and_export export_thresh).
    p_3d_nipple.add_argument(
        "--export-thresh",
        type=float,
        default=0.5,
        help="Voxel threshold (0~1) for mesh export. Matches --export-thresh in 3D_pipeline.py.",
    )
    # Voxel grid size — must match the checkpoint's training grid_size.
    p_3d_nipple.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Voxel grid resolution. Must match the training checkpoint (default: 64).",
    )
    # Remove very small disconnected voxel components.
    p_3d_nipple.add_argument(
        "--solidify-min-component",
        type=int,
        default=64,
        help="Remove isolated tiny voxel parts smaller than this size.",
    )
    # Morphology closing steps for smoother mesh.
    p_3d_nipple.add_argument(
        "--solidify-closing-iters",
        type=int,
        default=0,
        help="Smoothing steps to close small gaps in voxel results.",
    )
    # GT-based local ROI mask (mirrors evaluate_and_export eval_local_roi).
    p_3d_nipple.add_argument(
        "--eval-local-roi",
        action="store_true",
        default=True,
        help="Apply GT-derived nipple+areola ROI mask before export (mirrors evaluate_and_export).",
    )
    p_3d_nipple.add_argument("--local-roi-radius-ratio", type=float, default=0.15)
    p_3d_nipple.add_argument("--local-roi-depth-ratio",  type=float, default=0.10)
    p_3d_nipple.add_argument("--local-areola-extra-ratio", type=float, default=0.07)
    p_3d_nipple.add_argument("--local-roi-axis", type=str, default="z", choices=["x", "y", "z"])
    p_3d_nipple.add_argument("--local-roi-sign", type=int, default=1, choices=[-1, 1])
    # Limit sample count for quick debug. 0 means process all.
    p_3d_nipple.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    # If enabled, stop immediately when one sample fails.
    p_3d_nipple.add_argument("--strict", action="store_true",
                             help="Stop on first bad sample. Default is skip bad samples and continue.")
    # Force CPU mode even if CUDA is available.
    p_3d_nipple.add_argument("--cpu", action="store_true", help="Run on CPU only.")
    # Attach runtime handler for this sub-command.
    p_3d_nipple.set_defaults(func=lambda a: run_recon_3d(argparse.Namespace(**vars(a), layout="nipple_only")))

    # --------------------------
    # Command: classify
    # --------------------------
    # Run 3D cluster/classification script on reconstructed binvox folders.
    p_cls = sub.add_parser("classify", help="Classify reconstructed 3D binvox results")
    # Choose which classifier script to run.
    p_cls.add_argument("--classifier", choices=["only", "plus_meta"], required=True,
                       help="'only' -> Classify_3d_only.py, 'plus_meta' -> Classify_3d_plus_meta.py")
    # Reconstruction output root that contains binvoxFiles subfolders.
    p_cls.add_argument(
        "--recon-dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "recon3d_full"),
        help="Folder that contains reconstructed binvoxFiles. Relative paths are resolved from this project folder.",
    )
    # Optional direct override for left binvox folder.
    p_cls.add_argument(
        "--left-dir",
        type=str,
        default="",
        help="Optional custom left binvox folder. Relative paths are resolved from this project folder.",
    )
    # Optional direct override for right binvox folder.
    p_cls.add_argument(
        "--right-dir",
        type=str,
        default="",
        help="Optional custom right binvox folder. Relative paths are resolved from this project folder.",
    )
    # Metadata file used only in plus_meta mode.
    p_cls.add_argument(
        "--metadata",
        type=str,
        default=str(BASE_DIR / "metadata.xlsx"),
        help="Metadata Excel file used in plus_meta mode. Relative paths are resolved from this project folder.",
    )
    # Working folder where classifier outputs are written.
    p_cls.add_argument(
        "--work-dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "classification"),
        help="Output folder for classification results. Relative paths are resolved from this project folder.",
    )
    # Attach runtime handler for this sub-command.
    p_cls.set_defaults(func=run_classification)

    # Return configured parser to main().
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
