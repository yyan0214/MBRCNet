#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_nipple_cylinder_gt.py
===========================
Export a "nipple+areola" surrogate GT by masking full-breast GT with a
GT-derived *cylindrical* ROI (same cylinder logic as 3D_pipeline.py).

This script is intended for ablation training where:
  - Inputs are nipple-only 2D images (Dataset/Nipple_only)
  - Targets are full-breast GT cropped to a nipple-cylinder region

Important:
  - The cylinder center is estimated from GT occupancy projection
    (this is fine for creating a surrogate GT, but document it).
  - The exported GT stays on the same grid and uses the same binvox format.

Output layout (single_breast GT layout):
  <out_root>/gt_data/3D/<split>/{left,right}/{id}.binvox

Example:
  python3 Breast_Recon_Codes/tools/export_nipple_cylinder_gt.py \
    --single-gt-root "Breast_Recon_Codes/Dataset/Single Breast" \
    --splits train test \
    --out-root "Breast_Recon_Codes/Dataset/Single Breast (nipple_cyl_gt)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_module_from_path(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_gt_binvox(single_gt_root: Path, splits: Iterable[str]):
    for split in splits:
        for side in ("left", "right"):
            side_dir = single_gt_root / "gt_data" / "3D" / split / side
            if not side_dir.exists():
                continue
            for p in sorted(side_dir.glob("*.binvox")):
                yield split, side, p


def _save_binvox_bool(vox: np.ndarray, out_path: Path) -> None:
    """
    Save a boolean (D,H,W) volume as .binvox with simple RLE encoding.
    Header is compatible with the loader in 3D_pipeline.py.
    """
    vox_u8 = (vox > 0).astype(np.uint8)
    d, h, w = vox_u8.shape
    flat = vox_u8.reshape(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--single-gt-root",
        type=str,
        default="Breast_Recon_Codes/Dataset/Single Breast",
        help="Root that contains gt_data/3D/<split>/<side>/*.binvox (Single Breast).",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Which split folders under gt_data/3D to export.",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root. Will create gt_data/3D/<split>/<side>/*.binvox under it.",
    )

    # Cylinder parameters (match 3D_pipeline defaults)
    ap.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"])
    ap.add_argument("--sign", type=int, default=1, choices=[-1, 1])
    ap.add_argument("--radius-ratio", type=float, default=0.18)
    ap.add_argument("--depth-ratio", type=float, default=0.16)
    ap.add_argument("--areola-extra-ratio", type=float, default=0.07)
    return ap.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]  # Breast_Recon_Codes/
    p3 = _load_module_from_path("pipeline3d_mod", base_dir / "3D_pipeline.py")

    single_gt_root = Path(args.single_gt_root)
    out_root = Path(args.out_root)

    n_in = 0
    n_out = 0
    for split, side, gt_path in _iter_gt_binvox(single_gt_root, args.splits):
        n_in += 1
        sid = gt_path.stem

        gt_t = p3.load_binvox(str(gt_path), device=None)
        gt_bin = gt_t.cpu().numpy().astype(bool)

        cyl = p3.build_nipple_areola_cylinder_mask(
            gt_bin,
            axis=args.axis,
            sign=args.sign,
            radius_ratio=float(args.radius_ratio),
            depth_ratio=float(args.depth_ratio),
            areola_extra_ratio=float(args.areola_extra_ratio),
        )
        gt_cyl = (gt_bin & cyl).astype(bool)

        out_path = out_root / "gt_data" / "3D" / split / side / f"{sid}.binvox"
        _save_binvox_bool(gt_cyl, out_path)
        n_out += 1

    print(f"[ExportGT] in={n_in}  out={n_out}")
    print(f"[ExportGT] out_root={out_root}")


if __name__ == "__main__":
    main()

