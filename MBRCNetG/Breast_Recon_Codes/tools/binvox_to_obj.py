#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
binvox_to_obj.py
================
Convert a single .binvox file into a mesh .obj using the same marching-cubes
export function as Breast_Recon_Codes/3D_pipeline.py.

Example:
  python3 Breast_Recon_Codes/tools/binvox_to_obj.py \
    --binvox "Breast_Recon_Codes/Dataset/Single Breast (nipple_cyl_gt)/gt_data/3D/test/left/19.binvox" \
    --out "Breast_Recon_Codes/visualizations/19_left_nipple_cyl.obj"
"""

from __future__ import annotations

import os
import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binvox", type=str, required=True, help="Path to .binvox")
    ap.add_argument("--out", type=str, default="", help="Output .obj path (default: alongside binvox)")
    ap.add_argument("--level", type=float, default=0.5, help="Marching-cubes level")
    return ap.parse_args()


def main():
    # Work around OpenMP shared-memory issues in some macOS sandbox environments.
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    binvox_path = Path(args.binvox)
    if not binvox_path.exists():
        raise FileNotFoundError(binvox_path)

    out_path = Path(args.out) if args.out else binvox_path.with_suffix(".obj")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid importing torch to prevent OpenMP SHM issues; use numpy + skimage directly.
    import io
    from skimage import measure

    data = binvox_path.read_bytes()
    stream = io.BytesIO(data)
    line = stream.readline().strip()
    if not line.startswith(b"#binvox"):
        raise ValueError(f"Not a binvox file: {binvox_path}")

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
        raise ValueError(f"Malformed binvox header: {binvox_path}")

    raw = stream.read()
    total = depth * height * width
    values = np.empty(total, dtype=np.uint8)
    out_i = 0
    i = 0
    while i + 1 < len(raw) and out_i < total:
        value = raw[i]
        count = raw[i + 1]
        i += 2
        end = min(out_i + count, total)
        values[out_i:end] = value
        out_i = end
    if out_i < total:
        raise ValueError(f"RLE decode shorter than expected in {binvox_path}")

    vox = values.reshape((depth, height, width)).astype(np.float32)
    if vox.max() <= 0:
        print(f"[WARN] Empty voxel, skip mesh export: {binvox_path}")
        return

    verts, faces, normals, _ = measure.marching_cubes(vox, level=float(args.level))

    with out_path.open("w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        for face in faces:
            f.write(
                f"f {face[0]+1}//{face[0]+1} "
                f"{face[1]+1}//{face[1]+1} "
                f"{face[2]+1}//{face[2]+1}\n"
            )

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
