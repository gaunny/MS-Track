#!/usr/bin/env python
"""
DTD Forward Simulation: Generate LTE+STE signals from ground truth tractograms.

Simulates voxel-level diffusion tensors from streamline segments, then
generates LTE (Linear Tensor Encoding) and STE (Spherical Tensor Encoding)
signals using the Stejskal-Tanner equation.

Usage:
    python -m src.forward_simulation.dtd_forward \
        --gt_tck <ground_truth.tck> \
        --wm_mask <wm_mask.nii.gz> \
        --output_dir <output/> \
        --bval_lte 1000 --bval_ste 1000 --n_lte 32 --n_ste 32
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_tck
from dipy.tracking.streamline import transform_streamlines
from scipy.io import savemat
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fibonacci_sphere(samples=64):
    """Generate uniformly distributed directions on a sphere using Fibonacci spiral."""
    phi = np.pi * (3. - np.sqrt(5.))
    y = 1 - (np.arange(samples) / float(samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * np.arange(samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(description="DTD forward simulation")
    parser.add_argument("--gt_tck", required=True, help="Ground truth .tck file")
    parser.add_argument("--wm_mask", required=True, help="White matter mask .nii.gz")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--bval_lte", type=float, default=1000.0,
                        help="LTE b-value in s/mm^2 (default: 1000)")
    parser.add_argument("--bval_ste", type=float, default=1000.0,
                        help="STE b-value in s/mm^2 (default: 1000)")
    parser.add_argument("--n_lte", type=int, default=32, help="Number of LTE directions")
    parser.add_argument("--n_ste", type=int, default=32, help="Number of STE directions")
    parser.add_argument("--lambda_par", type=float, default=1.7e-3,
                        help="Parallel diffusivity in mm^2/s (default: 1.7e-3)")
    parser.add_argument("--lambda_perp", type=float, default=0.3e-3,
                        help="Perpendicular diffusivity in mm^2/s (default: 0.3e-3)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bval_lte = args.bval_lte
    bval_ste = args.bval_ste
    n_lte = args.n_lte
    n_ste = args.n_ste
    lambda_par = args.lambda_par
    lambda_perp = args.lambda_perp

    # Generate LTE directions using Fibonacci sphere
    lte_dirs = fibonacci_sphere(n_lte)

    # Load WM mask and streamlines
    wm_img = nib.load(args.wm_mask)
    wm_mask = wm_img.get_fdata() > 0
    affine = wm_img.affine
    shape = wm_mask.shape
    inv_affine = np.linalg.inv(affine)
    print(f"WM mask shape: {shape}, voxels: {np.sum(wm_mask)}")

    tck = load_tck(args.gt_tck)
    streamlines = [sl for sl in tck.streamlines if len(sl) >= 2]
    sl_vox = transform_streamlines(streamlines, inv_affine)
    print(f"Loaded {len(streamlines)} streamlines")

    # Initialize voxel signal and tensor accumulation
    data = np.zeros(shape + (n_lte + n_ste,), dtype=np.float32)
    voxel_tensors = np.zeros(shape + (3, 3), dtype=np.float32)
    voxel_counts = np.zeros(shape, dtype=np.float32)

    # Accumulate voxel-level diffusion tensors from streamline segments
    print("Accumulating voxel-level tensors...")
    for sl in tqdm(sl_vox):
        sl_np = np.array(sl)
        seg_vecs = np.diff(sl_np, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)[:, None]
        seg_dirs = seg_vecs / seg_lengths
        voxel_coords = np.round(sl_np[:-1]).astype(int)

        for i, v in enumerate(seg_dirs):
            x, y, z = voxel_coords[i]
            if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                continue
            if not wm_mask[x, y, z]:
                continue
            D = lambda_par * np.outer(v, v) + lambda_perp * (np.eye(3) - np.outer(v, v))
            voxel_tensors[x, y, z] += D
            voxel_counts[x, y, z] += 1

    # Average tensors
    mask_voxels = voxel_counts > 0
    voxel_tensors[mask_voxels] /= voxel_counts[mask_voxels][..., None, None]

    # Generate LTE and STE signals
    print("Computing voxel signals LTE + STE...")
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if not mask_voxels[x, y, z]:
                    continue
                D_vox = voxel_tensors[x, y, z]
                # LTE: exp(-b * g^T D g)
                for i, g in enumerate(lte_dirs):
                    data[x, y, z, i] = np.exp(-bval_lte * g @ D_vox @ g)
                # STE: exp(-b * trace(D)/3)
                for i in range(n_ste):
                    data[x, y, z, n_lte + i] = np.exp(-bval_ste * np.trace(D_vox) / 3)

    # Clip to physical range
    data = np.clip(data, 0.001, 0.999)

    # Save NIfTI
    nib.save(nib.Nifti1Image(data, affine),
             os.path.join(args.output_dir, "data.nii.gz"))

    # Build XPS (gradient encoding metadata)
    bt = np.zeros((n_lte + n_ste, 6))
    bvals = np.zeros((n_lte + n_ste, 1))
    b_delta = np.zeros((n_lte + n_ste, 1))
    u = np.zeros((n_lte + n_ste, 3))
    s_ind = np.ones((n_lte + n_ste, 1))

    for i in range(n_lte):
        g = lte_dirs[i]
        bvals[i] = bval_lte * 1e6  # Convert to SI (s/m^2)
        b_delta[i] = 1.0
        u[i] = g
        B = bvals[i, 0] * np.outer(g, g)
        bt[i] = [B[0, 0], B[1, 1], B[2, 2], B[0, 1], B[0, 2], B[1, 2]]

    bvals[n_lte:] = bval_ste * 1e6
    b_delta[n_lte:] = 0.0
    u[n_lte:] = 0
    bt[n_lte:] = np.zeros((n_ste, 6))

    xps = {
        "n": np.array([[n_lte + n_ste]]),
        "b": bvals,
        "b_delta": b_delta,
        "b_eta": np.zeros_like(b_delta),
        "u": u,
        "bt": bt,
        "s_ind": s_ind,
    }
    savemat(os.path.join(args.output_dir, "data_xps.mat"), {"xps": xps})

    print("Forward simulation done!")
    print(f"  NIfTI: {os.path.join(args.output_dir, 'data.nii.gz')}")
    print(f"  XPS: {os.path.join(args.output_dir, 'data_xps.mat')}")


if __name__ == "__main__":
    main()
