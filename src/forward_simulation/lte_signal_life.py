#!/usr/bin/env python
"""
LTE Signal Simulation using LiFE (Linear Fascicle Evaluation) model.

Simulates Linear Tensor Encoding (LTE) signals from ground truth
tractograms using the LiFE model from DIPY. Each streamline contributes
a signal weighted by segment length, and the signals are accumulated
per voxel.

Usage:
    python -m src.forward_simulation.lte_signal_life \
        --gt_tck <ground_truth.tck> \
        --wm_mask <wm_mask.nii.gz> \
        --grad_file <gradient_directions.txt> \
        --output_dir <output/> \
        --bval 650 --n_dir 64
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_tck
from dipy.core.gradients import gradient_table
from dipy.tracking.life import streamline_signal, voxel2streamline
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.utils import unique_rows
from scipy.io import savemat
from tqdm import tqdm
from numba import jit, prange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="LTE signal simulation with LiFE")
    parser.add_argument("--gt_tck", required=True, help="Ground truth .tck file")
    parser.add_argument("--wm_mask", required=True, help="White matter mask .nii.gz")
    parser.add_argument("--grad_file", required=True, help="Gradient direction file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--bval", type=float, default=650.0,
                        help="b-value in s/mm^2 (default: 650)")
    parser.add_argument("--n_dir", type=int, default=64,
                        help="Number of LTE directions (default: 64)")
    parser.add_argument("--lambda_par", type=float, default=1.7e-3,
                        help="Parallel diffusivity in mm^2/s")
    parser.add_argument("--lambda_perp", type=float, default=0.3e-3,
                        help="Perpendicular diffusivity in mm^2/s")
    parser.add_argument("--prefix", default="lte",
                        help="Output file prefix (default: 'lte')")
    return parser.parse_args()


@jit(nopython=True, parallel=True)
def calculate_segment_lengths_numba(sl_np):
    """Calculate segment lengths for a streamline (Numba-accelerated)."""
    n_nodes = sl_np.shape[0]
    segment_lengths = np.zeros(n_nodes, dtype=np.float32)
    for i in prange(1, n_nodes):
        dx = sl_np[i, 0] - sl_np[i - 1, 0]
        dy = sl_np[i, 1] - sl_np[i - 1, 1]
        dz = sl_np[i, 2] - sl_np[i - 1, 2]
        segment_lengths[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
    if n_nodes >= 2:
        segment_lengths[-1] = segment_lengths[-2]
    return segment_lengths


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = args.prefix
    bval_lte = args.bval
    n_dir = args.n_dir
    evals = (args.lambda_par, args.lambda_perp, args.lambda_perp)

    # Load WM mask
    wm_img = nib.load(args.wm_mask)
    wm_mask = wm_img.get_fdata() > 0
    affine = wm_img.affine
    shape = wm_mask.shape
    inv_affine = np.linalg.inv(affine)
    print(f"WM mask loaded: shape={shape}, valid voxels={np.sum(wm_mask)}")

    # Load gradient directions
    directions = np.loadtxt(args.grad_file)
    if directions.shape[0] == n_dir + 1:
        directions = directions[1:, :]  # Remove b0 row
    directions = np.asarray(directions, dtype=np.float64)
    assert directions.shape == (n_dir, 3), \
        f"Expected {n_dir}x3 directions, got {directions.shape}"
    dirs = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    print(f"Gradient directions: {n_dir} directions loaded")

    # Build gradient table
    bvals = np.full(n_dir, bval_lte, dtype=np.float64)
    gtab_lte = gradient_table(bvals=bvals, bvecs=dirs, b0_threshold=0.0)

    # Load ground truth streamlines
    tck = load_tck(args.gt_tck)
    streamlines = [sl for sl in tck.streamlines if len(sl) >= 2]
    print(f"Loaded {len(streamlines)} valid streamlines")

    # Transform to voxel space
    sl_vox = transform_streamlines(streamlines, inv_affine)
    all_coords = np.concatenate(sl_vox)
    vox_coords = unique_rows(np.round(all_coords).astype(int))
    vox_coords = [c for c in vox_coords
                 if 0 <= c[0] < shape[0] and
                 0 <= c[1] < shape[1] and
                 0 <= c[2] < shape[2]]
    vox_coords = np.array(vox_coords, dtype=int)

    # Build voxel-to-streamline mapping
    print("Building voxel-streamline mapping...")
    try:
        v2f, v2fn = voxel2streamline(
            streamlines=sl_vox, affine=np.eye(4), unique_idx=vox_coords)
    except TypeError:
        v2f, v2fn = voxel2streamline(
            streamline=sl_vox, affine=np.eye(4), unique_idx=vox_coords)

    # Precompute LTE signals per streamline
    print("Precomputing LTE signals per streamline...")
    sl_signals = []
    sl_seg_lengths = []
    for sl_idx in tqdm(range(len(streamlines)), desc="Precompute"):
        sl = streamlines[sl_idx]
        sl_np = np.array(sl, dtype=np.float32)
        seg_len = calculate_segment_lengths_numba(sl_np)
        sl_seg_lengths.append(seg_len)
        sl_sig = streamline_signal(sl, gtab_lte, evals=evals)
        if sl_sig.shape == (len(sl), n_dir):
            sl_signals.append(sl_sig)
        else:
            sl_signals.append(np.ones((len(sl), n_dir)) * 0.5)

    # Project streamlines to voxels
    lte_data = np.zeros(shape + (n_dir,), dtype=np.float32)
    voxel_hist = np.zeros(shape + (n_dir,), dtype=np.float32)

    valid_vox_indices = []
    valid_vox_xyz = []
    for idx in range(len(vox_coords)):
        x, y, z = vox_coords[idx]
        if wm_mask[x, y, z]:
            valid_vox_indices.append(idx)
            valid_vox_xyz.append((x, y, z))

    print(f"Projecting to {len(valid_vox_indices)} valid voxels...")
    for vox_idx, (x, y, z) in tqdm(zip(valid_vox_indices, valid_vox_xyz),
                                    total=len(valid_vox_indices),
                                    desc="Projecting"):
        if vox_idx not in v2f:
            continue
        sl_indices = v2f[vox_idx]
        for sl_idx in sl_indices:
            sl_sig = sl_signals[sl_idx]
            segment_lengths = sl_seg_lengths[sl_idx]
            node_indices = v2fn[sl_idx][vox_idx]
            if len(node_indices) == 0:
                continue
            node_indices_np = np.array(node_indices, dtype=int)
            valid_mask = (node_indices_np < sl_sig.shape[0]) & \
                         (node_indices_np < len(segment_lengths))
            valid_nodes = node_indices_np[valid_mask]
            if len(valid_nodes) == 0:
                continue
            weights = segment_lengths[valid_nodes]
            sig_nodes = sl_sig[valid_nodes]
            voxel_hist[x, y, z] += np.sum(weights[:, np.newaxis] * sig_nodes, axis=0)

    np.save(os.path.join(args.output_dir, "voxel_hist.npy"), voxel_hist)
    lte_data = voxel_hist

    # Physical scale calibration: target fiber region mean ~0.5
    fiber_mask = lte_data[..., 0] > 1e-6
    if np.sum(fiber_mask) > 0:
        fiber_mean = lte_data[fiber_mask].mean()
        scale_factor = 0.5 / fiber_mean
        lte_data = lte_data * scale_factor
    else:
        scale_factor = 1.0

    lte_data = np.clip(lte_data, 0.001, 0.999)

    print(f"LTE signal generation complete:")
    print(f"  Scale factor: {scale_factor:.2f}")
    print(f"  Signal range: [{lte_data.min():.6f}, {lte_data.max():.6f}]")

    # Save results
    nib.save(nib.Nifti1Image(lte_data, affine),
             os.path.join(args.output_dir, f"{prefix}_signal.nii.gz"))
    lte_data_scaled = (lte_data * 255).astype(np.uint8)
    nib.save(nib.Nifti1Image(lte_data_scaled, affine),
             os.path.join(args.output_dir, f"{prefix}_signal_scaled.nii.gz"))

    # Build XPS metadata
    xps = {
        "n": np.array([[n_dir]]),
        "b": np.zeros((n_dir, 1)),
        "b_delta": np.ones((n_dir, 1)),
        "u": dirs,
        "bt": np.zeros((n_dir, 6)),
        "s_ind": np.ones((n_dir, 1)),
    }
    for i in range(n_dir):
        b_si = bval_lte * 1e6
        xps["b"][i, 0] = b_si
        u_dir = dirs[i] / np.linalg.norm(dirs[i])
        B = b_si * np.outer(u_dir, u_dir)
        xps["bt"][i] = [B[0, 0], B[1, 1], B[2, 2], B[0, 1], B[0, 2], B[1, 2]]

    savemat(os.path.join(args.output_dir, f"{prefix}_xps.mat"), {"xps": xps})

    print(f"\nLTE simulation complete! Generated files:")
    print(f"  1. {args.output_dir}/{prefix}_signal.nii.gz")
    print(f"  2. {args.output_dir}/{prefix}_signal_scaled.nii.gz")
    print(f"  3. {args.output_dir}/{prefix}_xps.mat")


if __name__ == "__main__":
    main()
