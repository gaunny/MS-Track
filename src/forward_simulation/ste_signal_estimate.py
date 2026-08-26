#!/usr/bin/env python
"""
STE Signal Simulation using multi-compartment model.

Simulates Spherical Tensor Encoding (STE) signals from ground truth
tractograms using a three-compartment model:
  1. Intra-fiber water (anisotropic, D_iso derived from tensor)
  2. Free water (fully isotropic)
  3. Glial water (weakly anisotropic)

Formula (Topgaard 2019, Eq. 12):
  S_STE = sum(w_k * exp(-b * D_iso_k))

Usage:
    python -m src.forward_simulation.ste_signal_estimate \
        --gt_tck <ground_truth.tck> \
        --wm_mask <wm_mask.nii.gz> \
        --output_dir <output/> \
        --bval 650
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


def parse_args():
    parser = argparse.ArgumentParser(description="STE signal simulation")
    parser.add_argument("--gt_tck", required=True, help="Ground truth .tck file")
    parser.add_argument("--wm_mask", required=True, help="White matter mask .nii.gz")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--bval", type=float, default=650.0,
                        help="STE b-value in s/mm^2 (default: 650)")
    parser.add_argument("--lambda_par", type=float, default=1.7e-3,
                        help="Parallel diffusivity in mm^2/s")
    parser.add_argument("--lambda_perp", type=float, default=0.3e-3,
                        help="Perpendicular diffusivity in mm^2/s")
    parser.add_argument("--free_water_frac", type=float, default=0.18,
                        help="Free water volume fraction (default: 0.18)")
    parser.add_argument("--glial_frac", type=float, default=0.12,
                        help="Glial water volume fraction (default: 0.12)")
    parser.add_argument("--d_free", type=float, default=3.0e-3,
                        help="Free water diffusivity in mm^2/s")
    parser.add_argument("--d_glial", type=float, default=1.2e-3,
                        help="Glial water diffusivity in mm^2/s")
    parser.add_argument("--prefix", default="ste",
                        help="Output file prefix (default: 'ste')")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = args.prefix
    bval_ste = args.bval
    n_ste = 1  # Isotropic encoding: direction-agnostic

    lambda_par = args.lambda_par
    lambda_perp = args.lambda_perp
    D_iso_fiber = (lambda_par + 2 * lambda_perp) / 3
    D_iso_free = args.d_free
    D_iso_glial = args.d_glial
    free_water_fraction = args.free_water_frac
    glial_fraction = args.glial_frac
    fiber_fraction = 1.0 - free_water_fraction - glial_fraction

    STE_FIBER_TARGET_MEAN = 0.4

    # Load WM mask and streamlines
    wm_img = nib.load(args.wm_mask)
    wm_mask = wm_img.get_fdata() > 0
    affine = wm_img.affine
    shape = wm_mask.shape
    inv_affine = np.linalg.inv(affine)
    print(f"WM mask loaded: shape={shape}, valid voxels={np.sum(wm_mask)}")

    tck = load_tck(args.gt_tck)
    streamlines = [sl for sl in tck.streamlines if len(sl) >= 2]
    print(f"Loaded {len(streamlines)} valid streamlines")

    sl_vox = transform_streamlines(streamlines, inv_affine)

    # Calculate voxel fiber fraction from streamline density
    voxel_fiber_length = np.zeros(shape, dtype=np.float32)
    print("Calculating voxel fiber length...")
    for sl_idx in tqdm(range(len(streamlines)), desc="Processing"):
        sl = sl_vox[sl_idx]
        sl_np = np.array(sl, dtype=np.float32)
        seg_lengths = np.sqrt(np.sum(np.diff(sl_np, axis=0) ** 2, axis=1))
        sl_coords = np.round(sl_np).astype(int)
        valid_coords = []
        for coord in sl_coords:
            x, y, z = coord
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                valid_coords.append(coord)
        valid_coords = np.array(valid_coords)
        for i in range(len(valid_coords) - 1):
            x1, y1, z1 = valid_coords[i]
            x2, y2, z2 = valid_coords[i + 1]
            length = seg_lengths[i]
            voxel_fiber_length[x1, y1, z1] += length / 2
            voxel_fiber_length[x2, y2, z2] += length / 2

    # Normalize to fiber fraction [0, 1]
    fiber_length_max = np.max(voxel_fiber_length)
    if fiber_length_max > 0:
        voxel_fiber_fraction = voxel_fiber_length / fiber_length_max
    else:
        voxel_fiber_fraction = voxel_fiber_length
    print(f"Voxel fiber fraction: range [{voxel_fiber_fraction.min():.6f}, "
          f"{voxel_fiber_fraction.max():.6f}]")

    # Compute STE signal per voxel
    ste_data = np.zeros(shape + (n_ste,), dtype=np.float32)
    print("Computing STE signal...")
    pbar = tqdm(total=shape[2], desc="STE Signal")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if not wm_mask[x, y, z]:
                    continue
                ff = voxel_fiber_fraction[x, y, z]
                w_fiber = fiber_fraction * ff
                w_free = free_water_fraction
                w_glial = glial_fraction
                total_w = w_fiber + w_free + w_glial
                w_fiber /= total_w
                w_free /= total_w
                w_glial /= total_w

                s_fiber = np.exp(-bval_ste * D_iso_fiber)
                s_free = np.exp(-bval_ste * D_iso_free)
                s_glial = np.exp(-bval_ste * D_iso_glial)
                ste_signal = w_fiber * s_fiber + w_free * s_free + w_glial * s_glial
                ste_data[x, y, z, :] = ste_signal
        pbar.update(1)
    pbar.close()

    # Physical scale calibration
    fiber_mask = voxel_fiber_fraction > 0.5
    if np.sum(fiber_mask) > 0:
        raw_ste_mean = np.mean(ste_data[fiber_mask])
        scale_factor = STE_FIBER_TARGET_MEAN / raw_ste_mean
        ste_data = ste_data * scale_factor
        print(f"STE scale calibration: factor={scale_factor:.2f}, "
              f"target mean={STE_FIBER_TARGET_MEAN}")
    else:
        print("Warning: No fiber regions detected, skipping calibration")

    ste_data = np.clip(ste_data, 0.001, 0.999)
    print(f"STE signal range: [{ste_data.min():.6f}, {ste_data.max():.6f}]")

    # Save results
    nib.save(nib.Nifti1Image(ste_data, affine),
             os.path.join(args.output_dir, f"{prefix}_signal.nii.gz"))
    ste_scaled = (ste_data * 255).astype(np.uint8)
    nib.save(nib.Nifti1Image(ste_scaled, affine),
             os.path.join(args.output_dir, f"{prefix}_signal_scaled.nii.gz"))
    nib.save(nib.Nifti1Image(voxel_fiber_fraction, affine),
             os.path.join(args.output_dir, f"{prefix}_fiber_fraction.nii.gz"))

    # Build XPS metadata
    xps = {
        "n": np.array([[n_ste]]),
        "b": np.full((n_ste, 1), bval_ste * 1e6),
        "b_delta": np.zeros((n_ste, 1)),
        "u": np.zeros((n_ste, 3)),
        "bt": np.zeros((n_ste, 6)),
        "s_ind": np.ones((n_ste, 1)),
    }
    for i in range(n_ste):
        b_si = bval_ste * 1e6
        xps["bt"][i] = [b_si / 3, b_si / 3, b_si / 3, 0, 0, 0]
    savemat(os.path.join(args.output_dir, f"{prefix}_xps.mat"), {"xps": xps})

    print(f"\nSTE simulation complete! Generated files:")
    print(f"  1. {args.output_dir}/{prefix}_signal.nii.gz")
    print(f"  2. {args.output_dir}/{prefix}_signal_scaled.nii.gz")
    print(f"  3. {args.output_dir}/{prefix}_fiber_fraction.nii.gz")
    print(f"  4. {args.output_dir}/{prefix}_xps.mat")


if __name__ == "__main__":
    main()
