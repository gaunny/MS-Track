#!/usr/bin/env python
"""
Compute Fractional Anisotropy (FA) from LTE signal.

Fits a diffusion tensor model to the LTE portion of a merged
LTE+STE dataset and computes the fractional anisotropy map.
This is equivalent to FSL dtifit (v6.0.5) using DIPY's TensorModel,
which implements the same linear least-squares tensor estimation.

Usage:
    python -m src.preprocessing.compute_fa \
        --data <data.nii.gz> \
        --output <fa.nii.gz> \
        --n_lte 64 --bval 650
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fibonacci_sphere(samples=64):
    """Generate uniformly distributed directions on a sphere."""
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
    parser = argparse.ArgumentParser(description="Compute FA from LTE signal")
    parser.add_argument("--data", required=True, help="Merged LTE+STE data .nii.gz")
    parser.add_argument("--output", required=True, help="Output FA .nii.gz path")
    parser.add_argument("--n_lte", type=int, default=64,
                        help="Number of LTE directions (default: 64)")
    parser.add_argument("--bval", type=float, default=650.0,
                        help="LTE b-value in s/mm^2 (default: 650)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load merged data
    data_img = nib.load(args.data)
    data = data_img.get_fdata()
    affine = data_img.affine

    n_lte = args.n_lte
    lte_signal = data[..., :n_lte]
    print(f"Loaded data: shape={data.shape}, LTE channels={n_lte}")

    # Build gradient table
    bval_lte = args.bval
    lte_dirs = fibonacci_sphere(n_lte)
    bvals = np.full(n_lte, bval_lte)
    gtab = gradient_table(bvals, lte_dirs)

    # Fit DTI model
    print("Fitting diffusion tensor model...")
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(lte_signal)

    # Compute FA
    fa = fractional_anisotropy(ten_fit.evals)
    fa = np.clip(fa, 0, 1)

    # Save FA map
    nib.save(nib.Nifti1Image(fa, affine), args.output)
    print(f"FA saved: {args.output}")


if __name__ == "__main__":
    main()
