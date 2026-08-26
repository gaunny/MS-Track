#!/usr/bin/env python
"""
Merge LTE and STE signals into a single dataset.

Loads LTE and STE signal NIfTI files along with their XPS metadata
(.mat files), checks scale consistency, concatenates signals along
the last axis, and merges gradient encoding metadata.

Usage:
    python -m src.forward_simulation.merge_lte_ste \
        --lte_dir <lte_output/> \
        --ste_dir <ste_output/> \
        --output_dir <merged_output/> \
        --lte_signal_name lte_signal.nii.gz \
        --ste_signal_name ste_signal.nii.gz
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LTE+STE signals")
    parser.add_argument("--lte_dir", required=True, help="LTE output directory")
    parser.add_argument("--ste_dir", required=True, help="STE output directory")
    parser.add_argument("--output_dir", required=True, help="Merged output directory")
    parser.add_argument("--lte_signal_name", default="lte_signal.nii.gz",
                        help="LTE signal filename")
    parser.add_argument("--ste_signal_name", default="ste_signal.nii.gz",
                        help="STE signal filename")
    parser.add_argument("--lte_xps_name", default="lte_xps.mat",
                        help="LTE XPS filename")
    parser.add_argument("--ste_xps_name", default="ste_xps.mat",
                        help="STE XPS filename")
    return parser.parse_args()


def extract_xps_fields(xps_struct):
    """Extract fields from MATLAB XPS structure."""
    fields = {}
    fields["n"] = xps_struct["n"][0, 0].astype(np.int32)
    fields["b"] = xps_struct["b"][0, 0].astype(np.float64)
    fields["b_delta"] = xps_struct["b_delta"][0, 0].astype(np.float64)
    fields["b_eta"] = xps_struct["b_eta"][0, 0].astype(np.float64) \
        if "b_eta" in xps_struct.dtype.names else \
        np.zeros_like(fields["b_delta"])
    fields["u"] = xps_struct["u"][0, 0].astype(np.float64)
    fields["bt"] = xps_struct["bt"][0, 0].astype(np.float64)
    fields["s_ind"] = xps_struct["s_ind"][0, 0].astype(np.int32)
    return fields


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load LTE and STE signals
    print("Loading LTE/STE signals...")
    lte_img = nib.load(os.path.join(args.lte_dir, args.lte_signal_name))
    lte_data = lte_img.get_fdata()
    affine = lte_img.affine
    lte_data = np.clip(lte_data, 0.001, 0.999)

    ste_img = nib.load(os.path.join(args.ste_dir, args.ste_signal_name))
    ste_data = ste_img.get_fdata()
    ste_data = np.clip(ste_data, 0.001, 0.999)

    # Load XPS metadata
    lte_xps = loadmat(os.path.join(args.lte_dir, args.lte_xps_name))["xps"]
    ste_xps = loadmat(os.path.join(args.ste_dir, args.ste_xps_name))["xps"]
    n_lte = lte_xps["n"][0, 0][0, 0]
    n_ste = ste_xps["n"][0, 0][0, 0]

    # Scale consistency check
    print("Checking scale consistency...")
    fiber_mask = (lte_data[..., 0] > 0.1) & (ste_data[..., 0] > 0.1)
    if np.sum(fiber_mask) > 0:
        lte_fiber_mean = lte_data[fiber_mask].mean()
        ste_fiber_mean = ste_data[fiber_mask].mean()
        scale_ratio = lte_fiber_mean / ste_fiber_mean
        print(f"  LTE fiber mean: {lte_fiber_mean:.3f}")
        print(f"  STE fiber mean: {ste_fiber_mean:.3f}")
        print(f"  Scale ratio: {scale_ratio:.2f} (normal range: 1.0-1.5)")

        if scale_ratio > 1.5 or scale_ratio < 0.7:
            ste_data = ste_data * scale_ratio
            print(f"  STE rescaled: new mean={ste_data[fiber_mask].mean():.3f}")

    # Spatial dimension validation
    assert lte_data.shape[:3] == ste_data.shape[:3], "Spatial dimensions mismatch!"
    assert lte_data.shape[-1] == n_lte, "LTE direction count mismatch!"
    assert ste_data.shape[-1] == n_ste, "STE direction count mismatch!"

    # Merge signals
    print("Merging LTE+STE signals...")
    merged_data = np.concatenate([lte_data, ste_data], axis=-1)
    merged_data = np.clip(merged_data, 0.0, 1.0)

    nib.save(nib.Nifti1Image(merged_data, affine),
             os.path.join(args.output_dir, "data.nii.gz"))
    print(f"Merged signal saved: shape={merged_data.shape}")

    # Merge XPS metadata
    print("Merging XPS metadata...")
    lte_fields = extract_xps_fields(lte_xps)
    ste_fields = extract_xps_fields(ste_xps)

    merged_fields = {
        "n": np.array([[n_lte + n_ste]], dtype=np.int32),
        "b": np.vstack([lte_fields["b"], ste_fields["b"]]),
        "b_delta": np.vstack([lte_fields["b_delta"], ste_fields["b_delta"]]),
        "b_eta": np.vstack([lte_fields["b_eta"], ste_fields["b_eta"]]),
        "u": np.vstack([lte_fields["u"], ste_fields["u"]]),
        "bt": np.vstack([lte_fields["bt"], ste_fields["bt"]]),
        "s_ind": np.vstack([lte_fields["s_ind"], ste_fields["s_ind"]]),
    }
    savemat(os.path.join(args.output_dir, "data_xps.mat"),
            {"xps": merged_fields}, do_compression=True)

    print("\nLTE+STE merge complete!")
    print(f"  data.nii.gz: {merged_data.shape}")
    print(f"  data_xps.mat: {n_lte + n_ste} volumes")


if __name__ == "__main__":
    main()
