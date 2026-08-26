#!/usr/bin/env python
"""
Count spurious fibers in a tractogram.

Computes spurious fiber ratio using two criteria:
  1. Intersect criterion: streamline intersects any reference bundle mask
  2. Endpoint criterion (stricter): both endpoints fall within the
     reference bundle union mask

Usage:
    python -m src.evaluation.spurious_fibers \
        --tck <tractogram.tck> \
        --mask_dir <reference_bundle_masks/> \
        [--ref <reference.nii.gz>] \
        [--dilation 1] \
        --out <output.json>
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Count spurious fibers")
    parser.add_argument("--tck", required=True, help="Input tractogram .tck")
    parser.add_argument("--mask_dir", required=True,
                        help="Reference bundle mask directory (*.nii.gz)")
    parser.add_argument("--ref", default=None,
                        help="Reference image (default: first mask in mask_dir)")
    parser.add_argument("--dilation", type=int, default=1,
                        help="Mask dilation iterations for endpoint criterion")
    parser.add_argument("--out", default="spurious_fiber_stats.json",
                        help="Output JSON file")
    return parser.parse_args()


def is_bundle_mask(fname):
    """Check if filename is a valid bundle mask (excludes TOM_, peaks)."""
    base = os.path.basename(fname)
    if base.startswith("TOM_") or base.startswith("peaks"):
        return False
    return True


def merge_bundle_masks(mask_dir, ref_img):
    """Merge all bundle masks into one boolean reference mask."""
    shape = ref_img.shape
    merged = np.zeros(shape, dtype=bool)
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))
    mask_files = [f for f in mask_files if is_bundle_mask(f)]
    if not mask_files:
        raise FileNotFoundError(f"No bundle masks found in {mask_dir}")
    for mf in mask_files:
        m = nib.load(mf).get_fdata().astype(bool)
        if m.shape != shape:
            raise ValueError(f"Mask {mf} shape {m.shape} != reference {shape}")
        merged |= m
    return merged, mask_files


def to_vox(streamline, inv_aff):
    """Convert streamline from world to voxel coordinates."""
    sl = np.asarray(streamline)
    homog = np.hstack([sl, np.ones((sl.shape[0], 1))])
    vox = (inv_aff @ homog.T).T[:, :3]
    return np.round(vox).astype(np.int64)


def streamlines_intersect_mask(streamlines, mask, inv_aff):
    """Count streamlines that have at least one voxel in the mask."""
    matched = 0
    shape = mask.shape
    for sl in streamlines:
        sl = np.asarray(sl)
        if sl.shape[0] < 2:
            continue
        vox = to_vox(sl, inv_aff)
        inside = ((vox[:, 0] >= 0) & (vox[:, 0] < shape[0]) &
                  (vox[:, 1] >= 0) & (vox[:, 1] < shape[1]) &
                  (vox[:, 2] >= 0) & (vox[:, 2] < shape[2]))
        if not inside.any():
            continue
        if mask[vox[inside, 0], vox[inside, 1], vox[inside, 2]].any():
            matched += 1
    return matched


def streamlines_endpoint_valid(streamlines, mask, inv_aff, dilation_iters=1):
    """Count streamlines whose both endpoints fall within the mask."""
    from scipy.ndimage import binary_dilation
    m = binary_dilation(mask, iterations=dilation_iters) \
        if dilation_iters > 0 else mask
    valid = 0
    shape = m.shape

    def in_mask(v):
        x, y, z = v
        return (0 <= x < shape[0] and 0 <= y < shape[1] and
                0 <= z < shape[2] and m[x, y, z])

    for sl in streamlines:
        sl = np.asarray(sl)
        if sl.shape[0] < 2:
            continue
        vox = to_vox(sl, inv_aff)
        if in_mask(vox[0]) and in_mask(vox[-1]):
            valid += 1
    return valid


def main():
    args = parse_args()

    mask_files_sorted = sorted(glob.glob(os.path.join(args.mask_dir, "*.nii*")))
    mask_files_sorted = [f for f in mask_files_sorted if is_bundle_mask(f)]
    if not mask_files_sorted:
        raise FileNotFoundError(f"No bundle masks found in {args.mask_dir}")

    ref_img = nib.load(args.ref) if args.ref else nib.load(mask_files_sorted[0])
    ref_mask, mask_files = merge_bundle_masks(args.mask_dir, ref_img)
    inv_aff = np.linalg.inv(ref_img.affine)

    sft = load_tractogram(args.tck, ref_img, to_space=Space.RASMM)
    streamlines = list(sft.streamlines)
    n_total = len(streamlines)

    # Criterion 1: Intersect
    n_matched = streamlines_intersect_mask(streamlines, ref_mask, inv_aff)
    n_spurious = n_total - n_matched
    spurious_ratio = n_spurious / n_total if n_total > 0 else 0.0

    # Criterion 2: Endpoint validity
    n_valid_ep = streamlines_endpoint_valid(
        streamlines, ref_mask, inv_aff, dilation_iters=args.dilation
    )
    ep_invalid_ratio = 1.0 - n_valid_ep / n_total if n_total > 0 else 0.0

    result = {
        "tck": os.path.abspath(args.tck),
        "mask_dir": os.path.abspath(args.mask_dir),
        "n_masks": len(mask_files),
        "criterion_intersect": {
            "n_total": n_total,
            "n_matched": n_matched,
            "n_spurious": n_spurious,
            "spurious_ratio": spurious_ratio,
        },
        "criterion_endpoint": {
            "n_total": n_total,
            "n_valid": n_valid_ep,
            "n_invalid": n_total - n_valid_ep,
            "invalid_ratio": ep_invalid_ratio,
            "dilation_iters": args.dilation,
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[intersect] Total: {n_total}")
    print(f"[intersect] Matched: {n_matched}  Spurious: {n_spurious}  ratio: {spurious_ratio:.4f}")
    print(f"[endpoint]  Valid: {n_valid_ep}  Invalid: {n_total - n_valid_ep}  "
          f"invalid_ratio: {ep_invalid_ratio:.4f}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
