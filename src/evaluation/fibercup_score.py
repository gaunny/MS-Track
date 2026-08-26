#!/usr/bin/env python
"""
FiberCup region-based scoring.

Evaluates predicted tractograms by assigning each streamline to its
dominant ground truth region (parallel/crossing/terminal/bending),
then computing per-region Dice overlap.

Usage:
    python -m src.evaluation.fibercup_score \
        --tck <prediction.tck> \
        --gt_dir <gt_region_masks/> \
        --wmmask <wm_mask.nii.gz>
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="FiberCup region-based scoring")
    parser.add_argument("--tck", required=True, help="Predicted tractogram .tck")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory with GT region masks (fibercup_region_*.nii.gz)")
    parser.add_argument("--wmmask", required=True, help="White matter mask .nii.gz")
    return parser.parse_args()


def load_region_masks(gt_dir, ref_img):
    """Load the four GT region masks."""
    regions = ["parallel", "crossing", "terminal", "bending"]
    masks = {}
    for r in regions:
        p = os.path.join(gt_dir, f"fibercup_region_{r}.nii.gz")
        if not os.path.exists(p):
            print(f"  [warning] GT region mask missing: {p}")
            masks[r] = np.zeros(ref_img.shape, dtype=bool)
        else:
            masks[r] = nib.load(p).get_fdata().astype(bool)
    return masks


def vox_of_streamline(sl, inv_aff):
    """Convert streamline from RASMM to voxel coordinates (deduplicated)."""
    homog = np.hstack([sl, np.ones((sl.shape[0], 1))])
    vox = (inv_aff @ homog.T).T[:, :3]
    vox = np.clip(np.round(vox), 0, None)
    idx = np.unique(vox.astype(np.int64), axis=0)
    return idx


def main():
    args = parse_args()
    ref_img = nib.load(args.wmmask)
    shape = np.array(ref_img.shape)
    inv_aff = np.linalg.inv(ref_img.affine)

    gt = load_region_masks(args.gt_dir, ref_img)
    regions = ["parallel", "crossing", "terminal", "bending"]

    sft = load_tractogram(args.tck, ref_img, to_space=Space.RASMM)
    streams = list(sft.streamlines)

    pred_per_region = {r: np.zeros(shape, dtype=np.uint8) for r in regions}
    assignment = {r: 0 for r in regions}

    for sl in streams:
        sl = np.asarray(sl)
        if sl.shape[0] < 2:
            continue
        idx = vox_of_streamline(sl, inv_aff)
        inside = np.all((idx >= 0) & (idx < shape), axis=1)
        idx = idx[inside]
        if idx.size == 0:
            continue
        counts = {r: int(gt[r][idx[:, 0], idx[:, 1], idx[:, 2]].sum()) for r in regions}
        best = max(regions, key=lambda r: counts[r])
        if counts[best] == 0:
            continue
        assignment[best] += 1
        pred_per_region[best][idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    print(f"{'region':<10}{'dice':>8}{'pred_stream':>12}{'gt_vox':>10}{'pred_vox':>10}")
    print("-" * 52)
    for r in regions:
        g = gt[r]
        p = pred_per_region[r].astype(bool)
        inter = np.logical_and(p, g).sum()
        union = p.sum() + g.sum()
        d = 2.0 * inter / (union + 1e-8)
        print(f"{r:<10}{d:>8.4f}{assignment[r]:>12}{int(g.sum()):>10}{int(p.sum()):>10}")

    wm = ref_img.get_fdata().astype(bool)
    pred_all = np.zeros(shape, dtype=bool)
    for r in regions:
        pred_all |= pred_per_region[r].astype(bool)
    cov = np.logical_and(pred_all, wm).sum() / (wm.sum() + 1e-8)
    print("-" * 52)
    print(f"WM coverage (region-assigned): {cov:.4f}")


if __name__ == "__main__":
    main()
