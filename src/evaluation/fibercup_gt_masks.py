#!/usr/bin/env python
"""
Generate FiberCup ground truth region masks.

Creates four geometric region masks (parallel, crossing, terminal, bending)
from FiberCup bundle and ROI masks for use in region-based evaluation.

Usage:
    python -m src.evaluation.fibercup_gt_masks \
        --bundles_dir <bundle_masks/> \
        --rois_dir <roi_masks/> \
        --output_dir <output/>
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FiberCup GT region masks")
    parser.add_argument("--bundles_dir", required=True,
                        help="Directory containing bundle masks (*.nii.gz)")
    parser.add_argument("--rois_dir", required=True,
                        help="Directory containing ROI masks (*_head.nii.gz, *_tail.nii.gz)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load bundle masks
    bundle_files = sorted(glob(os.path.join(args.bundles_dir, "*.nii.gz")))
    bundle_masks = []
    for f in bundle_files:
        img = nib.load(f)
        bundle_masks.append(img.get_fdata().astype(bool))

    ref_img = nib.load(bundle_files[0])
    shape = ref_img.shape
    affine = ref_img.affine
    print(f"Loaded {len(bundle_masks)} bundle masks, shape={shape}")

    # Generate terminal region from head/tail ROIs
    head_files = sorted(glob(os.path.join(args.rois_dir, "*_head.nii.gz")))
    tail_files = sorted(glob(os.path.join(args.rois_dir, "*_tail.nii.gz")))

    terminal_mask = np.zeros(shape, dtype=bool)
    for h, t in zip(head_files, tail_files):
        h_mask = nib.load(h).get_fdata().astype(bool)
        t_mask = nib.load(t).get_fdata().astype(bool)
        terminal_mask |= h_mask | t_mask

    # Generate crossing region
    def safe_get(idx):
        return bundle_masks[idx] if idx < len(bundle_masks) else np.zeros(shape, dtype=bool)

    crossing_mask = np.zeros(shape, dtype=bool)
    crossing_mask |= safe_get(0) & safe_get(4)  # Bundle 0 & 4
    crossing_mask |= safe_get(1) & safe_get(5)  # Bundle 1 & 5
    crossing_mask |= safe_get(4) & safe_get(5)  # Bundle 4 & 5

    # Generate parallel region (bundle body minus terminal and crossing)
    parallel_mask = np.zeros(shape, dtype=bool)
    for b_mask in bundle_masks:
        parallel_mask |= b_mask & ~terminal_mask & ~crossing_mask

    # Generate bending region (remaining bundle voxels)
    bending_mask = np.zeros(shape, dtype=bool)
    for b_mask in bundle_masks:
        bending_mask |= b_mask & ~parallel_mask & ~crossing_mask & ~terminal_mask

    # Save masks
    mask_dict = {
        "parallel": parallel_mask,
        "crossing": crossing_mask,
        "terminal": terminal_mask,
        "bending": bending_mask,
    }

    for name, mask in mask_dict.items():
        out_file = os.path.join(args.output_dir, f"fibercup_region_{name}.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), out_file)
        print(f"Saved {name} mask: {out_file}")

    print("\nAll region masks generated successfully.")


if __name__ == "__main__":
    main()
