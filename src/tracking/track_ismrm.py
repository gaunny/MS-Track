#!/usr/bin/env python
"""
ISMRM 2015 challenge per-bundle tracking.

Tracks individual bundles using ROI masks with muFA/ASI-guided
termination criteria. Optimized for the ISMRM 2015 synthetic dataset.

Usage:
    python -m src.tracking.track_ismrm --fodf <fodf.nii.gz> --delta <delta.nii.gz> \
        --peaks <peaks.nii.gz> --asi <asi.nii.gz> --mask_dir <bundle_masks/> \
        --output_dir <output/>
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from glob import glob

from dipy.io.image import load_nifti, load_nifti_data
from dipy.direction.pmf import SHCoeffPmfGen
from dipy.data import default_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATASETS, ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="ISMRM per-bundle tracking")
    parser.add_argument("--fodf", default=None, help="fODF NIfTI path")
    parser.add_argument("--delta", default=None, help="muFA map NIfTI path")
    parser.add_argument("--peaks", default=None, help="Peaks NIfTI path")
    parser.add_argument("--asi", default=None, help="ASI map NIfTI path")
    parser.add_argument("--mask_dir", default=None, help="Bundle mask directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--dataset", default="ismrm",
                        choices=list(DATASETS.keys()),
                        help="Use predefined dataset config")
    parser.add_argument("--use_asi", action="store_true", default=True)
    parser.add_argument("--use_delta", action="store_true", default=True)
    parser.add_argument("--use_pmf", action="store_true", default=False)
    parser.add_argument("--probabilistic", action="store_true", default=True)
    parser.add_argument("--min_length_mm", type=float, default=30.0)
    parser.add_argument("--max_length_mm", type=float, default=200.0)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--max_angle", type=float, default=70.0)
    parser.add_argument("--seed_density", type=int, default=3)
    return parser.parse_args()


def generate_seeds_with_density(mask, density):
    """Generate seed points within a mask at given density."""
    indices = np.where(mask > 0)
    indices = np.array(indices).T

    if density <= 1:
        return indices.astype(np.float64) + 0.5

    offsets = np.linspace(1.0 / (2 * density), 1.0 - 1.0 / (2 * density), density)
    grid_offsets = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
    all_seeds = indices[:, np.newaxis, :] + grid_offsets[np.newaxis, :, :]
    return all_seeds.reshape(-1, 3)


def track_bundle(mask_data, seeds_voxel, peaks_res, delta_data, asi_data,
                 pmf_gen, flipped_vertices, cos_threshold, max_steps,
                 data_shape, params):
    """Track streamlines within a single bundle mask."""
    all_streamlines_v = []
    use_asi = params["use_asi"]
    use_delta = params["use_delta"]
    use_pmf = params["use_pmf"]
    probabilistic = params["probabilistic"]
    step_size = params["step_size"]
    max_boundary_steps = 8
    delta_thresh = 0.08
    asi_thresh = 0.85
    momentum_base = 4.0
    pmf_out = np.zeros(len(default_sphere.vertices), dtype=np.float64)

    for seed_v in seeds_voxel:
        tracks = []
        for direction_sign in [1, -1]:
            streamline = [seed_v.copy()]
            current_pos = seed_v.copy()
            prev_dir = None
            boundary_steps = 0

            for step in range(max_steps // 2):
                ix, iy, iz = np.round(current_pos).astype(int)

                if not (0 <= ix < data_shape[0] and
                        0 <= iy < data_shape[1] and
                        0 <= iz < data_shape[2]):
                    break
                if mask_data[ix, iy, iz] == 0:
                    break

                delta_val = delta_data[ix, iy, iz]
                asi_val = asi_data[ix, iy, iz]

                # muFA-based termination
                if use_delta and delta_val < delta_thresh:
                    boundary_steps += 1
                    if boundary_steps >= max_boundary_steps:
                        break
                else:
                    boundary_steps = max(0, boundary_steps - 1)

                # ASI-based probabilistic termination
                if use_asi and asi_val > asi_thresh:
                    if np.random.random() < 0.2:
                        break

                # Direction selection
                momentum_power = momentum_base
                if use_asi:
                    momentum_power *= (1.0 - 0.5 * asi_val)
                if use_delta:
                    momentum_power *= (0.5 + 0.5 * delta_val / 0.2)

                candidates, scores = [], []

                if use_pmf:
                    target_pos = np.ascontiguousarray(current_pos, dtype=np.float64)
                    pmf_gen.get_pmf(target_pos, pmf_out)
                    if prev_dir is None:
                        idx = np.argmax(pmf_out)
                        candidates.append(flipped_vertices[idx] * direction_sign)
                        scores.append(1.0)
                    else:
                        dot_prods = np.dot(flipped_vertices, prev_dir)
                        weights = pmf_out * (dot_prods > cos_threshold) * \
                            (np.maximum(0, dot_prods) ** momentum_power)
                        if np.max(weights) > 1e-7:
                            top_idx = np.argsort(weights)[-5:]
                            for idx in top_idx:
                                if weights[idx] > 0:
                                    candidates.append(flipped_vertices[idx])
                                    scores.append(weights[idx])
                else:
                    voxel_peaks = peaks_res[ix, iy, iz]
                    if prev_dir is None:
                        strengths = np.linalg.norm(voxel_peaks, axis=1)
                        if np.max(strengths) > 1e-5:
                            best_idx = np.argmax(strengths)
                            candidates.append(voxel_peaks[best_idx] * direction_sign)
                            scores.append(strengths[best_idx])
                    else:
                        for p in voxel_peaks:
                            norm_p = np.linalg.norm(p)
                            if norm_p < 1e-5:
                                continue
                            for v_dir in [p / norm_p, -p / norm_p]:
                                dot = np.dot(v_dir, prev_dir)
                                if dot > cos_threshold:
                                    candidates.append(v_dir)
                                    scores.append((dot ** momentum_power) * norm_p)

                if not candidates:
                    if boundary_steps > 3:
                        break
                    continue

                if not probabilistic:
                    best_dir = candidates[np.argmax(scores)]
                else:
                    temp = 0.15
                    scores_array = np.array(scores)
                    ps = np.exp(scores_array / temp)
                    ps = ps / np.sum(ps)
                    best_dir = candidates[np.random.choice(len(candidates), p=ps)]

                current_pos += best_dir * step_size
                streamline.append(current_pos.copy())
                prev_dir = best_dir

            tracks.append(streamline)

        # Merge bidirectional results
        if len(tracks) == 2 and len(tracks[0]) > 1 and len(tracks[1]) > 1:
            full_line_v = np.vstack([
                np.array(tracks[1])[::-1],
                np.array(tracks[0])[1:]
            ])
            line_length_mm = len(full_line_v) * step_size * params["voxel_mm"]
            if params["min_length_mm"] <= line_length_mm <= params["max_length_mm"]:
                all_streamlines_v.append(full_line_v)

    return all_streamlines_v


def main():
    args = parse_args()

    # Resolve paths from config or CLI args
    cfg = DATASETS.get(args.dataset, {})
    fodf_path = args.fodf or cfg.get("fodf_path")
    delta_path = args.delta or cfg.get("delta_path")
    peaks_path = args.peaks or cfg.get("peaks_path")
    asi_path = args.asi or cfg.get("asi_path")
    mask_dir = args.mask_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "ismrm", "scoring_data", "masks", "bundles", "2mm"
    )
    output_dir = args.output_dir or os.path.join(
        cfg.get("out_root", "."), "ismrm_optimized"
    )
    ensure_output_dir(output_dir)

    print("=" * 60)
    print("Loading ISMRM datasets...")
    print("=" * 60)

    data, affine, hardi_img = load_nifti(fodf_path, return_img=True)
    voxel_mm = np.mean(hardi_img.header.get_zooms()[:3])
    asi_data = load_nifti_data(asi_path)
    delta_data = load_nifti_data(delta_path)
    data_shape = asi_data.shape

    print(f"  FODF shape: {data.shape}")
    print(f"  muFA range: [{delta_data.min():.4f}, {delta_data.max():.4f}]")
    print(f"  ASI range: [{asi_data.min():.4f}, {asi_data.max():.4f}]")

    peaks_res = None
    if not args.use_pmf:
        peaks_res = load_nifti_data(peaks_path).reshape(data_shape + (-1, 3))

    pmf_gen = SHCoeffPmfGen(data.astype(np.float64), default_sphere, "tournier07")
    flipped_vertices = default_sphere.vertices * np.array([-1, 1, 1])

    max_steps = int(args.max_length_mm / (args.step_size * voxel_mm))
    cos_threshold = np.cos(np.deg2rad(args.max_angle))

    params = {
        "use_asi": args.use_asi,
        "use_delta": args.use_delta,
        "use_pmf": args.use_pmf,
        "probabilistic": args.probabilistic,
        "step_size": args.step_size,
        "min_length_mm": args.min_length_mm,
        "max_length_mm": args.max_length_mm,
        "voxel_mm": voxel_mm,
    }

    mask_files = sorted(glob(os.path.join(mask_dir, "*.nii.gz")))
    print(f"Found {len(mask_files)} bundles to track")

    for mask_path in mask_files:
        bundle_name = os.path.basename(mask_path).replace(".nii.gz", "")
        mask_data = load_nifti_data(mask_path)
        seeds_voxel = generate_seeds_with_density(mask_data, args.seed_density)

        sl_v_final = track_bundle(
            mask_data, seeds_voxel, peaks_res, delta_data, asi_data,
            pmf_gen, flipped_vertices, cos_threshold, max_steps,
            data_shape, params
        )

        sl_ras = []
        for s in sl_v_final:
            s_ras = np.dot(affine, np.hstack(
                [s, np.ones((len(s), 1))]).T).T[:, :3]
            sl_ras.append(s_ras.astype(np.float32))

        if sl_ras:
            sft = StatefulTractogram(sl_ras, hardi_img, Space.RASMM)
            sft.remove_invalid_streamlines()
            output_fn = os.path.join(output_dir, f"{bundle_name}_track.tck")
            save_tractogram(sft, output_fn)
            print(f"  {bundle_name}: {len(sl_ras)} streamlines")
        else:
            print(f"  {bundle_name}: no streamlines")

    print("\nAll ISMRM bundles processed!")


if __name__ == "__main__":
    main()
