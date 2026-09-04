#!/usr/bin/env python
"""
MS-Track: Multi-Scale Tracking with DTD-guided modifiers.

This is the main tractography algorithm that integrates three types of
microstructural information derived from Diffusion Tensor Distribution (DTD)
inversion to guide streamline propagation:

1. FA (Fractional Anisotropy) - from LTE (Linear Tensor Encoding) signal
2. muFA (microscopic fractional anisotropy) - microscopic diffusion anisotropy from DTD inversion
3. ASI (Asymmetry Index) - from asymmetric fODF peaks

The tracker classifies each voxel into one of four regions (parallel,
crossing, terminal, bending) and applies region-specific direction
filtering and score boosting.

Usage:
    python -m src.tracking.ms_track --dataset fibercup --fa_mod --delta_mod --asi_mod
    python -m src.tracking.ms_track --dataset ismrm --fa_mod --delta_mod --asi_mod
"""

import os
import sys
import time
import json
import argparse
import numpy as np

import nibabel as nib
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.data import default_sphere
from dipy.direction.pmf import SHCoeffPmfGen

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATASETS, TRACKING_PARAMS, ensure_output_dir
from src.fiber_classification.region_classification import patch_based_classification


def parse_args():
    parser = argparse.ArgumentParser(description="MS-Track: DTD-guided tractography")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Dataset name")
    parser.add_argument("--fa_mod", action="store_true", dest="use_fa",
                        help="Enable FA modifier")
    parser.add_argument("--no_fa_mod", action="store_false", dest="use_fa", default=False)
    parser.add_argument("--delta_mod", action="store_true", dest="use_delta",
                        help="Enable muFA modifier")
    parser.add_argument("--no_delta_mod", action="store_false", dest="use_delta", default=False)
    parser.add_argument("--asi_mod", action="store_true", dest="use_asi",
                        help="Enable ASI modifier")
    parser.add_argument("--no_asi_mod", action="store_false", dest="use_asi", default=False)
    parser.add_argument("--pmf", action="store_true", dest="use_pmf", default=False,
                        help="Use PMF instead of discrete peaks")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: auto from out_root)")
    return parser.parse_args()


def run_tracking(args):
    cfg = DATASETS[args.dataset]
    file_name = args.dataset

    fodf_path = cfg["fodf_path"]
    mask_path = cfg.get("mask_path", cfg["fa_path"])
    seed_mask_path = cfg.get("seed_mask_path", mask_path)
    fa_path = cfg["fa_path"]
    delta_path = cfg["delta_path"]
    peaks_path = cfg["peaks_path"]
    asi_path = cfg["asi_path"]
    IS_HUMAN_DATA = cfg["is_human"]

    USE_FA_MOD = args.use_fa
    USE_DELTA_MOD = args.use_delta
    USE_ASI_MOD = args.use_asi
    USE_PMF = args.use_pmf
    PROBABILISTIC = TRACKING_PARAMS["probabilistic"]
    USE_GT_MASK = TRACKING_PARAMS["use_gt_mask"]

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(cfg["out_root"], f"{USE_FA_MOD}_{USE_DELTA_MOD}_{USE_ASI_MOD}")
    ensure_output_dir(OUTPUT_DIR)
    out_tck = os.path.join(OUTPUT_DIR, f"{file_name}.tck")

    # =========================================================
    # Parameters
    # =========================================================
    step_size = TRACKING_PARAMS["step_size"]
    min_length = 30.0 if IS_HUMAN_DATA else 80.0
    max_length = TRACKING_PARAMS["max_length_mm"]
    seed_density = TRACKING_PARAMS["seed_density"]
    MAX_NO_PEAK_STEPS = 8 if IS_HUMAN_DATA else 3
    MAX_HIST = 5

    t_start = time.time()

    # =========================================================
    # Load data
    # =========================================================
    print(f"Loading data for dataset: {args.dataset}")
    fodf_img = nib.load(fodf_path)
    affine = fodf_img.affine
    voxel_size = np.mean(fodf_img.header.get_zooms()[:3])

    mask = nib.load(mask_path).get_fdata().astype(bool)
    seed_mask = nib.load(seed_mask_path).get_fdata().astype(bool)
    peaks_data = nib.load(peaks_path).get_fdata()
    X, Y, Z, C = peaks_data.shape
    peaks = peaks_data.reshape(X, Y, Z, C // 3, 3)

    fa_map = nib.load(fa_path).get_fdata()
    delta_map = nib.load(delta_path).get_fdata()
    asi_map = nib.load(asi_path).get_fdata()

    vertices = default_sphere.vertices
    pmf_gen = SHCoeffPmfGen(
        fodf_img.get_fdata().astype(np.float64),
        default_sphere, "tournier07"
    )
    flipped_vertices = vertices * np.array([-1, 1, 1])

    seeds_voxel = utils.seeds_from_mask(
        seed_mask, affine=np.eye(4), density=seed_density
    )

    # =========================================================
    # Generate region masks via patch-based classification
    # =========================================================
    if USE_GT_MASK:
        parallel_mask = nib.load(cfg["parallel_mask_path"]).get_fdata().astype(bool)
        crossing_mask = nib.load(cfg["crossing_mask_path"]).get_fdata().astype(bool)
        bending_mask = nib.load(cfg["bending_mask_path"]).get_fdata().astype(bool)
        terminal_mask = nib.load(cfg["terminal_mask_path"]).get_fdata().astype(bool)
    else:
        use_mod = USE_FA_MOD or USE_DELTA_MOD or USE_ASI_MOD
        if use_mod:
            region_map = patch_based_classification(
                mask, peaks, fa_map, delta_map, asi_map
            )
            parallel_mask = region_map == 1
            crossing_mask = region_map == 2
            terminal_mask = region_map == 3
            bending_mask = region_map == 4

            labels = ["parallel", "crossing", "terminal", "bending"]
            for idx, name in enumerate(labels):
                binary_map = (region_map == (idx + 1)).astype(np.uint8)
                binary_img = nib.Nifti1Image(binary_map, affine)
                nib.save(binary_img,
                         os.path.join(OUTPUT_DIR, f"{file_name}_region_{name}.nii.gz"))
                print(f"Saved {name} mask: {OUTPUT_DIR}/{file_name}_region_{name}.nii.gz")
        else:
            print("WARNING: All modifiers (FA, muFA, ASI) are disabled. "
                  "Skipping automatic region classification.")
            parallel_mask = crossing_mask = bending_mask = terminal_mask = None

    # =========================================================
    # Region classification utilities
    # =========================================================
    REGION_SIGMA = TRACKING_PARAMS["region_sigma"]
    PROTO_PARALLEL = np.array(TRACKING_PARAMS["proto_parallel"])
    PROTO_CROSS = np.array(TRACKING_PARAMS["proto_cross"])
    PROTO_TERMINAL = np.array(TRACKING_PARAMS["proto_terminal"])
    PROTO_BENDING = np.array(TRACKING_PARAMS["proto_bending"])

    def classify_region(i, j, k):
        """
        Return (normalized weight list, dominant region type, max weight).
        Region types: 0=parallel, 1=crossing, 2=terminal, 3=bending
        """
        clamp01 = lambda x: np.clip(x, 0.0, 1.0)

        if USE_GT_MASK:
            w_par = 1.0 if parallel_mask[i, j, k] else 0.0
            w_crs = 1.0 if crossing_mask[i, j, k] else 0.0
            w_end = 1.0 if terminal_mask[i, j, k] else 0.0
            w_bnd = 1.0 if bending_mask[i, j, k] else 0.0
        else:
            use_mod = USE_FA_MOD or USE_DELTA_MOD or USE_ASI_MOD
            if use_mod:
                v = np.array([
                    clamp01(fa_map[i, j, k]) if USE_FA_MOD else 0.0,
                    clamp01(delta_map[i, j, k]) if USE_DELTA_MOD else 0.0,
                    clamp01(asi_map[i, j, k]) if USE_ASI_MOD else 0.0
                ])
                w = lambda proto: np.exp(
                    -np.sum((v - proto) ** 2) / (2 * REGION_SIGMA ** 2)
                )
                w_par = w(PROTO_PARALLEL)
                w_crs = w(PROTO_CROSS)
                w_end = w(PROTO_TERMINAL)
                w_bnd = w(PROTO_BENDING)
            else:
                w_par = w_crs = w_end = w_bnd = 1.0

        s = w_par + w_crs + w_end + w_bnd + 1e-8
        w_par, w_crs, w_end, w_bnd = w_par / s, w_crs / s, w_end / s, w_bnd / s

        weight_list = [w_par, w_crs, w_end, w_bnd]
        main_region = np.argmax(weight_list)
        max_weight = weight_list[main_region]

        return (w_par, w_crs, w_end, w_bnd), main_region, max_weight

    # =========================================================
    # Tracking core
    # =========================================================
    def track_one_direction(seed_v, init_dir_sign):
        pos = seed_v.copy()
        prev_dir = None
        sl = [pos.copy()]
        no_peak_cnt = 0
        pmf_out = np.zeros(len(vertices))
        max_steps = int(max_length / (step_size * voxel_size))

        for _ in range(max_steps):
            # Voxel coordinate conversion and validity check
            i, j, k = np.round(pos).astype(int)
            if not (0 <= i < X and 0 <= j < Y and 0 <= k < Z) or not mask[i, j, k]:
                break

            # Get region weights and dominant type
            (_, _, _, _), main_region, max_weight = classify_region(i, j, k)
            asi_val = float(asi_map[i, j, k])

            # Terminal region: terminate immediately when the streamline is already
            # in a high-confidence terminal voxel (ASI >= 0.7) or the terminal
            # prototype dominates; otherwise the voxel is a transition voxel and
            # the 20 deg turning limit below keeps the streamline from sharp turns.
            if main_region == 2 and (asi_val >= 0.7 or max_weight > 0.6):
                break

            # Generate candidate directions
            if USE_PMF:
                pmf_gen.get_pmf(
                    np.ascontiguousarray(pos, dtype=np.float64), pmf_out
                )
                candidates = [flipped_vertices[np.argmax(pmf_out)]]
                scores = [1.0]
            else:
                cand_peaks = peaks[i, j, k]
                candidates = cand_peaks[np.linalg.norm(cand_peaks, axis=1) > 1e-5]
                scores = [np.linalg.norm(c) for c in candidates]

            # Handle no valid candidates
            if len(candidates) == 0:
                if prev_dir is None or no_peak_cnt >= MAX_NO_PEAK_STEPS:
                    break
                pos += step_size * prev_dir
                sl.append(pos.copy())
                no_peak_cnt += 1
                continue
            no_peak_cnt = 0

            # Region-specific direction filtering and scoring
            final_dirs, final_scores = [], []
            for d, s in zip(candidates, scores):
                for v in [d, -d]:
                    if prev_dir is not None:
                        dot = np.dot(v, prev_dir)
                        base_score = dot * s
                    else:
                        base_score = s
                        v = v * init_dir_sign

                    # Region-specific direction filtering
                    keep_dir = True
                    if main_region == 0:  # Parallel: 30 deg max turn
                        if prev_dir is not None and np.dot(v, prev_dir) < np.cos(np.deg2rad(30)):
                            keep_dir = False
                    elif main_region == 3:  # Bending: 70 deg max turn
                        if prev_dir is not None and np.dot(v, prev_dir) < np.cos(np.deg2rad(70)):
                            keep_dir = False
                    elif main_region == 2 and asi_val < 0.7:
                        # Terminal transition voxel: allow at most a 20 deg turn so
                        # the streamline does not make a sharp corner right before
                        # entering or after leaving the termination region.
                        if prev_dir is not None and np.dot(v, prev_dir) < np.cos(np.deg2rad(20)):
                            keep_dir = False
                    # Crossing (main_region=1): no filtering
                    # Terminal: already terminated above

                    if not keep_dir:
                        continue

                    # Region-specific peak sampling weights (manuscript Methods 3.2.1)
                    if main_region == 0:
                        # Parallel: primary peak 0.95, secondary 0.05
                        is_primary = (prev_dir is None) or (np.dot(v, prev_dir) >= 0)
                        peak_weight = 0.95 if is_primary else 0.05
                        score_boost = peak_weight
                    elif main_region == 1:
                        # Crossing: equal weights 0.5 for peaks above amplitude 0.1
                        score_boost = 0.5 if s > 0.1 else 0.0
                    elif main_region == 3:
                        # Bending: ASI-modulated boost for 15-45 deg deflection, factor 2.0
                        if prev_dir is not None:
                            angle = np.degrees(np.arccos(np.clip(np.dot(v, prev_dir), -1, 1)))
                            if 15 <= angle <= 45:
                                score_boost = 1.0 + 2.0 * asi_val
                            else:
                                score_boost = 1.0
                        else:
                            score_boost = 1.0
                    else:
                        # Terminal transition: minimal boost
                        score_boost = 1.0
                    final_score = base_score * score_boost

                    final_dirs.append(v)
                    final_scores.append(final_score)

            if not final_dirs:
                if prev_dir is None or no_peak_cnt >= MAX_NO_PEAK_STEPS:
                    break
                pos += step_size * prev_dir
                sl.append(pos.copy())
                no_peak_cnt += 1
                continue

            # Direction selection (probabilistic or deterministic)
            if PROBABILISTIC:
                temp = 0.1
                scores_array = np.nan_to_num(
                    np.array(final_scores), nan=0.0, posinf=1e10, neginf=0.0
                )
                if np.all(scores_array == 0):
                    ps = np.ones_like(scores_array) / len(scores_array)
                else:
                    ps = np.exp(scores_array / temp)
                    ps = np.nan_to_num(
                        ps / (np.sum(ps) + 1e-10),
                        nan=0.0, posinf=0.0, neginf=0.0
                    )
                    if np.sum(ps) == 0:
                        ps = np.ones_like(ps) / len(ps)
                new_dir = final_dirs[np.random.choice(len(final_dirs), p=ps)]
            else:
                new_dir = final_dirs[np.argmax(final_scores)]

            pos += step_size * new_dir
            sl.append(pos.copy())
            prev_dir = new_dir

        return sl

    # =========================================================
    # Run tracking
    # =========================================================
    final_streamlines = []
    n_seeds = len(seeds_voxel)
    print(f"Tracking {n_seeds} seeds...")

    t_track_start = time.time()

    for idx, seed in enumerate(seeds_voxel):
        fwd = track_one_direction(seed, +1)
        bwd = track_one_direction(seed, -1)
        full = np.array(bwd[::-1][:-1] + fwd)

        if len(full) * step_size * voxel_size >= min_length:
            ras = np.dot(
                affine,
                np.hstack([full, np.ones((len(full), 1))]).T
            ).T[:, :3]
            final_streamlines.append(ras.astype(np.float32))

        if (idx + 1) % 1000 == 0:
            print(f"{idx + 1}/{len(seeds_voxel)} | streamlines: {len(final_streamlines)}")

    t_track_end = time.time()
    track_seconds = t_track_end - t_track_start

    # Save tractogram
    if final_streamlines:
        sft = StatefulTractogram(final_streamlines, fodf_img, Space.RASMM)
        sft.remove_invalid_streamlines()
        save_tractogram(sft, out_tck)
        print(f"Saved to {out_tck}")
    else:
        print("No streamlines saved.")

    # Runtime report
    total_seconds = time.time() - t_start
    n_sl = len(final_streamlines)
    runtime_report = {
        "dataset": args.dataset,
        "modules": f"{USE_FA_MOD}_{USE_DELTA_MOD}_{USE_ASI_MOD}",
        "n_seeds": int(n_seeds),
        "n_streamlines": int(n_sl),
        "tracking_seconds": round(track_seconds, 2),
        "total_seconds_incl_io": round(total_seconds, 2),
        "seconds_per_1M_streamlines": round(
            track_seconds / n_sl * 1e6, 2
        ) if n_sl > 0 else None,
        "seconds_per_1M_seeds": round(
            track_seconds / n_seeds * 1e6, 2
        ) if n_seeds > 0 else None,
    }
    runtime_fn = os.path.join(OUTPUT_DIR, f"runtime_{file_name}.json")
    with open(runtime_fn, "w") as f:
        json.dump(runtime_report, f, indent=2)
    print("Runtime report:")
    for k, v in runtime_report.items():
        print(f"  {k}: {v}")
    print(f"Runtime written to {runtime_fn}")


if __name__ == "__main__":
    args = parse_args()
    run_tracking(args)
