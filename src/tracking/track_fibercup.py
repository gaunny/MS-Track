#!/usr/bin/env python
"""
FiberCup-specific tracking variant.

Alternative tracking script optimized for the FiberCup phantom dataset.
Uses adaptive angle thresholds and geometry health scoring.

Usage:
    python -m src.tracking.track_fibercup --output fibercup.tck
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib

from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.data import default_sphere
from dipy.direction.pmf import SHCoeffPmfGen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATASETS, TRACKING_PARAMS, ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="FiberCup tracking")
    parser.add_argument("--dataset", default="fibercup",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--output", default=None, help="Output .tck path")
    parser.add_argument("--use_asi", action="store_true", default=False)
    parser.add_argument("--use_delta", action="store_true", default=False)
    parser.add_argument("--use_pmf", action="store_true", default=False)
    parser.add_argument("--probabilistic", action="store_true", default=True)
    parser.add_argument("--use_geometry_health", action="store_true", default=True)
    parser.add_argument("--use_adaptive_theta", action="store_true", default=True)
    return parser.parse_args()


def geometry_health(curv_hist, smooth_hist, is_human):
    """Compute a geometry health score from curvature and smoothness history."""
    if len(curv_hist) == 0:
        return 1.0
    curv_coeff = np.deg2rad(45) if is_human else np.deg2rad(30)
    curv_score = np.exp(-np.mean(curv_hist) / curv_coeff)
    smooth_score = np.clip((np.mean(smooth_hist) + 1) / 2, 0, 1)
    if is_human:
        return 0.7 * curv_score + 0.3 * smooth_score
    return 0.5 * (curv_score + smooth_score)


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]

    fodf_path = cfg["fodf_path"]
    mask_path = cfg["mask_path"]
    seed_mask_path = cfg.get("seed_mask_path", mask_path)
    delta_pred_path = cfg["delta_path"]
    peaks_path = cfg["peaks_path"]
    asi_path = cfg["asi_path"]
    IS_HUMAN_DATA = cfg["is_human"]

    USE_ASI = args.use_asi
    USE_DELTA = args.use_delta
    USE_PMF = args.use_pmf
    PROBABILISTIC = args.probabilistic
    USE_GEOMETRY_HEALTH = args.use_geometry_health
    USE_ADAPTIVE_THETA = args.use_adaptive_theta

    out_tck = args.output or os.path.join(cfg["out_root"], f"{args.dataset}.tck")
    ensure_output_dir(os.path.dirname(out_tck))

    step_size = 0.5
    min_length = 30.0 if IS_HUMAN_DATA else 80.0
    max_length = TRACKING_PARAMS["max_length_mm"]
    # Base turning angles per manuscript Table 1 / Methods 3.2.2
    # (ms_track.py implements full per-region filtering; this variant
    #  uses a single modulated angle for the FiberCup phantom)
    theta_base = 70.0
    theta_min = 25.0
    seed_density = TRACKING_PARAMS["seed_density"]
    MAX_NO_PEAK_STEPS = 8 if IS_HUMAN_DATA else 3
    MAX_HIST = 5

    print("Loading datasets...")
    fodf_img = nib.load(fodf_path)
    affine = fodf_img.affine
    voxel_size = np.mean(fodf_img.header.get_zooms()[:3])

    mask = nib.load(mask_path).get_fdata().astype(bool)
    seed_mask = nib.load(seed_mask_path).get_fdata().astype(bool)
    delta = np.clip(nib.load(delta_pred_path).get_fdata(), 0, 1)
    asi = nib.load(asi_path).get_fdata() if USE_ASI else None

    peaks_data = nib.load(peaks_path).get_fdata()
    X, Y, Z, C = peaks_data.shape
    peaks = peaks_data.reshape(X, Y, Z, C // 3, 3)

    vertices = default_sphere.vertices
    pmf_gen = SHCoeffPmfGen(fodf_img.get_fdata().astype(np.float64),
                            default_sphere, "tournier07")
    flipped_vertices = vertices * np.array([-1, 1, 1])

    seeds_voxel = utils.seeds_from_mask(seed_mask, affine=np.eye(4),
                                        density=seed_density)

    def track_one_direction(seed_v, init_dir_sign):
        pos = seed_v.copy()
        prev_dir = None
        sl = [pos.copy()]
        curv_hist, smooth_hist = [], []
        no_peak_cnt = 0
        pmf_out = np.zeros(len(vertices))
        max_steps = int(max_length / (step_size * voxel_size))

        for _ in range(max_steps):
            i, j, k = np.round(pos).astype(int)
            if not (0 <= i < X and 0 <= j < Y and 0 <= k < Z) or not mask[i, j, k]:
                break
            if USE_DELTA and delta[i, j, k] < 0.1:
                break

            if USE_PMF:
                pmf_gen.get_pmf(np.ascontiguousarray(pos, dtype=np.float64), pmf_out)
                if prev_dir is None:
                    candidates = [flipped_vertices[np.argmax(pmf_out)] * init_dir_sign]
                    scores = [1.0]
                else:
                    dot_prods = np.dot(flipped_vertices, prev_dir)
                    w = pmf_out * (dot_prods > 0) * (dot_prods ** 4)
                    idx = np.argsort(w)[-3:]
                    candidates = [flipped_vertices[i] for i in idx if w[i] > 1e-5]
                    scores = [w[i] for i in idx if w[i] > 1e-5]
            else:
                cand_peaks = peaks[i, j, k]
                candidates = cand_peaks[np.linalg.norm(cand_peaks, axis=1) > 1e-5]
                scores = [np.linalg.norm(c) for c in candidates]

            if len(candidates) == 0:
                if prev_dir is None or no_peak_cnt >= MAX_NO_PEAK_STEPS:
                    break
                pos += step_size * prev_dir
                sl.append(pos.copy())
                no_peak_cnt += 1
                continue
            no_peak_cnt = 0

            geom_h = geometry_health(curv_hist, smooth_hist, IS_HUMAN_DATA) \
                if USE_GEOMETRY_HEALTH else 1.0

            if USE_ADAPTIVE_THETA:
                asi_val = asi[i, j, k] if USE_ASI else 1.0
                theta = np.deg2rad(max(theta_base * (0.7 + 0.15 * asi_val + 0.15 * geom_h),
                                      theta_min))
            else:
                theta = np.deg2rad(theta_base)

            final_scores = []
            valid_dirs = []
            for d, s in zip(candidates, scores):
                for v_dir in [d, -d] if not USE_PMF else [d]:
                    if prev_dir is not None:
                        dot = np.dot(v_dir, prev_dir)
                        ang = np.arccos(np.clip(dot, -1, 1))
                        if ang > theta:
                            continue
                        asi_power = 3.0 * (1.0 - (asi[i, j, k] if USE_ASI else 0.5)) + 1.0
                        f_score = (dot ** asi_power) * s
                    else:
                        f_score = s
                        v_dir = v_dir * init_dir_sign
                    valid_dirs.append(v_dir)
                    final_scores.append(f_score)

            if not valid_dirs:
                break

            if not PROBABILISTIC:
                new_dir = valid_dirs[np.argmax(final_scores)]
            else:
                ps = np.exp(np.array(final_scores) / 0.1)
                ps = ps / np.sum(ps)
                new_dir = valid_dirs[np.random.choice(len(valid_dirs), p=ps)]

            if prev_dir is not None:
                curv_hist.append(np.arccos(np.clip(np.dot(new_dir, prev_dir), -1, 1)))
                smooth_hist.append(np.dot(new_dir, prev_dir))
                curv_hist = curv_hist[-MAX_HIST:]
                smooth_hist = smooth_hist[-MAX_HIST:]

            pos += step_size * new_dir
            sl.append(pos.copy())
            prev_dir = new_dir

        return sl

    final_streamlines = []
    print(f"Starting bidirectional tracking on {len(seeds_voxel)} seeds...")

    for s_idx, seed_v in enumerate(seeds_voxel):
        sl_fwd = track_one_direction(seed_v, +1)
        sl_bwd = track_one_direction(seed_v, -1)
        full_sl_v = np.array(sl_bwd[::-1][:-1] + sl_fwd)

        if len(full_sl_v) * step_size * voxel_size >= min_length:
            full_sl_ras = np.dot(affine, np.hstack(
                [full_sl_v, np.ones((len(full_sl_v), 1))]).T).T[:, :3]
            final_streamlines.append(full_sl_ras.astype(np.float32))

        if (s_idx + 1) % 1000 == 0:
            print(f"Progress: {s_idx + 1}/{len(seeds_voxel)} | "
                  f"Streamlines: {len(final_streamlines)}")

    if final_streamlines:
        sft = StatefulTractogram(final_streamlines, fodf_img, Space.RASMM)
        sft.remove_invalid_streamlines()
        save_tractogram(sft, out_tck)
        print(f"Done! Saved to {out_tck}")
    else:
        print("No streamlines met the length criteria.")


if __name__ == "__main__":
    main()
