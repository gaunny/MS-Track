#!/usr/bin/env python
"""
Baseline tractography methods: iFOD2 (probabilistic) and SD_STREAM (deterministic).

Runs baseline methods under the same settings as MS-Track for fair comparison.

Usage:
    python -m src.tracking.baselines --dataset fibercup --method ifod2
    python -m src.tracking.baselines --dataset ismrm --method sdstream
    python -m src.tracking.baselines --dataset fibercup --method all
"""

import os
import sys
import json
import time
import argparse
import functools
import numpy as np
import nibabel as nib

from dipy.tracking import utils
from dipy.tracking.localtracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.data import default_sphere
from dipy.direction import (
    ProbabilisticDirectionGetter,
    DeterministicMaximumDirectionGetter,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATASETS, TRACKING_PARAMS, ensure_output_dir

# Tracking parameters aligned with ms_track.py
STEP_SIZE_VOX = TRACKING_PARAMS["step_size"]
SEED_DENSITY = TRACKING_PARAMS["seed_density"]
MAX_ANGLE_DEG = 30.0
PMF_THRESHOLD = 0.1
SH_ORDER = 8


def _patch_peak_directions():
    """
    Patch dipy's peak_directions to handle version mismatches between
    the Python layer and compiled Cython modules.
    """
    import sys as _sys
    import dipy.direction.peaks as _peaks
    if getattr(_peaks.peak_directions, "_ms_track_patched", False):
        return
    _orig = _peaks.peak_directions

    @functools.wraps(_orig)
    def wrapper(*args, **kwargs):
        kwargs.pop("sh_order", None)
        kwargs.pop("sh_basis_type", None)
        return _orig(*args, **kwargs)

    wrapper._ms_track_patched = True
    _peaks.peak_directions = wrapper
    for name, mod in list(_sys.modules.items()):
        if not name.startswith("dipy.direction"):
            continue
        bound = getattr(mod, "peak_directions", None)
        if bound is _orig:
            try:
                mod.peak_directions = wrapper
            except Exception:
                pass


def tracking_params(cfg):
    """Return length limits matching ms_track.py."""
    if cfg["is_human"]:
        return dict(min_mm=30.0, max_mm=250.0)
    return dict(min_mm=80.0, max_mm=250.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline tractography methods")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--method", required=True, choices=["ifod2", "sdstream", "all"])
    parser.add_argument("--max_angle", type=float, default=MAX_ANGLE_DEG,
                        help="Maximum turning angle in degrees (default: 30)")
    return parser.parse_args()


def main():
    _patch_peak_directions()
    args = parse_args()

    cfg = DATASETS[args.dataset]
    params = tracking_params(cfg)
    print(f"Dataset: {args.dataset} | human={cfg['is_human']} | "
          f"length [{params['min_mm']}, {params['max_mm']}] mm")

    fodf_img = nib.load(cfg["fodf_path"])
    affine = fodf_img.affine
    voxel_mm = float(np.mean(fodf_img.header.get_zooms()[:3]))
    step_mm = STEP_SIZE_VOX * voxel_mm
    seed_mask = nib.load(cfg.get("seed_mask_path", cfg["mask_path"])).get_fdata().astype(bool)
    print(f"voxel = {voxel_mm:.2f} mm, step = {step_mm:.2f} mm")

    seeds = utils.seeds_from_mask(seed_mask, affine=np.eye(4), density=SEED_DENSITY)
    n_seeds = len(seeds)
    print(f"Seeds: {n_seeds}")

    stop_mask = nib.load(cfg["mask_path"]).get_fdata().astype(bool)
    stopping_crit = BinaryStoppingCriterion(stop_mask)

    methods = ["ifod2", "sdstream"] if args.method == "all" else [args.method]
    out_root = os.path.join(cfg["out_root"], "baseline_runs")
    fodf_data = fodf_img.get_fdata().astype(np.float64)

    for method in methods:
        out_dir = os.path.join(out_root, method)
        ensure_output_dir(out_dir)
        out_tck = os.path.join(out_dir, f"{args.dataset}.tck")

        t0 = time.time()
        if method == "ifod2":
            dg = ProbabilisticDirectionGetter.from_shcoeff(
                fodf_data, max_angle=args.max_angle, sphere=default_sphere,
                sh_order=SH_ORDER, sh_basis_type="tournier07",
                pmf_threshold=PMF_THRESHOLD)
        else:
            dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                fodf_data, max_angle=args.max_angle, sphere=default_sphere,
                sh_order=SH_ORDER, sh_basis_type="tournier07",
                pmf_threshold=PMF_THRESHOLD)

        # Smoke test to catch dipy version issues early
        _ = dg.initial_direction(np.zeros(3, dtype=np.float64))

        streamlines_iter = LocalTracking(
            dg, stopping_crit, seeds=seeds, affine=np.eye(4),
            step_size=STEP_SIZE_VOX, max_cross=None, return_all=False)

        min_pts = max(2, int(round(params["min_mm"] / step_mm)))
        max_pts = int(round(params["max_mm"] / step_mm))
        final_streamlines = []
        for sl in streamlines_iter:
            n = len(sl)
            if n < min_pts or n > max_pts:
                continue
            ras = np.dot(affine, np.hstack([sl, np.ones((n, 1))]).T).T[:, :3]
            final_streamlines.append(ras.astype(np.float32))

        track_seconds = time.time() - t0

        if final_streamlines:
            sft = StatefulTractogram(final_streamlines, fodf_img, Space.RASMM)
            sft.remove_invalid_streamlines()
            save_tractogram(sft, out_tck)
            print(f"[{method}] Saved {len(final_streamlines)} streamlines -> {out_tck}")
        else:
            print(f"[{method}] No streamlines saved.")

        n_sl = len(final_streamlines)
        report = {
            "dataset": args.dataset,
            "modules": method,
            "tracker": method,
            "n_seeds": int(n_seeds),
            "n_streamlines": int(n_sl),
            "tracking_seconds": round(track_seconds, 2),
            "seconds_per_1M_streamlines": round(track_seconds / n_sl * 1e6, 2) if n_sl > 0 else None,
            "seconds_per_1M_seeds": round(track_seconds / n_seeds * 1e6, 2) if n_seeds > 0 else None,
            "step_mm": round(step_mm, 3),
            "max_angle_deg": args.max_angle,
            "seed_density": SEED_DENSITY,
        }
        runtime_fn = os.path.join(out_dir, f"runtime_{args.dataset}.json")
        with open(runtime_fn, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[{method}] Runtime -> {runtime_fn}")
        for k, v in report.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
