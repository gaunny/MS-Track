#!/usr/bin/env python
"""
Summarize runtime reports from tracking results.

Scans for runtime_*.json files under a results directory and produces
a comparison table of seeds, streamlines, and timing metrics.

Usage:
    python -m src.evaluation.runtime_summary \
        --root <tracking_results/> \
        --out <runtime_summary.json>
"""

import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime summary")
    parser.add_argument("--root", default="tracking_results",
                        help="Tracking results root directory")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: <root>/runtime_summary.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    pattern = os.path.join(args.root, "**", "runtime_*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No runtime_*.json found (pattern: {pattern})")
        return

    rows = []
    for fn in files:
        try:
            with open(fn) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {fn}: {e}")
            continue
        rel = os.path.relpath(fn, args.root)
        combo_dir = os.path.dirname(rel)
        rows.append({
            "condition": combo_dir,
            "dataset": d.get("dataset", "?"),
            "modules": d.get("modules", "?"),
            "n_seeds": d.get("n_seeds", 0),
            "n_streamlines": d.get("n_streamlines", 0),
            "tracking_seconds": d.get("tracking_seconds"),
            "sec_per_1M_sl": d.get("seconds_per_1M_streamlines"),
            "sec_per_1M_seeds": d.get("seconds_per_1M_seeds"),
        })

    out_fn = args.out or os.path.join(args.root, "runtime_summary.json")
    with open(out_fn, "w") as f:
        json.dump(rows, f, indent=2)

    hdr = (f"{'condition':<42}{'seeds':>10}{'streaml.':>10}"
           f"{'sec':>9}{'s/1M_sl':>10}{'s/1M_seed':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = r["tracking_seconds"]
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else "-"
        s_sl = r["sec_per_1M_sl"]
        s_sl_str = f"{s_sl:.2f}" if isinstance(s_sl, (int, float)) else "-"
        s_sd = r["sec_per_1M_seeds"]
        s_sd_str = f"{s_sd:.2f}" if isinstance(s_sd, (int, float)) else "-"
        print(f"{r['condition']:<42}{r['n_seeds']:>10}{r['n_streamlines']:>10}"
              f"{t_str:>9}{s_sl_str:>10}{s_sd_str:>11}")

    print(f"\nSummary written to {out_fn}")


if __name__ == "__main__":
    main()
