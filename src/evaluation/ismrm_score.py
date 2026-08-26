#!/usr/bin/env python
"""
ISMRM 2015 tractometer scoring.

Wrapper for the ISMRM 2015 tractography challenge scoring system.
Scores a tractogram submission following the method described in
https://doi.org/10.1101/084137

Usage:
    python -m src.evaluation.ismrm_score \
        <tractogram.tck> <scoring_data_dir> <output_dir> \
        [--compute_ic_ib] [--save_full_vc]

Note: Requires the ISMRM 2015 challenge scoring data and the
'challenge_scoring' Python package.
"""

import os
import sys
import argparse
import glob
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_args_parser():
    description = """
    Score a submission for the ISMRM 2015 tractography challenge.

    Based on the ISMRM 2015 tractography challenge:
    http://www.tractometer.org/ismrm_2015_challenge/

    The algorithm has 6 main steps:
        1: extract all streamlines classified as Valid Connections (VC)
        2: remove streamlines shorter than a threshold
        3: cluster the remaining streamlines
        4: remove singletons
        5: assign each cluster to closest ROIs (Invalid Connections, IC)
        6: remaining streamlines classified as No Connection (NC)
    """
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tractogram", metavar="TRACTS",
                   help="Tractogram file (.tck or .trk)")
    p.add_argument("base_dir", metavar="BASE_DIR",
                   help="Base directory for scoring data")
    p.add_argument("out_dir", metavar="OUT_DIR",
                   help="Directory for score output files")
    p.add_argument("--out_tract_type", choices=["tck", "trk"], default="tck")
    p.add_argument("--save_full_vc", action="store_true",
                   help="Save all VCs in one file")
    p.add_argument("--save_full_ic", action="store_true",
                   help="Save all ICs in one file")
    p.add_argument("--save_full_nc", action="store_true",
                   help="Save all NCs in one file")
    p.add_argument("--compute_ic_ib", action="store_true",
                   help="Segment rejected streamlines into NC + IC")
    p.add_argument("--save_ib", action="store_true", help="Save IBs independently")
    p.add_argument("--save_vb", action="store_true", help="Save VBs independently")
    p.add_argument("-f", dest="force", action="store_true", help="Overwrite output")
    p.add_argument("-v", dest="verbose", action="store_true", help="Verbose output")
    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    try:
        from challenge_scoring.io.results import save_results
        from challenge_scoring.metrics.scoring import score_submission
        from challenge_scoring.utils.attributes import load_attribs
        from challenge_scoring.utils.filenames import mkdir
    except ImportError:
        print("Error: 'challenge_scoring' package not found.")
        print("Install from: https://github.com/scilus/ismrm_2015_tractography_challenge_scoring")
        sys.exit(1)

    tractogram = args.tractogram
    base_dir = args.base_dir
    out_dir = args.out_dir

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(tractogram):
        parser.error(f'"{tractogram}" must be a file!')

    _, ext = os.path.splitext(tractogram)
    if ext not in (".tck", ".trk"):
        parser.error(f"Tractogram should be .tck or .trk, not {ext}")

    if not os.path.isdir(base_dir):
        parser.error(f'"{base_dir}" must be a directory!')

    scores_dir = mkdir(os.path.join(out_dir, "scores"))
    scores_filename = os.path.join(
        scores_dir,
        os.path.splitext(os.path.basename(tractogram))[0] + ".json"
    )

    if os.path.isfile(scores_filename):
        if not args.force:
            parser.error("Scores file exists. Use -f to overwrite.")
        else:
            os.remove(scores_filename)

    segments_dir = ""
    base_name = ""
    if any([args.save_full_vc, args.save_full_ic, args.save_full_nc,
            args.save_ib, args.save_vb]):
        segments_dir = mkdir(os.path.join(out_dir, "segmented"))
        base_name = os.path.splitext(os.path.basename(tractogram))[0]

    gt_bundles_attribs_path = os.path.join(base_dir, "gt_bundles_attributes.json")
    if not os.path.isfile(gt_bundles_attribs_path):
        parser.error('Missing "gt_bundles_attributes.json" in base directory.')

    basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

    scores = score_submission(
        tractogram, base_dir, basic_bundles_attribs,
        args.save_full_vc, args.save_full_ic, args.save_full_nc,
        args.compute_ic_ib, args.save_ib, args.save_vb,
        segments_dir, base_name, args.out_tract_type, args.verbose
    )

    if scores is not None:
        print(f"Saving results to {scores_filename}")
        save_results(scores_filename, scores)


if __name__ == "__main__":
    main()
