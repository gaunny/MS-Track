#!/usr/bin/env python
"""
Tractogram format conversion: .tck <-> .vtk <-> .tck.

Converts between MRtrix .tck format and VTK polydata format,
resampling streamlines to fixed-length point arrays as needed.

Usage:
    python -m src.utils.trk2vtk2tck --tck2vtk <input.tck> <output.vtk>
    python -m src.utils.trk2vtk2tck --vtk2tck <input.vtk> <output.tck>
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Tractogram format conversion")
    parser.add_argument("--tck2vtk", nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Convert .tck to .vtk")
    parser.add_argument("--vtk2tck", nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Convert .vtk to .tck")
    return parser.parse_args()


def tck_to_vtk(input_tck, output_vtk):
    """Convert .tck to .vtk format."""
    try:
        import pyvista as pv
    except ImportError:
        print("Error: pyvista not installed. Install with: pip install pyvista")
        sys.exit(1)

    sft = load_tractogram(input_tck, "same", to_space=Space.RASMM)
    streamlines = list(sft.streamlines)
    print(f"Loaded {len(streamlines)} streamlines from {input_tck}")

    polydata = pv.PolyData()
    lines = []
    points = []
    point_offset = 0

    for sl in streamlines:
        n = len(sl)
        lines.append(n)
        for i in range(n):
            lines.append(point_offset + i)
        points.extend(sl)
        point_offset += n

    polydata.points = np.array(points)
    polydata.lines = np.array(lines)
    polydata.save(output_vtk)
    print(f"Saved VTK to {output_vtk}")


def vtk_to_tck(input_vtk, output_tck):
    """Convert .vtk to .tck format."""
    try:
        import pyvista as pv
        from dipy.tracking.streamline import Streamlines
    except ImportError:
        print("Error: pyvista not installed. Install with: pip install pyvista")
        sys.exit(1)

    mesh = pv.read(input_vtk)
    streamlines = []
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        pts = np.array(cell.points)
        if pts.shape[0] > 1:
            streamlines.append(pts)

    print(f"Loaded {len(streamlines)} streamlines from {input_vtk}")

    if streamlines:
        ref = nib.Nifti1Image(np.zeros((1, 1, 1)), np.eye(4))
        sft = StatefulTractogram(streamlines, ref, Space.RASMM)
        sft.remove_invalid_streamlines()
        save_tractogram(sft, output_tck)
        print(f"Saved TCK to {output_tck}")
    else:
        print("No valid streamlines found.")


def main():
    args = parse_args()
    if args.tck2vtk:
        tck_to_vtk(args.tck2vtk[0], args.tck2vtk[1])
    elif args.vtk2tck:
        vtk_to_tck(args.vtk2tck[0], args.vtk2tck[1])
    else:
        print("Please specify --tck2vtk or --vtk2tck")


if __name__ == "__main__":
    main()
