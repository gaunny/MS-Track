#!/usr/bin/env python
"""
Fiber bundle geometric analysis.

Computes geometric metrics for streamlines in a tractogram:
  - Length, curvature, smoothness, torsion
  - Bundle consistency (mean pairwise MAM distance)
  - Voxel volume coverage
  - Endpoint distance distribution

Usage:
    python -m src.utils.fiber_analysis --file <tractogram.tck>
    python -m src.utils.fiber_analysis --folder <tractogram_dir/>
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.tracking.streamline import length as streamline_length
from dipy.tracking.distances import bundles_distances_mam
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Fiber bundle geometric analysis")
    parser.add_argument("--file", default=None, help="Single tractogram file")
    parser.add_argument("--folder", default=None, help="Directory of tractograms")
    parser.add_argument("--csv", default="streamline_metrics.csv",
                        help="Output CSV file")
    parser.add_argument("--plot_dir", default="plots", help="Output plot directory")
    return parser.parse_args()


def compute_length(streamline):
    return streamline_length(streamline)


def compute_curvature(streamline):
    if len(streamline) < 3:
        return 0
    angles = []
    for i in range(1, len(streamline) - 1):
        v1 = streamline[i] - streamline[i - 1]
        v2 = streamline[i + 1] - streamline[i]
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6
        angles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return np.mean(angles) if angles else 0


def compute_smoothness(streamline):
    if len(streamline) < 3:
        return 1
    sims = []
    for i in range(1, len(streamline) - 1):
        v1 = streamline[i] - streamline[i - 1]
        v2 = streamline[i + 1] - streamline[i]
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6
        sims.append(np.dot(v1, v2))
    return np.mean(sims) if sims else 1


def compute_bundle_consistency(streamlines, n_samples=200):
    if len(streamlines) < 2:
        return 0
    n = min(n_samples, len(streamlines))
    idx = np.random.choice(len(streamlines), n, replace=False)
    subs = [streamlines[i] for i in idx]
    dists = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            d = bundles_distances_mam([subs[i]], [subs[j]])[0][0]
            dists.append(d)
    return np.mean(dists) if dists else 0


def compute_voxel_volume(streamlines, voxel_size=1.0):
    points = np.vstack(streamlines)
    points = np.round(points / voxel_size).astype(int)
    unique_voxels = np.unique(points, axis=0)
    return unique_voxels.shape[0] * (voxel_size ** 3)


def compute_endpoint_distance(streamlines):
    if len(streamlines) < 2:
        return 0
    endpoints = np.array([[s[0], s[-1]] for s in streamlines]).reshape(-1, 3)
    tree = cKDTree(endpoints)
    dists, _ = tree.query(endpoints, k=2)
    return np.mean(dists[:, 1])


def compute_torsion(streamline):
    if len(streamline) < 4:
        return 0
    torsions = []
    for i in range(1, len(streamline) - 2):
        p0, p1, p2, p3 = (streamline[i - 1], streamline[i],
                          streamline[i + 1], streamline[i + 2])
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        b1 /= np.linalg.norm(b1) + 1e-6
        b2 /= np.linalg.norm(b2) + 1e-6
        b3 /= np.linalg.norm(b3) + 1e-6
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        if norm_n1 < 1e-6 or norm_n2 < 1e-6:
            continue
        torsion = np.arccos(np.clip(
            np.dot(n1, n2) / (norm_n1 * norm_n2), -1, 1))
        torsions.append(torsion)
    return np.mean(torsions) if torsions else 0


def load_vtk_streamlines(file_path):
    import pyvista as pv
    mesh = pv.read(file_path)
    streamlines = []
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        points = cell.points
        if points.shape[0] > 1:
            streamlines.append(np.array(points))
    return streamlines


def analyze_streamlines(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".trk", ".tck"]:
        tractogram = nib.streamlines.load(file_path)
        streamlines = tractogram.streamlines
    elif ext == ".vtk":
        streamlines = load_vtk_streamlines(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    lengths = [compute_length(s) for s in streamlines]
    curvatures = [compute_curvature(s) for s in streamlines]
    smooths = [compute_smoothness(s) for s in streamlines]
    torsions = [compute_torsion(s) for s in streamlines]

    metrics = {
        "num_streamlines": len(streamlines),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        "curvature_mean": np.mean(curvatures),
        "curvature_std": np.std(curvatures),
        "smoothness_mean": np.mean(smooths),
        "smoothness_std": np.std(smooths),
        "torsion_mean": np.mean(torsions),
        "bundle_consistency": compute_bundle_consistency(streamlines),
        "endpoint_distance_mean": compute_endpoint_distance(streamlines),
        "voxel_volume": compute_voxel_volume(streamlines),
    }
    distributions = {
        "lengths": lengths, "curvatures": curvatures,
        "smoothness": smooths, "torsions": torsions,
    }
    return metrics, distributions


def save_metrics_to_csv(metrics, file_name, csv_file):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["file"] + list(metrics.keys()))
        writer.writerow([file_name] + list(metrics.values()))
    print(f"Saved to {csv_file}")


def plot_distributions(distributions, file_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for key, values in distributions.items():
        plt.figure()
        plt.hist(values, bins=50, alpha=0.7)
        plt.title(f"{key} distribution: {file_name}")
        plt.xlabel(key)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{file_name}_{key}.png"))
        plt.close()


def analyze(folder_path=None, file_path=None, csv_file="streamline_metrics.csv",
            plot_dir="plots"):
    if file_path is not None:
        metrics, distributions = analyze_streamlines(file_path)
        save_metrics_to_csv(metrics, os.path.basename(file_path), csv_file)
        plot_distributions(distributions, os.path.basename(file_path), plot_dir)
    elif folder_path is not None:
        for fname in os.listdir(folder_path):
            if fname.endswith((".trk", ".tck", ".vtk")):
                fpath = os.path.join(folder_path, fname)
                metrics, distributions = analyze_streamlines(fpath)
                save_metrics_to_csv(metrics, fname, csv_file)
                plot_distributions(distributions, fname, plot_dir)
    else:
        raise ValueError("Either --file or --folder must be provided.")


if __name__ == "__main__":
    args = parse_args()
    analyze(args.folder, args.file, args.csv, args.plot_dir)
