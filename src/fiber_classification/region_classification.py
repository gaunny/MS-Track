"""
Fiber region classification module.

Classifies each white matter voxel into one of four geometric regions
based on fODF peak configuration and microstructural metrics (FA, muFA, ASI):

  1 = Parallel    (single dominant fiber direction)
  2 = Crossing    (2+ fiber directions with significant angular separation)
  3 = Terminal    (high asymmetry, likely near fiber endpoints)
  4 = Bending    (two directions at moderate angle)

This classification is used by the MS-Track tracker to apply
region-specific direction filtering and score boosting.
"""

import numpy as np


PATCH_SIZE = 3
MIN_ANGLE_PARALLEL = 160.0   # Angle > 160 degrees -> parallel
MIN_ANGLE_BENDING = 90.0     # Angle > 90 degrees -> bending
TERMINAL_ASI_THRESH = 0.7    # ASI threshold for terminal classification


def clamp01(x):
    """Clip values to [0, 1] range."""
    return np.clip(x, 0.0, 1.0)


def angle_between(v1, v2):
    """Return angle in degrees between two vectors, guaranteed [0, 180]."""
    v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    cos_theta = np.clip(np.sum(v1_norm * v2_norm, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def classify_voxel(voxel_peaks, fa, delta, asi, edge=False):
    """
    Classify a single voxel based on its fODF peaks and microstructural metrics.

    Parameters
    ----------
    voxel_peaks : ndarray (N_peaks, 3)
        Peak directions for this voxel.
    fa : float
        Fractional anisotropy value.
    delta : float
        Microscopic MSD anisotropy value.
    asi : float
        Asymmetry index value.
    edge : bool
        Whether this voxel is on the white matter boundary.

    Returns
    -------
    label : int
        0=unclassified, 1=parallel, 2=crossing, 3=terminal, 4=bending
    """
    nonzero = np.linalg.norm(voxel_peaks, axis=-1) > 1e-5
    peaks_vec = voxel_peaks[nonzero]
    n_fam = len(peaks_vec)

    label = 0

    # Edge constraint: cannot be crossing at boundary
    if edge and n_fam >= 3:
        n_fam = 2

    if n_fam >= 3:
        label = 2  # Crossing
    elif n_fam == 2:
        ang = angle_between(peaks_vec[0], peaks_vec[1])
        if ang >= MIN_ANGLE_PARALLEL:
            label = 1  # Parallel
        elif ang >= MIN_ANGLE_BENDING:
            label = 4  # Bending
        else:
            label = 2  # Crossing
    elif n_fam == 1:
        if asi >= TERMINAL_ASI_THRESH:
            label = 3  # Terminal
        else:
            label = 1  # Parallel
    else:
        label = 0  # Empty voxel

    return label


def patch_based_classification(mask, peaks, fa_map, delta_map, asi_map):
    """
    Classify all white matter voxels into geometric regions using
    patch-based neighborhood context.

    Parameters
    ----------
    mask : ndarray (X, Y, Z)
        Boolean white matter mask.
    peaks : ndarray (X, Y, Z, N_peaks, 3)
        fODF peak directions.
    fa_map : ndarray (X, Y, Z)
        Fractional anisotropy map.
    delta_map : ndarray (X, Y, Z)
        Microscopic MSD anisotropy map.
    asi_map : ndarray (X, Y, Z)
        Asymmetry index map.

    Returns
    -------
    region_map : ndarray (X, Y, Z) uint8
        Region labels: 0=background, 1=parallel, 2=crossing,
        3=terminal, 4=bending
    """
    region_map = np.zeros(mask.shape, dtype=np.uint8)
    pad = PATCH_SIZE // 2

    padded_mask = np.pad(mask, pad, mode="constant", constant_values=0)
    padded_peaks = np.pad(
        peaks, ((pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)),
        mode="constant", constant_values=0
    )
    padded_fa = np.pad(fa_map, pad, mode="constant", constant_values=0)
    padded_delta = np.pad(delta_map, pad, mode="constant", constant_values=0)
    padded_asi = np.pad(asi_map, pad, mode="constant", constant_values=0)

    X, Y, Z = mask.shape

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if not mask[i, j, k]:
                    continue

                patch_mask = padded_mask[i:i + PATCH_SIZE,
                                         j:j + PATCH_SIZE,
                                         k:k + PATCH_SIZE]
                on_edge = np.any(patch_mask == 0)

                label = classify_voxel(
                    peaks[i, j, k],
                    fa_map[i, j, k],
                    delta_map[i, j, k],
                    asi_map[i, j, k],
                    edge=on_edge
                )
                region_map[i, j, k] = label

    return region_map
