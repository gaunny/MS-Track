"""
Centralized path configuration for ASIDTD-Track.

Data is NOT bundled in this repository. Users must download the
FiberCup and ISMRM 2015 datasets from their original sources and
set DATA_ROOT below to point to the local directory containing them.

Expected directory layout under DATA_ROOT:

    <DATA_ROOT>/
    ├── fibercup/
    │   ├── fodf/
    │   │   ├── fibercup_fodf_filter_asym.nii.gz
    │   │   ├── fibercup_peaks.nii.gz
    │   │   └── asi_map_wm.nii.gz
    │   ├── masks/
    │   │   └── fibercup_wm.nii.gz
    │   ├── anat/
    │   │   └── fibercup_fa.nii.gz
    │   └── DTDsimulation_results/
    │       └── fibercup_lte_ste_merged/
    │           └── dtd_msddelta.nii.gz
    └── ismrm/
        ├── fodf/
        │   ├── ismrm2015_fodf_filter_asym.nii.gz
        │   ├── ismrm2015_peaks.nii.gz
        │   └── asi_map.nii.gz
        ├── masks/
        │   └── ismrm2015_wm.nii.gz
        ├── anat/
        │   └── fa.nii.gz
        └── DTDsimulation_results/
            └── ismrm_lte_ste_merged/
                └── dtd_msddelta_wm_smooth_corrected.nii.gz

Data download links:
    FiberCup: https://www.tractometer.org/fibercup/
    ISMRM 2015: http://www.tractometer.org/ismrm_2015_challenge/

For custom datasets (e.g., clinical data), add new entries to DATASETS
following the same structure.
"""

import os
from pathlib import Path

# =========================================================
# Project root (auto-detected, no need to modify)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# =========================================================
# >>> USER CONFIGURATION <<<
# Set this to the directory where you placed the downloaded
# FiberCup and ISMRM datasets.
# =========================================================
DATA_ROOT = Path(os.environ.get(
    "ASIDTD_DATA_ROOT",
    str(Path.home() / "data" / "tractography"),
))

# Output root for tracking results
OUTPUT_ROOT = PROJECT_ROOT / "output"

# =========================================================
# Dataset configurations
# Each dataset defines paths to all required inputs:
#   - fodf:   fODF (fiber orientation distribution function) NIfTI
#   - mask:   White matter (or tracking) mask NIfTI
#   - fa:     Fractional anisotropy map NIfTI
#   - delta:  Microscopic fractional anisotropy (muFA) NIfTI
#   - peaks:  Asymmetric fODF peaks NIfTI
#   - asi:    Asymmetry index map NIfTI
#   - is_human: Whether the dataset is human brain (affects parameters)
# =========================================================

DATASETS = {
    # ------------------------------------------------------------------
    # FiberCup phantom (open-source)
    # Download: https://www.tractometer.org/fibercup/
    # ------------------------------------------------------------------
    "fibercup": {
        "fodf_path":   str(DATA_ROOT / "fibercup" / "fodf" / "fibercup_fodf_filter_asym.nii.gz"),
        "mask_path":   str(DATA_ROOT / "fibercup" / "masks" / "fibercup_wm.nii.gz"),
        "fa_path":     str(DATA_ROOT / "fibercup" / "anat" / "fibercup_fa.nii.gz"),
        "delta_path":  str(DATA_ROOT / "fibercup" / "DTDsimulation_results" / "fibercup_lte_ste_merged" / "dtd_msddelta.nii.gz"),
        "peaks_path":  str(DATA_ROOT / "fibercup" / "fodf" / "fibercup_peaks.nii.gz"),
        "asi_path":    str(DATA_ROOT / "fibercup" / "fodf" / "asi_map_wm.nii.gz"),
        "is_human": False,
        "out_root":    str(OUTPUT_ROOT / "tracking_results" / "fibercup"),
    },

    # ------------------------------------------------------------------
    # ISMRM 2015 challenge (open-source)
    # Download: http://www.tractometer.org/ismrm_2015_challenge/
    # ------------------------------------------------------------------
    "ismrm": {
        "fodf_path":   str(DATA_ROOT / "ismrm" / "fodf" / "ismrm2015_fodf_filter_asym.nii.gz"),
        "mask_path":   str(DATA_ROOT / "ismrm" / "masks" / "ismrm2015_wm.nii.gz"),
        "fa_path":     str(DATA_ROOT / "ismrm" / "anat" / "fa.nii.gz"),
        "delta_path":  str(DATA_ROOT / "ismrm" / "DTDsimulation_results" / "ismrm_lte_ste_merged" / "dtd_msddelta_wm_smooth_corrected.nii.gz"),
        "peaks_path":  str(DATA_ROOT / "ismrm" / "fodf" / "ismrm2015_peaks.nii.gz"),
        "asi_path":    str(DATA_ROOT / "ismrm" / "fodf" / "asi_map.nii.gz"),
        "is_human": True,
        "out_root":    str(OUTPUT_ROOT / "tracking_results" / "ismrm"),
    },
}

# =========================================================
# Forward simulation parameters
# =========================================================
FORWARD_PARAMS = {
    # Diffusion tensor eigenvalues (mm^2/s)
    "lambda_par":  1.7e-3,
    "lambda_perp": 0.3e-3,

    # FiberCup LTE/STE
    "fibercup_lte_bval": 650.0,
    "fibercup_lte_ndir": 64,
    "fibercup_ste_bval": 650.0,
    "fibercup_ste_ndir": 1,

    # ISMRM LTE/STE
    "ismrm_lte_bval": 1000.0,
    "ismrm_lte_ndir": 32,
    "ismrm_ste_bval": 1000.0,
    "ismrm_ste_ndir": 32,
}

# =========================================================
# Tracking parameters (shared across datasets)
# =========================================================
TRACKING_PARAMS = {
    "step_size": 0.1,           # in voxel units
    "seed_density": 1,
    "max_length_mm": 250.0,
    "probabilistic": True,
    "use_gt_mask": False,
    # Region-specific thresholds
    "region_sigma": 0.25,
    # Prototype feature vectors for region classification
    "proto_parallel": [0.8, 0.3, 0.1],   # FA=0.80, muFA=0.30, ASI=0.10
    "proto_cross":    [0.4, 0.6, 0.1],   # FA=0.40, muFA=0.60, ASI=0.10
    "proto_terminal": [0.2, 0.1, 0.9],   # FA=0.20, muFA=0.10, ASI=0.90
    "proto_bending":  [0.5, 0.3, 0.6],   # FA=0.50, muFA=0.30, ASI=0.60
}

# =========================================================
# Helper functions
# =========================================================

def get_dataset(name):
    """Get dataset configuration by name."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name]


def ensure_output_dir(path):
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path
