#!/bin/bash
# =========================================================
# ISMRM 2015 Pipeline: Tracking -> Tractometer Scoring
# =========================================================
# Runs the ISMRM 2015 challenge pipeline:
#   1. Run MS-Track per-bundle tracking with muFA/ASI guidance
#   2. Run baseline methods (iFOD2, SD_STREAM)
#   3. Score all submissions using the ISMRM 2015 tractometer
#
# Prerequisites:
#   - Set ASIDTD_DATA_ROOT to your data directory, or edit config.py
#   - Download ISMRM 2015 data from
#     http://www.tractometer.org/ismrm_2015_challenge/
# =========================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Data root (set via environment variable or edit config.py)
DATA_ROOT="${ASIDTD_DATA_ROOT:-$HOME/data/tractography}"
DATASET="ismrm"
ISMRM_DATA="${DATA_ROOT}/ismrm"

# Input paths (adjust to your local data layout)
FODF="${ISMRM_DATA}/fodf/ismrm2015_fodf.nii.gz"
WM_MASK="${ISMRM_DATA}/masks/ismrm2015_wm.nii.gz"
DELTA_MAP="${ISMRM_DATA}/DTDsimulation_results/ismrm_lte_ste_merged/dtd_msddelta_wm_smooth_corrected.nii.gz"
PEAKS="${ISMRM_DATA}/fodf/ismrm2015_peaks.nii.gz"
ASI="${ISMRM_DATA}/fodf/asi_map.nii.gz"
BUNDLE_MASKS="${ISMRM_DATA}/scoring_data/masks/bundles/2mm"
SCORING_DATA="${ISMRM_DATA}/scoring_data"
TRACKING_OUT="${PROJECT_ROOT}/output/tracking_results/ismrm"

echo "========================================================"
echo "  ISMRM 2015 Pipeline"
echo "  DATA_ROOT: ${DATA_ROOT}"
echo "========================================================"

# Verify data exists
if [ ! -f "$FODF" ]; then
    echo "ERROR: fODF not found at ${FODF}"
    echo "Please download ISMRM 2015 data and set ASIDTD_DATA_ROOT."
    echo "  export ASIDTD_DATA_ROOT=/path/to/your/data"
    exit 1
fi

# Step 1: Run MS-Track (FA + muFA + ASI all on)
echo -e "\n[Step 1] Running MS-Track (FA+muFA+ASI all on)..."
python -m src.tracking.ms_track \
    --dataset ismrm \
    --fa_mod --delta_mod --asi_mod \
    --output_dir "${TRACKING_OUT}/ms_track_optimized"

# Step 2: Run MS-Track ablation (muFA only, no ASI, no FA)
echo -e "\n[Step 2] Running MS-Track (muFA only)..."
python -m src.tracking.ms_track \
    --dataset ismrm \
    --delta_mod --no_fa_mod --no_asi_mod \
    --output_dir "${TRACKING_OUT}/ms_track_delta_only"

# Step 3: Run MS-Track ablation (no modifiers)
echo -e "\n[Step 3] Running MS-Track (no modifiers)..."
python -m src.tracking.ms_track \
    --dataset ismrm \
    --no_fa_mod --no_delta_mod --no_asi_mod \
    --output_dir "${TRACKING_OUT}/ms_track_no_modifiers"

# Step 4: Run baselines
echo -e "\n[Step 4] Running baseline methods..."
python -m src.tracking.baselines \
    --dataset "$DATASET" --method all

# Step 5: Merge per-bundle tractograms and score
echo -e "\n[Step 5] Scoring submissions..."

# Helper: merge per-bundle tck files into one
merge_bundles() {
    local bundle_dir="$1"
    local output_tck="$2"
    local tmp_dir=$(mktemp -d)
    for tck in "${bundle_dir}"/*.tck; do
        [ -f "$tck" ] || continue
        local name=$(basename "$tck" .tck)
        python -c "
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import nibabel as nib
ref = nib.load('$FODF')
sft = load_tractogram('$tck', ref, to_space=Space.RASMM)
save_tractogram(sft, '${tmp_dir}/${name}.trk')
" 2>/dev/null
    done
    # Merge using mrtrix
    if command -v tckedit &> /dev/null; then
        tckedit "${tmp_dir}"/*.trk "$output_tck" -force 2>/dev/null || true
    fi
    rm -rf "$tmp_dir"
}

for run_dir in "${TRACKING_OUT}/ms_track_optimized" \
               "${TRACKING_OUT}/ms_track_delta_only" \
               "${TRACKING_OUT}/ms_track_no_modifiers" \
               "${TRACKING_OUT}/baseline_runs/ifod2" \
               "${TRACKING_OUT}/baseline_runs/sdstream"; do
    tck="${run_dir}/${DATASET}.tck"
    # If per-bundle tcks exist, merge them
    if [ ! -f "$tck" ] && ls "${run_dir}"/*.tck &>/dev/null; then
        echo "  Merging per-bundle tractograms in ${run_dir}..."
        merge_bundles "$run_dir" "$tck"
    fi
    if [ -f "$tck" ]; then
        echo "  Scoring: $tck"
        python -m src.evaluation.ismrm_score \
            "$tck" "$SCORING_DATA" "${run_dir}/tractometer_out" \
            --compute_ic_ib --save_full_vc -f
    fi
done

# Step 6: Count spurious fibers
echo -e "\n[Step 6] Counting spurious fibers..."
for run_dir in "${TRACKING_OUT}/ms_track_optimized" \
               "${TRACKING_OUT}/baseline_runs/ifod2" \
               "${TRACKING_OUT}/baseline_runs/sdstream"; do
    tck="${run_dir}/${DATASET}.tck"
    if [ -f "$tck" ]; then
        echo "  Spurious: $tck"
        python -m src.evaluation.spurious_fibers \
            --tck "$tck" \
            --mask_dir "$BUNDLE_MASKS" \
            --out "${run_dir}/spurious_fiber_stats.json"
    fi
done

echo -e "\n========================================================"
echo "  ISMRM 2015 pipeline complete!"
echo "========================================================"
