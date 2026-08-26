#!/bin/bash
# =========================================================
# FiberCup Pipeline: Forward Simulation -> Tracking -> Scoring
# =========================================================
# Runs the complete FiberCup experiment pipeline:
#   1. Simulate LTE and STE signals from ground truth
#   2. Merge LTE+STE signals
#   3. Compute FA from LTE signal
#   4. Run MS-Track with all three modifiers (FA, muFA, ASI)
#   5. Run baseline methods (iFOD2, SD_STREAM)
#   6. Score all tractograms
#
# Prerequisites:
#   - Set ASIDTD_DATA_ROOT to your data directory, or edit config.py
#   - Download FiberCup data from https://www.tractometer.org/fibercup/
# =========================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Data root (set via environment variable or edit config.py)
DATA_ROOT="${ASIDTD_DATA_ROOT:-$HOME/data/tractography}"
DATASET="fibercup"
FC_DATA="${DATA_ROOT}/fibercup"

# Input paths (adjust to your local data layout)
GT_TCK="${FC_DATA}/gt/fibercup_bundle_all.tck"
WM_MASK="${FC_DATA}/masks/fibercup_wm.nii.gz"
GRAD_FILE="${FC_DATA}/grad/fibercup_dirs.txt"
SIM_DIR="${FC_DATA}/DTDsimulation_results"
SCORING_DIR="${FC_DATA}/scoring_data"
TRACKING_OUT="${PROJECT_ROOT}/output/tracking_results/fibercup"

echo "========================================================"
echo "  FiberCup Pipeline"
echo "  DATA_ROOT: ${DATA_ROOT}"
echo "========================================================"

# Verify data exists
if [ ! -f "$WM_MASK" ]; then
    echo "ERROR: White matter mask not found at ${WM_MASK}"
    echo "Please download FiberCup data and set ASIDTD_DATA_ROOT."
    echo "  export ASIDTD_DATA_ROOT=/path/to/your/data"
    exit 1
fi

# Step 1: Simulate LTE signal
echo -e "\n[Step 1] Simulating LTE signal..."
python -m src.forward_simulation.lte_signal_life \
    --gt_tck "$GT_TCK" \
    --wm_mask "$WM_MASK" \
    --grad_file "$GRAD_FILE" \
    --output_dir "${SIM_DIR}/fibercup_lte_only" \
    --bval 650 --n_dir 64 \
    --prefix fibercup_lte

# Step 2: Simulate STE signal
echo -e "\n[Step 2] Simulating STE signal..."
python -m src.forward_simulation.ste_signal_estimate \
    --gt_tck "$GT_TCK" \
    --wm_mask "$WM_MASK" \
    --output_dir "${SIM_DIR}/fibercup_ste_only" \
    --bval 650 \
    --prefix fibercup_ste

# Step 3: Merge LTE+STE
echo -e "\n[Step 3] Merging LTE+STE signals..."
python -m src.forward_simulation.merge_lte_ste \
    --lte_dir "${SIM_DIR}/fibercup_lte_only" \
    --ste_dir "${SIM_DIR}/fibercup_ste_only" \
    --output_dir "${SIM_DIR}/fibercup_lte_ste_merged" \
    --lte_signal_name fibercup_lte_signal.nii.gz \
    --ste_signal_name fibercup_ste_signal.nii.gz

# Step 4: Compute FA
echo -e "\n[Step 4] Computing FA from LTE signal..."
python -m src.preprocessing.compute_fa \
    --data "${SIM_DIR}/fibercup_lte_ste_merged/data.nii.gz" \
    --output "${FC_DATA}/anat/fibercup_fa.nii.gz" \
    --n_lte 64 --bval 650

# Step 5: Run MS-Track (all modifiers enabled)
echo -e "\n[Step 5] Running MS-Track (FA+muFA+ASI)..."
python -m src.tracking.ms_track \
    --dataset "$DATASET" --fa_mod --delta_mod --asi_mod

# Step 6: Run MS-Track (ablation: FA+muFA only)
echo -e "\n[Step 6] Running MS-Track (FA+muFA only)..."
python -m src.tracking.ms_track \
    --dataset "$DATASET" --fa_mod --delta_mod --no_asi_mod

# Step 7: Run baselines
echo -e "\n[Step 7] Running baseline methods..."
python -m src.tracking.baselines \
    --dataset "$DATASET" --method all

# Step 8: Score FiberCup results
echo -e "\n[Step 8] Scoring tractograms..."
for dir in "${TRACKING_OUT}/True_True_True" "${TRACKING_OUT}/True_True_False" \
           "${TRACKING_OUT}/baseline_runs/ifod2" "${TRACKING_OUT}/baseline_runs/sdstream"; do
    tck="${dir}/${DATASET}.tck"
    if [ -f "$tck" ]; then
        echo "  Scoring: $tck"
        python -m src.evaluation.fibercup_score \
            --tck "$tck" \
            --gt_dir "$dir" \
            --wmmask "$WM_MASK"
    fi
done

echo -e "\n========================================================"
echo "  FiberCup pipeline complete!"
echo "========================================================"
