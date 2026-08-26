#!/bin/bash
# =========================================================
# Baseline tracking methods for FiberCup and ISMRM datasets
# =========================================================
# Runs iFOD2 (probabilistic) and SD_STREAM (deterministic)
# baseline methods on both open-source datasets.
# =========================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================"
echo "  Baseline Methods Pipeline"
echo "========================================================"

# FiberCup baselines
echo -e "\n[FiberCup] Running baselines..."
python -m src.tracking.baselines --dataset fibercup --method all

# ISMRM baselines
echo -e "\n[ISMRM] Running baselines..."
python -m src.tracking.baselines --dataset ismrm --method all

echo -e "\n========================================================"
echo "  Baseline methods complete!"
echo "========================================================"
