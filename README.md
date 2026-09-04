# ASIDTD-Track: DTD-Guided Multi-Scale Tractography

MS-Track (Multi-Scale Tracking) is a tractography algorithm that integrates
microstructural information from Diffusion Tensor Distribution (DTD) inversion
to guide streamline propagation. Three types of microstructural modifiers are used:

1. **FA** (Fractional Anisotropy) - derived from LTE (Linear Tensor Encoding) signal
2. **μFA** (microscopic fractional anisotropy) - microscopic diffusion anisotropy from DTD inversion
3. **ASI** (Asymmetry Index) - from asymmetric fODF peak analysis

> **Note:** The a-fODF (asymmetric fiber Orientation Distribution Function) and ASI
> maps used in this pipeline are pre-computed by the unsupervised deep-learning
> model from [a-fODFs-recursive-net](https://github.com/gaunny/a-fODFs-recursive-net).
> This repository consumes those pre-computed NIfTI outputs as input; it does not
> re-estimate a-fODFs or ASI internally. μFA is estimated externally using the
> [md-dmri](https://github.com/daniel-topgaard/md-dmri) MATLAB framework from the
> full DTD acquisition (LTE and STE b-tensors at b = 100, 700, 1400, 2000 s/mm²
> with 6/6/12/16 directions per shell). The DTD is estimated with a Monte Carlo
> non-negative least squares scheme: 200 axisymmetric tensor nodes (nin = 200)
> randomly initialized on a log-uniform diffusivity grid (5×10⁻¹² – 5×10⁻⁹ m²/s,
> uniform orientations), refined by 20 proliferation iterations (each adding 200
> random nodes followed by NNLS weight re-estimation) and 20 extinction iterations
> (node selection and Gaussian mutation with relative diffusivity perturbation 0.1
> and angular perturbation 0.1·2π), finally retaining the nout = 20 highest-weighted
> components. FA is estimated from the linear-encoded diffusion data using FSL
> `dtifit` (v6.0.5); the `compute_fa.py` script in this repo provides an equivalent
> DIPY-based computation for users without FSL.

The tracker classifies each white matter voxel into one of four geometric regions
(parallel, crossing, terminal, bending) and applies region-specific direction
filtering and score boosting.

## Repository Structure

```
ASIDTD-Track/
├── config.py                          # Centralized path & parameter configuration
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore
├── src/
│   ├── forward_simulation/
│   │   ├── dtd_forward.py             # LTE+STE forward simulation from ground truth
│   │   ├── lte_signal_life.py         # LTE signal simulation using LiFE model
│   │   ├── ste_signal_estimate.py     # STE signal simulation (multi-compartment)
│   │   └── merge_lte_ste.py           # Merge LTE+STE signals and metadata
│   ├── preprocessing/
│   │   └── compute_fa.py              # Compute FA from LTE signal
│   ├── fiber_classification/
│   │   └── region_classification.py  # Voxel region classification (4 types)
│   ├── tracking/
│   │   ├── ms_track.py                # Unified MS-Track tracker (all datasets)
│   │   ├── track_fibercup.py          # Deprecated: use ms_track.py --dataset fibercup
│   │   ├── track_ismrm.py             # Deprecated: use ms_track.py --dataset ismrm
│   │   └── baselines.py              # iFOD2 and SD_STREAM baselines
│   ├── evaluation/
│   │   ├── fibercup_gt_masks.py       # Generate FiberCup GT region masks
│   │   ├── fibercup_score.py          # FiberCup region-based Dice scoring
│   │   ├── ismrm_score.py             # ISMRM 2015 tractometer scoring
│   │   ├── spurious_fibers.py         # Spurious fiber counting
│   │   └── runtime_summary.py         # Runtime comparison report
│   └── utils/
│       ├── fiber_analysis.py          # Streamline geometric analysis
│       └── trk2vtk2tck.py             # Tractogram format conversion
└── scripts/
    ├── run_fibercup_pipeline.sh       # Complete FiberCup pipeline
    ├── run_ismrm_pipeline.sh          # Complete ISMRM pipeline
    └── run_baselines.sh               # Baseline methods on both datasets
```

## Installation

### Prerequisites

- Python >= 3.8
- [MRtrix3](https://www.mrtrix.org/) (for `tckmap`, `tckedit` commands)
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (v6.0.5, for `dtifit` FA estimation)
- [TractSeg](https://github.com/MIC-DKFZ/TractSeg) (for bundle-specific mask generation in clinical data)
- [ISMRM 2015 Challenge Scoring](https://github.com/scilus/ismrm_2015_tractography_challenge_scoring) (for ISMRM evaluation)
- [a-fODFs-recursive-net](https://github.com/gaunny/a-fODFs-recursive-net) (for pre-computing a-fODF peaks and ASI maps)
- [md-dmri](https://github.com/daniel-topgaard/md-dmri) (MATLAB, for pre-computing μFA via DTD inversion)

### Setup

```bash
git clone <repository_url>
cd ASIDTD-Track
pip install -r requirements.txt
```

## Data Download (Required)

This repository does **not** include any data. You must download the
open-source phantom datasets yourself:

| Dataset | Description | Download |
|---------|-------------|----------|
| **FiberCup** | Synthetic phantom with known ground truth | [tractometer.org/fibercup](https://www.tractometer.org/fibercup/) |
| **ISMRM 2015** | ISMRM 2015 tractography challenge | [tractometer.org/ismrm_2015_challenge](http://www.tractometer.org/ismrm_2015_challenge/) |

### Configure Data Paths

After downloading, set the `ASIDTD_DATA_ROOT` environment variable to point
to the directory containing the datasets:

```bash
export ASIDTD_DATA_ROOT=/path/to/your/data
```

Or edit `DATA_ROOT` in `config.py` directly.

Expected directory layout under your data root:

```
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
    ├── DTDsimulation_results/
    │   └── ismrm_lte_ste_merged/
    │       └── dtd_msddelta_wm_smooth_corrected.nii.gz
    └── scoring_data/                  # ISMRM tractometer scoring package
        ├── gt_bundles_attributes.json
        └── masks/
```

**Note:** Clinical data used in the manuscript are **not**
publicly available due to data sharing restrictions. The pipeline can be applied
to any dMRI dataset by adding entries to `config.py`.

## Usage

### Quick Start: Run MS-Track on FiberCup

```bash
python -m src.tracking.ms_track --dataset fibercup --fa_mod --delta_mod --asi_mod
```

### Quick Start: Run Baselines

```bash
python -m src.tracking.baselines --dataset fibercup --method all
python -m src.tracking.baselines --dataset ismrm --method all
```

### Full Pipeline

```bash
# FiberCup: simulation -> tracking -> scoring
bash scripts/run_fibercup_pipeline.sh

# ISMRM: per-bundle tracking -> tractometer scoring
bash scripts/run_ismrm_pipeline.sh
```

### Ablation Study

MS-Track supports independent toggling of each modifier:

```bash
# FA + μFA + ASI (full model)
python -m src.tracking.ms_track --dataset fibercup --fa_mod --delta_mod --asi_mod

# FA + μFA only
python -m src.tracking.ms_track --dataset fibercup --fa_mod --delta_mod --no_asi_mod

# FA only
python -m src.tracking.ms_track --dataset fibercup --fa_mod --no_delta_mod --no_asi_mod

# No modifiers (pure fODF tracking)
python -m src.tracking.ms_track --dataset fibercup --no_fa_mod --no_delta_mod --no_asi_mod
```

### Custom Datasets

Add new dataset entries to `config.py`:

```python
DATASETS["my_data"] = {
    "fodf_path":  "/path/to/fodf.nii.gz",
    "mask_path":  "/path/to/wm_mask.nii.gz",
    "fa_path":    "/path/to/fa.nii.gz",
    "delta_path": "/path/to/dtd_msddelta.nii.gz",
    "peaks_path": "/path/to/peaks.nii.gz",
    "asi_path":   "/path/to/asi.nii.gz",
    "is_human":   True,
    "out_root":   "output/tracking_results/my_data",
}
```

Then run:
```bash
python -m src.tracking.ms_track --dataset my_data --fa_mod --delta_mod --asi_mod
```

## Pipeline Overview

### 1. Forward Simulation

Simulate LTE and STE signals from ground truth tractograms:

```bash
python -m src.forward_simulation.lte_signal_life \
    --gt_tck <ground_truth.tck> --wm_mask <wm.nii.gz> \
    --grad_file <directions.txt> --output_dir <out/> --bval 650

python -m src.forward_simulation.ste_signal_estimate \
    --gt_tck <ground_truth.tck> --wm_mask <wm.nii.gz> \
    --output_dir <out/> --bval 650

python -m src.forward_simulation.merge_lte_ste \
    --lte_dir <lte_out/> --ste_dir <ste_out/> --output_dir <merged_out/>
```

### 2. Preprocessing

```bash
# Compute FA from LTE signal
python -m src.preprocessing.compute_fa --data <merged.nii.gz> --output <fa.nii.gz>
```

### 3. Tracking

```bash
# MS-Track (main tracker)
python -m src.tracking.ms_track --dataset fibercup --fa_mod --delta_mod --asi_mod

# ISMRM 2015 tracking (unified)
python -m src.tracking.ms_track --dataset ismrm --fa_mod --delta_mod --asi_mod

# Baselines
python -m src.tracking.baselines --dataset fibercup --method all
```

### 4. Evaluation

```bash
# FiberCup region scoring
python -m src.evaluation.fibercup_score \
    --tck <prediction.tck> --gt_dir <gt_masks/> --wmmask <wm.nii.gz>

# ISMRM tractometer scoring
python -m src.evaluation.ismrm_score <prediction.tck> <scoring_data/> <output/>

# Spurious fiber counting
python -m src.evaluation.spurious_fibers \
    --tck <prediction.tck> --mask_dir <bundle_masks/>

# Runtime comparison
python -m src.evaluation.runtime_summary --root <results_dir/>

# Geometric analysis
python -m src.utils.fiber_analysis --file <tractogram.tck>
```


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- FiberCup phantom: [Tractometer](https://www.tractometer.org/fibercup/)
- ISMRM 2015 Challenge: [Tractometer](http://www.tractometer.org/ismrm_2015_challenge/)
- DIPY: [dipy.org](https://dipy.org/)
- MRtrix3: [mrtrix.org](https://www.mrtrix.org/)
- a-fODF estimation: [a-fODFs-recursive-net](https://github.com/gaunny/a-fODFs-recursive-net)
- md-dmri (DTD inversion): [github.com/daniel-topgaard/md-dmri](https://github.com/daniel-topgaard/md-dmri)
- FSL: [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
- TractSeg: [github.com/MIC-DKFZ/TractSeg](https://github.com/MIC-DKFZ/TractSeg)
