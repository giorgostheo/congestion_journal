# Transition-Aware Short-Term Congestion Prediction

This repository contains the feature-engineered XGBoost method used in the paper
“Transition-Aware Short-Term Traffic Congestion Prediction: Efficient Gradient
Boosting versus Spatio-Temporal Graph Neural Networks.” It reproduces the proposed
Naive, Global, Local, and Transfer experiments on the two public datasets used in
the study: PEMS-BAY and METR-LA. The deep spatio-temporal baselines (DGCRN, Graph
WaveNet, MTGNN) are evaluated in the paper with the authors’ published
implementations under identical labels, splits, target sensors, and transition-aware
threshold tuning, and are not part of this repository. The NDW rows in the paper’s
tables rely on the access-restricted NDW dataset and are likewise out of scope.

## Repository contents

```text
.
├── README.md
├── environment.yml
├── reproduce.sh
├── checksums.sha256
├── data/
│   ├── metr/
│   │   ├── METR-LA.csv
│   │   └── adj_mx_METR-LA.pkl
│   └── pems/
│       ├── PEMS-BAY.csv
│       └── adj_mx_bay.pkl
└── method/
    ├── shared.py
    ├── origins.py
    ├── naive.py
    ├── global_model.py
    ├── local.py
    ├── transfer.py
    └── run.sh
```

## Congestion-origin targets

The ten target sensors per dataset are the top-k congestion origins mined from
historical data by the deterministic graph procedure of Section 4.1 of the paper:
a sensor is an *active origin* at the timestamps at which it is congested, none
of its downstream neighbors is congested, and at least one upstream neighbor is
congested; sensors are ranked by the number of such timestamps, and the top-k
(k = 10) form the evaluation target set. To re-derive them:

```bash
python method/origins.py --dataset pems
python method/origins.py --dataset metr
```

The fixed target lists used by the reported experiments are embedded in
`method/shared.py` (`TARGET_SENSOR_IDS_BY_DATASET`).

## Environment

The dataset files are tracked with Git LFS. Install Git LFS before cloning, or run
`git lfs pull` after cloning, so the CSV and adjacency files are materialized rather
than left as pointer files. The reproduction wrapper verifies their SHA-256 hashes.

The recorded environment uses Python 3.12 and XGBoost 3.0.2. Create it with:

```bash
conda env create -f environment.yml
conda activate transition-aware-congestion
```

XGBoost training uses CUDA by default. The installed XGBoost build and host driver
must support GPU execution. To force CPU execution, set `CUDA_VISIBLE_DEVICES=-1`;
this should preserve the experimental logic but will be substantially slower and
will not reproduce the paper's training-time measurements.

## Reproduce the experiments

Run both public datasets:

```bash
./reproduce.sh
```

Run one dataset:

```bash
./reproduce.sh --dataset pems
./reproduce.sh --dataset metr
```

The wrapper verifies the Python dependencies and input checksums before starting.
It then runs, in order, the three naive predictors and the Global, Local, and
Transfer XGBoost regimes for horizons 10, 20, 30, 40, 50, and 60 minutes. Learned
models are trained with transition weights 1 and 3, and validation thresholds are
selected using TRSP, matching the reported protocol.

The full run performs feature selection and hyperparameter search and can take
several hours. The paper's timing measurements were collected on an NVIDIA A100;
accuracy should be close on another supported GPU, but wall-clock and inference
measurements are hardware-dependent.

## Outputs

Per-sensor results and intermediate transfer artifacts are written under:

```text
method/results/pems/
method/results/metr/
```

Aggregate accuracy and training-time CSVs are written under:

```text
results/
```

Every run uses a shared timestamp so the output files from all stages can be
identified as one experiment. Generated results are ignored by Git.

## Reproduction controls

The defaults used by `reproduce.sh` include:

- chronological 70%/10%/20% train/validation/test splits;
- the ten congestion-origin target sensors per dataset (derivation:
  `method/origins.py`; fixed lists: `method/shared.py`);
- congestion threshold `0.65 × training-split q90`;
- zero-speed missing-value handling and removal of incomplete target tails;
- TRSP validation selection with window 1 and transition weight 0.5;
- transition-training weights 1 and 3;
- random seed 42; and
- all feature families enabled.

Advanced configuration is available through the environment variables documented
by `method/run.sh --help`. Changing those variables defines a different experiment
from the paper reproduction.
