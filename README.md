<div align="center">

# T-BiAS
### Hybrid TCN–BiGRU–Attention and Semantic XGBoost for Near-Failure Detection

**NASA C-MAPSS · FD001–FD004 · Predictive Maintenance · Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-14833B)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-1D5B9B)](https://data.nasa.gov/)

*A research implementation for precision-aware, engine-level near-failure classification.*

</div>

---

## Overview

**T-BiAS** combines a temporal feature encoder with interpretable statistical and trend descriptors to identify turbofan engines approaching failure. It operates on multivariate **40-cycle windows** and evaluates near-failure status at the **final observed cycle** of each test engine.

The pipeline has two complementary branches: a **TCN → BiGRU → attention pooling** branch that learns a 128-dimensional temporal embedding, and a **semantic-feature** branch that computes 204 descriptors. Their concatenation forms a **332-dimensional** representation for a five-member XGBoost ensemble.

> **Research scope:** This is near-failure *classification* with the positive class defined as **RUL ≤ 30 cycles**. It is not a remaining-useful-life regression model or a demonstration of real-world aircraft deployment.

## Architecture

```text
                 NASA C-MAPSS sensor streams
                            │
                 40-cycle input window
                   (40 time steps × 17)
                            │
              ┌─────────────┴─────────────┐
              │                           │
       Temporal encoder             Semantic features
              │                           │
      Dilated causal TCN          12 descriptors × 17
              │                           │
           2-layer                    204-D vector
            BiGRU                         │
              │                           │
       Attention pooling                  │
              │                           │
       128-D embedding                    │
              └─────────────┬─────────────┘
                            │
                   Concatenation (332-D)
                            │
                  5-model XGBoost ensemble
                            │
                 Validation-selected threshold
                            │
                 Normal / Near-failure
```

## Experimental protocol

| Component | Configuration |
|:--|:--|
| Dataset | NASA C-MAPSS FD001, FD002, FD003, FD004 |
| Inputs | 3 operating settings + 14 selected sensors |
| Window / stride | 40 cycles / 1 cycle |
| Positive class | RUL ≤ 30 |
| Training / validation | 80% / 20%, split by engine |
| Temporal encoder | TCN (64–128–128–64), 2-layer BiGRU, attention |
| Temporal representation | 128 dimensions |
| Semantic representation | 204 dimensions (12 descriptors × 17 channels) |
| Fused representation | 332 dimensions |
| Classifier | Ensemble of 5 XGBoost models |
| Threshold selection | Maximize validation F1 subject to precision ≥ 0.90 when feasible |
| Test evaluation | Last available 40-cycle window of each test engine |
| Default seed | 42 |

**Threshold policy.** Candidate thresholds are derived from the validation precision–recall curve. If no candidate meets the 0.90 precision requirement, the implementation records a fallback. Thresholds are **not tuned on the test set**.

**Implementation-specific note.** The supplied script does **not** implement isotonic probability calibration. Please ensure the manuscript describes the exact version of the code and experiments being released.

## Repository structure

```text
T-BiAS-CMAPSS/
├── README.md                # Project overview and instructions
├── requirements.txt         # Direct Python dependencies
├── CITATION.cff             # Software citation metadata
├── .gitignore
├── src/
│   └── t_bias_cmapss.py     # Research implementation
└── data/
    └── README.md            # Dataset preparation instructions
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/begum26/T-BiAS-CMAPSS.git
cd T-BiAS-CMAPSS
```

### 2. Install dependencies

Use a Python environment with a compatible PyTorch installation:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

For Windows, activate the environment with `.venv\Scripts\activate` instead. GPU acceleration is used when CUDA is available; otherwise the script runs on CPU.

### 3. Prepare NASA C-MAPSS

Download the **NASA C-MAPSS turbofan engine degradation simulation dataset** from its official distribution source. The dataset is **not redistributed** in this repository.

Place the files for each subset together in the configured data directory:

```text
data/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

See [`data/README.md`](data/README.md) for the expected filenames. The loader also searches `/content`, `/content/data`, `/content/CMAPSSData`, and the current working directory.

### 4. Configure paths and run

The script was originally configured for **Google Colab**. Before running locally, edit the `CFG` dictionary in [`src/t_bias_cmapss.py`](src/t_bias_cmapss.py):

```python
"data_dir": "./data",
"artifact_dir": "./artifacts",
"datasets": ["FD001", "FD002", "FD003", "FD004"],
```

Then execute:

```bash
python src/t_bias_cmapss.py
```

To run only one subset, set `"datasets": ["FD001"]`. Full experiments can be computationally expensive, especially the XGBoost hyperparameter search.

## Reproducibility

The implementation includes fixed random seeds, engine-disjoint validation, training-only preprocessing, and validation-based threshold selection. The default configuration uses **60 encoder epochs**, **batch size 256**, **learning rate 1e−3**, and a **five-member XGBoost ensemble**.

Generated experiment outputs and model artifacts are written to `CFG["artifact_dir"]`. Exact metrics may vary with hardware, library versions, and nondeterministic operations.

> **Verification status:** This repository package has been prepared from the supplied source code, but the full experiments have **not** been rerun or independently verified as part of packaging. Before claiming exact reproduction of published tables, run the released script and compare its outputs with the manuscript.

## Results

Results are intentionally **not hard-coded into this README** until they are verified against this exact release. For reported performance, consult the accompanying manuscript and the output artifacts generated by the script. In particular, check that any reported calibration method, threshold-selection rule, and multi-seed statistics correspond to the released code.

## Citation

If you use this implementation, please cite the associated paper once published and the archived software release (preferably with a Zenodo DOI).

GitHub supports the **“Cite this repository”** button through [`CITATION.cff`](CITATION.cff). The metadata should be updated with the publication details and release DOI when available.

**Repository:** https://github.com/begum26/T-BiAS-CMAPSS

## Authors

- **Begüm Özcan**

*Additional contributors and institutional affiliations should be added after confirmation by the respective authors.*

## License and data rights

No software license has been selected for this package. Reuse rights are therefore **not granted by default**; choose a license after confirming permission from all relevant rights holders. The NASA dataset is distributed separately under its own terms.

## Acknowledgment

This repository supports research on near-failure detection and predictive maintenance using the NASA C-MAPSS benchmark.
