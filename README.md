# AC-PFL: Adaptive Clustered Personalized Federated Learning

Code and experiment logs for a thesis/journal project on **AC-PFL**, a
clustered personalized federated learning method, evaluated against five
baselines on the NASA C-MAPSS turbofan degradation (Remaining Useful Life)
datasets.

## Methods compared

| Method | Description |
|---|---|
| FedAvg | Standard federated averaging baseline |
| FedProx | FedAvg with a proximal term for client drift |
| FedPer | Personalization via locally-kept output heads |
| Ditto | Personalized + global model trained jointly |
| CFL | Clustered federated learning (hard client clustering) |
| **AC-PFL** | Adaptive clustered personalized FL (this work) |

## Datasets

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/) turbofan
degradation simulation data, subsets **FD001–FD004** (varying operating
conditions and fault modes). Each engine's sensor trajectory is used to
predict Remaining Useful Life (RUL), capped at 130 cycles (`RUL_CAP` in
[`src/config.py`](src/config.py)) following the standard piecewise-linear RUL
convention.

## Repository structure

```
├── src/                       # Core, reusable implementation
│   ├── config.py              #   dataset paths / registry
│   ├── utils.py                #   data loading, engine clustering, condition-regime labeling
│   ├── preprocess.py          #   RUL calculation, feature preprocessing
│   └── run_experiment.py      #   FedAvg / FedProx / FedPer / Ditto / CFL / AC-PFL training loops
├── notebooks/
│   ├── main_experiments.ipynb    # FD001 & FD004 experiment runs (incl. condition-subset diagnostics)
│   └── fd002_experiments.ipynb   # FD002 experiment runs (archived, older src snapshot — see note inside)
├── results/
│   ├── results_summary.csv    # mean ± std of Test MAE / NASA score per method per dataset
│   ├── parsed_results.csv     # every individual run (146 total), by method/dataset/seed
│   └── raw_logs/              # original Flower training logs (one file per dataset)
└── requirements.txt
```

## Running an experiment

The notebooks were developed on Kaggle, where `src/*.py` files are written to
the working directory via `%%writefile` before being imported. To run this
code elsewhere:

```bash
pip install -r requirements.txt
```

```python
import sys
sys.path.insert(0, "src")

from run_experiment import run_simulation, run_acpfl

# Baselines
result = run_simulation("fedavg", "FD001", seed=42)

# AC-PFL
result = run_acpfl("FD004", seed=42, num_clusters=2)
```

You will also need the C-MAPSS dataset files (`train_FD00X.txt`,
`test_FD00X.txt`, `RUL_FD00X.txt`) and to update `DATA_DIR` in
[`src/config.py`](src/config.py) to point at them.

## Results

These are the results as reported in the paper (Table 2), reproduced here
verbatim as the authoritative source. Mean ± sample standard deviation (ddof=1)
across seeds; seed counts vary by comparison (see paper Sections 4.6, 8.1).
Subsets are grouped by operating-condition heterogeneity level (low: FD001,
FD003; high: FD002, FD004) — see [Manuscript](#manuscript) below for the full
statistical analysis (paired significance tests, mixed-effects model, ablation).

> **Note on `results/`:** [`results/parsed_results.csv`](results/parsed_results.csv)
> and [`results/results_summary.csv`](results/results_summary.csv) are derived
> from the four raw logs in [`results/raw_logs/`](results/raw_logs/) and are
> provided for inspecting individual seed-level runs. They closely match, but
> are not byte-identical to, Table 2 below — the logs are missing a small
> number of runs present in the paper's final dataset (e.g. FD002 has 9 AC-PFL
> seeds here vs. 10 in the paper; a few FD004 baselines differ slightly). Treat
> **Table 2 in the paper as the results of record**; the CSVs are a useful but
> not fully reconciled supplementary artifact pending the missing seed logs.

#### FD001 (1 condition, 1 fault mode — low heterogeneity), n = 5

| Method | Test MAE | NASA Score |
|---|---|---|
| FedAvg | 11.87 ± 0.73 | 375.8 ± 48.5 |
| FedProx | 11.49 ± 0.72 | 371.7 ± 75.8 |
| FedPer | 12.74 ± 1.12 | 434.3 ± 115.8 |
| **AC-PFL** | 12.88 ± 0.93 | 507.2 ± 209.3 |
| Ditto | 14.44 ± 1.14 | 588.0 ± 260.9 |
| CFL | 14.74 ± 1.54 | 1,179.7 ± 565.4 |

#### FD003 (1 condition, 2 fault modes — low heterogeneity), n = 5

| Method | Test MAE | NASA Score |
|---|---|---|
| FedAvg | 12.11 ± 1.30 | 640.0 ± 221.3 |
| FedProx | 12.61 ± 1.76 | 1,037.2 ± 712.2 |
| FedPer | 14.02 ± 1.54 | 907.1 ± 361.3 |
| **AC-PFL** | 13.65 ± 1.04 | 4,300.8 ± 5,510.5 |
| Ditto | 14.28 ± 0.76 | 9,051.0 ± 4,938.4 |
| CFL | 14.25 ± 0.69 | 13,213.0 ± 3,557.8 |

#### FD002 (6 conditions, 1 fault mode — high heterogeneity), n = 10 (AC-PFL/FedAvg/FedProx), n = 5 (others)

| Method | Test MAE | NASA Score |
|---|---|---|
| FedAvg | 19.18 ± 1.35 | 11,941.8 ± 3,725.5 |
| FedProx | 19.41 ± 1.10 | 22,299.7 ± 24,963.9 |
| FedPer | 18.86 ± 0.70 | 11,149.8 ± 5,495.6 |
| **AC-PFL** | 19.66 ± 0.98 | 8,648.0 ± 3,490.7 |
| Ditto | 21.18 ± 0.76 | 13,411.3 ± 4,769.3 |
| CFL | 22.85 ± 1.27 | 37,638.2 ± 22,627.9 |

#### FD004 (6 conditions, 2 fault modes — high heterogeneity), n = 10 (AC-PFL/FedAvg/FedProx), n = 5 (others)

| Method | Test MAE | NASA Score |
|---|---|---|
| FedAvg | 22.89 ± 0.77 | 47,039.8 ± 29,816.4 |
| FedProx | 22.57 ± 0.83 | 67,975.8 ± 37,437.8 |
| FedPer | 22.14 ± 0.89 | 39,385.0 ± 22,622.4 |
| **AC-PFL** | 22.17 ± 1.06 | 28,616.1 ± 13,048.1 |
| Ditto | 23.13 ± 0.52 | 31,631.6 ± 9,788.1 |
| CFL | 24.99 ± 0.84 | 99,979.6 ± 54,281.1 |

**Summary (from the paper):** AC-PFL's advantage is heterogeneity-conditional,
not universal. Under low operating-condition heterogeneity (FD001, FD003), it
provides no measurable benefit over FedAvg/FedProx and carries a real MAE cost
— an ablation shows removing clustering entirely (FedPer) actually *improves*
NASA score by 14.4% on FD001, since fixed clustering fragments an
already-small per-cluster client pool with no offsetting benefit. Under high
operating-condition heterogeneity (FD002, FD004), AC-PFL achieves the best
NASA score of all six methods — most consistently significant against FedProx
— while MAE rankings remain mixed, indicating it specifically reduces
systematic late-prediction errors rather than improving average accuracy
uniformly. A pooled mixed-effects analysis localizes this to
**operating-condition** diversity specifically (β = −0.488, p < 0.01), not
fault-mode diversity (β = −0.041, p = 0.71). CFL is consistently the
worst-performing and highest-variance method throughout, reflecting the cost
of one-shot, non-adaptive clustering.

## Manuscript

The full paper — including significance testing (Wilcoxon/t-test), the
pooled mixed-effects heterogeneity analysis, the component-level ablation
study, and a discussion of limitations — is:

> Kaiser, F., Hasan, E., & Rakib, S. M. *When Does Personalization Help? A
> Heterogeneity-Conditional Evaluation of Adaptive Clustered Federated
> Learning for Turbofan Engine Prognostics.* Preprint submitted to Elsevier.

A preliminary version of AC-PFL (static clustering, single dataset, MAE only)
was presented at ICCIT 2025: Kaiser, F. *Adaptive clustered personalized
federated learning for non-IID remaining useful life prediction in edge-based
industrial systems.* ICCIT 2025, pp. 365–370. doi:
10.1109/ICCIT68739.2025.11491433.

## Notes on the code

- `run_experiment.py`'s `run_acpfl` intentionally keeps its own copy of the
  data-loading logic rather than calling the shared `load_dataset()` — see
  the docstring for why.
- Condition-regime filtering (`condition_subset` in `load_dataset` /
  `run_acpfl`) recovers operating-condition labels via a *separate* KMeans
  clustering from the one used for client partitioning. The paper documents
  (Section 4.1) that this approach to constructing intermediate heterogeneity
  levels was ultimately not usable for the main results: engines in
  FD002/FD004 cycle through all six operating conditions within a single
  trajectory, so the great majority of engines majority-vote to the same
  dominant condition, leaving too few engines to build a usable subset
  federation.
  This code is retained as a diagnostic used during development.
- `RUL_CAP = 130` in `config.py` (not the 125-cycle cap used in some prior
  C-MAPSS studies) — this affects the magnitude, but not the direction, of
  NASA score comparisons; see paper Section 4.3.
