# AC-PFL: Adaptive Clustered Personalized Federated Learning

Official research code and experimental artifacts accompanying the manuscript:

> **When Does Personalization Help? A Heterogeneity-Conditional Evaluation of Adaptive Clustered Federated Learning for Turbofan Engine Prognostics**

**Manuscript status:** Under review at *Reliability Engineering & System Safety (RESS)*.

This repository contains the implementation, experimental configurations, logs, notebooks, and summarized results used to evaluate **Adaptive Clustered Personalized Federated Learning (AC-PFL)** for Remaining Useful Life (RUL) prediction under heterogeneous industrial data distributions.

The study investigates a central question:

> **When does personalization in federated learning actually help, and under what forms of heterogeneity?**

Rather than assuming that personalization or clustering is universally beneficial, the experiments evaluate how different federated learning strategies behave across the heterogeneous NASA C-MAPSS turbofan engine datasets.

---

## 1. Research Overview

Predictive maintenance systems increasingly rely on distributed industrial data, where raw sensor measurements cannot always be centralized because of privacy, communication, ownership, or deployment constraints.

Federated Learning (FL) provides a framework for training a shared model without directly exchanging raw training data. However, industrial clients can exhibit substantial **statistical heterogeneity**, meaning that a single globally shared model may not perform equally well across all clients.

This work evaluates whether **adaptive clustering combined with personalized model components** can provide advantages under such heterogeneous conditions.

The proposed **AC-PFL** framework combines:

* Federated learning
* Client clustering
* Personalized model components
* Adaptive cluster assignment/reclustering
* Non-IID industrial RUL prediction

The experimental study compares AC-PFL with several established federated learning approaches across **NASA C-MAPSS FD001–FD004**.

---

## 2. Methods Compared

The repository contains implementations and experimental results for six federated learning strategies:

| Method      | Description                                                                              |
| ----------- | ---------------------------------------------------------------------------------------- |
| **FedAvg**  | Standard federated averaging with a globally shared model                                |
| **FedProx** | FedAvg with a proximal regularization term to improve robustness to client heterogeneity |
| **FedPer**  | Personalized federated learning using shared and private model components                |
| **Ditto**   | Personalized federated learning through local personalized models and a global model     |
| **CFL**     | Clustered Federated Learning based on client similarity                                  |
| **AC-PFL**  | Adaptive Clustered Personalized Federated Learning proposed in this work                 |

---

## 3. Dataset

Experiments use the **NASA C-MAPSS turbofan engine simulation datasets**:

* FD001
* FD002
* FD003
* FD004

The datasets differ in their operating-condition and fault-mode characteristics, allowing the evaluation to cover multiple levels and forms of data heterogeneity.

The prediction task is **Remaining Useful Life (RUL) estimation** from multivariate engine sensor measurements.

### RUL Configuration

The experiments use an RUL upper cap of:

**130 cycles**

---

## 4. Experimental Design

The experiments were conducted using multiple independent random seeds.

The number of seeds is intentionally different across datasets:

| Dataset   | Independent Seeds per Method |
| --------- | ---------------------------: |
| **FD001** |                            5 |
| **FD002** |                           10 |
| **FD003** |                            5 |
| **FD004** |                           10 |

**Important:** These seed counts correspond to the experimental runs reported in the manuscript. All methods within a given dataset use the same number of seeds.

Therefore:

* FD001 → 5 runs per method
* FD002 → 10 runs per method
* FD003 → 5 runs per method
* FD004 → 10 runs per method

Reported results are presented as **mean ± standard deviation** across the corresponding independent runs.

---

## 5. Repository Structure

```text
AC_PFL_Journal_Code/
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── preprocess.py
│   └── run_experiment.py
│
├── notebooks/
│   ├── main_experiments.ipynb
│   └── fd002_experiments.ipynb
│
├── results/
│   ├── results_summary.csv
│   ├── parsed_results.csv
│   └── raw_logs/
│
├── requirements.txt
└── README.md
```

### `src/`

Contains the main implementation and experiment utilities.

* `config.py` — experiment configuration and hyperparameters
* `utils.py` — utility functions used throughout the experiments
* `preprocess.py` — dataset preprocessing and preparation
* `run_experiment.py` — experiment execution

### `notebooks/`

Contains notebooks used for running and inspecting experiments.

* `main_experiments.ipynb`
* `fd002_experiments.ipynb`

### `results/`

Contains summarized experimental results and raw experiment logs.

* `results_summary.csv`
* `parsed_results.csv`
* `raw_logs/`

---

## 6. Model Architecture

The RUL prediction model uses a recurrent neural network architecture based on LSTM layers.

The architecture consists of:

```text
Input Sequence
      │
      ▼
LSTM (64)
      │
 Layer Normalization
      │
   Dropout
      │
      ▼
LSTM (32)
      │
 Layer Normalization
      │
      ▼
Personalized / Prediction Head
      │
      ├── Dense
      ├── Dense
      ├── Dropout
      └── Dense (1)
      │
      ▼
Predicted RUL
```

The model operates on sequential sensor data and predicts the remaining useful life of the engine.

---

## 7. Federated Learning Setup

The experiments simulate a distributed industrial environment in which data are partitioned across multiple clients.

The training process proceeds through communication rounds:

1. Clients receive the relevant global/cluster model.
2. Each client performs local training using its private data.
3. Client updates are communicated to the server.
4. Federated aggregation is performed.
5. AC-PFL evaluates client similarity and cluster structure.
6. Cluster assignments can be adapted during training.
7. Personalized components remain associated with individual clients/clusters rather than being globally averaged.

This allows the framework to investigate whether clients with different data distributions benefit from receiving different model parameters.

---

## 8. AC-PFL

The proposed AC-PFL framework combines **clustering and personalization** within the federated learning process.

The central idea is that clients may not all benefit equally from a single global model.

Instead, AC-PFL seeks to:

* identify similarities between clients,
* organize similar clients into clusters,
* maintain personalized components,
* adapt cluster structure during federated training,
* and provide models that better reflect heterogeneous client distributions.

The framework is therefore intended to address situations in which **client heterogeneity determines whether personalization is useful**.

---

## 9. Evaluation Metrics

The experiments primarily evaluate predictive performance using:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and true RUL:

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

Lower values indicate better prediction accuracy.

### NASA C-MAPSS RUL Score

The NASA scoring function is also reported to capture the asymmetric cost associated with early and late RUL predictions.

Lower scores indicate better performance.

Both metrics are reported to provide complementary views of model performance.

---

# 10. Experimental Results

## FD001

| Method  |          MAE |     NASA Score |
| ------- | -----------: | -------------: |
| FedAvg  | 11.87 ± 0.73 |   375.8 ± 48.5 |
| FedProx | 11.49 ± 0.72 |   371.7 ± 75.8 |
| FedPer  | 12.74 ± 1.12 |  434.3 ± 115.8 |
| AC-PFL  | 12.88 ± 0.93 |  507.2 ± 209.3 |
| Ditto   | 14.44 ± 1.14 |  588.0 ± 260.9 |
| CFL     | 14.74 ± 1.54 | 1179.7 ± 565.4 |

**Seeds:** 5 per method.

---

## FD003

| Method  |          MAE |       NASA Score |
| ------- | -----------: | ---------------: |
| FedAvg  | 12.11 ± 1.30 |    640.0 ± 221.3 |
| FedProx | 12.61 ± 1.76 |   1037.2 ± 712.2 |
| FedPer  | 14.02 ± 1.54 |    907.1 ± 361.3 |
| AC-PFL  | 13.65 ± 1.04 |  4300.8 ± 5510.5 |
| Ditto   | 14.28 ± 0.76 |  9051.0 ± 4938.4 |
| CFL     | 14.25 ± 0.69 | 13213.0 ± 3557.8 |

**Seeds:** 5 per method.

---

## FD002

| Method  |          MAE |        NASA Score |
| ------- | -----------: | ----------------: |
| FedAvg  | 19.18 ± 1.35 |  11941.8 ± 3725.5 |
| FedProx | 19.41 ± 1.10 | 22299.7 ± 24963.9 |
| FedPer  | 18.86 ± 0.70 |  11149.8 ± 5495.6 |
| AC-PFL  | 19.66 ± 0.98 |   8648.0 ± 3490.7 |
| Ditto   | 21.18 ± 0.76 |  13411.3 ± 4769.3 |
| CFL     | 22.85 ± 1.27 | 37638.2 ± 22627.9 |

**Seeds:** 10 per method.

---

## FD004

| Method  |          MAE |        NASA Score |
| ------- | -----------: | ----------------: |
| FedAvg  | 22.89 ± 0.77 | 47039.8 ± 29816.4 |
| FedProx | 22.57 ± 0.83 | 67975.8 ± 37437.8 |
| FedPer  | 22.14 ± 0.89 | 39385.0 ± 22622.4 |
| AC-PFL  | 22.17 ± 1.06 | 28616.1 ± 13048.1 |
| Ditto   | 23.13 ± 0.52 |  31631.6 ± 9788.1 |
| CFL     | 24.99 ± 0.84 | 99979.6 ± 54281.1 |

**Seeds:** 10 per method.

---

# 11. Main Research Question

The study does **not** assume that personalization automatically improves federated learning.

Instead, it investigates the more specific question:

> **Under what types of data heterogeneity does personalization and adaptive clustering provide an advantage?**

The results indicate that the effectiveness of personalization is **heterogeneity-dependent** rather than universally superior to global federated learning.

This distinction is central to the manuscript.

---

# 12. Heterogeneity Analysis

The analysis considers different sources of heterogeneity in the C-MAPSS datasets, including:

* operating-condition heterogeneity
* fault-mode heterogeneity
* differences in client data distributions

The study evaluates whether the relationship between heterogeneity and personalization effectiveness is statistically meaningful.

The pooled analysis reported in the manuscript includes interaction terms between learning strategy and heterogeneity characteristics.

The results indicate a statistically meaningful interaction associated with operating-condition heterogeneity, whereas the corresponding fault-mode interaction was not statistically significant.

This supports the central argument that **the usefulness of personalization depends on the nature of the underlying client heterogeneity**.

---

# 13. Condition-Subset Analysis

An additional diagnostic analysis examined the possibility of separating engines according to operating-condition assignments in FD002 and FD004.

The analysis showed that individual engines can traverse multiple operating conditions during their lifetime. Consequently, assigning each engine to a single operating-condition category can produce highly imbalanced subsets rather than cleanly separated engine populations.

For this reason, intermediate condition subsets were not treated as independent engine populations for the primary analysis.

This is important when interpreting operating-condition heterogeneity in C-MAPSS and avoids overstating what the dataset can support.

---

# 14. Ablation Studies

The repository also contains analyses examining the contribution of important components of AC-PFL.

The ablation experiments investigate components including:

* clustering,
* adaptive/dynamic reclustering,
* and personalized model components.

Examples reported in the manuscript include:

* Removing clustering improved the FD001 NASA score by approximately **14.4%**, indicating that clustering is not universally beneficial.
* Removing dynamic reclustering increased the FD004 NASA score by approximately **97.4%**.
* Removing private personalized heads increased the FD004 NASA score by approximately **63.4%**.

These results support the broader finding that the contribution of individual AC-PFL components varies with dataset heterogeneity.

---

# 15. Reproducibility

To facilitate reproducibility, the repository provides:

* Source code
* Experiment notebooks
* Configuration files
* Result summaries
* Parsed results
* Raw experiment logs
* Multiple independent random seeds

The experiments use the same seed allocation as the reported manuscript results:

| Dataset | Number of Seeds |
| ------- | --------------: |
| FD001   |               5 |
| FD002   |              10 |
| FD003   |               5 |
| FD004   |              10 |

The seed count is **dataset-dependent by design**, rather than being a discrepancy between the manuscript and implementation.

---

# 16. Installation

Clone the repository:

```bash
git clone https://github.com/PseudoGod541/AC_PFL_Journal_Code.git
cd AC_PFL_Journal_Code
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 17. Running the Experiments

The primary experiment implementation is located in:

```text
src/run_experiment.py
```

Configuration parameters can be found in:

```text
src/config.py
```

The notebooks provide additional experiment-specific workflows:

```text
notebooks/main_experiments.ipynb
notebooks/fd002_experiments.ipynb
```

Results generated during experimentation are stored under:

```text
results/
```

---

# 18. Citation

If you use this repository, the experimental methodology, or AC-PFL in academic work, please cite the corresponding manuscript when it becomes publicly available.

### Journal Manuscript

**Fardin Kaiser.**
*When Does Personalization Help? A Heterogeneity-Conditional Evaluation of Adaptive Clustered Federated Learning for Turbofan Engine Prognostics.*
Manuscript under review at *Reliability Engineering & System Safety*.

### Related Conference Publication

**Fardin Kaiser.**
“Adaptive Clustered Personalized Federated Learning for Non-IID Remaining Useful Life Prediction in Edge-Based Industrial Systems.”
*2025 International Conference on Computer and Information Technology (ICCIT)*, pp. 365–370.

DOI:

```text
10.1109/ICCIT68739.2025.11491433
```

---

# 19. Repository Status

This repository corresponds to the experimental code and artifacts used for the manuscript currently under review at:

**Reliability Engineering & System Safety (RESS)**

The repository is maintained to support transparency and reproducibility of the reported experimental study.

---

## License

Please refer to the repository license for terms governing the use and redistribution of the code.
