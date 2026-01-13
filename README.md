# OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering

This repository contains the implementation of **OT-CFM** (Optimal Transport Coupled Flow Matching), a novel approach for multi-view clustering that combines flow matching with optimal transport for handling incomplete and unaligned multi-view data.

## Overview

OT-CFM addresses the challenging problem of multi-view clustering where views may be:
- **Incomplete**: Some samples have missing views
- **Unaligned**: Views lack one-to-one sample correspondences

Key features:
- Flow matching for generative modeling and missing view imputation
- Gromov-Wasserstein distance for alignment-free multi-view fusion
- End-to-end clustering with alternating optimization
- Support for multiple benchmark datasets
- **Optuna hyperparameter tuning** for optimal performance
- **External SOTA baselines** integration (MFLVC, SURE, DealMVC, GCFAggMVC, DCG, MRG-UMC, CANDY)

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or on macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
# Clone the repository
git clone https://github.com/yourusername/OT-CFM.git
cd OT-CFM

# Create virtual environment and install dependencies (one command)
uv sync

# Or install with all extras (including Optuna for tuning)
uv sync --all-extras
```

### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/yourusername/OT-CFM.git
cd OT-CFM

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install with tuning support
pip install -e ".[tuning]"
```

## Requirements

The project uses `pyproject.toml` for dependency management. Key dependencies:

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
pot>=0.9.0          # Python Optimal Transport
pandas>=2.0.0
optuna>=3.0.0       # Optional: for hyperparameter tuning
```

## Project Structure

```
OT-CFM/
├── src/
│   └── otcfm/                    # Core package
│       ├── __init__.py           # Package exports
│       ├── config.py             # Configuration dataclasses
│       ├── datasets.py           # Dataset loaders and preprocessing
│       ├── models.py             # Neural network components
│       ├── losses.py             # Loss functions (CFM, GW, clustering)
│       ├── metrics.py            # Evaluation metrics (ACC, NMI, ARI, etc.)
│       ├── ot_cfm.py             # Main OT-CFM model
│       ├── trainer.py            # Training pipeline
│       ├── baselines.py          # Internal baseline methods
│       ├── external_baselines.py # External SOTA methods wrapper
│       ├── ablation.py           # Ablation study runner
│       ├── utils.py              # Helper functions
│       └── visualization.py      # Visualization utilities
├── scripts/
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_optuna_tuning.py      # Optuna hyperparameter tuning
│   ├── tune_all_datasets.py      # Batch tuning for all datasets
│   ├── run_robustness_test.py    # Robustness testing (incomplete & unaligned data)
│   └── run_ablation.py           # Comprehensive ablation study
├── external_methods/             # External SOTA baselines (clone here)
│   ├── MFLVC/                    # CVPR 2022
│   ├── SURE/                     # TPAMI 2022
│   ├── DealMVC/                  # CVPR 2023
│   ├── GCFAggMVC/                # CVPR 2023
│   ├── 2025-AAAI-DCG/            # AAAI 2025
│   ├── MRG-UMC/                  # TNNLS 2025
│   └── 2024-NeurIPS-CANDY/       # NeurIPS 2024
├── config/                       # Tuned hyperparameters
│   └── tuned_params.json         # Optuna tuning results
├── results/                      # CSV experiment results
├── data/                         # Dataset directory
├── experiments/                  # Experiment outputs
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Quick Start

### Training OT-CFM

```python
from otcfm import OTCFM, get_default_config, MultiViewDataset, Trainer, get_dataloader
from otcfm.datasets import load_synthetic

# Load configuration
config = get_default_config()

# Load data
data = load_synthetic(n_samples=1000, n_clusters=10)
dataset = MultiViewDataset(data['views'], data['labels'])
train_loader = get_dataloader(dataset, batch_size=256)

# Create model
model = OTCFM(
    view_dims=[v.shape[1] for v in data['views']],
    latent_dim=128,
    num_clusters=10
)

# Train
trainer = Trainer(model, config.training, experiment_dir='experiments/test')
results = trainer.train(train_loader, data['labels'])

print(f"Final ACC: {results['final']['acc']:.4f}")
```

### Running Experiments via Command Line

All commands use `uv run` to automatically manage the Python environment:

```bash
# Basic training
uv run python scripts/run_experiment.py --mode train --dataset Synthetic --epochs 200

# Compare with internal baselines
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --epochs 200

# Compare with external SOTA methods (MFLVC, SURE, DealMVC, GCFAggMVC, DCG)
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --epochs 200 --include_external

# Compare with external methods only (exclude internal baselines)
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --epochs 200 --include_external --no_internal

# Run ablation study
uv run python scripts/run_experiment.py --mode ablation --dataset Coil20 --epochs 100

# Multi-dataset experiment
uv run python scripts/run_experiment.py --mode multi --datasets Synthetic Scene15 Handwritten

# Specify results directory
uv run python scripts/run_experiment.py --mode compare --dataset Handwritten --results_dir my_results
```

### Hyperparameter Tuning (Optuna)

OT-CFM supports automatic hyperparameter tuning using [Optuna](https://optuna.org/).

```bash
# Install tuning dependencies
uv add optuna

# Tune single dataset (100 trials, ~50 epochs per trial)
uv run python scripts/run_optuna_tuning.py --dataset Handwritten --n_trials 100

# Tune with fewer trials for quick testing
uv run python scripts/run_optuna_tuning.py --dataset Coil20 --n_trials 20 --tuning_epochs 30

# Tune all datasets (saves to config/tuned_params.json)
uv run python scripts/tune_all_datasets.py --n_trials 100

# Use Optuna-tuned parameters for training
uv run python scripts/run_experiment.py --dataset Handwritten --use_tuned

# Use tuned parameters with comparison mode
uv run python scripts/run_experiment.py --mode compare --dataset Handwritten --use_tuned --include_external
```

#### Robustness-Aware Tuning

For better performance in robustness tests (incomplete/unaligned data), use **robustness-aware tuning**. This optimizes hyperparameters across multiple challenging conditions simultaneously, producing parameters that work well under various missing rates and unaligned rates.

```bash
# Tune for incomplete data robustness (missing views)
uv run python scripts/run_optuna_tuning.py --dataset Scene15 --n_trials 100 --robustness incomplete

# Tune for unaligned data robustness (shuffled samples)
uv run python scripts/run_optuna_tuning.py --dataset Scene15 --n_trials 100 --robustness unaligned

# Tune for both conditions (recommended for comprehensive robustness)
uv run python scripts/run_optuna_tuning.py --dataset Scene15 --n_trials 100 --robustness both

# Use robustness-tuned parameters in robustness testing
uv run python scripts/run_robustness_test.py --dataset Scene15 --test_type incomplete \
    --use_tuned --tuned_key scene15_robust_incomplete

# Full robustness test with tuned parameters
uv run python scripts/run_robustness_test.py --dataset Scene15 --test_type both \
    --use_tuned --tuned_key scene15_robust_both --include_external
```

**Robustness tuning modes:**
| Mode | Description | Optimizes For |
|------|-------------|---------------|
| `none` | Standard tuning on complete data | Baseline performance |
| `incomplete` | Tuning across missing rates [0%, 10%, 30%, 50%] | Missing view handling |
| `unaligned` | Tuning across unaligned rates [0%, 20%, 40%] | Sample alignment |
| `both` | Combined conditions including stress tests | Overall robustness |

Tuned parameters are saved to `config/tuned_params.json` and can be reused across experiments.

**Tuned hyperparameters include:**
| Category | Parameters |
|----------|------------|
| Model Architecture | `latent_dim`, `hidden_dims`, `flow_hidden_dim`, `flow_num_layers`, `time_dim`, `ode_steps` |
| Loss Weights | `lambda_gw`, `lambda_cluster`, `lambda_recon`, `lambda_contrastive` |
| Training | `learning_rate`, `weight_decay`, `batch_size`, `dropout` |
| Kernel | `kernel_type`, `kernel_gamma` |

### Results Export

All experiment results are automatically saved to CSV files:

```bash
# Results saved to results/{dataset}_{timestamp}.csv
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --include_external

# Custom results directory
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --results_dir my_results
```

CSV format includes: Method, ACC, NMI, ARI, Purity, F1

### Handling Missing Views

```python
# Create dataset with missing views (30% missing rate)
dataset = MultiViewDataset(
    views=views,
    labels=labels,
    missing_rate=0.3
)

# Model automatically handles imputation during training
```

### Handling Unaligned Views

```python
# Create dataset with unaligned samples (20% unaligned)
dataset = MultiViewDataset(
    views=views,
    labels=labels,
    unaligned_rate=0.2
)

# GW alignment handles cross-view correspondences
```

### Robustness Testing

OT-CFM provides comprehensive robustness testing to evaluate model performance under challenging conditions. **By default, robustness tests compare OT-CFM against external SOTA methods only.**

外部方法按专长领域分类，测试时自动筛选相关方法进行公平对比：
- **Incomplete Test**: OT-CFM vs Incomplete专用方法 + 通用方法
- **Unaligned Test**: OT-CFM vs Unaligned专用方法 + 通用方法

| Category | Methods |
|----------|---------|
| Incomplete | COMPLETER, SURE, DealMVC, DCG |
| Unaligned | MRG-UMC, CANDY |
| General | MFLVC, GCFAggMVC |

#### Incomplete Data Test (Missing Views)
Tests how well the model handles missing view data at various missing rates η ∈ {0.1, 0.3, 0.5, 0.7}.

```bash
# Run incomplete data test (compares with external methods by default)
uv run python scripts/run_robustness_test.py --test_type incomplete --dataset Scene15

# Custom missing rates
uv run python scripts/run_robustness_test.py --test_type incomplete --dataset Scene15 \
    --missing_rates 0.0 0.2 0.4 0.6 0.8

# Also include internal baselines
uv run python scripts/run_robustness_test.py --test_type incomplete --dataset Scene15 \
    --include_internal --epochs 100 --num_runs 3
```

#### Unaligned Data Test (Shuffled Samples)  
Tests the model's ability to handle sample misalignment across views at shuffle rates p ∈ {0%, 20%, 40%, 60%}.

```bash
# Run unaligned data test (compares with external methods by default)
uv run python scripts/run_robustness_test.py --test_type unaligned --dataset Scene15

# Custom unaligned rates
uv run python scripts/run_robustness_test.py --test_type unaligned --dataset Scene15 \
    --unaligned_rates 0.0 0.2 0.4 0.6 0.8

# Also include internal baselines
uv run python scripts/run_robustness_test.py --test_type unaligned --dataset Scene15 \
    --include_internal --epochs 100 --num_runs 3
```

#### Run Both Tests
```bash
# Run both incomplete and unaligned tests (external methods by default)
uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 \
    --epochs 100 --num_runs 3

# Full robustness test command
uv run python scripts/run_robustness_test.py --test_type both \
    --dataset Scene15 \
    --epochs 100 \
    --num_runs 3 \
    --save_dir results/robustness

# Disable external methods (OT-CFM only)
uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 --no_external
```

#### Using Tuned Parameters for Better Robustness

For optimal robustness test results, first run robustness-aware tuning, then use those parameters:

```bash
# Step 1: Tune for robustness (run once, takes time)
uv run python scripts/run_optuna_tuning.py --dataset Scene15 --n_trials 100 --robustness both

# Step 2: Run robustness test with tuned parameters
uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 \
    --use_tuned --tuned_key scene15_robust_both --epochs 100 --num_runs 3
```

#### Output Files
Results are saved to `results/robustness/`:
- **CSV**: `{dataset}_{test_type}_{timestamp}.csv` - Tabular results
- **JSON**: `{dataset}_{test_type}_{timestamp}.json` - Structured data for analysis
- **PNG/PDF**: `{dataset}_{test_type}_{timestamp}.png` - Visualization plots
- **Combined**: `{dataset}_robustness_combined_{timestamp}.png` - 2x2 overview plot

#### Expected Results
- **Incomplete Data**: All methods degrade as missing rate increases, but OT-CFM should show the most gradual decline due to its flow-based imputation.
- **Unaligned Data**: Most methods collapse when p > 0, while OT-CFM maintains high performance due to its Landmark OT alignment mechanism - this is the key differentiator.

## Datasets

Supported datasets:
| Dataset | Views | Samples | Clusters | Description |
|---------|-------|---------|----------|-------------|
| **Synthetic** | 3 | 1000 | 10 | Generated multi-view data |
| **Handwritten** | 6 | 2000 | 10 | Handwritten digit features |
| **Coil20** | 3 | 1440 | 20 | Object images |
| **Scene15** | 3 | 4485 | 15 | Scene categories |
| **NoisyMNIST** | 2 | 70000 | 10 | Noisy MNIST digits |
| **Caltech101** | 6 | 9144 | 101 | Object categories |
| **BDGP** | 2 | 2500 | 5 | Gene expression |
| **Reuters** | 5 | 18758 | 6 | Multilingual documents |

Place datasets in `./data/` directory in `.mat` format.

## External Baseline Methods

OT-CFM integrates several SOTA multi-view clustering methods for comparison:

| Method | Venue | Paper |
|--------|-------|-------|
| **MFLVC** | CVPR 2022 | Multi-level Feature Learning for Contrastive MVC |
| **SURE** | TPAMI 2022 | Stable and Unified Representation Enhancement |
| **DealMVC** | CVPR 2023 | Dual Contrastive Calibration for MVC |
| **GCFAggMVC** | CVPR 2023 | Global and Cross-view Feature Aggregation |
| **CANDY** | NeurIPS 2024 | Robust Contrastive MVC against Dual Noisy Correspondence |
| **DCG** | AAAI 2025 | Diffusion-based Incomplete MVC |
| **MRG-UMC** | TNNLS 2025 | Multi-level Reliable Guidance for Unpaired MVC |

To use external methods, clone them to `external_methods/`:

```bash
cd external_methods

# MFLVC (CVPR 2022)
git clone https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git MFLVC

# SURE (TPAMI 2022)
git clone https://github.com/XLearning-SCU/2022-NeurIPS-SURE.git SURE

# DealMVC (CVPR 2023)
git clone https://github.com/SubmissionsIn/DealMVC.git DealMVC

# GCFAggMVC (CVPR 2023)
git clone https://github.com/Galaxy922/GCFAggMVC.git GCFAggMVC

# DCG (AAAI 2025)
git clone https://github.com/zhangyuanyang21/2025-AAAI-DCG.git 2025-AAAI-DCG

# MRG-UMC (TNNLS 2025)
git clone https://github.com/LikeXin94/MRG-UMC.git MRG-UMC

# CANDY (NeurIPS 2024)
git clone https://github.com/XLearning-SCU/2024-NeurIPS-CANDY.git 2024-NeurIPS-CANDY
```

Then run comparison:

```bash
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --include_external
```

## Algorithm Overview

OT-CFM consists of three main components:

### 1. Latent Space Construction
View-specific encoders map each view to a shared latent space:
```
z_v = f_θ(x_v) for each view v
```

### 2. Gromov-Wasserstein Guided Flow
The vector field is learned to transform samples while preserving geometric structure:
```
dz/dt = v_θ(z_t, t, c)
```
where the condition `c` comes from available views.

### 3. Alternating Optimization
Training alternates between:
- **E-step**: Update cluster assignments
- **M-step**: Update network parameters

### Training Pipeline
1. **Phase 1**: Reconstruction pretraining (encoder-decoder)
2. **Phase 2**: Single-View DEC pretraining (cluster-separable latent space)
3. **Phase 3**: Full OT-CFM training with all loss components

## Ablation Study

OT-CFM provides comprehensive ablation study tools to evaluate component contributions. Use the dedicated `run_ablation.py` script for systematic experiments.

### Component Ablation

Tests the contribution of each model component by removing it:

```bash
# Run full component ablation
uv run python scripts/run_ablation.py --dataset Scene15 --epochs 100

# Run specific ablation modes
uv run python scripts/run_ablation.py --dataset Handwritten --modes full no_gw no_flow no_cluster

# Quick test with fewer runs
uv run python scripts/run_ablation.py --dataset Coil20 --epochs 50 --num_runs 2
```

**Ablation modes:**
| Mode | Description |
|------|-------------|
| `full` | Complete OT-CFM model (baseline) |
| `no_gw` | Without Gromov-Wasserstein alignment loss |
| `no_cluster` | Without clustering loss |
| `no_ot` | Without optimal transport (random coupling) |
| `no_flow` | Without flow matching (direct prediction) |
| `no_contrastive` | Without contrastive loss |
| `no_recon` | Without reconstruction loss |

### Lambda Sensitivity Analysis

Evaluates how sensitive the model is to different loss weight settings:

```bash
# Analyze single lambda parameter
uv run python scripts/run_ablation.py --dataset Scene15 --analysis lambda_sensitivity \
    --lambda_name lambda_gw

# Custom lambda values
uv run python scripts/run_ablation.py --dataset Coil20 --analysis lambda_sensitivity \
    --lambda_name lambda_cluster --lambda_values 0.0 0.1 0.3 0.5 1.0
```

**Lambda parameters:**
- `lambda_gw`: Gromov-Wasserstein loss weight
- `lambda_cluster`: Clustering loss weight
- `lambda_recon`: Reconstruction loss weight
- `lambda_contrastive`: Contrastive loss weight

### Architecture Ablation

Tests different architecture configurations:

```bash
# Run architecture ablation
uv run python scripts/run_ablation.py --dataset Scene15 --analysis architecture
```

**Architecture variants tested:**
- Latent dimensions: 64, 128, 256
- Encoder depths: shallow, baseline, deep
- Flow network depths: 2, 4, 8 layers

### Run All Analyses

```bash
# Run all ablation analyses (component + lambda sensitivity + architecture)
uv run python scripts/run_ablation.py --dataset Scene15 --analysis all --epochs 50 --num_runs 2
```

### Output Files

Results are saved to `results/ablation/`:
- **CSV**: `{dataset}_{analysis_type}_{timestamp}.csv`
- **JSON**: `{dataset}_{analysis_type}_{timestamp}.json`
- **PNG**: Visualization plots for each analysis type

### Programmatic Usage

```python
from otcfm.ablation import AblationStudy, AblationConfig

ablation_config = AblationConfig(
    modes=["full", "no_gw", "no_cluster", "no_flow", "no_contrastive"],
    num_runs=3
)

study = AblationStudy(config, ablation_config)
results = study.run(train_loader, labels, view_dims)
```

## Internal Baseline Methods

Comparison baselines implemented in `src/otcfm/baselines.py`:
- **Concat-KMeans**: Concatenate views and apply KMeans
- **Multi-View Spectral**: Spectral clustering on multi-view data
- **CCA-Clustering**: Canonical Correlation Analysis + KMeans
- **Weighted-View**: Weighted combination of views
- **DMVC**: Deep Multi-View Clustering
- **Contrastive-MVC**: Contrastive learning based MVC
- **Incomplete-MVC**: Handles missing views
- **Unaligned-MVC**: Handles unaligned samples

## Evaluation Metrics

All metrics are implemented in `src/otcfm/metrics.py`:
- **ACC**: Clustering accuracy (Hungarian matching)
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Purity**: Cluster purity
- **F1**: Macro F1-score
- **Silhouette**: Silhouette coefficient
- **Davies-Bouldin**: Davies-Bouldin index
- **Calinski-Harabasz**: Calinski-Harabasz index

## Command Line Reference

```bash
# Full argument list
uv run python scripts/run_experiment.py --help

# Key arguments:
#   --mode          : train, compare, ablation, multi
#   --dataset       : Synthetic, Handwritten, Coil20, Scene15, NoisyMNIST, etc.
#   --epochs        : Number of training epochs (default: 200)
#   --batch_size    : Batch size (default: 256)
#   --lr            : Learning rate (default: 1e-3)
#   --device        : cuda, mps, cpu (auto-detect if not specified)
#   --include_external : Include external SOTA baselines
#   --no_internal   : Exclude internal baselines
#   --use_tuned     : Use Optuna-tuned hyperparameters
#   --results_dir   : Directory for CSV results (default: results)
#   --missing_rate  : Missing view rate (default: 0.0)
#   --unaligned_rate: Unaligned sample rate (default: 0.0)
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{otcfm2024,
  title={OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering with Incomplete and Unaligned Views},
  author={},
  journal={},
  year={2024}
}
```

## License

MIT License

## Acknowledgements

This work builds upon:
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Gromov-Wasserstein Learning](https://arxiv.org/abs/2011.01012)
- [Python Optimal Transport](https://pythonot.github.io/)
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
