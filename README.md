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

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
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
torchvision>=0.15.0
```

## Project Structure

```
OT-CFM/
├── src/
│   └── otcfm/              # Core package
│       ├── __init__.py     # Package exports
│       ├── config.py       # Configuration dataclasses
│       ├── datasets.py     # Dataset loaders and preprocessing
│       ├── models.py       # Neural network components
│       ├── losses.py       # Loss functions (CFM, GW, clustering)
│       ├── metrics.py      # Evaluation metrics (ACC, NMI, ARI, etc.)
│       ├── ot_cfm.py       # Main OT-CFM model
│       ├── trainer.py      # Training pipeline
│       ├── baselines.py    # Baseline methods for comparison
│       ├── ablation.py     # Ablation study runner
│       ├── utils.py        # Helper functions
│       └── visualization.py # Visualization utilities
├── scripts/
│   └── run_experiment.py   # Main experiment runner
├── data/                   # Dataset directory (add your datasets here)
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── experiments/            # Experiment outputs
├── pyproject.toml          # Project configuration (uv/pip)
├── requirements.txt        # Legacy requirements
├── main.tex                # Paper source
└── README.md               # This file
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

```bash
# Activate virtual environment first
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Basic training
python scripts/run_experiment.py --mode train --dataset Synthetic --epochs 200

# Compare with baselines
python scripts/run_experiment.py --mode compare --dataset Scene15 --epochs 200

# Compare with baselines (including external methods: MFLVC, SURE, GCFAggMVC, etc.)
python scripts/run_experiment.py --mode compare --dataset Scene15 --epochs 200 --include_external

# Run ablation study
python scripts/run_experiment.py --mode ablation --dataset Caltech101 --epochs 100

# Multi-dataset experiment
python scripts/run_experiment.py --mode multi --datasets Synthetic Scene15 Caltech101
```

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

## Datasets

Supported datasets:
- **Caltech101**: 101 object categories
- **Scene15**: 15 scene categories
- **NoisyMNIST**: Noisy version of MNIST
- **BDGP**: Drosophila gene expression
- **CUB**: Caltech-UCSD Birds
- **Reuters**: Multilingual text documents
- **Synthetic**: Generated multi-view data

Place datasets in `./data/` directory.

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

## Ablation Study

Run ablation to evaluate component contributions:

```python
from otcfm.ablation import AblationStudy, AblationConfig

ablation_config = AblationConfig(
    modes=["full", "no_gw", "no_cluster", "no_flow"],
    num_runs=3
)

study = AblationStudy(config, ablation_config)
results = study.run(train_loader, labels, view_dims)
```

Available ablation modes:
- `full`: Complete model
- `no_gw`: Without Gromov-Wasserstein loss
- `no_cluster`: Without clustering loss
- `no_ot`: Without optimal transport
- `no_flow`: Without flow matching

## Baseline Methods

Comparison baselines implemented in `src/otcfm/baselines.py`:
- Concat-KMeans
- Multi-View Spectral Clustering
- CCA-Clustering
- Deep Multi-View Clustering (DMVC)
- Contrastive Multi-View Clustering
- Incomplete Multi-View Clustering
- Unaligned Multi-View Clustering

## Evaluation Metrics

All metrics are implemented in `src/otcfm/metrics.py`:
- **ACC**: Clustering accuracy (Hungarian matching)
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Purity**: Cluster purity
- **F1**: Macro F1-score

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
