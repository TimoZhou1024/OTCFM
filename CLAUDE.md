# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OT-CFM (Optimal Transport Coupled Flow Matching) is a research project for multi-view clustering that handles incomplete (missing views) and unaligned (mismatched correspondences) data. It combines flow matching with Gromov-Wasserstein optimal transport for robust multi-view learning.

## Build & Development Commands

```bash
# Install dependencies (recommended)
uv sync                     # Basic installation
uv sync --all-extras        # With dev tools and Optuna tuning

# Alternative: pip
pip install -e "."          # Basic
pip install -e ".[all]"     # With all extras

# Run experiments
uv run python scripts/run_experiment.py --mode train --dataset Synthetic --epochs 200
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --include_external

# Hyperparameter tuning
uv run python scripts/run_optuna_tuning.py --dataset Handwritten --n_trials 100

# Robustness testing
uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 --include_external

# Code quality (dev extras required)
uv run black src/ scripts/
uv run ruff check src/ scripts/
uv run mypy src/
uv run pytest tests/
```

## Architecture

### Core Components (src/otcfm/)

**Training Pipeline** operates in three phases:
1. **Phase 1**: Encoder-decoder pretraining (reconstruction loss)
2. **Phase 2**: Single-view DEC pretraining (cluster-separable latent space)
3. **Phase 3**: Full OT-CFM with alternating E-step (cluster assignment) / M-step (network update)

**Key Modules:**
- `ot_cfm.py` - Main OTCFM model class combining all components
- `models.py` - Neural networks: MultiViewEncoderDecoder, VectorFieldNetwork (flow), ClusteringModule, ODESolver
- `trainer.py` - Training loop with 3-phase pipeline
- `losses.py` - Combined loss: L = λ_gw * L_gw + λ_cluster * L_cluster + λ_recon * L_recon + λ_contrastive * L_contrastive
- `datasets.py` - Dataset loaders supporting missing_rate and unaligned_rate parameters
- `baselines.py` - 8 internal baseline methods (Concat-KMeans, DMVC, etc.)
- `external_baselines.py` - Wrappers for 7 external SOTA methods (MFLVC, SURE, DealMVC, etc.)

### Scripts (scripts/)

- `run_experiment.py` - Main entry point with modes: train, compare, ablation, multi
- `run_optuna_tuning.py` - Bayesian hyperparameter optimization
- `tune_all_datasets.py` - Batch tuning across all datasets
- `run_robustness_test.py` - Missing view and unaligned data evaluation

## Datasets

Place `.mat` files in `data/`. Supported: Synthetic, Handwritten, Coil20, Scene15, NoisyMNIST, Caltech101, BDGP, Reuters, NUS-WIDE, CUB.

## External Methods

Clone external baselines to `external_methods/` (MFLVC, SURE, DealMVC, GCFAggMVC, DCG, MRG-UMC, CANDY). See README.md for git URLs.

## Key Configuration

- Default hyperparameters in `config.py` (latent_dim=128, hidden_dims=[512,256], etc.)
- Tuned parameters saved to `config/tuned_params.json`
- Experiment outputs go to `experiments/` with timestamps
- CSV results exported to `results/`
