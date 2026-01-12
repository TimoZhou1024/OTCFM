# Copilot Instructions for OT-CFM

## Project Overview
OT-CFM (Optimal Transport Coupled Flow Matching) is a research project for multi-view clustering that handles **incomplete** (missing views) and **unaligned** (mismatched correspondences) data. It combines flow matching with Gromov-Wasserstein optimal transport.

## Architecture

### Training Pipeline (Three Phases)
The training in [trainer.py](src/otcfm/trainer.py) operates in three phases:
1. **Phase 1**: Encoder-decoder pretraining with reconstruction loss only
2. **Phase 2**: Single-view DEC pretraining (cluster-separable latent space per view)
3. **Phase 3**: Full OT-CFM with alternating E-step (cluster assignment) / M-step (network update)

### Core Components (`src/otcfm/`)
| Module | Purpose |
|--------|---------|
| `ot_cfm.py` | Main `OTCFM` model class combining encoder-decoder, flow, and clustering |
| `models.py` | Neural networks: `MultiViewEncoderDecoder`, `VectorFieldNetwork`, `ClusteringModule`, `ODESolver` |
| `losses.py` | Loss functions: CFM, GW structural alignment, KL clustering, reconstruction, contrastive |
| `trainer.py` | Training loop implementing the 3-phase pipeline |
| `datasets.py` | Dataset loaders with `missing_rate` and `unaligned_rate` simulation |
| `baselines.py` | 8 internal baseline methods (Concat-KMeans, DMVC, etc.) |
| `external_baselines.py` | Wrappers for 7 external SOTA methods (MFLVC, SURE, DealMVC, etc.) |

### Key Design Pattern: Aligned vs Unaligned
The `is_aligned` flag controls behavior throughout the codebase:
- **Aligned (IMVC)**: Consensus computed by averaging latents across views; contrastive loss enabled
- **Unaligned (UMVC)**: Consensus is INVALID; use cluster centroids for conditioning; contrastive loss disabled

```python
# In ot_cfm.py - critical pattern
if self.is_aligned:
    consensus = self.encoder_decoder.fuse_latents(latents, mask)  # Safe to average
else:
    consensus = None  # Averaging is mathematically invalid for UMVC!
```

## Commands

```bash
# Install (uv recommended)
uv sync --all-extras

# Basic training
uv run python scripts/run_experiment.py --mode train --dataset Scene15 --epochs 200

# Compare with all baselines (internal + external)
uv run python scripts/run_experiment.py --mode compare --dataset Scene15 --include_external

# Hyperparameter tuning (saves to config/tuned_params.json)
uv run python scripts/run_optuna_tuning.py --dataset Handwritten --n_trials 100

# Robustness testing (incomplete/unaligned data)
uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 --include_external

# Code quality
uv run black src/ scripts/ && uv run ruff check src/ scripts/
```

## Datasets
Place `.mat` files in `data/`. Supported: Synthetic, Handwritten, Coil20, Scene15, NoisyMNIST, Caltech101, BDGP, Reuters, NUS-WIDE, CUB.

Dataset loaders are in [datasets.py](src/otcfm/datasets.py) with a `DATASET_LOADERS` mapping in experiment scripts.

## External Methods
Clone to `external_methods/` (e.g., MFLVC, SURE, DealMVC). Wrappers in [external_baselines.py](src/otcfm/external_baselines.py) handle integration.

External methods are categorized by specialty:
- **Incomplete-specialized**: COMPLETER, SURE, DealMVC, DCG
- **Unaligned-specialized**: MRG-UMC, CANDY
- **General**: MFLVC, GCFAggMVC

## Key Configuration

### Loss Weights (in `config.py` / `ot_cfm.py`)
```python
lambda_gw=0.2          # Gromov-Wasserstein structural alignment
lambda_cluster=1.0     # Clustering loss (KL divergence)
lambda_recon=0.5       # Reconstruction loss
lambda_contrastive=0.3 # Contrastive (ONLY for aligned data)
```

### Hyperparameters
- `latent_dim=128`, `hidden_dims=[512, 256]`, `flow_hidden_dim=256`
- `ode_steps=10` (Euler integration steps for flow)
- Tuned parameters saved to `config/tuned_params.json`

## Code Conventions

1. **Device handling**: Use `config.training.device` (auto-detects CUDA/MPS/CPU)
2. **Experiment outputs**: Go to `experiments/ot_cfm_{timestamp}/`
3. **Results CSV**: Exported to `results/` with format: Method, ACC, NMI, ARI, Purity, F1
4. **Metrics**: Use `evaluate_clustering()` from [metrics.py](src/otcfm/metrics.py)

## Adding New Baselines
See [add_new_baselines_guide.md](docs/add_new_baselines_guide.md). Key steps:
1. Inherit from `BaseClusteringMethod` in `baselines.py`
2. Implement `fit_predict(views, mask, **kwargs) -> np.ndarray`
3. Register in `get_baseline_methods()` or `get_external_baselines()`
