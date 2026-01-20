# Sensitivity Analysis Guide for OT-CFM

This guide describes how to perform publication-quality sensitivity analysis for OT-CFM, suitable for top-tier conferences (ICML, NeurIPS, CVPR).

## Overview

The sensitivity analysis framework systematically evaluates how model performance varies with different hyperparameter settings. This is crucial for:
- Understanding which hyperparameters matter most
- Identifying robust parameter ranges
- Providing evidence for publication claims
- Guiding hyperparameter tuning

## Quick Start

### Single Parameter Analysis

Test how one parameter affects performance:

```bash
# Analyze lambda_gw (Gromov-Wasserstein weight)
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Scene15 \
    --mode single \
    --param lambda_gw \
    --epochs 100 \
    --num_runs 3

# Analyze learning rate
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Handwritten \
    --mode single \
    --param learning_rate \
    --epochs 100
```

**Output:**
- CSV file with all results
- JSON file with metadata
- 2×2 plot showing ACC, NMI, ARI, F1 curves
- Statistical report with best values and sensitivity scores

### Two-Parameter Grid Search

Analyze interactions between two parameters:

```bash
# Analyze lambda_gw vs lambda_cluster
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Scene15 \
    --mode grid \
    --param lambda_gw lambda_cluster \
    --epochs 80 \
    --num_runs 2

# With 3D visualization
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Coil20 \
    --mode grid \
    --param learning_rate latent_dim \
    --plot_3d
```

**Output:**
- CSV with grid search results
- JSON with metadata
- 2×2 heatmap (one per metric)
- Optional 3D surface plots
- Statistical report

### Full Analysis

Comprehensive analysis of all parameters:

```bash
# Full analysis (takes time)
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Scene15 \
    --mode full \
    --epochs 100

# Quick mode (fewer values per parameter)
uv run python scripts/run_sensitivity_analysis.py \
    --dataset Synthetic \
    --mode full \
    --quick_mode \
    --epochs 50
```

**Output:**
- Separate results for each parameter
- All plots and statistics per parameter

## Parameters Analyzed

### Loss Weights
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lambda_gw` | 0.2 | [0.01, 1.0] | Gromov-Wasserstein alignment weight |
| `lambda_cluster` | 1.0 | [0.1, 10.0] | Clustering loss weight |
| `lambda_recon` | 0.5 | [0.1, 3.0] | Reconstruction loss weight |
| `lambda_contrastive` | 0.3 | [0.01, 1.0] | Contrastive learning weight |

### Architecture
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `latent_dim` | 128 | [32, 384] | Latent space dimension |
| `flow_hidden_dim` | 256 | [128, 512] | Flow network hidden dimension |
| `ode_steps` | 10 | [5, 50] | ODE integration steps |

### Training
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 0.001 | [0.0001, 0.01] | Adam learning rate |
| `dropout` | 0.1 | [0.0, 0.5] | Dropout rate |

## Output Files

All outputs are saved to `results/sensitivity/`:

### CSV Format
```
parameter,value,ACC_mean,ACC_std,NMI_mean,NMI_std,ARI_mean,ARI_std,F1_mean,F1_std
lambda_gw,0.01,0.7234,0.0123,0.6543,0.0098,0.5432,0.0145,0.6123,0.0112
lambda_gw,0.02,0.7456,0.0134,0.6678,0.0102,0.5567,0.0156,0.6234,0.0121
...
```

### JSON Format
```json
{
  "dataset": "Scene15",
  "analysis_type": "single_lambda_gw",
  "timestamp": "20260114_123456",
  "num_experiments": 8,
  "data": [...]
}
```

### Plots

#### Single Parameter
- **Line plots** with error bands (mean ± std)
- **Best value annotations**
- Automatic log scale for lambda parameters
- Separate subplot for each metric (ACC, NMI, ARI, F1)

#### Two-Parameter Grid
- **Heatmaps** showing performance surface
- Color-coded performance values
- Annotated cells with exact values
- Inverted y-axis for readability

#### 3D Visualization (Optional)
- **Surface plots** showing 3D performance landscape
- Contour projection on base plane
- Interactive rotation (if displayed)

### Statistical Report

Text file containing:
- **Best/worst values** for each metric
- **Performance ranges** (max - min)
- **Sensitivity scores** (normalized range)
- **Correlations** with parameter values
- **Statistical significance** (p-values)

Example:
```
ACC Statistics:
--------------------------------------------------
  Best value: 0.2000
  Best performance: 0.8456 ± 0.0123
  Worst value: 0.0100
  Worst performance: 0.7234 ± 0.0145
  Performance range: 0.1222
  Sensitivity score: 0.1523
  Pearson correlation: 0.8765 (p=1.23e-05)
```

## Advanced Usage

### Custom Parameter Ranges

Edit `PARAM_RANGES` in the script:

```python
PARAM_RANGES = {
    'lambda_gw': np.logspace(-2, 0, 12),  # More granular
    'latent_dim': [64, 128, 256, 512],    # Custom values
}
```

### Different Datasets

```bash
# Small dataset (fast)
uv run python scripts/run_sensitivity_analysis.py --dataset Synthetic

# Medium dataset
uv run python scripts/run_sensitivity_analysis.py --dataset Handwritten

# Large dataset (slow)
uv run python scripts/run_sensitivity_analysis.py --dataset Scene15
```

### Computational Considerations

- **Single parameter (8 values, 3 runs)**: ~4-8 hours on GPU
- **Grid search (8×8, 2 runs)**: ~16-24 hours on GPU
- **Full analysis (9 params, quick mode)**: ~12-20 hours on GPU

**Tips:**
- Use `--quick_mode` for initial exploration
- Use `--num_runs 2` instead of 3 to save time
- Use `--epochs 80` for faster experiments
- Run on GPU for best performance

### Batch Processing

Create a batch script for multiple analyses:

```bash
#!/bin/bash
# analyze_all.sh

DATASETS="Synthetic Handwritten Scene15"
PARAMS="lambda_gw lambda_cluster learning_rate latent_dim"

for dataset in $DATASETS; do
    for param in $PARAMS; do
        uv run python scripts/run_sensitivity_analysis.py \
            --dataset $dataset \
            --mode single \
            --param $param \
            --epochs 100 \
            --num_runs 3
    done
done
```

## Interpreting Results

### Sensitivity Score

The sensitivity score measures how much performance varies with parameter changes:

- **Score < 0.05**: Parameter is **insensitive** (robust)
- **Score 0.05-0.15**: **Moderate** sensitivity (typical)
- **Score > 0.15**: **High** sensitivity (needs careful tuning)

### Correlation Analysis

Pearson correlation shows if there's a linear relationship:

- **|r| > 0.7**: Strong correlation
- **|r| 0.3-0.7**: Moderate correlation
- **|r| < 0.3**: Weak correlation

P-value < 0.05 indicates statistical significance.

### Best Practices for Publication

1. **Run multiple seeds** (`--num_runs 3` minimum)
2. **Report error bars** (automatically included in plots)
3. **Test key parameters** (loss weights, architecture dims)
4. **Include statistical tests** (use the generated reports)
5. **Use consistent styling** (PDF outputs for LaTeX papers)

### Example Publication Statements

Based on sensitivity analysis results, you can write:

> "We performed systematic sensitivity analysis by varying λ_GW from 0.01 to 1.0 (8 values, 3 runs each). 
> Performance was relatively stable across the range [0.1, 0.5] with ACC varying by only ±2.3%. 
> The optimal value was λ_GW = 0.2, achieving ACC = 84.56 ± 1.23%."

> "Figure X shows the performance surface for λ_GW and λ_cluster. The model exhibits a 
> robustness plateau in the region [0.1, 0.3] × [0.5, 2.0], suggesting these parameters 
> can be safely tuned within this range without significant performance degradation."

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 128

# Reduce number of runs
--num_runs 2

# Use smaller dataset
--dataset Synthetic
```

### Slow Execution

```bash
# Use quick mode
--quick_mode

# Fewer epochs
--epochs 50

# Single run (not recommended for publication)
--num_runs 1
```

### Missing Dependencies

```bash
# Install visualization dependencies
uv add seaborn matplotlib scipy

# For 3D plots
uv add mpl_toolkits
```

## Citation

If you use this sensitivity analysis framework in your publication, please cite:

```bibtex
@inproceedings{otcfm2024,
  title={OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering},
  author={Your Name},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## References

- Sensitivity analysis: Saltelli et al., "Global Sensitivity Analysis: The Primer", 2008
- Statistical testing: Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 2006
- Visualization: Tufte, "The Visual Display of Quantitative Information", 2001
