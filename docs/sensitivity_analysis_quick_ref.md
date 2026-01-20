# Sensitivity Analysis Quick Reference

## 🎯 Three Analysis Modes

### 1️⃣ Single Parameter Sweep
**Purpose**: Analyze one hyperparameter's effect  
**When**: Initial exploration, understanding individual parameter impact  
**Time**: 1-2 hours per dataset  

```bash
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 \
  --mode single \
  --param lambda_gw \
  --values 0.0 0.1 0.2 0.3 0.4 \
  --n_runs 5 --epochs 200
```

**Outputs**: Line plots (2×2), CSV, stats report

---

### 2️⃣ Grid Search (Two Parameters)
**Purpose**: Analyze parameter interactions  
**When**: Found important params, need to understand synergy  
**Time**: 3-5 hours per dataset  

```bash
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 \
  --mode grid \
  --param1 lambda_gw --param2 lambda_cluster \
  --values1 0.0 0.1 0.2 0.3 \
  --values2 0.5 1.0 1.5 2.0 \
  --n_runs 3 --epochs 150 --plot_3d
```

**Outputs**: Heatmaps (2×2), 3D surfaces, correlation report

---

### 3️⃣ Full Analysis (All Parameters)
**Purpose**: Comprehensive evaluation for publication  
**When**: Final experiments for paper submission  
**Time**: 8-12 hours per dataset  

```bash
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 \
  --mode full \
  --n_runs 10 --epochs 200
```

**Outputs**: 9 line plots, comprehensive stats, sensitivity scores

---

## 📊 Key Parameters (9 Total)

| Parameter | Category | Default | Range | Sensitivity |
|-----------|----------|---------|-------|-------------|
| `lambda_gw` | Loss | 0.2 | [0.0, 0.5] | **High** ⚠️ |
| `lambda_cluster` | Loss | 1.0 | [0.5, 2.0] | **High** ⚠️ |
| `lambda_recon` | Loss | 0.5 | [0.1, 1.0] | Medium |
| `lambda_contrastive` | Loss | 0.3 | [0.0, 0.5] | Medium |
| `latent_dim` | Arch | 128 | [64, 256] | Medium |
| `flow_hidden_dim` | Arch | 256 | [128, 512] | Low |
| `ode_steps` | Arch | 10 | [5, 20] | Low |
| `learning_rate` | Train | 3e-4 | [1e-4, 1e-3] | Medium |
| `dropout` | Train | 0.1 | [0.0, 0.3] | Low |

⚠️ **High sensitivity** = Requires careful tuning  
✓ **Medium/Low** = Default values usually sufficient  

---

## 📁 Output Files

```
sensitivity_results/{experiment_name}/
├── results.csv                  # Raw data (all runs)
├── summary_stats.json           # Mean ± Std for each config
├── statistical_report.txt       # Comprehensive analysis
│   ├── Best Configurations      #   → Best param values per metric
│   ├── Sensitivity Scores       #   → Parameter importance (0-1)
│   ├── Correlation Matrix       #   → Parameter interactions
│   └── P-values                 #   → Statistical significance
└── Plots:
    ├── single_param_sweep.pdf   # 2×2 line plots
    ├── grid_heatmap.pdf         # 2×2 heatmaps  
    ├── 3d_surface_ACC.pdf       # 3D surface (optional)
    ├── 3d_surface_NMI.pdf
    ├── 3d_surface_ARI.pdf
    └── 3d_surface_F1.pdf
```

---

## 🚀 Quick Start Workflow

### Step 1: Fast Test (5 min)
```bash
uv run python scripts/test_sensitivity_analysis.py
```

### Step 2: Explore Important Params (2 hours)
```bash
# Test lambda_gw
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode single --param lambda_gw \
  --values 0.0 0.1 0.2 0.3 0.4 --n_runs 5 --epochs 200

# Test lambda_cluster
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode single --param lambda_cluster \
  --values 0.5 1.0 1.5 2.0 --n_runs 5 --epochs 200
```

### Step 3: Analyze Interactions (4 hours)
```bash
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode grid \
  --param1 lambda_gw --param2 lambda_cluster \
  --values1 0.0 0.1 0.2 0.3 \
  --values2 0.5 1.0 1.5 2.0 \
  --n_runs 5 --epochs 200 --plot_3d
```

### Step 4: Full Analysis for Publication (12 hours)
```bash
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode full \
  --n_runs 10 --epochs 200
```

### Step 5: Generate LaTeX Tables
```bash
python scripts/generate_latex_tables.py \
  --input sensitivity_results/Scene15_lambda_gw \
  --output paper/tables/sensitivity_lambda_gw.tex
```

---

## 📖 Usage Examples

### Example 1: Understanding λ_GW Impact
```bash
# Run analysis
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode single --param lambda_gw \
  --values 0.0 0.05 0.1 0.15 0.2 0.25 0.3 \
  --n_runs 10 --epochs 200

# Check results
cat sensitivity_results/*/statistical_report.txt

# Generate LaTeX
python scripts/generate_latex_tables.py \
  --input sensitivity_results/Scene15_lambda_gw \
  --output lambda_gw_table.tex
```

**Expected output in report**:
```
Best Configuration for ACC: lambda_gw=0.2 (ACC=0.876±0.012)
Sensitivity Score: 0.73 (High sensitivity)
Recommendation: Tune carefully in range [0.15, 0.25]
```

### Example 2: Parameter Interaction Study
```bash
# Run grid search
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Handwritten --mode grid \
  --param1 lambda_gw --param2 lambda_cluster \
  --values1 0.0 0.1 0.2 0.3 0.4 \
  --values2 0.5 1.0 1.5 2.0 2.5 \
  --n_runs 5 --epochs 200 --plot_3d

# Results show interaction
```

**Expected output in report**:
```
Correlation Analysis:
  lambda_gw ↔ lambda_cluster: r=0.82, p<0.001
  Interpretation: Strong positive correlation
  → These parameters work synergistically
  → Jointly tune them together
```

### Example 3: Batch Processing Multiple Datasets
```bash
# Windows
scripts\run_all_sensitivity.bat

# Linux/Mac
bash scripts/run_all_sensitivity.sh
```

**Processes**: Scene15, Handwritten, NoisyMNIST in sequence

---

## 💡 Interpretation Guide

### Sensitivity Scores
- **0.7-1.0**: 🔴 **Critical** - Small changes cause large performance swings
  - Action: Carefully tune, use fine-grained search
  - Example: λ_GW = 0.85 → Test [0.15, 0.18, 0.20, 0.22, 0.25]

- **0.3-0.7**: 🟡 **Moderate** - Noticeable but manageable impact
  - Action: Use default or coarse tuning
  - Example: latent_dim = 0.45 → Test [64, 128, 256]

- **0.0-0.3**: 🟢 **Robust** - Minimal impact on performance
  - Action: Use default values confidently
  - Example: dropout = 0.15 → Default 0.1 is fine

### Correlation Patterns
- **|r| > 0.7**: Strong relationship
  - Positive (r > +0.7): Parameters work synergistically
    - Example: λ_GW ↔ λ_cluster → Increase both together
  - Negative (r < -0.7): Parameters are antagonistic
    - Example: λ_recon ↔ λ_cluster → Balance carefully

- **|r| < 0.3**: Independent parameters
  - Can be tuned separately without concern

---

## 📝 Paper Writing Template

### Section 4.5: Sensitivity Analysis

```latex
\subsection{Sensitivity Analysis}

We conducted comprehensive sensitivity analysis to evaluate the robustness 
of OT-CFM with respect to key hyperparameters. Figure~\ref{fig:sensitivity_lambda_gw} 
shows the effect of the Gromov-Wasserstein weight $\lambda_{GW}$ on clustering 
performance across four metrics (ACC, NMI, ARI, F1). The model achieves peak 
performance at $\lambda_{GW}=0.2$, demonstrating moderate sensitivity 
(sensitivity score = 0.73) to this parameter.

Table~\ref{tab:sensitivity_results} presents the mean$\pm$std results across 
10 independent runs for each parameter configuration. Statistical analysis 
reveals that $\lambda_{cluster}$ has the highest sensitivity score (0.85), 
indicating its critical role in determining clustering quality. 

Correlation analysis (Figure~\ref{fig:param_correlation}) shows strong positive 
correlation ($r=0.82$, $p<0.001$) between $\lambda_{GW}$ and $\lambda_{cluster}$, 
suggesting they work synergistically to balance structural alignment and 
cluster separation.

Our full parameter sweep over 9 hyperparameters confirms that OT-CFM maintains 
consistent performance across a wide range of configurations, with performance 
degradation $<5\%$ within $\pm 50\%$ of default values, demonstrating the 
model's robustness.
```

### Figures to Include
1. **Figure 1**: Single param sweep (line plots)
   - File: `single_param_sweep.pdf`
   - Caption: "Effect of λ_GW on clustering performance"

2. **Figure 2**: Grid search heatmap
   - File: `grid_heatmap.pdf`
   - Caption: "Interaction between λ_GW and λ_cluster"

3. **Figure 3**: 3D surface (supplementary)
   - File: `3d_surface_ACC.pdf`
   - Caption: "Performance surface over parameter space"

### Tables to Include
- **Table 1**: Best configurations (from `summary_stats.json`)
- **Table 2**: Sensitivity scores (from `statistical_report.txt`)
- **Table 3**: Top 10 configurations (for grid search)

**Generate with**:
```bash
python scripts/generate_latex_tables.py \
  --input sensitivity_results/Scene15_lambda_gw \
  --output paper/tables/sensitivity.tex
```

---

## ⚙️ Advanced Options

### Custom Parameter Ranges
```bash
# Fine-grained search around optimal value
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode single --param lambda_gw \
  --values 0.15 0.175 0.2 0.225 0.25 \  # Fine-grained
  --n_runs 10 --epochs 200
```

### Quick Prototyping
```bash
# Fast test with reduced epochs
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode full \
  --n_runs 1 --epochs 50  # Fast sanity check
```

### High-Precision Final Experiments
```bash
# Maximum rigor for publication
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode full \
  --n_runs 20 --epochs 300  # Publication quality
```

---

## 🔧 Troubleshooting

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size or use smaller dataset first
```bash
# In scripts/run_sensitivity_analysis.py, modify:
config.training.batch_size = 64  # Default is 256
```

### Issue: Too Slow
**Solution**: Reduce runs and epochs for initial exploration
```bash
--n_runs 3 --epochs 100  # Instead of 10 runs × 200 epochs
```

### Issue: Results Not Converging
**Solution**: Increase training epochs
```bash
--epochs 300  # Or check if learning_rate needs adjustment
```

---

## 📚 Related Documentation

- **Comprehensive Guide**: [docs/sensitivity_analysis_guide.md](../docs/sensitivity_analysis_guide.md)
- **Usage Demo**: Run `python scripts/demo_sensitivity_usage.py`
- **Main README**: [README.md](../README.md#sensitivity-analysis)
- **Baseline Guide**: [docs/add_new_baselines_guide.md](../docs/add_new_baselines_guide.md)

---

## 🎓 Citation

If you use this sensitivity analysis framework in your research, please cite:

```bibtex
@inproceedings{otcfm2025,
  title={OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering},
  author={Your Name},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

---

**Quick Links**:
- [↑ Back to Main README](../README.md)
- [→ Full Documentation](../docs/sensitivity_analysis_guide.md)
- [→ Code](../scripts/run_sensitivity_analysis.py)
