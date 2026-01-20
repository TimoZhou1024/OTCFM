# Sensitivity Analysis Workflow Diagram

```
                    🎯 OT-CFM Sensitivity Analysis Framework
                              (ICML-Quality)
                                    │
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌─────────────────┐            ┌─────────────────┐
          │  Setup Phase    │            │  Quick Start    │
          │  (5 minutes)    │            │  (5 minutes)    │
          └────────┬────────┘            └────────┬────────┘
                   │                              │
                   │ uv sync                      │ test_sensitivity
                   │                              │   _analysis.py
                   ▼                              ▼
          ┌─────────────────┐            ┌─────────────────┐
          │ Environment OK  │            │  Tests Passed   │
          └────────┬────────┘            └────────┬────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Choose Analysis Mode  │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
    ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Mode 1: Single    │ │ Mode 2: Grid │ │ Mode 3: Full │
    │ Parameter Sweep   │ │    Search    │ │   Analysis   │
    │                   │ │              │ │              │
    │ Time: 1-2 hours   │ │ Time: 3-5h   │ │ Time: 8-12h  │
    │ Params: 1         │ │ Params: 2    │ │ Params: 9    │
    └─────────┬─────────┘ └──────┬───────┘ └──────┬───────┘
              │                  │                 │
              │ --mode single    │ --mode grid     │ --mode full
              │ --param X        │ --param1 X      │
              │ --values ...     │ --param2 Y      │
              │                  │ --plot_3d       │
              ▼                  ▼                 ▼
    ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Execute Sweep   │ │ Execute Grid │ │  Execute All │
    │                   │ │              │ │              │
    │ • Train models    │ │ • Train all  │ │ • Train all  │
    │ • Record metrics  │ │   combos     │ │   params     │
    │ • Compute stats   │ │ • 2D + 3D    │ │ • Generate   │
    │                   │ │   analysis   │ │   all plots  │
    └─────────┬─────────┘ └──────┬───────┘ └──────┬───────┘
              │                  │                 │
              └──────────────────┼─────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Results Generated     │
                    │                         │
                    │ sensitivity_results/    │
                    │   └── {experiment}/     │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
    ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
    │  CSV Files    │   │  PDF Figures │   │ Statistics   │
    │               │   │              │   │              │
    │ • results.csv │   │ • Line plots │   │ • Best vals  │
    │ • summary.    │   │ • Heatmaps   │   │ • Scores     │
    │   json        │   │ • 3D surface │   │ • P-values   │
    └───────┬───────┘   └──────┬───────┘   └──────┬───────┘
            │                  │                   │
            └──────────────────┼───────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Post-Processing     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
    ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Generate      │ │   Visualize  │ │   Analyze    │
    │ LaTeX Tables  │ │    Results   │ │  Statistics  │
    │               │ │              │ │              │
    │ generate_     │ │ Open PDFs    │ │ Read TXT     │
    │ latex_        │ │              │ │ reports      │
    │ tables.py     │ │              │ │              │
    └───────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                 │
            └────────────────┼─────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Paper Integration   │
                  │                      │
                  │ • Figures (PDF)      │
                  │ • Tables (LaTeX)     │
                  │ • Text (from report) │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Section 4.5:         │
                  │ Sensitivity Analysis │
                  │                      │
                  │ ✓ Figures: 2-4       │
                  │ ✓ Tables: 2-3        │
                  │ ✓ Text: 1-2 pages    │
                  │ ✓ Stats: Complete    │
                  └──────────────────────┘


═══════════════════════════════════════════════════════════════════

                        📊 Output Structure

sensitivity_results/
│
├── {dataset}_{param}/              # Single Parameter Mode
│   ├── results.csv                 # Raw: run_id, param_value, metrics
│   ├── summary_stats.json          # Stats: mean ± std per config
│   ├── single_param_sweep.pdf      # Plot: 2×2 line plots
│   └── statistical_report.txt      # Report: best, scores, interpretation
│
├── {dataset}_grid_{p1}_{p2}/       # Grid Search Mode
│   ├── results.csv                 # Raw: run_id, p1, p2, metrics
│   ├── summary_stats.json          # Stats: mean ± std per combo
│   ├── grid_heatmap.pdf            # Plot: 2×2 heatmaps
│   ├── 3d_surface_ACC.pdf          # 3D: ACC surface (if --plot_3d)
│   ├── 3d_surface_NMI.pdf          # 3D: NMI surface
│   ├── 3d_surface_ARI.pdf          # 3D: ARI surface
│   ├── 3d_surface_F1.pdf           # 3D: F1 surface
│   └── statistical_report.txt      # Report: correlations, best combos
│
└── {dataset}_full_analysis/        # Full Analysis Mode
    ├── results.csv                 # Raw: all experiments
    ├── summary_stats.json          # Stats: comprehensive
    ├── single_param_sweep_lambda_gw.pdf
    ├── single_param_sweep_lambda_cluster.pdf
    ├── single_param_sweep_lambda_recon.pdf
    ├── single_param_sweep_lambda_contrastive.pdf
    ├── single_param_sweep_latent_dim.pdf
    ├── single_param_sweep_flow_hidden_dim.pdf
    ├── single_param_sweep_ode_steps.pdf
    ├── single_param_sweep_learning_rate.pdf
    ├── single_param_sweep_dropout.pdf
    └── statistical_report.txt      # Report: full analysis


═══════════════════════════════════════════════════════════════════

                    🎯 Decision Tree: Which Mode?

                         START
                           │
                           ▼
                  ┌─────────────────┐
                  │ What's your     │
                  │ objective?      │
                  └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Explore  │      │ Analyze  │      │ Complete │
  │ single   │      │ param    │      │ study    │
  │ param    │      │ interact │      │ for      │
  │ effect   │      │ -ions    │      │ paper    │
  └────┬─────┘      └────┬─────┘      └────┬─────┘
       │                 │                  │
       ▼                 ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Mode:    │      │ Mode:    │      │ Mode:    │
  │ single   │      │ grid     │      │ full     │
  │          │      │          │      │          │
  │ Time:    │      │ Time:    │      │ Time:    │
  │ 1-2h     │      │ 3-5h     │      │ 8-12h    │
  └──────────┘      └──────────┘      └──────────┘


═══════════════════════════════════════════════════════════════════

                    📈 Parameter Sensitivity Map

         High Sensitivity (>0.7)          Medium (0.3-0.7)           Low (<0.3)
         Need Careful Tuning              Use Defaults OK            Very Robust
                 │                              │                         │
    ┌────────────┴────────────┐   ┌────────────┴────────────┐   ┌────────┴────────┐
    │                         │   │                         │   │                 │
    │ • lambda_gw (0.85)     │   │ • lambda_recon (0.45)  │   │ • dropout (0.18)│
    │ • lambda_cluster (0.81)│   │ • latent_dim (0.42)    │   │ • ode_steps     │
    │                         │   │ • learning_rate (0.38) │   │   (0.22)        │
    │                         │   │ • lambda_contrast      │   │ • flow_hidden   │
    │                         │   │   (0.35)               │   │   _dim (0.15)   │
    │                         │   │                         │   │                 │
    └─────────────────────────┘   └─────────────────────────┘   └─────────────────┘
              │                              │                         │
              ▼                              ▼                         ▼
    Fine-grained search            Coarse search                Keep defaults
    [0.15, 0.18, 0.20,            [64, 128, 256]               No tuning needed
     0.22, 0.25]


═══════════════════════════════════════════════════════════════════

                    📝 Paper Writing Workflow

    Step 1: Run Experiments
         │
         ├─► Single param sweeps (identify important params)
         ├─► Grid search (analyze interactions)
         └─► Full analysis (comprehensive evaluation)
         │
         ▼
    Step 2: Generate Outputs
         │
         ├─► CSV files (data backup)
         ├─► PDF figures (for paper)
         ├─► LaTeX tables (using generate_latex_tables.py)
         └─► Statistical reports (for text)
         │
         ▼
    Step 3: Draft Section 4.5
         │
         ├─► Subsection 4.5.1: Individual Parameters
         │   • Figure: single_param_sweep.pdf
         │   • Text: Best values, sensitivity scores
         │
         ├─► Subsection 4.5.2: Parameter Interactions
         │   • Figure: grid_heatmap.pdf
         │   • Text: Correlation analysis
         │
         └─► Subsection 4.5.3: Robustness
             • Table: sensitivity_scores
             • Text: Robustness claims
         │
         ▼
    Step 4: Supplementary Material
         │
         ├─► Appendix A: Full parameter sweep
         ├─► Appendix B: Additional datasets
         └─► Appendix C: 3D visualizations
         │
         ▼
    Step 5: Final Checks
         │
         ├─► All figures in PDF format ✓
         ├─► All tables in LaTeX format ✓
         ├─► Statistical significance reported ✓
         └─► Reproducibility info included ✓
         │
         ▼
    📄 Ready for Submission!


═══════════════════════════════════════════════════════════════════

                    ⚙️ Computational Pipeline

    Input: Dataset + Parameters
         │
         ▼
    ┌────────────────────────────┐
    │ SensitivityAnalyzer        │
    │                            │
    │ 1. Load dataset            │
    │ 2. Initialize model        │
    │ 3. Create config variants  │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ For each configuration:    │
    │                            │
    │   For run in 1..n_runs:   │
    │     • Set random seed      │
    │     • Create model         │
    │     • Train (epochs)       │
    │     • Evaluate             │
    │     • Record metrics       │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ Statistical Analysis:      │
    │                            │
    │ • Compute mean ± std       │
    │ • Find best configs        │
    │ • Calculate sensitivity    │
    │ • Correlation analysis     │
    │ • P-value tests            │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ Visualization:             │
    │                            │
    │ • Generate line plots      │
    │ • Generate heatmaps        │
    │ • Generate 3D surfaces     │
    │ • Export to PDF            │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ Export Results:            │
    │                            │
    │ • Save CSV (raw data)      │
    │ • Save JSON (summary)      │
    │ • Save TXT (report)        │
    │ • Save PDF (figures)       │
    └────────────────────────────┘


═══════════════════════════════════════════════════════════════════

                    🎓 Statistical Analysis Pipeline

    Raw Results (CSV)
         │
         ├─► Metric: ACC, NMI, ARI, F1
         │
         ▼
    ┌────────────────────────────┐
    │ 1. Descriptive Statistics  │
    │                            │
    │ • Mean (μ)                 │
    │ • Std Dev (σ)              │
    │ • Min / Max                │
    │ • Quartiles                │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ 2. Sensitivity Scores      │
    │                            │
    │ Score = (max - min) / max  │
    │                            │
    │ Interpretation:            │
    │ • >0.7: High sensitivity   │
    │ • 0.3-0.7: Medium          │
    │ • <0.3: Low (robust)       │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ 3. Correlation Analysis    │
    │                            │
    │ Pearson's r:               │
    │   r = cov(X,Y) /           │
    │       (σ_X × σ_Y)          │
    │                            │
    │ Interpretation:            │
    │ • |r| > 0.7: Strong        │
    │ • |r| 0.3-0.7: Moderate    │
    │ • |r| < 0.3: Weak          │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ 4. Significance Testing    │
    │                            │
    │ P-value (two-tailed):      │
    │ • p < 0.001: ***           │
    │ • p < 0.01: **             │
    │ • p < 0.05: *              │
    │ • p ≥ 0.05: ns             │
    └──────────┬─────────────────┘
               │
               ▼
    Statistical Report (TXT)


═══════════════════════════════════════════════════════════════════

                    🔍 Troubleshooting Flowchart

    Problem?
       │
       ├─► ImportError
       │      │
       │      └─► Check PYTHONPATH
       │          export PYTHONPATH="${PWD}:${PYTHONPATH}"
       │
       ├─► CUDA OOM
       │      │
       │      └─► Reduce batch_size
       │          config.training.batch_size = 64
       │
       ├─► Not reproducible
       │      │
       │      └─► Set fixed seed
       │          config.training.seed = 42
       │
       ├─► Plots pixelated
       │      │
       │      └─► Increase DPI
       │          fig.savefig(..., dpi=600)
       │
       ├─► LaTeX errors
       │      │
       │      └─► Add packages
       │          \usepackage{booktabs}
       │
       └─► Experiments too slow
              │
              ├─► Option 1: Reduce epochs
              │   --epochs 100
              │
              ├─► Option 2: Fewer runs
              │   --n_runs 3
              │
              └─► Option 3: Use GPU
                  (automatic if available)


═══════════════════════════════════════════════════════════════════

Quick Commands Summary:

# Test (5 min)
uv run python scripts/test_sensitivity_analysis.py

# Demo
python scripts/demo_sensitivity_usage.py

# Single param (1-2h)
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode single --param lambda_gw \
  --n_runs 5 --epochs 200

# Grid search (3-5h)
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode grid \
  --param1 lambda_gw --param2 lambda_cluster \
  --n_runs 5 --epochs 200 --plot_3d

# Full analysis (8-12h)
uv run python scripts/run_sensitivity_analysis.py \
  --dataset Scene15 --mode full \
  --n_runs 10 --epochs 200

# Generate LaTeX
python scripts/generate_latex_tables.py \
  --input sensitivity_results/Scene15_lambda_gw \
  --output tables.tex

═══════════════════════════════════════════════════════════════════
```

## Visual Legend

### Symbols
- `│` `└` `├` `┌` `┐` `┴` `┬` : Box drawing characters
- `▼` `►` : Flow direction
- `✓` : Completed/Success
- `•` : Bullet point
- `📊` : Data/Statistics
- `📈` : Analysis/Trends
- `📝` : Writing/Documentation
- `🎯` : Goal/Target
- `⚙️` : Processing/Computation
- `🔍` : Troubleshooting
- `🎓` : Academic/Research

### Box Types
- `┌────────┐` : Process/Action
- `╔════════╗` : Important highlight
- `├────────┤` : Decision point

### File Structure Notation
```
directory/
├── file1.ext           # Description
├── subdirectory/
│   ├── file2.ext       # Description
│   └── file3.ext       # Description
└── file4.ext           # Description
```

---

**Note**: This is a text-based diagram. For interactive version, see the demo script:
```bash
python scripts/demo_sensitivity_usage.py
```
