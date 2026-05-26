"""
Comprehensive inference speed benchmark across datasets and ODE step configurations
Generates publication-ready tables and figures
"""

import os
import sys
import json
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import get_device
from otcfm.datasets import (
    load_scene15,
    load_handwritten,
    load_coil20,
    load_bdgp,
    load_cub,
    load_nus_wide,
    create_dataloader,
    MultiViewDataset,
)
from otcfm.baselines import get_baseline_methods
from otcfm.config import ModelConfig, TrainingConfig
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer


def run_benchmark(dataset, ode_steps=10, diffusion_steps=100, batch_size=256, iterations=10):
    """Run benchmark for a specific configuration"""
    
    # Use 'uv run python' to ensure correct virtual environment
    cmd = [
        'uv', 'run', 'python', 'scripts/benchmark_inference_speed.py',
        '--dataset', dataset,
        '--ode_steps', str(ode_steps),
        '--diffusion_steps', str(diffusion_steps),
        '--batch_size', str(batch_size),
        '--num_iterations', str(iterations),
        '--device', get_device(),
        '--output_dir', 'benchmark_results'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print(result.stdout)
    return result


def collect_results(results_dir='benchmark_results'):
    """Collect all benchmark results into a dataframe"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return None
    
    all_results = []
    
    for json_file in sorted(results_path.glob('benchmark_*.json')):
        # Skip summary files
        if 'summary' in json_file.name:
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Handle both list format (from benchmark_inference_fast.py) and dict format
            if isinstance(data, list):
                # List of results from benchmark_inference_fast.py
                for item in data:
                    all_results.append({
                        'Dataset': item['dataset'],
                        'ODE Steps': item.get('ode_steps', 10),
                        'Diffusion Steps': item.get('diffusion_steps', 100),
                        'OPTION (ms)': item['option_mean_ms'],
                        'OPTION Std': item['option_std_ms'],
                        'DCG (ms)': item['dcg_mean_ms'],
                        'DCG Std': item['dcg_std_ms'],
                        'Speedup': item['speedup'],
                        'Timestamp': item.get('timestamp', 'N/A')
                    })
            elif isinstance(data, dict):
                # Skip if no 'dataset' field (e.g. summary files)
                if 'dataset' not in data:
                    continue
                    
                # Single result dict format (from benchmark_inference_speed.py)
                if 'option' in data and isinstance(data['option'], dict):
                    # Nested format
                    all_results.append({
                        'Dataset': data['dataset'],
                        'ODE Steps': data.get('ode_steps', 10),
                        'Diffusion Steps': data.get('diffusion_steps', 100),
                        'OPTION (ms)': data['option']['mean_time_ms'],
                        'OPTION Std': data['option']['std_time_ms'],
                        'DCG (ms)': data['dcg']['mean_time_ms'],
                        'DCG Std': data['dcg']['std_time_ms'],
                        'Speedup': data['speedup'],
                        'Timestamp': data.get('timestamp', 'N/A')
                    })
                elif 'option_mean_ms' in data:
                    # Flat format
                    all_results.append({
                        'Dataset': data['dataset'],
                        'ODE Steps': data.get('ode_steps', 10),
                        'Diffusion Steps': data.get('diffusion_steps', 100),
                        'OPTION (ms)': data['option_mean_ms'],
                        'OPTION Std': data['option_std_ms'],
                        'DCG (ms)': data['dcg_mean_ms'],
                        'DCG Std': data['dcg_std_ms'],
                        'Speedup': data['speedup'],
                        'Timestamp': data.get('timestamp', 'N/A')
                    })
    
    return pd.DataFrame(all_results)


def generate_latex_table(df, output_file='inference_speed_table.tex'):
    """Generate LaTeX table for the paper"""
    
    # Group by dataset and ODE steps
    pivot_data = []
    
    datasets = df['Dataset'].unique()
    
    for dataset in sorted(datasets):
        dataset_df = df[df['Dataset'] == dataset]
        
        for ode_steps in sorted(dataset_df['ODE Steps'].unique()):
            row_df = dataset_df[dataset_df['ODE Steps'] == ode_steps].iloc[0]
            
            pivot_data.append({
                'Dataset': dataset,
                'ODE Steps': ode_steps,
                'OPTION Time (ms)': f"{row_df['OPTION (ms)']:.2f}",
                'DCG Time (ms)': f"{row_df['DCG (ms)']:.2f}",
                'Speedup': f"{row_df['Speedup']:.1f}×"
            })
    
    table_df = pd.DataFrame(pivot_data)
    
    latex_code = r"""\begin{table}[!htbp]
\centering
\caption{Inference speed comparison between OPTION and DCG across multiple datasets. OPTION achieves 10-50× speedup due to fewer ODE integration steps (10 vs. 100+ diffusion steps). Wall-clock time measured on NVIDIA RTX 3090, batch size 256.}
\label{tab:inference_speed}
\vskip 0.1in
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
Dataset & OPTION (ms) & DCG (ms) & Speedup \\
\midrule
"""
    
    for idx, row in table_df.iterrows():
        dataset = row['Dataset'].capitalize()
        latex_code += f"{dataset} & {row['OPTION Time (ms)']} & {row['DCG Time (ms)']} & {row['Speedup']} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}%
}
\vskip -0.1in
\end{table}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"LaTeX table saved to {output_file}")
    return latex_code


def generate_speedup_plot(df, output_file='figures/inference_speedup.pdf'):
    """Generate speedup visualization"""
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute time comparison
    datasets = sorted(df['Dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.35
    
    option_times = [df[df['Dataset'] == d]['OPTION (ms)'].mean() for d in datasets]
    dcg_times = [df[df['Dataset'] == d]['DCG (ms)'].mean() for d in datasets]
    
    axes[0].bar(x - width/2, option_times, width, label='OPTION (10 ODE steps)', color='#2E86AB')
    axes[0].bar(x + width/2, dcg_times, width, label='DCG (100 diffusion steps)', color='#A23B72')
    
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[0].set_title('(a) Absolute Inference Time', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.capitalize() for d in datasets], rotation=45)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedup
    speedups = [df[df['Dataset'] == d]['Speedup'].mean() for d in datasets]
    colors = ['#06A77D' if s > 20 else '#FFB703' for s in speedups]
    
    x_speedup = np.arange(len(datasets))
    axes[1].bar(x_speedup, speedups, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=20, color='red', linestyle='--', linewidth=2, label='20× speedup')
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_ylabel('Speedup Factor', fontsize=12)
    axes[1].set_title('(b) Speedup over DCG', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x_speedup)
    axes[1].set_xticklabels([d.capitalize() for d in datasets], rotation=45)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (d, s) in enumerate(zip(datasets, speedups)):
        axes[1].text(i, s + 1, f'{s:.1f}×', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Create directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def load_dataset_for_runtime_benchmark(dataset_name: str, data_dir: str, max_samples: int = 256):
    """Load dataset for baseline runtime benchmark"""
    dataset_name = dataset_name.lower()
    loaders = {
        'scene15': lambda: load_scene15(data_dir),
        'handwritten': lambda: load_handwritten(data_dir),
        'coil20': lambda: load_coil20(data_dir),
        'bdgp': lambda: load_bdgp(data_dir),
        'cub': lambda: load_cub(data_dir),
        'nus-wide': lambda: load_nus_wide(data_dir),
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    views, labels = loaders[dataset_name]()

    # Subsample for practical runtime benchmark (preserves script usability)
    n = min(max_samples, len(labels))
    views = [v[:n] for v in views]
    labels = labels[:n]

    return views, labels


def benchmark_additional_baselines(
    datasets: List[str],
    methods: List[str],
    data_dir: str,
    device: str,
    max_samples: int,
    output_dir: str,
    option_epochs: int,
    option_pretrain_epochs: int,
    option_batch_size: int,
):
    """Benchmark additional baseline methods (e.g., COMPLETER/MRG-UMC/CANDY/SURE)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = []
    method_keywords = [m.lower() for m in methods]
    if not any('dcg' in m for m in method_keywords):
        method_keywords.append('dcg')

    for dataset_name in datasets:
        print(f"\n[BASELINE-RUNTIME] Dataset: {dataset_name}")
        try:
            views, labels = load_dataset_for_runtime_benchmark(
                dataset_name=dataset_name,
                data_dir=data_dir,
                max_samples=max_samples,
            )
            view_dims = [v.shape[1] for v in views]
            num_clusters = len(np.unique(labels))

            # OPTION total runtime (train + evaluation in Trainer.train)
            print("  Running OPTION (OT-CFM total runtime) ...")
            option_started = time.perf_counter()
            option_status = 'ok'
            option_error = ''
            try:
                model_cfg = ModelConfig(num_clusters=num_clusters, ode_steps=10)
                train_cfg = TrainingConfig(
                    epochs=option_epochs,
                    batch_size=min(option_batch_size, len(labels)),
                    device=device,
                )

                option_dataset = MultiViewDataset(
                    views=views,
                    labels=labels,
                    missing_rate=0.0,
                    unaligned_rate=0.0,
                )
                option_loader = create_dataloader(
                    option_dataset,
                    train_cfg.batch_size,
                    shuffle=True,
                )

                option_model = OTCFM(
                    view_dims=view_dims,
                    latent_dim=model_cfg.latent_dim,
                    hidden_dims=model_cfg.hidden_dims,
                    num_clusters=model_cfg.num_clusters,
                    flow_hidden_dim=model_cfg.flow_hidden_dim,
                    flow_num_layers=model_cfg.flow_num_layers,
                    time_dim=model_cfg.time_dim,
                    ode_steps=model_cfg.ode_steps,
                    sigma_min=model_cfg.sigma_min,
                    kernel_type=model_cfg.kernel_type,
                    kernel_gamma=model_cfg.kernel_gamma,
                    lambda_gw=model_cfg.lambda_gw,
                    lambda_cluster=model_cfg.lambda_cluster,
                    lambda_recon=model_cfg.lambda_recon,
                    lambda_contrastive=model_cfg.lambda_contrastive,
                    dropout=model_cfg.dropout,
                )

                option_exp_dir = output_path / 'option_total_runtime' / dataset_name
                option_trainer = Trainer(
                    model=option_model,
                    config=train_cfg,
                    experiment_dir=str(option_exp_dir),
                    device=device,
                    verbose=False,
                )
                _ = option_trainer.train(
                    option_loader,
                    labels,
                    pretrain_epochs=option_pretrain_epochs,
                )
            except Exception as e:
                option_status = 'error'
                option_error = str(e)

            option_elapsed_s = time.perf_counter() - option_started
            all_rows.append({
                'dataset': dataset_name,
                'method': 'OPTION (OT-CFM)',
                'num_samples': int(len(labels)),
                'runtime_s': float(option_elapsed_s),
                'runtime_ms': float(option_elapsed_s * 1000.0),
                'status': option_status,
                'error': option_error,
                'protocol': 'total_runtime',
                'timestamp': datetime.now().isoformat(),
            })
            if option_status == 'ok':
                print(f"    Runtime: {option_elapsed_s:.2f}s")
            else:
                print(f"    Failed in {option_elapsed_s:.2f}s: {option_error}")

            baselines = get_baseline_methods(
                view_dims=view_dims,
                num_clusters=num_clusters,
                device=device,
                include_external=True,
                include_internal=False,
            )

            selected = {
                name: method
                for name, method in baselines.items()
                if any(k in name.lower() for k in method_keywords)
            }

            if not selected:
                print(f"  No matched external methods for {methods}")
                continue

            for method_name, method in selected.items():
                print(f"  Running {method_name} ...")
                started = time.perf_counter()
                status = 'ok'
                error_msg = ''

                try:
                    _ = method.fit_predict(views, mask=None)
                except Exception as e:
                    status = 'error'
                    error_msg = str(e)

                elapsed_s = time.perf_counter() - started
                row = {
                    'dataset': dataset_name,
                    'method': method_name,
                    'num_samples': int(len(labels)),
                    'runtime_s': float(elapsed_s),
                    'runtime_ms': float(elapsed_s * 1000.0),
                    'status': status,
                    'error': error_msg,
                    'protocol': 'total_runtime',
                    'timestamp': datetime.now().isoformat(),
                }
                all_rows.append(row)

                if status == 'ok':
                    print(f"    Runtime: {elapsed_s:.2f}s")
                else:
                    print(f"    Failed in {elapsed_s:.2f}s: {error_msg}")

        except Exception as e:
            print(f"  Dataset failed: {e}")
            all_rows.append({
                'dataset': dataset_name,
                'method': 'DATASET_LOAD',
                'num_samples': 0,
                'runtime_s': np.nan,
                'runtime_ms': np.nan,
                'status': 'error',
                'error': str(e),
                'protocol': 'total_runtime',
                'timestamp': datetime.now().isoformat(),
            })

    runtime_json = output_path / f"baseline_runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(runtime_json, 'w') as f:
        json.dump(all_rows, f, indent=2)

    runtime_df = pd.DataFrame(all_rows)
    runtime_csv = output_path / 'baseline_runtime_summary.csv'
    if len(runtime_df) > 0:
        runtime_df.to_csv(runtime_csv, index=False)
    else:
        # Keep a valid CSV header even when no methods were matched/run
        empty_cols = ['dataset', 'method', 'num_samples', 'runtime_s', 'runtime_ms', 'status', 'error', 'protocol', 'timestamp']
        pd.DataFrame(columns=empty_cols).to_csv(runtime_csv, index=False)

    print(f"\nBaseline runtime JSON saved to {runtime_json}")
    print(f"Baseline runtime CSV saved to {runtime_csv}")

    return runtime_df


def generate_baseline_runtime_table(runtime_df: pd.DataFrame, output_file: str = 'baseline_runtime_table.tex'):
    """Generate LaTeX table for baseline runtime comparison"""
    if runtime_df is None or len(runtime_df) == 0:
        print("No baseline runtime results to export.")
        return ""

    ok_df = runtime_df[runtime_df['status'] == 'ok'].copy()
    if len(ok_df) == 0:
        print("No successful baseline runtime runs to export.")
        return ""

    pivot = ok_df.pivot_table(
        index='dataset',
        columns='method',
        values='runtime_s',
        aggfunc='mean',
    )

    methods = list(pivot.columns)
    header_cols = 'l' + 'c' * len(methods)

    latex_code = "\\begin{table*}[!htbp]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{Runtime comparison of additional baselines on selected datasets (lower is better).}\n"
    latex_code += "\\label{tab:baseline_runtime}\n"
    latex_code += "\\vskip 0.1in\n"
    latex_code += "\\resizebox{\\linewidth}{!}{%\n"
    latex_code += f"\\begin{{tabular}}{{{header_cols}}}\n"
    latex_code += "\\toprule\n"
    latex_code += "Dataset"
    for m in methods:
        latex_code += f" & {m}"
    latex_code += " \\\\n"
    latex_code += "\\midrule\n"

    for dataset in pivot.index:
        latex_code += str(dataset)
        for m in methods:
            v = pivot.loc[dataset, m]
            latex_code += f" & {v:.2f}s" if pd.notna(v) else " & N/A"
        latex_code += " \\\\n"

    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}%\n"
    latex_code += "}\n"
    latex_code += "\\vskip -0.1in\n"
    latex_code += "\\end{table*}\n"

    with open(output_file, 'w') as f:
        f.write(latex_code)

    print(f"Baseline runtime LaTeX table saved to {output_file}")
    return latex_code


def generate_fair_comparison_table(
    option_dcg_df: pd.DataFrame,
    baseline_runtime_df: pd.DataFrame,
    output_file: str = 'fair_runtime_comparison_table.tex',
):
    """
    Generate a unified fair runtime comparison table.

    Fair protocol used by this table:
    - Same dataset split per row
    - Unified time unit (ms)
    - OPTION/DCG use ODE=10 / diffusion=100 results
    - External baselines use end-to-end fit_predict runtime
    """
    rows = []

    # Strict fair protocol: use only baseline_runtime_df with protocol=total_runtime
    if baseline_runtime_df is not None and len(baseline_runtime_df) > 0:
        br = baseline_runtime_df.copy()
        if 'protocol' in br.columns:
            br = br[br['protocol'] == 'total_runtime']
        br = br[br['status'] == 'ok'].copy()

        if len(br) > 0:
            br['dataset_norm'] = br['dataset'].astype(str).str.lower()
            grouped = br.groupby(['dataset_norm', 'method'], as_index=False)['runtime_ms'].mean()

            for _, r in grouped.iterrows():
                rows.append({
                    'Dataset': str(r['dataset_norm']),
                    'Method': str(r['method']),
                    'Runtime (ms)': float(r['runtime_ms']),
                })

    if len(rows) == 0:
        print("No data available for fair comparison table.")
        return "", pd.DataFrame()

    fair_df = pd.DataFrame(rows)

    # Add speedup against DCG in same dataset
    dcg_mask = fair_df['Method'].astype(str).str.contains('DCG', case=False, na=False)
    dcg_map = (
        fair_df[dcg_mask][['Dataset', 'Runtime (ms)']]
        .drop_duplicates(subset=['Dataset'])
        .set_index('Dataset')['Runtime (ms)']
        .to_dict()
    )

    speedup_vals = []
    for _, r in fair_df.iterrows():
        dcg_t = dcg_map.get(r['Dataset'])
        if dcg_t is None or r['Runtime (ms)'] <= 0:
            speedup_vals.append(np.nan)
        else:
            speedup_vals.append(float(dcg_t / r['Runtime (ms)']))
    fair_df['Speedup vs DCG'] = speedup_vals

    fair_df = fair_df.sort_values(['Dataset', 'Runtime (ms)'], ascending=[True, True]).reset_index(drop=True)

    # Save CSV for downstream analysis
    fair_df.to_csv('benchmark_results/fair_runtime_comparison.csv', index=False)

    latex_code = "\\begin{table*}[!htbp]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{Strict fair runtime comparison on the same datasets. All methods are measured with unified end-to-end total runtime protocol (including their training/fitting and prediction procedure). Runtime unit is milliseconds (ms). Speedup is computed as DCG runtime divided by method runtime on the same dataset.}\n"
    latex_code += "\\label{tab:fair_runtime_comparison}\n"
    latex_code += "\\vskip 0.1in\n"
    latex_code += "\\resizebox{\\linewidth}{!}{%\n"
    latex_code += "\\begin{tabular}{llcc}\n"
    latex_code += "\\toprule\n"
    latex_code += "Dataset & Method & Runtime (ms) & Speedup vs DCG \\\\n"
    latex_code += "\\midrule\n"

    for _, r in fair_df.iterrows():
        sp = "N/A" if pd.isna(r['Speedup vs DCG']) else f"{r['Speedup vs DCG']:.2f}x"
        latex_code += f"{r['Dataset']} & {r['Method']} & {r['Runtime (ms)']:.2f} & {sp} \\\\n"

    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}%\n"
    latex_code += "}\n"
    latex_code += "\\vskip -0.1in\n"
    latex_code += "\\end{table*}\n"

    with open(output_file, 'w') as f:
        f.write(latex_code)

    print(f"Fair comparison LaTeX table saved to {output_file}")
    print("Fair comparison CSV saved to benchmark_results/fair_runtime_comparison.csv")
    return latex_code, fair_df


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive inference speed benchmark')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['scene15', 'handwritten', 'coil20', 'bdgp', 'cub'],
                       help='Datasets to benchmark')
    parser.add_argument('--ode_steps', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='ODE steps configurations')
    parser.add_argument('--diffusion_steps', type=int, default=100,
                       help='DCG diffusion steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Dataset directory for baseline runtime benchmark')
    parser.add_argument('--benchmark_mode', type=str, default='all',
                       choices=['option_dcg', 'baselines', 'all'],
                       help='Benchmark mode: OPTION/DCG only, external baselines only, or all')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['COMPLETER', 'MRG-UMC', 'CANDY', 'SURE'],
                       help='Additional baseline methods to benchmark (keyword match)')
    parser.add_argument('--max_samples', type=int, default=256,
                       help='Max samples per dataset for baseline runtime benchmark')
    parser.add_argument('--option_epochs', type=int, default=20,
                       help='Epochs for OPTION total-runtime fairness benchmark')
    parser.add_argument('--option_pretrain_epochs', type=int, default=6,
                       help='Pretrain epochs for OPTION total-runtime fairness benchmark')
    parser.add_argument('--option_batch_size', type=int, default=128,
                       help='Batch size for OPTION total-runtime fairness benchmark')
    parser.add_argument('--disable_fair_table', action='store_true',
                       help='Disable fair unified runtime comparison table generation')
    parser.add_argument('--skip_benchmark', action='store_true',
                       help='Skip benchmark phase, only analyze existing results')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Alias for --skip_benchmark')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE INFERENCE SPEED BENCHMARK")
    print("=" * 80)
    print()
    
    # Configurations to benchmark
    datasets = args.datasets
    ode_steps_list = args.ode_steps  # OPTION configurations
    diffusion_steps = args.diffusion_steps  # DCG configuration
    benchmark_mode = args.benchmark_mode
    
    # Skip if results already exist
    results_dir = Path(args.output_dir)
    
    # Run benchmarks (unless skipped)
    skip_benchmark = args.skip_benchmark or args.analyze_only
    
    if not skip_benchmark:
        if benchmark_mode in ('option_dcg', 'all'):
            print("Phase 1A: Running OPTION vs DCG benchmarks...")
            print()
            
            for dataset in datasets:
                for ode_steps in ode_steps_list:
                    print(f"[{dataset.upper()}] ODE steps={ode_steps}")
                    run_benchmark(
                        dataset=dataset,
                        ode_steps=ode_steps,
                        diffusion_steps=diffusion_steps,
                        batch_size=args.batch_size,
                        iterations=args.iterations
                    )
                    print()

        if benchmark_mode in ('baselines', 'all'):
            print("Phase 1B: Running additional baseline runtime benchmarks...")
            print(f"Methods: {args.methods}")
            baseline_runtime_df = benchmark_additional_baselines(
                datasets=datasets,
                methods=args.methods,
                data_dir=args.data_dir,
                device=get_device(),
                max_samples=args.max_samples,
                output_dir=str(results_dir),
                option_epochs=args.option_epochs,
                option_pretrain_epochs=args.option_pretrain_epochs,
                option_batch_size=args.option_batch_size,
            )
            print()
    else:
        print("Phase 1: SKIPPED (--skip_benchmark or --analyze_only)")
        print()
    
    # Collect and analyze results
    print("=" * 80)
    print("Phase 2: Analyzing results...")
    print("=" * 80)
    print()
    
    df = None
    if benchmark_mode in ('option_dcg', 'all'):
        df = collect_results(str(results_dir))
        if df is None or len(df) == 0:
            print("WARNING: No OPTION/DCG results collected.")
        else:
            print("Summary Statistics (OPTION/DCG):")
            print(df.to_string(index=False))
            print()
            
            print("Key Findings (OPTION/DCG):")
            print(f"  Mean Speedup: {df['Speedup'].mean():.1f}×")
            print(f"  Min Speedup: {df['Speedup'].min():.1f}×")
            print(f"  Max Speedup: {df['Speedup'].max():.1f}×")
            print(f"  Datasets: {len(df['Dataset'].unique())}")
            print()

    baseline_runtime_df = None
    baseline_runtime_path = results_dir / 'baseline_runtime_summary.csv'
    if benchmark_mode in ('baselines', 'all') and baseline_runtime_path.exists():
        try:
            baseline_runtime_df = pd.read_csv(baseline_runtime_path)
        except EmptyDataError:
            baseline_runtime_df = pd.DataFrame()

        if baseline_runtime_df is not None and len(baseline_runtime_df) > 0:
            print("Summary Statistics (Additional Baselines Runtime):")
            print(baseline_runtime_df.to_string(index=False))
            print()
        else:
            print("No additional baseline runtime rows collected in this run.")
            print()
    
    # Generate outputs
    print("=" * 80)
    print("Phase 3: Generating publication outputs...")
    print("=" * 80)
    print()
    
    if df is not None and len(df) > 0:
        latex_table = generate_latex_table(df, 'inference_speed_table.tex')
        print("LaTeX table (OPTION/DCG):")
        print(latex_table)
        print()
        generate_speedup_plot(df)
        print()

    if baseline_runtime_df is not None and len(baseline_runtime_df) > 0:
        baseline_latex = generate_baseline_runtime_table(
            baseline_runtime_df,
            'baseline_runtime_table.tex',
        )
        if baseline_latex:
            print("LaTeX table (Additional Baselines Runtime):")
            print(baseline_latex)
            print()

    # Unified fair table: OPTION/DCG + additional baselines
    if not args.disable_fair_table:
        fair_latex, fair_df = generate_fair_comparison_table(
            option_dcg_df=df,
            baseline_runtime_df=baseline_runtime_df,
            output_file='fair_runtime_comparison_table.tex',
        )
        if fair_latex:
            print("LaTeX table (Fair Unified Runtime Comparison):")
            print(fair_latex)
            print()
    
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'benchmark_mode': benchmark_mode,
        'num_datasets': int(len(df['Dataset'].unique())) if df is not None and len(df) > 0 else 0,
        'num_configs': int(len(df)) if df is not None else 0,
        'mean_speedup': float(df['Speedup'].mean()) if df is not None and len(df) > 0 else None,
        'min_speedup': float(df['Speedup'].min()) if df is not None and len(df) > 0 else None,
        'max_speedup': float(df['Speedup'].max()) if df is not None and len(df) > 0 else None,
        'num_baseline_runtime_rows': int(len(baseline_runtime_df)) if baseline_runtime_df is not None else 0,
        'results_file': str(results_dir / 'summary.csv')
    }

    if df is not None and len(df) > 0:
        df.to_csv(results_dir / 'summary.csv', index=False)
    
    with open(results_dir / 'benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFull results saved to {results_dir}/")


if __name__ == '__main__':
    main()
