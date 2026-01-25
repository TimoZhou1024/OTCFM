"""
Generate publication-quality sensitivity analysis figures for ICML paper.

This script generates:
1. Main text figure: 2x2 panel for top-4 parameters (lambda_gw, lambda_cluster, lambda_recon, ode_steps)
2. Appendix figure: 3x3 panel for all 9 parameters

Usage:
    uv run python scripts/generate_sensitivity_figures.py --results_dir sensitivity_results/Scene15_full_analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ICML-style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'text.usetex': False,  # Set True if LaTeX is available
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# Color palette for ICML (colorblind-friendly)
COLORS = {
    'main': '#2E86AB',      # Blue
    'accent': '#E94F37',    # Red
    'secondary': '#F6AE2D', # Yellow
    'tertiary': '#86BA90',  # Green
    'gray': '#666666'
}


def load_sensitivity_data(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all sensitivity analysis CSV files from results directory."""
    results_path = Path(results_dir)
    data = {}
    
    param_files = {
        'lambda_gw': 'lambda_gw',
        'lambda_cluster': 'lambda_cluster',
        'lambda_recon': 'lambda_recon',
        'lambda_contrastive': 'lambda_contrastive',
        'ode_steps': 'ode_steps',
        'latent_dim': 'latent_dim',
        'flow_hidden_dim': 'flow_hidden_dim',
        'learning_rate': 'learning_rate',
        'dropout': 'dropout'
    }
    
    for param_name, file_pattern in param_files.items():
        csv_files = list(results_path.glob(f"*_full_{file_pattern}_*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            data[param_name] = df
            print(f"Loaded {param_name}: {len(df)} configurations")
        else:
            print(f"Warning: No CSV found for {param_name}")
    
    return data


def load_statistics(results_dir: str) -> Dict[str, Dict]:
    """Load statistics from JSON files."""
    results_path = Path(results_dir)
    stats = {}
    
    json_files = list(results_path.glob("*_full_*.json"))
    for json_file in json_files:
        param_name = json_file.stem.split('_full_')[1].rsplit('_', 1)[0]
        with open(json_file, 'r') as f:
            stats[param_name] = json.load(f)
    
    return stats


def aggregate_by_param(df: pd.DataFrame, param_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate results by parameter value, computing mean and std of ACC."""
    # CSV files use 'value' column for parameter values, or the ACC_mean/ACC_std columns already
    if 'value' in df.columns:
        # If we have raw run data, aggregate by value
        if 'ACC' in df.columns:
            grouped = df.groupby('value')['ACC'].agg(['mean', 'std']).reset_index()
            param_values = grouped['value'].values
            means = grouped['mean'].values
            stds = grouped['std'].values
        else:
            # Already aggregated: columns are value, ACC_mean, ACC_std
            param_values = df['value'].values
            means = df['ACC_mean'].values
            stds = df['ACC_std'].values
    elif param_col in df.columns:
        grouped = df.groupby(param_col)['ACC'].agg(['mean', 'std']).reset_index()
        param_values = grouped[param_col].values
        means = grouped['mean'].values
        stds = grouped['std'].values
    else:
        raise KeyError(f"Cannot find parameter column. Available: {df.columns.tolist()}")
    
    # Sort by parameter value
    sort_idx = np.argsort(param_values)
    return param_values[sort_idx], means[sort_idx], stds[sort_idx]


def plot_single_sensitivity(ax, param_values: np.ndarray, means: np.ndarray, stds: np.ndarray,
                           param_name: str, best_value: float, xlabel: str = None,
                           use_log_scale: bool = False, show_optimal: bool = True):
    """Plot a single sensitivity curve with error bands."""
    
    # Main line
    ax.plot(param_values, means, 'o-', color=COLORS['main'], linewidth=1.5, 
            markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    
    # Error band (±1 std)
    ax.fill_between(param_values, means - stds, means + stds, 
                   alpha=0.2, color=COLORS['main'], linewidth=0)
    
    # Mark optimal point
    if show_optimal:
        best_idx = np.argmin(np.abs(param_values - best_value))
        ax.scatter([param_values[best_idx]], [means[best_idx]], 
                  color=COLORS['accent'], s=80, zorder=5, marker='*',
                  edgecolors='white', linewidths=0.5)
        ax.annotate(f'{best_value:.3g}', 
                   xy=(param_values[best_idx], means[best_idx]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, color=COLORS['accent'])
    
    # Formatting
    if use_log_scale:
        ax.set_xscale('log')
    
    ax.set_xlabel(xlabel or param_name)
    ax.set_ylabel('ACC')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    y_min, y_max = means.min() - stds.max() * 1.5, means.max() + stds.max() * 1.5
    ax.set_ylim(max(0, y_min), min(1, y_max))


def create_main_figure(data: Dict[str, pd.DataFrame], stats: Dict[str, Dict], 
                       output_path: str):
    """Create the main 2x2 figure for the paper body."""
    
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))  # ICML column width is ~3.25 inches
    axes = axes.flatten()
    
    # Top 4 parameters by sensitivity
    params_config = [
        ('lambda_gw', r'$\lambda_{gw}$', True),
        ('lambda_cluster', r'$\lambda_{cluster}$', True),
        ('lambda_recon', r'$\lambda_{recon}$', True),
        ('ode_steps', 'ODE Steps', False),
    ]
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for ax, (param_name, xlabel, use_log), label in zip(axes, params_config, subplot_labels):
        if param_name not in data:
            ax.text(0.5, 0.5, f'No data for {param_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        df = data[param_name]
        param_col = param_name
        
        param_values, means, stds = aggregate_by_param(df, param_col)
        
        # Get best value from statistics
        best_value = float(stats.get(param_name, {}).get('ACC', {}).get('best_value', param_values[np.argmax(means)]))
        
        plot_single_sensitivity(ax, param_values, means, stds, param_name, 
                               best_value, xlabel, use_log_scale=use_log)
        
        # Add subplot label
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for ext in ['pdf', 'png']:
        save_path = f"{output_path}_main_4panel.{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def create_appendix_figure(data: Dict[str, pd.DataFrame], stats: Dict[str, Dict],
                           output_path: str):
    """Create the full 3x3 figure for the appendix."""
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    axes = axes.flatten()
    
    # All 9 parameters in order of sensitivity
    params_config = [
        ('lambda_gw', r'$\lambda_{gw}$', True),
        ('lambda_recon', r'$\lambda_{recon}$', True),
        ('lambda_cluster', r'$\lambda_{cluster}$', True),
        ('ode_steps', 'ODE Steps', False),
        ('latent_dim', 'Latent Dim', False),
        ('dropout', 'Dropout', False),
        ('learning_rate', 'Learning Rate', True),
        ('lambda_contrastive', r'$\lambda_{contrastive}$', True),
        ('flow_hidden_dim', 'Flow Hidden Dim', False),
    ]
    
    for ax, (param_name, xlabel, use_log) in zip(axes, params_config):
        if param_name not in data:
            ax.text(0.5, 0.5, f'No data for\n{param_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xlabel(xlabel)
            continue
        
        df = data[param_name]
        param_col = param_name
        
        param_values, means, stds = aggregate_by_param(df, param_col)
        
        # Get best value from statistics
        best_value = float(stats.get(param_name, {}).get('ACC', {}).get('best_value', param_values[np.argmax(means)]))
        
        plot_single_sensitivity(ax, param_values, means, stds, param_name,
                               best_value, xlabel, use_log_scale=use_log)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for ext in ['pdf', 'png']:
        save_path = f"{output_path}_full_9panel.{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def create_sensitivity_bar_chart(stats: Dict[str, Dict], output_path: str):
    """Create a bar chart showing sensitivity scores for all parameters."""
    
    # Extract sensitivity scores
    param_scores = []
    for param_name, param_stats in stats.items():
        if 'ACC' in param_stats and 'sensitivity_score' in param_stats['ACC']:
            param_scores.append({
                'param': param_name,
                'sensitivity': param_stats['ACC']['sensitivity_score']
            })
    
    if not param_scores:
        print("No sensitivity scores found in statistics")
        return
    
    # Sort by sensitivity
    param_scores = sorted(param_scores, key=lambda x: x['sensitivity'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(6.5, 3))
    
    params = [p['param'] for p in param_scores]
    scores = [p['sensitivity'] for p in param_scores]
    
    # Create bar chart
    bars = ax.barh(range(len(params)), scores, color=COLORS['main'], alpha=0.8, edgecolor='white')
    
    # Highlight top 4
    for i in range(min(4, len(bars))):
        bars[i].set_color(COLORS['accent'])
    
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p.replace('_', r'\_') for p in params])
    ax.set_xlabel('Sensitivity Score $S$')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add values on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        save_path = f"{output_path}_sensitivity_ranking.{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate sensitivity analysis figures for ICML paper')
    parser.add_argument('--results_dir', type=str, 
                       default='sensitivity_results/Scene15_full_analysis',
                       help='Directory containing sensitivity analysis results')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for generated figures')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / 'sensitivity')
    
    print(f"Loading data from: {args.results_dir}")
    
    # Load data
    data = load_sensitivity_data(args.results_dir)
    stats = load_statistics(args.results_dir)
    
    if not data:
        print("Error: No sensitivity data found!")
        return
    
    print(f"\nLoaded {len(data)} parameters")
    print(f"Generating figures...")
    
    # Generate figures
    create_main_figure(data, stats, output_path)
    create_appendix_figure(data, stats, output_path)
    create_sensitivity_bar_chart(stats, output_path)
    
    print(f"\nDone! Figures saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {output_path}_main_4panel.pdf  (for main text Figure)")
    print(f"  - {output_path}_full_9panel.pdf  (for Appendix Figure)")
    print(f"  - {output_path}_sensitivity_ranking.pdf  (optional bar chart)")


if __name__ == '__main__':
    main()
