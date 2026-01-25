"""
Visualize multi-seed convergence analysis with mean ± std plots
Creates publication-quality figures with shaded error regions.

Usage:
    uv run python scripts/plot_multi_seed_convergence.py --results_dir multi_seed_results/Handwritten_20260122_120000
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """Setup publication-quality plot style"""
    # Use seaborn style with white background
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.3)
    
    # Set matplotlib parameters
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'lines.linewidth': 2.0,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_convergence_with_std(aggregated_data, output_path, dataset_name='Dataset'):
    """
    Create 2-subplot figure with mean ± std visualization
    
    Top subplot: ACC, NMI, ARI metrics
    Bottom subplot: Total loss and loss components
    """
    setup_plot_style()
    
    # Get data
    epochs = np.array(aggregated_data['epochs'])
    n_seeds = aggregated_data['n_seeds']
    
    # Use seaborn deep palette for distinct colors
    colors = sns.color_palette('deep', 8)
    
    # Create figure with 2 vertical subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f'Convergence Analysis: {dataset_name}\n(Mean ± Std across {n_seeds} runs)',
                 fontweight='bold', fontsize=16)
    
    # ==================== Top subplot: Metrics ====================
    ax1 = axes[0]
    
    # Plot ACC
    acc_mean = np.array(aggregated_data['metrics']['acc']['mean']) * 100
    acc_std = np.array(aggregated_data['metrics']['acc']['std']) * 100
    ax1.plot(epochs, acc_mean, label='ACC', color=colors[0], linewidth=2.5, zorder=3)
    ax1.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                     alpha=0.2, color=colors[0], zorder=1)
    
    # Plot NMI
    nmi_mean = np.array(aggregated_data['metrics']['nmi']['mean']) * 100
    nmi_std = np.array(aggregated_data['metrics']['nmi']['std']) * 100
    ax1.plot(epochs, nmi_mean, label='NMI', color=colors[1], linewidth=2.5, zorder=3)
    ax1.fill_between(epochs, nmi_mean - nmi_std, nmi_mean + nmi_std,
                     alpha=0.2, color=colors[1], zorder=1)
    
    # Plot ARI
    ari_mean = np.array(aggregated_data['metrics']['ari']['mean']) * 100
    ari_std = np.array(aggregated_data['metrics']['ari']['std']) * 100
    ax1.plot(epochs, ari_mean, label='ARI', color=colors[2], linewidth=2.5, zorder=3)
    ax1.fill_between(epochs, ari_mean - ari_std, ari_mean + ari_std,
                     alpha=0.2, color=colors[2], zorder=1)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Metric Value (%)', fontweight='bold')
    ax1.set_title('Clustering Metrics', fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])
    
    # Add final values as text
    final_acc = acc_mean[-1]
    final_nmi = nmi_mean[-1]
    final_ari = ari_mean[-1]
    ax1.text(0.02, 0.98, 
             f'Final: ACC={final_acc:.1f}%, NMI={final_nmi:.1f}%, ARI={final_ari:.1f}%',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== Bottom subplot: Losses ====================
    ax2 = axes[1]
    
    # Plot Total Loss
    total_mean = np.array(aggregated_data['losses']['loss']['mean'])
    total_std = np.array(aggregated_data['losses']['loss']['std'])
    ax2.plot(epochs, total_mean, label='Total Loss', color=colors[3], 
             linewidth=2.5, linestyle='-', zorder=3)
    ax2.fill_between(epochs, total_mean - total_std, total_mean + total_std,
                     alpha=0.2, color=colors[3], zorder=1)
    
    # Plot Reconstruction Loss
    recon_mean = np.array(aggregated_data['losses']['recon']['mean'])
    recon_std = np.array(aggregated_data['losses']['recon']['std'])
    ax2.plot(epochs, recon_mean, label='Reconstruction', color=colors[4],
             linewidth=2.0, linestyle='--', zorder=3)
    ax2.fill_between(epochs, recon_mean - recon_std, recon_mean + recon_std,
                     alpha=0.15, color=colors[4], zorder=1)
    
    # Plot GW Loss (gc_loss in the requirement)
    gw_mean = np.array(aggregated_data['losses']['gw']['mean'])
    gw_std = np.array(aggregated_data['losses']['gw']['std'])
    ax2.plot(epochs, gw_mean, label='GW Alignment', color=colors[5],
             linewidth=2.0, linestyle='--', zorder=3)
    ax2.fill_between(epochs, gw_mean - gw_std, gw_mean + gw_std,
                     alpha=0.15, color=colors[5], zorder=1)
    
    # Plot Clustering Loss (cc_loss in the requirement)
    cluster_mean = np.array(aggregated_data['losses']['cluster']['mean'])
    cluster_std = np.array(aggregated_data['losses']['cluster']['std'])
    ax2.plot(epochs, cluster_mean, label='Clustering', color=colors[6],
             linewidth=2.0, linestyle='--', zorder=3)
    ax2.fill_between(epochs, cluster_mean - cluster_std, cluster_mean + cluster_std,
                     alpha=0.15, color=colors[6], zorder=1)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss Value', fontweight='bold')
    ax2.set_title('Loss Components', fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, shadow=True, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure saved to: {output_path}")
    
    plt.close()


def print_statistics_table(aggregated_data):
    """Print a formatted statistics table"""
    print("\n" + "="*80)
    print(f"{'Statistical Summary':^80}")
    print("="*80)
    
    # Metrics table
    print(f"\n{'Clustering Metrics (%)':^80}")
    print("-"*80)
    print(f"{'Metric':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    
    for metric in ['acc', 'nmi', 'ari', 'purity', 'f1']:
        stats = aggregated_data['final_stats']['metrics'][metric]
        print(f"{metric.upper():<12} "
              f"{stats['mean']*100:>10.2f}% "
              f"{stats['std']*100:>10.2f}% "
              f"{stats['min']*100:>10.2f}% "
              f"{stats['max']*100:>10.2f}%")
    
    # Losses table
    print(f"\n{'Loss Components':^80}")
    print("-"*80)
    print(f"{'Loss':<15} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)
    
    for loss in ['loss', 'recon', 'gw', 'cluster', 'contrastive', 'cfm']:
        stats = aggregated_data['final_stats']['losses'][loss]
        print(f"{loss.capitalize():<15} "
              f"{stats['mean']:>13.4f} "
              f"{stats['std']:>13.4f} "
              f"{stats['min']:>13.4f} "
              f"{stats['max']:>13.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-seed convergence analysis')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing aggregated_results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output figure path (default: {results_dir}/convergence_plot.pdf)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load aggregated results
    agg_file = results_dir / 'aggregated_results.json'
    if not agg_file.exists():
        print(f"❌ Aggregated results file not found: {agg_file}")
        print("Please run run_multi_seed_convergence.py first.")
        return
    
    print(f"📂 Loading aggregated results from: {agg_file}")
    with open(agg_file, 'r') as f:
        aggregated_data = json.load(f)
    
    print(f"✅ Loaded data: {aggregated_data['n_seeds']} seeds × {aggregated_data['n_epochs']} epochs")
    
    # Load experiment config to get dataset name
    config_file = results_dir / 'experiment_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            exp_config = json.load(f)
        dataset_name = exp_config['dataset']
    else:
        dataset_name = results_dir.name.split('_')[0]
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / 'convergence_plot.pdf'
    
    # Print statistics
    print_statistics_table(aggregated_data)
    
    # Create plot
    print(f"\n📊 Creating convergence plot...")
    plot_convergence_with_std(aggregated_data, output_path, dataset_name)
    
    # Also save to figures directory for paper
    figures_dir = Path('figures')
    if figures_dir.exists():
        paper_output = figures_dir / f'{dataset_name}_multi_seed_convergence.pdf'
        plot_convergence_with_std(aggregated_data, paper_output, dataset_name)
        print(f"✅ Paper figure saved to: {paper_output}")
    
    print(f"\n{'='*80}")
    print(f"✅ Multi-seed convergence visualization complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
