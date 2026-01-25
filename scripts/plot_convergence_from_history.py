"""
Generate convergence analysis plots from standard training experiment
Usage: 
    1. Run: uv run python scripts/run_experiment.py --mode train --dataset Handwritten --epochs 100
    2. Run: uv run python scripts/plot_convergence_from_history.py --exp_dir experiments/ot_cfm_YYYYMMDD_HHMMSS
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_history(exp_dir: Path) -> pd.DataFrame:
    """Load training history from experiment directory"""
    history_file = exp_dir / 'history.json'
    
    if not history_file.exists():
        raise FileNotFoundError(f"No history.json found in {exp_dir}")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return pd.DataFrame(history)


def detect_phases(df: pd.DataFrame) -> dict:
    """Detect training phases from loss patterns"""
    phases = {}
    
    # Phase 1: Reconstruction (first 10 epochs, usually)
    # Typically loss decreases rapidly, no cluster/flow losses
    phase1_mask = df['epoch'] < 10
    phases['recon'] = {'start': 0, 'end': 10, 'color': '#ffdddd'}
    
    # Phase 2: DEC pretraining (next 10 epochs)
    # Look for sudden changes in loss patterns
    if len(df) >= 20:
        phases['dec'] = {'start': 10, 'end': 20, 'color': '#ddffdd'}
        phases['full'] = {'start': 20, 'end': len(df), 'color': '#ddddff'}
    else:
        phases['full'] = {'start': 10, 'end': len(df), 'color': '#ddddff'}
    
    return phases


def plot_convergence(df: pd.DataFrame, phases: dict, dataset_name: str, output_path: Path):
    """Generate 2x2 convergence plot"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Convergence Analysis: {dataset_name}', fontsize=14, fontweight='bold')
    
    # Add phase backgrounds
    for ax in axes.flat:
        for phase_name, phase_info in phases.items():
            ax.axvspan(phase_info['start'], phase_info['end'], 
                      alpha=0.2, color=phase_info['color'], zorder=0)
            # Add phase label at top
            mid_point = (phase_info['start'] + phase_info['end']) / 2
            ax.text(mid_point, ax.get_ylim()[1] * 0.98, phase_name.upper(),
                   ha='center', va='top', fontsize=8, style='italic', alpha=0.7)
    
    # Plot 1: Total loss
    ax1 = axes[0, 0]
    if 'loss' in df.columns:
        ax1.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Total Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss components
    ax2 = axes[0, 1]
    loss_components = []
    if 'recon_loss' in df.columns:
        ax2.plot(df['epoch'], df['recon_loss'], label='Reconstruction', linewidth=1.5)
        loss_components.append('recon_loss')
    if 'gw_loss' in df.columns:
        ax2.plot(df['epoch'], df['gw_loss'], label='Gromov-Wasserstein', linewidth=1.5)
        loss_components.append('gw_loss')
    if 'cluster_loss' in df.columns:
        ax2.plot(df['epoch'], df['cluster_loss'], label='Clustering', linewidth=1.5)
        loss_components.append('cluster_loss')
    if 'cfm_loss' in df.columns:
        ax2.plot(df['epoch'], df['cfm_loss'], label='Flow Matching', linewidth=1.5)
        loss_components.append('cfm_loss')
    
    if loss_components:
        ax2.set_ylabel('Loss Value', fontsize=11)
        ax2.set_title('Loss Components', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Clustering metrics (ACC, NMI, ARI)
    ax3 = axes[1, 0]
    if 'acc' in df.columns:
        ax3.plot(df['epoch'], df['acc'] * 100, 'r-', linewidth=2, label='ACC', marker='o', markersize=2)
    if 'nmi' in df.columns:
        ax3.plot(df['epoch'], df['nmi'] * 100, 'g-', linewidth=2, label='NMI', marker='s', markersize=2)
    if 'ari' in df.columns:
        ax3.plot(df['epoch'], df['ari'] * 100, 'b-', linewidth=2, label='ARI', marker='^', markersize=2)
    
    ax3.set_ylabel('Score (%)', fontsize=11)
    ax3.set_title('Clustering Metrics', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Additional metrics (Purity, F1)
    ax4 = axes[1, 1]
    if 'purity' in df.columns:
        ax4.plot(df['epoch'], df['purity'] * 100, 'm-', linewidth=2, label='Purity', marker='d', markersize=2)
    if 'f1' in df.columns:
        ax4.plot(df['epoch'], df['f1'] * 100, 'c-', linewidth=2, label='F1-Score', marker='x', markersize=3)
    
    ax4.set_ylabel('Score (%)', fontsize=11)
    ax4.set_title('Additional Metrics', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])
    
    # Set x-labels for bottom plots
    for ax in axes[1, :]:
        ax.set_xlabel('Epoch', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Convergence plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate convergence plots from experiment history')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Path to experiment directory (e.g., experiments/ot_cfm_20260122_120000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF path (default: figures/{dataset}_convergence.pdf)')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    print(f"📂 Loading history from: {exp_dir}")
    
    try:
        df = load_history(exp_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Add epoch column if not present
    if 'epoch' not in df.columns:
        df['epoch'] = range(len(df))
    
    print(f"✅ Loaded {len(df)} epochs of training history")
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Detect phases
    phases = detect_phases(df)
    print(f"📊 Detected phases:")
    for phase_name, phase_info in phases.items():
        print(f"   - {phase_name}: epochs {phase_info['start']}-{phase_info['end']}")
    
    # Determine dataset name from config or directory name
    config_file = exp_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            dataset_name = config.get('data', {}).get('name', 'Unknown')
    else:
        # Try to extract from directory name
        dataset_name = exp_dir.name.split('_')[0] if '_' in exp_dir.name else 'Unknown'
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('figures')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'{dataset_name}_convergence.pdf'
    
    # Generate plot
    plot_convergence(df, phases, dataset_name, output_path)
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    if 'acc' in df.columns:
        final_acc = df['acc'].iloc[-1] * 100
        best_acc = df['acc'].max() * 100
        print(f"   - Final ACC: {final_acc:.2f}%")
        print(f"   - Best ACC: {best_acc:.2f}%")
    
    if 'loss' in df.columns:
        initial_loss = df['loss'].iloc[0]
        final_loss = df['loss'].iloc[-1]
        print(f"   - Initial loss: {initial_loss:.4f}")
        print(f"   - Final loss: {final_loss:.4f}")
        print(f"   - Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    # Find convergence point (where ACC plateaus within 1% of final)
    if 'acc' in df.columns and len(df) > 20:
        final_acc = df['acc'].iloc[-1]
        plateau_mask = df['acc'] >= (final_acc - 0.01)
        if plateau_mask.any():
            convergence_epoch = df[plateau_mask]['epoch'].iloc[0]
            print(f"   - Convergence epoch: {convergence_epoch} (ACC within 1% of final)")


if __name__ == '__main__':
    main()
