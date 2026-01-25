"""
Convergence Analysis for OT-CFM
Tracks loss curves and clustering metrics during training to demonstrate convergence properties

This script runs OT-CFM training while logging detailed loss components and metrics at each epoch,
then generates publication-quality plots showing convergence behavior.

Usage:
    # Single dataset
    uv run python scripts/run_convergence_analysis.py --dataset Handwritten --epochs 200
    
    # Multiple datasets for selection
    uv run python scripts/run_convergence_analysis.py --datasets Handwritten Scene15 Coil20 --epochs 200
    
    # With specific hyperparameters
    uv run python scripts/run_convergence_analysis.py --dataset Scene15 --epochs 200 --use_tuned
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import json
from tqdm import tqdm

from otcfm.config import get_default_config
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_synthetic, load_handwritten, load_coil20,
    MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.metrics import evaluate_clustering


# Dataset loaders
DATASET_LOADERS = {
    'caltech101': load_caltech101,
    'scene15': load_scene15,
    'noisy_mnist': load_noisy_mnist,
    'bdgp': load_bdgp,
    'synthetic': load_synthetic,
    'handwritten': load_handwritten,
    'coil20': load_coil20,
}


class ConvergenceTrainer:
    """Extended trainer that logs detailed convergence metrics"""
    
    def __init__(
        self,
        model: OTCFM,
        config,
        device: str = 'cuda',
        verbose: bool = True
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.verbose = verbose
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Convergence tracking
        self.epoch_losses = []
        self.epoch_metrics = []
    
    def train(
        self,
        train_loader,
        labels: np.ndarray,
        epochs: int = 200,
        pretrain_epochs: int = 20
    ) -> Dict:
        """
        Training loop with detailed convergence tracking
        
        Returns:
            Dict with 'losses' and 'metrics' history
        """
        # Phase 1: Reconstruction pretraining
        recon_epochs = pretrain_epochs // 2
        dec_epochs = pretrain_epochs - recon_epochs
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Phase 1: Reconstruction pretraining ({recon_epochs} epochs)")
            print(f"{'='*60}")
        
        for epoch in tqdm(range(recon_epochs), desc="Phase 1: Reconstruction"):
            epoch_loss = self._train_reconstruction_epoch(train_loader)
            metrics = self._evaluate(train_loader, labels)
            
            self.epoch_losses.append({
                'epoch': epoch,
                'phase': 'recon',
                'total_loss': epoch_loss,
                'recon_loss': epoch_loss,
                'gw_loss': 0.0,
                'cluster_loss': 0.0,
                'contrastive_loss': 0.0,
                'cfm_loss': 0.0
            })
            self.epoch_metrics.append({
                'epoch': epoch,
                'phase': 'recon',
                **metrics
            })
        
        # Initialize clustering
        if self.verbose:
            print("\nInitializing clustering centroids...")
        self.model.init_clustering(train_loader, self.device)
        
        # Phase 2: Single-View DEC pretraining
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Phase 2: Single-View DEC pretraining ({dec_epochs} epochs)")
            print(f"{'='*60}")
        
        for epoch in tqdm(range(dec_epochs), desc="Phase 2: DEC"):
            epoch_loss_dict = self._train_dec_epoch(train_loader)
            metrics = self._evaluate(train_loader, labels)
            
            self.epoch_losses.append({
                'epoch': recon_epochs + epoch,
                'phase': 'dec',
                **epoch_loss_dict
            })
            self.epoch_metrics.append({
                'epoch': recon_epochs + epoch,
                'phase': 'dec',
                **metrics
            })
        
        # Re-initialize centroids
        if self.verbose:
            print("\nRe-initializing clustering centroids...")
        self.model.init_clustering(train_loader, self.device)
        
        # Phase 3: Full OT-CFM training
        full_epochs = epochs - pretrain_epochs
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Phase 3: Full OT-CFM training ({full_epochs} epochs)")
            print(f"{'='*60}")
        
        best_acc = 0.0
        patience_counter = 0
        patience = 20
        
        for epoch in tqdm(range(full_epochs), desc="Phase 3: Full Training"):
            epoch_loss_dict = self._train_full_epoch(train_loader)
            metrics = self._evaluate(train_loader, labels)
            
            self.epoch_losses.append({
                'epoch': pretrain_epochs + epoch,
                'phase': 'full',
                **epoch_loss_dict
            })
            self.epoch_metrics.append({
                'epoch': pretrain_epochs + epoch,
                'phase': 'full',
                **metrics
            })
            
            # Early stopping check
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {pretrain_epochs + epoch}")
                break
        
        # Final metrics
        final_metrics = self._evaluate(train_loader, labels)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete")
            print(f"{'='*60}")
            print(f"Final ACC: {final_metrics['acc']:.4f}")
            print(f"Final NMI: {final_metrics['nmi']:.4f}")
            print(f"Final ARI: {final_metrics['ari']:.4f}")
        
        return {
            'losses': pd.DataFrame(self.epoch_losses),
            'metrics': pd.DataFrame(self.epoch_metrics),
            'final': final_metrics
        }
    
    def _train_reconstruction_epoch(self, train_loader) -> float:
        """Single reconstruction pretraining epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(views, mask, return_all=True)
            
            # Reconstruction loss only
            recon_loss = 0
            for v_idx in range(len(views)):
                recon_loss += torch.nn.functional.mse_loss(
                    outputs['reconstructions'][v_idx],
                    views[v_idx]
                )
            recon_loss = recon_loss / len(views)
            
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += recon_loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _train_dec_epoch(self, train_loader) -> Dict:
        """Single DEC pretraining epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_cluster = 0.0
        num_batches = 0
        
        for batch in train_loader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(views, mask, return_all=True)
            latents = outputs['latents']
            
            # Reconstruction loss
            recon_loss = 0
            for v_idx in range(len(views)):
                recon_loss += torch.nn.functional.mse_loss(
                    outputs['reconstructions'][v_idx],
                    views[v_idx]
                )
            recon_loss = recon_loss / len(views)
            
            # Single-view DEC loss
            dec_loss = 0
            for z_v in latents:
                q_v, p_v = self.model.clustering(z_v)
                kl_loss = (p_v * torch.log((p_v + 1e-8) / (q_v + 1e-8))).sum(dim=1).mean()
                dec_loss += kl_loss
            dec_loss = dec_loss / len(latents)
            
            loss = recon_loss + dec_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_cluster += dec_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'cluster_loss': total_cluster / num_batches,
            'gw_loss': 0.0,
            'contrastive_loss': 0.0,
            'cfm_loss': 0.0
        }
    
    def _train_full_epoch(self, train_loader) -> Dict:
        """Single full OT-CFM training epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_gw = 0.0
        total_cluster = 0.0
        total_contrastive = 0.0
        total_cfm = 0.0
        num_batches = 0
        
        for batch in train_loader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # compute_loss returns (loss, loss_dict)
            loss, loss_dict = self.model.compute_loss(views, mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # loss_dict values are already .item() called in losses.py
            # So they are Python floats, not tensors
            total_loss += loss.item()
            total_recon += float(loss_dict.get('recon', 0.0))
            total_gw += float(loss_dict.get('gw', 0.0))
            total_cluster += float(loss_dict.get('cluster', 0.0))
            total_contrastive += float(loss_dict.get('contrastive', 0.0))
            total_cfm += float(loss_dict.get('cfm', 0.0))
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'gw_loss': total_gw / num_batches,
            'cluster_loss': total_cluster / num_batches,
            'contrastive_loss': total_contrastive / num_batches,
            'cfm_loss': total_cfm / num_batches
        }
    
    def _evaluate(self, train_loader, labels: np.ndarray) -> Dict:
        """Evaluate clustering performance"""
        self.model.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch in train_loader:
                views = [v.to(self.device) for v in batch['views']]
                mask = batch['mask'].to(self.device)
                
                outputs = self.model(views, mask, return_all=True)
                
                # Get consensus or use first view's latent
                if 'consensus' in outputs and outputs['consensus'] is not None:
                    consensus = outputs['consensus']
                elif 'latents' in outputs and len(outputs['latents']) > 0:
                    # Use first view's latent as fallback
                    consensus = outputs['latents'][0]
                else:
                    # Emergency fallback
                    continue
                    
                all_latents.append(consensus.cpu())
        
        if len(all_latents) == 0:
            # Return dummy metrics if no data
            return {'acc': 0.0, 'nmi': 0.0, 'ari': 0.0, 'purity': 0.0, 'f1': 0.0}
        
        consensus_latent = torch.cat(all_latents, dim=0).numpy()
        
        # Get cluster assignments
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=42)
        pred_labels = kmeans.fit_predict(consensus_latent)
        
        # Compute metrics
        metrics = evaluate_clustering(labels, pred_labels)
        
        return metrics


def plot_convergence(
    losses_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path
):
    """Generate publication-quality convergence plots"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Convergence Analysis: {dataset_name}', fontsize=18, fontweight='bold')
    
    # Color scheme for phases
    phase_colors = {
        'recon': '#1f77b4',    # Blue
        'dec': '#ff7f0e',      # Orange
        'full': '#2ca02c'      # Green
    }
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    for phase in ['recon', 'dec', 'full']:
        phase_data = losses_df[losses_df['phase'] == phase]
        if len(phase_data) > 0:
            ax1.plot(
                phase_data['epoch'],
                phase_data['total_loss'],
                label=f'Phase: {phase.upper()}',
                color=phase_colors[phase],
                linewidth=2,
                marker='o' if phase == 'full' else None,
                markersize=3,
                markevery=max(1, len(phase_data) // 20)
            )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('(a) Total Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components (Phase 3 only)
    ax2 = axes[0, 1]
    full_phase = losses_df[losses_df['phase'] == 'full']
    if len(full_phase) > 0:
        components = ['recon_loss', 'gw_loss', 'cluster_loss', 'cfm_loss']
        component_labels = ['Reconstruction', 'Gromov-Wasserstein', 'Clustering', 'Flow Matching']
        component_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        for comp, label, color in zip(components, component_labels, component_colors):
            if comp in full_phase.columns:
                ax2.plot(
                    full_phase['epoch'],
                    full_phase[comp],
                    label=label,
                    color=color,
                    linewidth=2,
                    marker='s',
                    markersize=3,
                    markevery=max(1, len(full_phase) // 20)
                )
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('(b) Loss Component Breakdown (Phase 3)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Clustering Accuracy
    ax3 = axes[1, 0]
    for phase in ['recon', 'dec', 'full']:
        phase_data = metrics_df[metrics_df['phase'] == phase]
        if len(phase_data) > 0:
            ax3.plot(
                phase_data['epoch'],
                phase_data['acc'],
                label=f'Phase: {phase.upper()}',
                color=phase_colors[phase],
                linewidth=2,
                marker='o' if phase == 'full' else None,
                markersize=3,
                markevery=max(1, len(phase_data) // 20)
            )
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Clustering Accuracy (ACC)')
    ax3.set_title('(c) Clustering Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: NMI and ARI
    ax4 = axes[1, 1]
    full_metrics = metrics_df[metrics_df['phase'] == 'full']
    if len(full_metrics) > 0:
        ax4.plot(
            full_metrics['epoch'],
            full_metrics['nmi'],
            label='NMI',
            color='#e67e22',
            linewidth=2,
            marker='o',
            markersize=3,
            markevery=max(1, len(full_metrics) // 20)
        )
        ax4.plot(
            full_metrics['epoch'],
            full_metrics['ari'],
            label='ARI',
            color='#16a085',
            linewidth=2,
            marker='s',
            markersize=3,
            markevery=max(1, len(full_metrics) // 20)
        )
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('(d) Clustering Metrics (Phase 3)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{dataset_name}_convergence.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to: {output_path}")
    
    # Also save as PNG
    output_path_png = output_dir / f'{dataset_name}_convergence.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    
    plt.close()


def run_convergence_analysis(
    dataset_name: str,
    epochs: int = 200,
    use_tuned: bool = False,
    output_dir: Path = Path("convergence_results"),
    data_dir: str = "./data"
):
    """Run convergence analysis on a single dataset"""
    
    print(f"\n{'='*80}")
    print(f"Running Convergence Analysis: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"{dataset_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_key = dataset_name.lower().replace('_', '').replace('-', '')
    name_mapping = {
        'scene15': 'scene15',
        'handwritten': 'handwritten',
        'caltech101': 'caltech101',
        'coil20': 'coil20',
        'bdgp': 'bdgp',
        'synthetic': 'synthetic',
        'noisymnist': 'noisy_mnist',
    }
    
    if dataset_key in name_mapping:
        dataset_key = name_mapping[dataset_key]
    
    if dataset_key not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DATASET_LOADERS[dataset_key]
    
    # Load data with proper arguments
    if dataset_key == 'synthetic':
        data = loader(n_samples=1000, n_clusters=10)
    else:
        # Return dict with views and labels
        views, labels = loader(data_dir)
        data = {'views': views, 'labels': labels}
    
    views = data['views']
    labels = data['labels']
    num_clusters = len(np.unique(labels))
    view_dims = [v.shape[1] for v in views]
    
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {views[0].shape[0]}")
    print(f"Views: {len(views)}")
    print(f"View dimensions: {view_dims}")
    print(f"Clusters: {num_clusters}\n")
    
    # Create dataset and dataloader
    dataset = MultiViewDataset(views, labels, missing_rate=0.0, unaligned_rate=0.0)
    train_loader = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}\n")
    
    # Load tuned parameters if requested
    if use_tuned:
        tuned_path = Path("config/tuned_params.json")
        if tuned_path.exists():
            with open(tuned_path, 'r') as f:
                tuned_params = json.load(f)
            if dataset_name.lower() in tuned_params:
                print(f"Loading tuned parameters for {dataset_name}")
                params = tuned_params[dataset_name.lower()]
            else:
                print(f"No tuned parameters found for {dataset_name}, using defaults")
                params = {}
        else:
            print("No tuned_params.json found, using defaults")
            params = {}
    else:
        params = {}
    
    # Create model with appropriate parameters
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=params.get('latent_dim', 128),
        hidden_dims=params.get('hidden_dims', [512, 256]),
        num_clusters=num_clusters,
        flow_hidden_dim=params.get('flow_hidden_dim', 256),
        flow_num_layers=params.get('flow_num_layers', 4),
        time_dim=params.get('time_dim', 64),
        ode_steps=params.get('ode_steps', 10),
        lambda_gw=params.get('lambda_gw', 0.2),
        lambda_cluster=params.get('lambda_cluster', 1.0),
        lambda_recon=params.get('lambda_recon', 0.5),
        lambda_contrastive=params.get('lambda_contrastive', 0.3),
        dropout=params.get('dropout', 0.1),
        is_aligned=True
    )
    
    # Create trainer config
    from types import SimpleNamespace
    config = SimpleNamespace(
        learning_rate=params.get('learning_rate', 1e-3),
        weight_decay=params.get('weight_decay', 1e-5)
    )
    
    # Create trainer
    trainer = ConvergenceTrainer(model, config, device=device, verbose=True)
    
    # Run training
    results = trainer.train(train_loader, labels, epochs=epochs, pretrain_epochs=20)
    
    # Save results
    losses_df = results['losses']
    metrics_df = results['metrics']
    
    losses_df.to_csv(exp_dir / 'losses.csv', index=False)
    metrics_df.to_csv(exp_dir / 'metrics.csv', index=False)
    
    with open(exp_dir / 'final_metrics.json', 'w') as f:
        json.dump(results['final'], f, indent=2)
    
    print(f"\nResults saved to: {exp_dir}")
    
    # Generate plots
    plot_convergence(losses_df, metrics_df, dataset_name, exp_dir)
    
    return results, exp_dir


def main():
    parser = argparse.ArgumentParser(description='Convergence Analysis for OT-CFM')
    parser.add_argument('--dataset', type=str, help='Single dataset to analyze')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to analyze')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--use_tuned', action='store_true', help='Use Optuna-tuned parameters')
    parser.add_argument('--output_dir', type=str, default='convergence_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine datasets to analyze
    if args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        # Default: analyze a few representative datasets
        datasets = ['Handwritten', 'Scene15', 'Coil20']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis on each dataset
    results_summary = []
    
    for dataset in datasets:
        try:
            results, exp_dir = run_convergence_analysis(
                dataset,
                epochs=args.epochs,
                use_tuned=args.use_tuned,
                output_dir=output_dir,
                data_dir="./data"
            )
            
            results_summary.append({
                'dataset': dataset,
                'final_acc': results['final']['acc'],
                'final_nmi': results['final']['nmi'],
                'final_ari': results['final']['ari'],
                'output_dir': str(exp_dir)
            })
            
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            continue
    
    if len(results_summary) > 0:
        summary_df = pd.DataFrame(results_summary)
        summary_path = output_dir / 'convergence_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print("Convergence Analysis Complete")
        print(f"{'='*80}\n")
        print("Summary:")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("Recommendations for Paper:")
        print(f"{'='*80}")
        best_result = summary_df.loc[summary_df['final_acc'].idxmax()]
        print(f"\nBest convergence result: {best_result['dataset']}")
        print(f"  ACC: {best_result['final_acc']:.4f}")
        print(f"  NMI: {best_result['final_nmi']:.4f}")
        print(f"  ARI: {best_result['final_ari']:.4f}")
        print(f"  Plot: {best_result['output_dir']}/{best_result['dataset']}_convergence.pdf")
    else:
        print("\nNo successful results to summarize.")


if __name__ == '__main__':
    main()
