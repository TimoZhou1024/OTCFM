"""
Comprehensive Ablation Study Script for OT-CFM

This script performs systematic ablation experiments to evaluate the contribution
of each component in the OT-CFM model.

Usage:
    # Run full ablation study on a dataset
    uv run python scripts/run_ablation.py --dataset Scene15 --epochs 100
    
    # Run specific ablation modes
    uv run python scripts/run_ablation.py --dataset Handwritten --modes full no_gw no_flow
    
    # Run lambda sensitivity analysis
    uv run python scripts/run_ablation.py --dataset Coil20 --analysis lambda_sensitivity
    
    # Run all analysis types
    uv run python scripts/run_ablation.py --dataset Scene15 --analysis all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import ExperimentConfig, get_default_config
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_synthetic, load_handwritten, load_coil20,
    MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.ablation import AblationStudy, AblationConfig, ABLATION_DESCRIPTIONS


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

# All ablation modes
ALL_ABLATION_MODES = [
    "full",           # Complete model (baseline)
    "no_gw",          # Without Gromov-Wasserstein loss
    "no_cluster",     # Without clustering loss  
    "no_ot",          # Without optimal transport
    "no_flow",        # Without flow matching
    "no_contrastive", # Without contrastive loss
    "no_recon",       # Without reconstruction loss
]

# Lambda parameters for sensitivity analysis
LAMBDA_PARAMS = {
    'lambda_gw': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'lambda_cluster': [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0],
    'lambda_recon': [0.0, 0.5, 1.0, 2.0, 5.0],
    'lambda_contrastive': [0.0, 0.05, 0.1, 0.2, 0.5],
}


class ComprehensiveAblation:
    """
    Comprehensive ablation study for OT-CFM
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        epochs: int = 100,
        batch_size: int = 256,
        num_runs: int = 3,
        device: str = None,
        save_dir: str = "results/ablation",
        verbose: bool = True
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.verbose = verbose
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load dataset
        self._load_data()
        
    def _load_data(self):
        """Load the dataset"""
        dataset_key = self.dataset_name.lower()
        if dataset_key not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        loader = DATASET_LOADERS[dataset_key]
        
        if dataset_key == 'synthetic':
            data = loader(n_samples=1000, n_clusters=10)
            self.views = data['views']
            self.labels = data['labels']
        else:
            result = loader(self.data_root)
            if isinstance(result, tuple):
                self.views, self.labels = result
            else:
                self.views = result['views']
                self.labels = result['labels']
        
        self.view_dims = [v.shape[1] for v in self.views]
        self.num_clusters = len(np.unique(self.labels))
        
        if self.verbose:
            print(f"Loaded {self.dataset_name}:")
            print(f"  Samples: {len(self.labels)}")
            print(f"  Views: {len(self.views)}")
            print(f"  View dims: {self.view_dims}")
            print(f"  Clusters: {self.num_clusters}")
    
    def _create_dataloader(self):
        """Create data loader"""
        dataset = MultiViewDataset(
            views=self.views,
            labels=self.labels
        )
        return create_dataloader(dataset, self.batch_size, shuffle=True)
    
    def _create_config(self) -> ExperimentConfig:
        """Create experiment config"""
        config = get_default_config()
        config.data.dataset_name = self.dataset_name
        config.training.epochs = self.epochs
        config.training.batch_size = self.batch_size
        config.training.device = self.device
        config.model.num_clusters = self.num_clusters
        return config
    
    def run_component_ablation(
        self,
        modes: List[str] = None
    ) -> Dict:
        """
        Run component ablation study
        
        Tests the contribution of each component by removing it
        """
        if modes is None:
            modes = ALL_ABLATION_MODES
        
        print("\n" + "="*70)
        print("Component Ablation Study")
        print("="*70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Modes: {modes}")
        print(f"Runs per mode: {self.num_runs}")
        print(f"Epochs: {self.epochs}")
        
        train_loader = self._create_dataloader()
        config = self._create_config()
        
        all_results = {}
        
        for mode in modes:
            print(f"\n{'─'*60}")
            print(f"Mode: {mode}")
            print(f"Description: {ABLATION_DESCRIPTIONS.get(mode, 'Custom mode')}")
            print(f"{'─'*60}")
            
            mode_metrics = {'acc': [], 'nmi': [], 'ari': []}
            
            for run in range(self.num_runs):
                seed = 42 + run
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                if self.verbose:
                    print(f"  Run {run+1}/{self.num_runs}...", end=" ")
                
                # Create model with appropriate lambda settings based on mode
                model_kwargs = self._get_model_kwargs_for_mode(mode)
                model = OTCFM(
                    view_dims=self.view_dims,
                    **model_kwargs
                )
                
                # Create trainer
                exp_dir = self.save_dir / f"temp_{mode}_run{run}"
                trainer = Trainer(
                    model=model,
                    config=config.training,
                    experiment_dir=str(exp_dir),
                    device=self.device,
                    verbose=False
                )
                
                try:
                    results = trainer.train(
                        train_loader=train_loader,
                        labels=self.labels,
                        ablation_mode=mode
                    )
                    
                    mode_metrics['acc'].append(results['best']['acc'])
                    mode_metrics['nmi'].append(results['best']['nmi'])
                    mode_metrics['ari'].append(results['best'].get('ari', 0))
                    
                    if self.verbose:
                        print(f"ACC={results['best']['acc']:.4f}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                finally:
                    # Clean up temp directory
                    import shutil
                    if exp_dir.exists():
                        shutil.rmtree(exp_dir)
            
            # Aggregate results
            if mode_metrics['acc']:
                all_results[mode] = {
                    'acc_mean': np.mean(mode_metrics['acc']),
                    'acc_std': np.std(mode_metrics['acc']),
                    'nmi_mean': np.mean(mode_metrics['nmi']),
                    'nmi_std': np.std(mode_metrics['nmi']),
                    'ari_mean': np.mean(mode_metrics['ari']),
                    'ari_std': np.std(mode_metrics['ari']),
                }
                
                print(f"\n  Result: ACC={all_results[mode]['acc_mean']:.4f}±{all_results[mode]['acc_std']:.4f}, "
                      f"NMI={all_results[mode]['nmi_mean']:.4f}±{all_results[mode]['nmi_std']:.4f}")
        
        return all_results
    
    def _get_model_kwargs_for_mode(self, mode: str) -> Dict:
        """Get model kwargs based on ablation mode"""
        config = self._create_config()
        
        kwargs = {
            'latent_dim': config.model.latent_dim,
            'hidden_dims': config.model.hidden_dims,
            'num_clusters': self.num_clusters,
            'flow_hidden_dim': config.model.flow_hidden_dim,
            'flow_num_layers': config.model.flow_num_layers,
            'time_dim': config.model.time_dim,
            'ode_steps': config.model.ode_steps,
            'sigma_min': config.model.sigma_min,
            'kernel_type': config.model.kernel_type,
            'kernel_gamma': config.model.kernel_gamma,
            'lambda_gw': config.model.lambda_gw,
            'lambda_cluster': config.model.lambda_cluster,
            'lambda_recon': config.model.lambda_recon,
            'lambda_contrastive': config.model.lambda_contrastive,
            'dropout': config.model.dropout,
        }
        
        # Modify based on mode (set lambda to 0 for disabled components)
        if mode == 'no_gw':
            kwargs['lambda_gw'] = 0.0
        elif mode == 'no_cluster':
            kwargs['lambda_cluster'] = 0.0
        elif mode == 'no_recon':
            kwargs['lambda_recon'] = 0.0
        elif mode == 'no_contrastive':
            kwargs['lambda_contrastive'] = 0.0
        
        return kwargs
    
    def run_lambda_sensitivity(
        self,
        lambda_name: str = 'lambda_gw',
        values: List[float] = None
    ) -> Dict:
        """
        Run sensitivity analysis for a specific lambda parameter
        """
        if values is None:
            values = LAMBDA_PARAMS.get(lambda_name, [0.0, 0.1, 0.5, 1.0])
        
        print("\n" + "="*70)
        print(f"Lambda Sensitivity Analysis: {lambda_name}")
        print("="*70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Values: {values}")
        print(f"Runs per value: {self.num_runs}")
        
        train_loader = self._create_dataloader()
        config = self._create_config()
        
        all_results = {}
        
        for val in values:
            print(f"\n{'─'*40}")
            print(f"{lambda_name} = {val}")
            
            val_metrics = {'acc': [], 'nmi': [], 'ari': []}
            
            for run in range(self.num_runs):
                seed = 42 + run
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                if self.verbose:
                    print(f"  Run {run+1}/{self.num_runs}...", end=" ")
                
                # Create model with specific lambda
                model_kwargs = self._get_model_kwargs_for_mode('full')
                model_kwargs[lambda_name] = val
                
                model = OTCFM(
                    view_dims=self.view_dims,
                    **model_kwargs
                )
                
                exp_dir = self.save_dir / f"temp_sensitivity_{lambda_name}_{val}_run{run}"
                trainer = Trainer(
                    model=model,
                    config=config.training,
                    experiment_dir=str(exp_dir),
                    device=self.device,
                    verbose=False
                )
                
                try:
                    results = trainer.train(train_loader, self.labels)
                    
                    val_metrics['acc'].append(results['best']['acc'])
                    val_metrics['nmi'].append(results['best']['nmi'])
                    val_metrics['ari'].append(results['best'].get('ari', 0))
                    
                    if self.verbose:
                        print(f"ACC={results['best']['acc']:.4f}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    import shutil
                    if exp_dir.exists():
                        shutil.rmtree(exp_dir)
            
            if val_metrics['acc']:
                all_results[val] = {
                    'acc_mean': np.mean(val_metrics['acc']),
                    'acc_std': np.std(val_metrics['acc']),
                    'nmi_mean': np.mean(val_metrics['nmi']),
                    'nmi_std': np.std(val_metrics['nmi']),
                    'ari_mean': np.mean(val_metrics['ari']),
                    'ari_std': np.std(val_metrics['ari']),
                }
        
        return all_results
    
    def run_all_lambda_sensitivity(self) -> Dict:
        """Run sensitivity analysis for all lambda parameters"""
        all_results = {}
        
        for lambda_name, values in LAMBDA_PARAMS.items():
            print(f"\n\n{'='*70}")
            print(f"Analyzing {lambda_name}")
            print(f"{'='*70}")
            
            results = self.run_lambda_sensitivity(lambda_name, values)
            all_results[lambda_name] = results
        
        return all_results
    
    def run_architecture_ablation(self) -> Dict:
        """
        Ablation on architecture choices
        """
        print("\n" + "="*70)
        print("Architecture Ablation Study")
        print("="*70)
        
        train_loader = self._create_dataloader()
        config = self._create_config()
        
        # Architecture variants
        variants = {
            'baseline': {'latent_dim': 128, 'hidden_dims': [512, 256], 'flow_num_layers': 4},
            'small_latent': {'latent_dim': 64, 'hidden_dims': [512, 256], 'flow_num_layers': 4},
            'large_latent': {'latent_dim': 256, 'hidden_dims': [512, 256], 'flow_num_layers': 4},
            'shallow_encoder': {'latent_dim': 128, 'hidden_dims': [256], 'flow_num_layers': 4},
            'deep_encoder': {'latent_dim': 128, 'hidden_dims': [1024, 512, 256], 'flow_num_layers': 4},
            'shallow_flow': {'latent_dim': 128, 'hidden_dims': [512, 256], 'flow_num_layers': 2},
            'deep_flow': {'latent_dim': 128, 'hidden_dims': [512, 256], 'flow_num_layers': 8},
        }
        
        all_results = {}
        
        for name, arch_kwargs in variants.items():
            print(f"\n{'─'*40}")
            print(f"Variant: {name}")
            print(f"Config: {arch_kwargs}")
            
            var_metrics = {'acc': [], 'nmi': [], 'ari': []}
            
            for run in range(self.num_runs):
                seed = 42 + run
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                if self.verbose:
                    print(f"  Run {run+1}/{self.num_runs}...", end=" ")
                
                # Create model with variant architecture
                model_kwargs = self._get_model_kwargs_for_mode('full')
                model_kwargs.update(arch_kwargs)
                
                model = OTCFM(
                    view_dims=self.view_dims,
                    **model_kwargs
                )
                
                exp_dir = self.save_dir / f"temp_arch_{name}_run{run}"
                trainer = Trainer(
                    model=model,
                    config=config.training,
                    experiment_dir=str(exp_dir),
                    device=self.device,
                    verbose=False
                )
                
                try:
                    results = trainer.train(train_loader, self.labels)
                    
                    var_metrics['acc'].append(results['best']['acc'])
                    var_metrics['nmi'].append(results['best']['nmi'])
                    var_metrics['ari'].append(results['best'].get('ari', 0))
                    
                    if self.verbose:
                        print(f"ACC={results['best']['acc']:.4f}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    import shutil
                    if exp_dir.exists():
                        shutil.rmtree(exp_dir)
            
            if var_metrics['acc']:
                all_results[name] = {
                    'config': arch_kwargs,
                    'acc_mean': np.mean(var_metrics['acc']),
                    'acc_std': np.std(var_metrics['acc']),
                    'nmi_mean': np.mean(var_metrics['nmi']),
                    'nmi_std': np.std(var_metrics['nmi']),
                    'ari_mean': np.mean(var_metrics['ari']),
                    'ari_std': np.std(var_metrics['ari']),
                }
        
        return all_results


def save_ablation_results(
    results: Dict,
    analysis_type: str,
    dataset_name: str,
    save_dir: str = "results/ablation"
) -> Tuple[str, str]:
    """Save ablation results to CSV and JSON"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to DataFrame
    rows = []
    for key, metrics in results.items():
        if isinstance(metrics, dict) and 'acc_mean' in metrics:
            row = {
                'Setting': key,
                'ACC': f"{metrics['acc_mean']:.4f}±{metrics['acc_std']:.4f}",
                'ACC_mean': metrics['acc_mean'],
                'ACC_std': metrics['acc_std'],
                'NMI': f"{metrics['nmi_mean']:.4f}±{metrics['nmi_std']:.4f}",
                'NMI_mean': metrics['nmi_mean'],
                'NMI_std': metrics['nmi_std'],
                'ARI': f"{metrics['ari_mean']:.4f}±{metrics['ari_std']:.4f}",
                'ARI_mean': metrics['ari_mean'],
                'ARI_std': metrics['ari_std'],
            }
            if 'config' in metrics:
                row['Config'] = str(metrics['config'])
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON
    json_path = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Convert keys to strings for JSON
        json_results = {str(k): v for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    return csv_path, json_path


def plot_component_ablation(results: Dict, dataset_name: str, save_path: str = None):
    """Plot component ablation results"""
    modes = list(results.keys())
    acc_means = [results[m]['acc_mean'] for m in modes]
    acc_stds = [results[m]['acc_std'] for m in modes]
    nmi_means = [results[m]['nmi_mean'] for m in modes]
    nmi_stds = [results[m]['nmi_std'] for m in modes]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(modes))
    width = 0.6
    
    # ACC plot
    colors = ['#2ecc71' if m == 'full' else '#e74c3c' for m in modes]
    bars1 = axes[0].bar(x, acc_means, width, yerr=acc_stds, capsize=5, color=colors, alpha=0.8)
    axes[0].set_ylabel('ACC', fontsize=12)
    axes[0].set_title(f'{dataset_name} - Component Ablation (ACC)', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=45, ha='right')
    axes[0].axhline(y=results['full']['acc_mean'], color='green', linestyle='--', alpha=0.5, label='Full model')
    axes[0].legend()
    
    # NMI plot
    bars2 = axes[1].bar(x, nmi_means, width, yerr=nmi_stds, capsize=5, color=colors, alpha=0.8)
    axes[1].set_ylabel('NMI', fontsize=12)
    axes[1].set_title(f'{dataset_name} - Component Ablation (NMI)', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, rotation=45, ha='right')
    axes[1].axhline(y=results['full']['nmi_mean'], color='green', linestyle='--', alpha=0.5, label='Full model')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_lambda_sensitivity(
    results: Dict,
    lambda_name: str,
    dataset_name: str,
    save_path: str = None
):
    """Plot lambda sensitivity results"""
    values = sorted([float(k) for k in results.keys()])
    acc_means = [results[v]['acc_mean'] for v in values]
    acc_stds = [results[v]['acc_std'] for v in values]
    nmi_means = [results[v]['nmi_mean'] for v in values]
    nmi_stds = [results[v]['nmi_std'] for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(values, acc_means, yerr=acc_stds, marker='o', label='ACC', capsize=5, linewidth=2, markersize=8)
    ax.errorbar(values, nmi_means, yerr=nmi_stds, marker='s', label='NMI', capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel(lambda_name, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{dataset_name} - {lambda_name} Sensitivity', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark best values
    best_acc_idx = np.argmax(acc_means)
    ax.scatter([values[best_acc_idx]], [acc_means[best_acc_idx]], 
               color='red', s=200, zorder=5, marker='*', label=f'Best ACC: {values[best_acc_idx]}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def print_ablation_table(results: Dict, title: str = "Ablation Results"):
    """Print results as formatted table"""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"{'Setting':<20} {'ACC':<20} {'NMI':<20} {'ARI':<20}")
    print("-"*80)
    
    for key, metrics in results.items():
        if isinstance(metrics, dict) and 'acc_mean' in metrics:
            acc = f"{metrics['acc_mean']:.4f}±{metrics['acc_std']:.4f}"
            nmi = f"{metrics['nmi_mean']:.4f}±{metrics['nmi_std']:.4f}"
            ari = f"{metrics['ari_mean']:.4f}±{metrics['ari_std']:.4f}"
            print(f"{str(key):<20} {acc:<20} {nmi:<20} {ari:<20}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ablation Study for OT-CFM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run component ablation
  uv run python scripts/run_ablation.py --dataset Scene15 --epochs 100
  
  # Run specific modes only
  uv run python scripts/run_ablation.py --dataset Handwritten --modes full no_gw no_flow no_cluster
  
  # Run lambda sensitivity analysis
  uv run python scripts/run_ablation.py --dataset Coil20 --analysis lambda_sensitivity --lambda_name lambda_gw
  
  # Run architecture ablation
  uv run python scripts/run_ablation.py --dataset Scene15 --analysis architecture
  
  # Run all analyses
  uv run python scripts/run_ablation.py --dataset Scene15 --analysis all --epochs 50 --num_runs 2
        """
    )
    
    parser.add_argument('--dataset', type=str, default='Synthetic',
                        choices=list(DATASET_LOADERS.keys()) + [k.title() for k in DATASET_LOADERS.keys()],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per run')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per setting')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detect if not specified)')
    parser.add_argument('--save_dir', type=str, default='results/ablation',
                        help='Directory to save results')
    
    # Analysis type
    parser.add_argument('--analysis', type=str, default='component',
                        choices=['component', 'lambda_sensitivity', 'architecture', 'all'],
                        help='Type of ablation analysis')
    
    # Component ablation options
    parser.add_argument('--modes', nargs='+', default=None,
                        help='Specific ablation modes to run (default: all)')
    
    # Lambda sensitivity options
    parser.add_argument('--lambda_name', type=str, default='lambda_gw',
                        choices=list(LAMBDA_PARAMS.keys()),
                        help='Lambda parameter for sensitivity analysis')
    parser.add_argument('--lambda_values', nargs='+', type=float, default=None,
                        help='Custom lambda values to test')
    
    # Output options
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize ablation study
    ablation = ComprehensiveAblation(
        dataset_name=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        device=args.device,
        save_dir=args.save_dir,
        verbose=args.verbose
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run selected analysis
    if args.analysis == 'component' or args.analysis == 'all':
        print("\n" + "#"*70)
        print("# COMPONENT ABLATION STUDY")
        print("#"*70)
        
        modes = args.modes if args.modes else ALL_ABLATION_MODES
        component_results = ablation.run_component_ablation(modes)
        
        print_ablation_table(component_results, "Component Ablation Results")
        save_ablation_results(component_results, 'component', args.dataset, args.save_dir)
        
        if not args.no_plot and 'full' in component_results:
            plot_path = os.path.join(args.save_dir, f"{args.dataset}_component_{timestamp}.png")
            plot_component_ablation(component_results, args.dataset, plot_path)
    
    if args.analysis == 'lambda_sensitivity' or args.analysis == 'all':
        print("\n" + "#"*70)
        print("# LAMBDA SENSITIVITY ANALYSIS")
        print("#"*70)
        
        if args.analysis == 'all':
            # Run all lambda parameters
            all_lambda_results = ablation.run_all_lambda_sensitivity()
            
            for lambda_name, results in all_lambda_results.items():
                print_ablation_table(results, f"{lambda_name} Sensitivity")
                save_ablation_results(results, f'lambda_{lambda_name}', args.dataset, args.save_dir)
                
                if not args.no_plot:
                    plot_path = os.path.join(args.save_dir, f"{args.dataset}_{lambda_name}_{timestamp}.png")
                    plot_lambda_sensitivity(results, lambda_name, args.dataset, plot_path)
        else:
            # Run single lambda parameter
            values = args.lambda_values if args.lambda_values else LAMBDA_PARAMS.get(args.lambda_name)
            lambda_results = ablation.run_lambda_sensitivity(args.lambda_name, values)
            
            print_ablation_table(lambda_results, f"{args.lambda_name} Sensitivity")
            save_ablation_results(lambda_results, f'lambda_{args.lambda_name}', args.dataset, args.save_dir)
            
            if not args.no_plot:
                plot_path = os.path.join(args.save_dir, f"{args.dataset}_{args.lambda_name}_{timestamp}.png")
                plot_lambda_sensitivity(lambda_results, args.lambda_name, args.dataset, plot_path)
    
    if args.analysis == 'architecture' or args.analysis == 'all':
        print("\n" + "#"*70)
        print("# ARCHITECTURE ABLATION STUDY")
        print("#"*70)
        
        arch_results = ablation.run_architecture_ablation()
        
        print_ablation_table(arch_results, "Architecture Ablation Results")
        save_ablation_results(arch_results, 'architecture', args.dataset, args.save_dir)
    
    print("\n" + "="*70)
    print("Ablation study completed!")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
