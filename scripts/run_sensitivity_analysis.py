"""
Sensitivity Analysis for OT-CFM Multi-View Clustering
Systematic analysis of hyperparameter sensitivity for publication-quality results

This script performs comprehensive sensitivity analysis by varying key hyperparameters
and measuring their impact on clustering performance. Results are saved in multiple formats
suitable for publication in top-tier conferences (e.g., ICML, NeurIPS, CVPR).

Usage:
    # Single parameter sweep
    uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --param lambda_gw
    
    # Two-parameter grid search (generates heatmaps)
    uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --param lambda_gw lambda_cluster
    
    # Full sensitivity analysis (all parameters)
    uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --mode full
    
    # With 3D visualization
    uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --param lambda_gw lambda_cluster --plot_3d
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
from typing import Dict, List, Tuple, Optional
import torch
from tqdm import tqdm
import json
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from otcfm.config import get_default_config
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_synthetic, load_handwritten, load_coil20,
    load_cub, load_nus_wide,
    MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
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
    'cub': load_cub,
    'nus_wide': load_nus_wide,
    'nus-wide': load_nus_wide,
    'nuswide': load_nus_wide,
}

# Parameter ranges for sensitivity analysis
PARAM_RANGES = {
    'lambda_gw': np.logspace(-2, 0, 8),           # [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    'lambda_cluster': np.logspace(-1, 1, 8),      # [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    'lambda_recon': np.logspace(-1, 0.5, 7),      # [0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0]
    'lambda_contrastive': np.logspace(-2, 0, 7),  # [0.01, 0.03, 0.1, 0.2, 0.5, 0.8, 1.0]
    'latent_dim': [32, 64, 96, 128, 192, 256, 384],
    'flow_hidden_dim': [128, 192, 256, 384, 512],
    'ode_steps': [5, 7, 10, 15, 20, 30, 50],
    'learning_rate': np.logspace(-4, -2, 7),      # [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    'dropout': np.linspace(0.0, 0.5, 6),          # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

# Default values (used when parameter is not being varied)
DEFAULT_VALUES = {
    'lambda_gw': 0.2,
    'lambda_cluster': 1.0,
    'lambda_recon': 0.5,
    'lambda_contrastive': 0.3,
    'latent_dim': 128,
    'flow_hidden_dim': 256,
    'ode_steps': 10,
    'learning_rate': 0.001,
    'dropout': 0.1,
}


class SensitivityAnalyzer:
    """Systematic sensitivity analysis for OT-CFM"""
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        epochs: int = 100,
        batch_size: int = 256,
        device: str = None,
        num_runs: int = 3,
        seed: int = 42,
        verbose: bool = True
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.seed = seed
        self.verbose = verbose
        
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
        
        # Results storage
        self.results = []
        
    def _load_data(self):
        """Load the dataset"""
        dataset_key = self.dataset_name.lower().replace('_', '')  # Handle NoisyMNIST -> noisymnist
        
        # Map common variations
        name_mapping = {
            'scene15': 'scene15',
            'noisymnist': 'noisy_mnist',
            'handwritten': 'handwritten',
            'caltech101': 'caltech101',
            'coil20': 'coil20',
            'bdgp': 'bdgp',
            'cub': 'cub',
            'nuswide': 'nus_wide',
            'nus-wide': 'nus_wide',
            'synthetic': 'synthetic',
        }
        
        if dataset_key in name_mapping:
            dataset_key = name_mapping[dataset_key]
        
        if dataset_key not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Available: {list(DATASET_LOADERS.keys())}")
        
        loader = DATASET_LOADERS[dataset_key]
        
        if dataset_key == 'synthetic':
            result = loader(n_samples=1000, n_clusters=10)
        else:
            result = loader(self.data_root)
        
        # Handle both dict and tuple returns
        if isinstance(result, dict):
            self.views = result['views']
            self.labels = result['labels']
        else:  # Tuple: (views, labels)
            self.views, self.labels = result
        
        self.view_dims = [v.shape[1] for v in self.views]
        self.num_clusters = len(np.unique(self.labels))
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {self.dataset_name}")
            print(f"  Samples: {self.views[0].shape[0]}")
            print(f"  Views: {len(self.views)}")
            print(f"  View dims: {self.view_dims}")
            print(f"  Clusters: {self.num_clusters}")
            print(f"{'='*60}\n")
    
    def _train_with_config(self, config_dict: Dict) -> Dict:
        """Train OT-CFM with specific configuration and return metrics"""
        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Create config
        config = get_default_config()
        config.data.dataset_name = self.dataset_name
        config.data.data_root = self.data_root
        config.training.epochs = self.epochs
        config.training.batch_size = self.batch_size
        config.training.device = self.device
        config.model.num_clusters = self.num_clusters
        
        # Apply custom parameters
        for key, value in config_dict.items():
            if key.startswith('lambda_'):
                setattr(config.model, key, value)
            elif key == 'learning_rate':
                config.training.learning_rate = value
            elif key in ['latent_dim', 'flow_hidden_dim', 'ode_steps', 'dropout']:
                setattr(config.model, key, value)
        
        # Create dataset
        dataset = MultiViewDataset(
            views=self.views,
            labels=self.labels,
            missing_rate=0.0,
            unaligned_rate=0.0
        )
        train_loader = create_dataloader(dataset, self.batch_size, shuffle=True)
        
        # Create model
        model = OTCFM(
            view_dims=self.view_dims,
            num_clusters=self.num_clusters,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            flow_hidden_dim=config.model.flow_hidden_dim,
            flow_num_layers=config.model.flow_num_layers,
            time_dim=config.model.time_dim,
            ode_steps=config.model.ode_steps,
            lambda_gw=config.model.lambda_gw,
            lambda_cluster=config.model.lambda_cluster,
            lambda_recon=config.model.lambda_recon,
            lambda_contrastive=config.model.lambda_contrastive,
            dropout=config.model.dropout,
            sigma_min=1e-4
        )
        
        # Train
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model, config.training, tmpdir)
            trainer.quiet = not self.verbose
            results = trainer.train(train_loader, self.labels)
        
        return results['final']
    
    def single_param_sweep(
        self,
        param_name: str,
        param_values: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Sweep a single parameter while keeping others at default values
        
        Args:
            param_name: Name of parameter to vary
            param_values: Values to test (uses PARAM_RANGES if None)
        
        Returns:
            DataFrame with results
        """
        if param_values is None:
            param_values = PARAM_RANGES[param_name]
        
        print(f"\n{'='*60}")
        print(f"Single Parameter Sweep: {param_name}")
        print(f"Values: {param_values}")
        print(f"Runs per value: {self.num_runs}")
        print(f"{'='*60}\n")
        
        results = []
        
        for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
            # Create config with default values + current parameter
            config = DEFAULT_VALUES.copy()
            config[param_name] = value
            
            # Run multiple times for statistical significance
            run_metrics = []
            for run in range(self.num_runs):
                self.seed = 42 + run
                metrics = self._train_with_config(config)
                run_metrics.append(metrics)
            
            # Aggregate metrics
            result = {
                'parameter': param_name,
                'value': value,
                'ACC_mean': np.mean([m['acc'] for m in run_metrics]),
                'ACC_std': np.std([m['acc'] for m in run_metrics]),
                'NMI_mean': np.mean([m['nmi'] for m in run_metrics]),
                'NMI_std': np.std([m['nmi'] for m in run_metrics]),
                'ARI_mean': np.mean([m['ari'] for m in run_metrics]),
                'ARI_std': np.std([m['ari'] for m in run_metrics]),
                'F1_mean': np.mean([m['f1'] for m in run_metrics]),
                'F1_std': np.std([m['f1'] for m in run_metrics]),
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        self.results.append(df)
        return df
    
    def two_param_grid(
        self,
        param1_name: str,
        param2_name: str,
        param1_values: Optional[List] = None,
        param2_values: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Grid search over two parameters
        
        Args:
            param1_name: First parameter name
            param2_name: Second parameter name
            param1_values: Values for first parameter
            param2_values: Values for second parameter
        
        Returns:
            DataFrame with results
        """
        if param1_values is None:
            param1_values = PARAM_RANGES[param1_name]
        if param2_values is None:
            param2_values = PARAM_RANGES[param2_name]
        
        print(f"\n{'='*60}")
        print(f"Two-Parameter Grid Search")
        print(f"  Parameter 1: {param1_name} ({len(param1_values)} values)")
        print(f"  Parameter 2: {param2_name} ({len(param2_values)} values)")
        print(f"  Total combinations: {len(param1_values) * len(param2_values)}")
        print(f"  Runs per combination: {self.num_runs}")
        print(f"{'='*60}\n")
        
        results = []
        
        for val1, val2 in tqdm(
            list(product(param1_values, param2_values)),
            desc=f"Grid {param1_name} x {param2_name}"
        ):
            # Create config
            config = DEFAULT_VALUES.copy()
            config[param1_name] = val1
            config[param2_name] = val2
            
            # Run multiple times
            run_metrics = []
            for run in range(self.num_runs):
                self.seed = 42 + run
                metrics = self._train_with_config(config)
                run_metrics.append(metrics)
            
            # Aggregate
            result = {
                'param1_name': param1_name,
                'param1_value': val1,
                'param2_name': param2_name,
                'param2_value': val2,
                'ACC_mean': np.mean([m['acc'] for m in run_metrics]),
                'ACC_std': np.std([m['acc'] for m in run_metrics]),
                'NMI_mean': np.mean([m['nmi'] for m in run_metrics]),
                'NMI_std': np.std([m['nmi'] for m in run_metrics]),
                'ARI_mean': np.mean([m['ari'] for m in run_metrics]),
                'ARI_std': np.std([m['ari'] for m in run_metrics]),
                'F1_mean': np.mean([m['f1'] for m in run_metrics]),
                'F1_std': np.std([m['f1'] for m in run_metrics]),
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        self.results.append(df)
        return df
    
    def full_analysis(self, quick_mode: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive sensitivity analysis for all parameters
        
        Args:
            quick_mode: Use fewer values for faster execution
        
        Returns:
            Dictionary mapping parameter names to DataFrames
        """
        print(f"\n{'='*60}")
        print("Full Sensitivity Analysis")
        print(f"Mode: {'Quick' if quick_mode else 'Comprehensive'}")
        print(f"{'='*60}\n")
        
        results_dict = {}
        
        for param_name in PARAM_RANGES.keys():
            if quick_mode:
                # Use fewer values in quick mode
                values = PARAM_RANGES[param_name][::2]  # Every other value
            else:
                values = PARAM_RANGES[param_name]
            
            df = self.single_param_sweep(param_name, values)
            results_dict[param_name] = df
        
        return results_dict


def save_results(
    results: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    analysis_type: str
) -> Tuple[str, str]:
    """
    Save results to CSV and JSON
    
    Returns:
        Tuple of (csv_path, json_path)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = Path(output_dir) / f"{dataset_name}_sensitivity_{analysis_type}_{timestamp}.csv"
    results.to_csv(csv_path, index=False, float_format='%.6f')
    
    # Save JSON (with metadata)
    json_path = Path(output_dir) / f"{dataset_name}_sensitivity_{analysis_type}_{timestamp}.json"
    metadata = {
        'dataset': dataset_name,
        'analysis_type': analysis_type,
        'timestamp': timestamp,
        'num_experiments': len(results),
        'columns': list(results.columns),
        'data': results.to_dict(orient='records')
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    return str(csv_path), str(json_path)


def plot_single_param_sweep(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    show: bool = False
) -> str:
    """
    Create publication-quality plots for single parameter sweep
    
    Returns:
        Path to saved figure
    """
    param_name = df['parameter'].iloc[0]
    values = df['value'].values
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sensitivity Analysis: {param_name} ({dataset_name})', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['ACC', 'NMI', 'ARI', 'F1']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for ax, metric, color in zip(axes.flat, metrics, colors):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        means = df[mean_col].values
        stds = df[std_col].values
        
        # Plot line with error bands
        ax.plot(values, means, 'o-', color=color, linewidth=2, 
                markersize=8, label=f'{metric}')
        ax.fill_between(values, means - stds, means + stds, 
                        alpha=0.2, color=color)
        
        # Formatting
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # Log scale for lambda parameters
        if param_name.startswith('lambda_') or param_name == 'learning_rate':
            ax.set_xscale('log')
        
        # Add value annotations for best performance
        best_idx = np.argmax(means)
        ax.annotate(f'Best: {means[best_idx]:.3f}',
                   xy=(values[best_idx], means[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = Path(output_dir) / f"{dataset_name}_sensitivity_{param_name}_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication
    pdf_path = fig_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"\nPlots saved:")
    print(f"  PNG: {fig_path}")
    print(f"  PDF: {pdf_path}")
    
    return str(fig_path)


def plot_two_param_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    show: bool = False
) -> str:
    """
    Create heatmap for two-parameter grid search
    
    Returns:
        Path to saved figure
    """
    param1_name = df['param1_name'].iloc[0]
    param2_name = df['param2_name'].iloc[0]
    
    # Create pivot tables for each metric
    metrics = ['ACC', 'NMI', 'ARI', 'F1']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Sensitivity Heatmap: {param1_name} vs {param2_name} ({dataset_name})',
                 fontsize=16, fontweight='bold')
    
    for ax, metric in zip(axes.flat, metrics):
        # Create pivot table
        pivot = df.pivot_table(
            values=f'{metric}_mean',
            index='param2_value',
            columns='param1_value',
            aggfunc='first'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': metric}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel(param1_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(param2_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Performance', fontsize=14, fontweight='bold')
        
        # Invert y-axis for better readability
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = Path(output_dir) / f"{dataset_name}_heatmap_{param1_name}_{param2_name}_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    pdf_path = fig_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"\nHeatmap saved:")
    print(f"  PNG: {fig_path}")
    print(f"  PDF: {pdf_path}")
    
    return str(fig_path)


def plot_3d_surface(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    metric: str = 'ACC',
    show: bool = False
) -> str:
    """
    Create 3D surface plot for two-parameter analysis
    
    Args:
        df: DataFrame from two_param_grid
        output_dir: Directory to save plots
        dataset_name: Name of dataset
        metric: Metric to plot ('ACC', 'NMI', 'ARI', 'F1')
        show: Whether to display plot
    
    Returns:
        Path to saved figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    param1_name = df['param1_name'].iloc[0]
    param2_name = df['param2_name'].iloc[0]
    
    # Create pivot table
    pivot = df.pivot_table(
        values=f'{metric}_mean',
        index='param2_value',
        columns='param1_value',
        aggfunc='first'
    )
    
    # Create meshgrid
    X = pivot.columns.values
    Y = pivot.index.values
    X, Y = np.meshgrid(X, Y)
    Z = pivot.values
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
    
    # Contour plot on bottom
    ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
    
    # Labels
    ax.set_xlabel(param1_name, fontsize=12, fontweight='bold')
    ax.set_ylabel(param2_name, fontsize=12, fontweight='bold')
    ax.set_zlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'3D Surface: {metric} vs {param1_name} & {param2_name} ({dataset_name})',
                fontsize=14, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Log scale if needed
    if param1_name.startswith('lambda_') or param1_name == 'learning_rate':
        ax.set_xscale('log')
    if param2_name.startswith('lambda_') or param2_name == 'learning_rate':
        ax.set_yscale('log')
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = Path(output_dir) / f"{dataset_name}_3d_{metric}_{param1_name}_{param2_name}_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    pdf_path = fig_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"\n3D plot saved:")
    print(f"  PNG: {fig_path}")
    print(f"  PDF: {pdf_path}")
    
    return str(fig_path)


def generate_statistical_report(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    analysis_type: str
) -> str:
    """
    Generate statistical analysis report
    
    Returns:
        Path to saved report
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"{dataset_name}_statistics_{analysis_type}_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"SENSITIVITY ANALYSIS STATISTICAL REPORT\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Analysis Type: {analysis_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        if 'parameter' in df.columns:
            # Single parameter analysis
            param_name = df['parameter'].iloc[0]
            f.write(f"Parameter: {param_name}\n")
            f.write(f"Number of values tested: {len(df)}\n")
            f.write(f"Value range: [{df['value'].min():.4f}, {df['value'].max():.4f}]\n\n")
            
            for metric in ['ACC', 'NMI', 'ARI', 'F1']:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                
                means = df[mean_col].values
                stds = df[std_col].values
                
                best_idx = np.argmax(means)
                worst_idx = np.argmin(means)
                
                f.write(f"\n{metric} Statistics:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Best value: {df['value'].iloc[best_idx]:.4f}\n")
                f.write(f"  Best performance: {means[best_idx]:.4f} ± {stds[best_idx]:.4f}\n")
                f.write(f"  Worst value: {df['value'].iloc[worst_idx]:.4f}\n")
                f.write(f"  Worst performance: {means[worst_idx]:.4f} ± {stds[worst_idx]:.4f}\n")
                f.write(f"  Performance range: {means.max() - means.min():.4f}\n")
                f.write(f"  Mean across all values: {means.mean():.4f}\n")
                f.write(f"  Std across all values: {means.std():.4f}\n")
                
                # Sensitivity score (normalized range)
                sensitivity = (means.max() - means.min()) / means.mean()
                f.write(f"  Sensitivity score: {sensitivity:.4f}\n")
                
                # Correlation with parameter value (if numeric)
                if df['value'].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    corr, p_value = stats.pearsonr(df['value'].values, means)
                    f.write(f"  Pearson correlation: {corr:.4f} (p={p_value:.4e})\n")
        
        else:
            # Two-parameter analysis
            param1_name = df['param1_name'].iloc[0]
            param2_name = df['param2_name'].iloc[0]
            
            f.write(f"Parameters: {param1_name} x {param2_name}\n")
            f.write(f"Grid size: {len(df['param1_value'].unique())} x {len(df['param2_value'].unique())}\n")
            f.write(f"Total experiments: {len(df)}\n\n")
            
            for metric in ['ACC', 'NMI', 'ARI', 'F1']:
                mean_col = f'{metric}_mean'
                means = df[mean_col].values
                
                best_idx = np.argmax(means)
                
                f.write(f"\n{metric} Statistics:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Best {param1_name}: {df['param1_value'].iloc[best_idx]:.4f}\n")
                f.write(f"  Best {param2_name}: {df['param2_value'].iloc[best_idx]:.4f}\n")
                f.write(f"  Best performance: {means[best_idx]:.4f}\n")
                f.write(f"  Performance range: {means.max() - means.min():.4f}\n")
                f.write(f"  Mean performance: {means.mean():.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\nStatistical report saved: {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for OT-CFM')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='Scene15',
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Path to data directory')
    
    # Analysis mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'grid', 'full'],
                       help='Analysis mode: single param, grid search, or full analysis')
    parser.add_argument('--param', type=str, nargs='+', default=['lambda_gw'],
                       help='Parameter(s) to analyze')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Runs per configuration for statistical significance')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto-detect if not specified)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='results/sensitivity',
                       help='Directory to save results')
    parser.add_argument('--show_plots', action='store_true',
                       help='Display plots interactively')
    parser.add_argument('--plot_3d', action='store_true',
                       help='Generate 3D plots (only for grid mode)')
    parser.add_argument('--quick_mode', action='store_true',
                       help='Use fewer parameter values for faster execution')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(
        dataset_name=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        num_runs=args.num_runs,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Run analysis
    if args.mode == 'single':
        # Single parameter sweep
        if len(args.param) != 1:
            raise ValueError("Single mode requires exactly 1 parameter")
        
        param_name = args.param[0]
        if param_name not in PARAM_RANGES:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        df = analyzer.single_param_sweep(param_name)
        
        # Save results
        save_results(df, args.output_dir, args.dataset, f"single_{param_name}")
        
        # Plot
        plot_single_param_sweep(df, args.output_dir, args.dataset, args.show_plots)
        
        # Statistical report
        generate_statistical_report(df, args.output_dir, args.dataset, f"single_{param_name}")
    
    elif args.mode == 'grid':
        # Two-parameter grid search
        if len(args.param) != 2:
            raise ValueError("Grid mode requires exactly 2 parameters")
        
        param1, param2 = args.param
        if param1 not in PARAM_RANGES or param2 not in PARAM_RANGES:
            raise ValueError(f"Unknown parameter(s)")
        
        df = analyzer.two_param_grid(param1, param2)
        
        # Save results
        save_results(df, args.output_dir, args.dataset, f"grid_{param1}_{param2}")
        
        # Plot heatmap
        plot_two_param_heatmap(df, args.output_dir, args.dataset, args.show_plots)
        
        # Optional 3D plots
        if args.plot_3d:
            for metric in ['ACC', 'NMI']:
                plot_3d_surface(df, args.output_dir, args.dataset, metric, args.show_plots)
        
        # Statistical report
        generate_statistical_report(df, args.output_dir, args.dataset, f"grid_{param1}_{param2}")
    
    elif args.mode == 'full':
        # Full sensitivity analysis
        results_dict = analyzer.full_analysis(args.quick_mode)
        
        # Save and plot each parameter
        for param_name, df in results_dict.items():
            save_results(df, args.output_dir, args.dataset, f"full_{param_name}")
            plot_single_param_sweep(df, args.output_dir, args.dataset, False)
            generate_statistical_report(df, args.output_dir, args.dataset, f"full_{param_name}")
    
    print("\n" + "="*60)
    print("Sensitivity Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
