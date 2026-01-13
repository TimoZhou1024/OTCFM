"""
Robustness Testing for OT-CFM Multi-View Clustering
Tests performance under incomplete data and unaligned data conditions

Usage:
    # Test incomplete data (missing views)
    uv run python scripts/run_robustness_test.py --test_type incomplete --dataset Scene15
    
    # Test unaligned data (shuffled samples)
    uv run python scripts/run_robustness_test.py --test_type unaligned --dataset Scene15
    
    # Run both tests
    uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15
    
    # Include external baselines for comparison
    uv run python scripts/run_robustness_test.py --test_type both --dataset Scene15 --include_external
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

from otcfm.config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, get_default_config
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_synthetic, load_handwritten, load_coil20, load_nus_wide, load_cub,
    MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.baselines import get_baseline_methods, run_baseline_comparison
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
    'nus_wide': load_nus_wide,
    'cub': load_cub,
}

# Default test rates
MISSING_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]
UNALIGNED_RATES = [0.0, 0.2, 0.4, 0.6]

# ============================================================
# External Method Categories for Robustness Testing
# ============================================================

# Methods specialized for handling INCOMPLETE data (missing views)
INCOMPLETE_METHODS = {
    "COMPLETER (CVPR21)",   # Incomplete Multi-view Clustering via Contrastive Prediction
    "SURE (TPAMI22)",       # Robust Multi-View Clustering with Incomplete Information
    "DealMVC (CVPR23)",     # Dual Contrastive Prediction for Incomplete MVC
    "DCG (AAAI25)",         # Diffusion-based Cross-view Generation for Incomplete MVC
}

# Methods specialized for handling UNALIGNED data (cross-view correspondence)
UNALIGNED_METHODS = {
    "MRG-UMC (TNNLS25)",    # Multi-level Reliable Guidance for Unpaired MVC
    "CANDY (NeurIPS24)",    # Robust Contrastive MVC against Dual Noisy Correspondence
}

# General methods (work on standard aligned complete data)
GENERAL_METHODS = {
    "MFLVC (CVPR22)",       # Multi-level Feature Learning for Contrastive MVC
    "GCFAggMVC (CVPR23)",   # Global and Cross-view Feature Aggregation
}

def filter_methods_by_test_type(method_names: List[str], test_type: str) -> List[str]:
    """
    Filter methods based on test type for fair comparison.
    
    - incomplete test: include INCOMPLETE_METHODS + GENERAL_METHODS
    - unaligned test: include UNALIGNED_METHODS + GENERAL_METHODS
    
    Args:
        method_names: List of method names to filter
        test_type: 'incomplete' or 'unaligned'
    
    Returns:
        Filtered list of method names
    """
    if test_type == 'incomplete':
        allowed = INCOMPLETE_METHODS | GENERAL_METHODS
    elif test_type == 'unaligned':
        allowed = UNALIGNED_METHODS | GENERAL_METHODS
    else:
        # For other test types, include all methods
        return method_names
    
    return [m for m in method_names if m in allowed]


class RobustnessTest:
    """Run robustness tests with various missing/unaligned rates"""
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        epochs: int = 100,
        batch_size: int = 256,
        device: str = None,
        num_runs: int = 3,
        include_external: bool = False,
        include_internal: bool = True,
        verbose: bool = True,
        seed: int = 42,
        use_tuned: bool = False,
        tuned_key: str = None
    ):
        """
        Initialize robustness test
        
        Args:
            dataset_name: Name of dataset to test
            data_root: Path to data directory
            epochs: Training epochs
            batch_size: Batch size
            device: Device (auto-detect if None)
            num_runs: Number of runs per setting for averaging
            include_external: Include external baseline methods
            include_internal: Include internal baseline methods
            verbose: Print detailed output
            seed: Base random seed
            use_tuned: Use Optuna-tuned hyperparameters
            tuned_key: Specific key for tuned params (e.g., 'scene15_robust_incomplete')
        """
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.include_external = include_external
        self.include_internal = include_internal
        self.verbose = verbose
        self.base_seed = seed
        self.use_tuned = use_tuned
        self.tuned_key = tuned_key
        self.tuned_params = None
        
        # Load tuned parameters if requested
        if use_tuned:
            self._load_tuned_params()
        
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
    
    def _load_tuned_params(self):
        """Load tuned hyperparameters from config"""
        try:
            from run_optuna_tuning import load_tuned_params
            self.tuned_params = load_tuned_params(
                self.dataset_name, 
                config_dir="config",
                tuned_key=self.tuned_key
            )
            if self.tuned_params:
                print(f"Loaded tuned parameters: {self.tuned_key or self.dataset_name.lower()}")
            else:
                print("Warning: Could not load tuned parameters, using defaults")
        except ImportError:
            print("Warning: Could not import tuning module, using default parameters")
            self.tuned_params = None
    
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
    
    def _create_config(self, missing_rate: float = 0.0, unaligned_rate: float = 0.0) -> ExperimentConfig:
        """Create experiment config with specified rates"""
        config = get_default_config()
        config.data.dataset_name = self.dataset_name
        config.data.data_root = self.data_root
        config.data.missing_rate = missing_rate
        config.data.unaligned_rate = unaligned_rate
        config.training.epochs = self.epochs
        config.training.batch_size = self.batch_size
        config.training.device = self.device
        config.model.num_clusters = self.num_clusters
        return config
    
    def _train_otcfm(
        self,
        missing_rate: float = 0.0,
        unaligned_rate: float = 0.0,
        seed: int = 42
    ) -> Dict:
        """Train OT-CFM with specified settings and return metrics"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create dataset with perturbation
        dataset = MultiViewDataset(
            views=self.views,
            labels=self.labels,
            missing_rate=missing_rate,
            unaligned_rate=unaligned_rate,
            seed=seed
        )
        
        train_loader = create_dataloader(dataset, self.batch_size, shuffle=True)
        
        # Get model parameters (use tuned if available)
        if self.tuned_params:
            model_params = {
                'latent_dim': self.tuned_params.get('latent_dim', 128),
                'hidden_dims': eval(self.tuned_params['hidden_dims']) if isinstance(self.tuned_params.get('hidden_dims'), str) else self.tuned_params.get('hidden_dims', [512, 256]),
                'flow_hidden_dim': self.tuned_params.get('flow_hidden_dim', 256),
                'flow_num_layers': self.tuned_params.get('flow_num_layers', 4),
                'time_dim': self.tuned_params.get('time_dim', 64),
                'ode_steps': self.tuned_params.get('ode_steps', 10),
                'kernel_type': self.tuned_params.get('kernel_type', 'rbf'),
                'kernel_gamma': self.tuned_params.get('kernel_gamma', 1.0),
                'lambda_gw': self.tuned_params.get('lambda_gw', 0.1),
                'lambda_cluster': self.tuned_params.get('lambda_cluster', 0.5),
                'lambda_recon': self.tuned_params.get('lambda_recon', 1.0),
                'lambda_contrastive': self.tuned_params.get('lambda_contrastive', 0.1),
                'dropout': self.tuned_params.get('dropout', 0.1),
            }
        else:
            model_params = {
                'latent_dim': 128,
                'hidden_dims': [512, 256],
                'flow_hidden_dim': 256,
                'flow_num_layers': 4,
                'time_dim': 64,
                'ode_steps': 10,
                'kernel_type': 'rbf',
                'kernel_gamma': 1.0,
                'lambda_gw': 0.1,
                'lambda_cluster': 0.5,
                'lambda_recon': 1.0,
                'lambda_contrastive': 0.1,
                'dropout': 0.1,
            }
        
        # Create model
        model = OTCFM(
            view_dims=self.view_dims,
            num_clusters=self.num_clusters,
            sigma_min=1e-4,
            **model_params
        )
        
        # Create trainer (with quiet mode if not verbose)
        exp_dir = f"experiments/robustness_temp_{seed}"
        os.makedirs(exp_dir, exist_ok=True)
        
        config = self._create_config(missing_rate, unaligned_rate)
        
        # Apply tuned training params if available
        if self.tuned_params:
            if 'learning_rate' in self.tuned_params:
                config.training.learning_rate = self.tuned_params['learning_rate']
            if 'weight_decay' in self.tuned_params:
                config.training.weight_decay = self.tuned_params['weight_decay']
            if 'batch_size' in self.tuned_params:
                config.training.batch_size = self.tuned_params['batch_size']
        
        trainer = Trainer(
            model=model,
            config=config.training,
            experiment_dir=exp_dir,
            device=self.device,
            verbose=self.verbose
        )
        
        # Train
        results = trainer.train(train_loader, self.labels)
        
        # Clean up temp directory
        import shutil
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        
        return results['best']
    
    def _run_baselines(
        self,
        missing_rate: float = 0.0,
        unaligned_rate: float = 0.0,
        seed: int = 42
    ) -> Dict:
        """Run baseline methods with specified settings"""
        np.random.seed(seed)

        # Create dataset with perturbation
        dataset = MultiViewDataset(
            views=self.views,
            labels=self.labels,
            missing_rate=missing_rate,
            unaligned_rate=unaligned_rate,
            seed=seed
        )

        # Get missing mask
        mask = dataset.missing_mask.numpy() if missing_rate > 0 else None

        # For baselines, we pass the UNALIGNED views directly
        # This simulates the real scenario where baselines don't know the correct alignment
        # The perturbed views have the same indices but different content due to shuffling
        perturbed_views = []
        for v_idx in range(len(self.views)):
            # Apply the alignment permutation to create the unaligned view
            perm = dataset.alignment_perm[v_idx].numpy()
            view_data = dataset.views[v_idx].numpy()
            # The view_data is already normalized, apply permutation to get unaligned version
            perturbed_view = view_data[perm]

            # Apply missing mask: zero out missing entries
            # This simulates the real scenario where missing views are unavailable
            if mask is not None:
                view_mask = mask[:, v_idx]  # [N] mask for this view
                perturbed_view = perturbed_view.copy()  # Don't modify original
                perturbed_view[view_mask == 0] = 0.0  # Zero out missing samples

            perturbed_views.append(perturbed_view)

        results = run_baseline_comparison(
            perturbed_views,
            self.labels,
            self.num_clusters,
            self.device,
            mask,
            include_external=self.include_external,
            include_internal=self.include_internal
        )

        return results
    
    def run_incomplete_test(
        self,
        missing_rates: List[float] = None
    ) -> Dict:
        """
        Test with incomplete data (missing views)
        
        Args:
            missing_rates: List of missing rates to test
        
        Returns:
            Dictionary with results for each method and rate
        """
        if missing_rates is None:
            missing_rates = MISSING_RATES
        
        print("\n" + "="*60)
        print("Robustness Test: Incomplete Data (Missing Views)")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Missing rates: {missing_rates}")
        print(f"Runs per setting: {self.num_runs}")
        if self.include_external:
            print(f"External methods: INCOMPLETE + GENERAL categories")
            print(f"  - Incomplete: {', '.join(sorted(INCOMPLETE_METHODS))}")
            print(f"  - General: {', '.join(sorted(GENERAL_METHODS))}")
        
        all_results = {}
        
        for rate in missing_rates:
            print(f"\n--- Missing Rate: {rate:.0%} ---")
            rate_results = {}
            
            # Run OT-CFM multiple times
            otcfm_metrics = {'acc': [], 'nmi': [], 'ari': []}
            for run in range(self.num_runs):
                seed = self.base_seed + run
                if self.verbose:
                    print(f"  OT-CFM run {run+1}/{self.num_runs}...")
                metrics = self._train_otcfm(missing_rate=rate, seed=seed)
                otcfm_metrics['acc'].append(metrics['acc'])
                otcfm_metrics['nmi'].append(metrics['nmi'])
                otcfm_metrics['ari'].append(metrics.get('ari', 0))
            
            rate_results['OT-CFM'] = {
                'acc': np.mean(otcfm_metrics['acc']),
                'acc_std': np.std(otcfm_metrics['acc']),
                'nmi': np.mean(otcfm_metrics['nmi']),
                'nmi_std': np.std(otcfm_metrics['nmi']),
                'ari': np.mean(otcfm_metrics['ari']),
                'ari_std': np.std(otcfm_metrics['ari']),
            }
            print(f"  OT-CFM: ACC={rate_results['OT-CFM']['acc']:.4f}±{rate_results['OT-CFM']['acc_std']:.4f}")
            
            # Run baselines (once, as they're deterministic or we average internally)
            if self.include_external or self.include_internal:
                print("  Running baselines...")
                baseline_results = self._run_baselines(missing_rate=rate, seed=self.base_seed)
                
                # Filter external methods: only include INCOMPLETE + GENERAL methods
                for name, metrics in baseline_results.items():
                    if 'error' not in metrics:
                        # Check if it's an external method that should be filtered
                        is_external = any(tag in name for tag in ['CVPR', 'TPAMI', 'NeurIPS', 'AAAI', 'TNNLS'])
                        if is_external and name not in INCOMPLETE_METHODS and name not in GENERAL_METHODS:
                            if self.verbose:
                                print(f"  [Skip] {name} (not designed for incomplete data)")
                            continue
                        
                        rate_results[name] = {
                            'acc': metrics['acc'],
                            'nmi': metrics['nmi'],
                            'ari': metrics.get('ari', 0),
                            'acc_std': 0, 'nmi_std': 0, 'ari_std': 0
                        }
                        print(f"  {name}: ACC={metrics['acc']:.4f}")
            
            all_results[rate] = rate_results
        
        return all_results
    
    def run_unaligned_test(
        self,
        unaligned_rates: List[float] = None
    ) -> Dict:
        """
        Test with unaligned data (shuffled samples)
        
        Args:
            unaligned_rates: List of unaligned rates to test
        
        Returns:
            Dictionary with results for each method and rate
        """
        if unaligned_rates is None:
            unaligned_rates = UNALIGNED_RATES
        
        print("\n" + "="*60)
        print("Robustness Test: Unaligned Data (Shuffled Samples)")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Unaligned rates: {unaligned_rates}")
        print(f"Runs per setting: {self.num_runs}")
        if self.include_external:
            print(f"External methods: UNALIGNED + GENERAL categories")
            print(f"  - Unaligned: {', '.join(sorted(UNALIGNED_METHODS))}")
            print(f"  - General: {', '.join(sorted(GENERAL_METHODS))}")
        
        all_results = {}
        
        for rate in unaligned_rates:
            print(f"\n--- Unaligned Rate: {rate:.0%} ---")
            rate_results = {}
            
            # Run OT-CFM multiple times
            otcfm_metrics = {'acc': [], 'nmi': [], 'ari': []}
            for run in range(self.num_runs):
                seed = self.base_seed + run
                if self.verbose:
                    print(f"  OT-CFM run {run+1}/{self.num_runs}...")
                metrics = self._train_otcfm(unaligned_rate=rate, seed=seed)
                otcfm_metrics['acc'].append(metrics['acc'])
                otcfm_metrics['nmi'].append(metrics['nmi'])
                otcfm_metrics['ari'].append(metrics.get('ari', 0))
            
            rate_results['OT-CFM'] = {
                'acc': np.mean(otcfm_metrics['acc']),
                'acc_std': np.std(otcfm_metrics['acc']),
                'nmi': np.mean(otcfm_metrics['nmi']),
                'nmi_std': np.std(otcfm_metrics['nmi']),
                'ari': np.mean(otcfm_metrics['ari']),
                'ari_std': np.std(otcfm_metrics['ari']),
            }
            print(f"  OT-CFM: ACC={rate_results['OT-CFM']['acc']:.4f}±{rate_results['OT-CFM']['acc_std']:.4f}")
            
            # Run baselines
            if self.include_external or self.include_internal:
                print("  Running baselines...")
                baseline_results = self._run_baselines(unaligned_rate=rate, seed=self.base_seed)
                
                # Filter external methods: only include UNALIGNED + GENERAL methods
                for name, metrics in baseline_results.items():
                    if 'error' not in metrics:
                        # Check if it's an external method that should be filtered
                        is_external = any(tag in name for tag in ['CVPR', 'TPAMI', 'NeurIPS', 'AAAI', 'TNNLS'])
                        if is_external and name not in UNALIGNED_METHODS and name not in GENERAL_METHODS:
                            if self.verbose:
                                print(f"  [Skip] {name} (not designed for unaligned data)")
                            continue
                        
                        rate_results[name] = {
                            'acc': metrics['acc'],
                            'nmi': metrics['nmi'],
                            'ari': metrics.get('ari', 0),
                            'acc_std': 0, 'nmi_std': 0, 'ari_std': 0
                        }
                        print(f"  {name}: ACC={metrics['acc']:.4f}")
            
            all_results[rate] = rate_results
        
        return all_results


def save_robustness_results(
    results: Dict,
    test_type: str,
    dataset_name: str,
    save_dir: str = "results/robustness"
) -> Tuple[str, str]:
    """
    Save robustness test results to CSV and JSON
    
    Returns:
        Tuple of (csv_path, json_path)
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to DataFrame
    rows = []
    for rate, methods in results.items():
        for method, metrics in methods.items():
            rows.append({
                'Rate': rate,
                'Method': method,
                'ACC': metrics['acc'],
                'ACC_std': metrics.get('acc_std', 0),
                'NMI': metrics['nmi'],
                'NMI_std': metrics.get('nmi_std', 0),
                'ARI': metrics.get('ari', 0),
                'ARI_std': metrics.get('ari_std', 0),
            })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"{dataset_name}_{test_type}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON (for plotting)
    json_path = os.path.join(save_dir, f"{dataset_name}_{test_type}_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Convert float keys to strings for JSON
        json_results = {str(k): v for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    return csv_path, json_path


def plot_robustness_results(
    results: Dict,
    test_type: str,
    dataset_name: str,
    save_dir: str = "results/robustness",
    show: bool = False
) -> str:
    """
    Plot robustness test results
    
    Args:
        results: Dictionary with results
        test_type: 'incomplete' or 'unaligned'
        dataset_name: Dataset name
        save_dir: Directory to save plots
        show: Whether to display the plot
    
    Returns:
        Path to saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data for plotting
    rates = sorted([float(r) for r in results.keys()])
    methods = set()
    for rate_results in results.values():
        methods.update(rate_results.keys())
    methods = sorted(methods)
    
    # Define colors and markers for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Create figure with 2 subplots (ACC and NMI)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_label = "Missing Rate (η)" if test_type == 'incomplete' else "Unaligned Rate (p)"
    title_prefix = "Incomplete Data" if test_type == 'incomplete' else "Unaligned Data"
    
    for ax_idx, (ax, metric) in enumerate(zip(axes, ['ACC', 'NMI'])):
        for m_idx, method in enumerate(methods):
            metric_values = []
            metric_stds = []
            valid_rates = []
            
            for rate in rates:
                # Try both float and string keys
                rate_key = rate if rate in results else str(rate)
                if rate_key in results and method in results[rate_key]:
                    metric_values.append(results[rate_key][method][metric.lower()])
                    metric_stds.append(results[rate_key][method].get(f'{metric.lower()}_std', 0))
                    valid_rates.append(rate)
            
            if metric_values:
                color = colors[m_idx % len(colors)]
                marker = markers[m_idx % len(markers)]
                
                # Highlight OT-CFM
                linewidth = 2.5 if method == 'OT-CFM' else 1.5
                markersize = 10 if method == 'OT-CFM' else 6
                
                ax.errorbar(
                    valid_rates, metric_values,
                    yerr=metric_stds if max(metric_stds) > 0 else None,
                    label=method,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=linewidth,
                    capsize=3
                )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{title_prefix} - {metric} on {dataset_name}', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Format x-axis as percentage
        ax.set_xticks(rates)
        ax.set_xticklabels([f'{r:.0%}' for r in rates])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(save_dir, f"{dataset_name}_{test_type}_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for paper
    pdf_path = os.path.join(save_dir, f"{dataset_name}_{test_type}_{timestamp}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Plots saved to:")
    print(f"  PNG: {fig_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig_path


def plot_combined_results(
    incomplete_results: Dict,
    unaligned_results: Dict,
    dataset_name: str,
    save_dir: str = "results/robustness",
    show: bool = False
) -> str:
    """
    Create a combined 2x2 plot for paper
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect all methods
    all_methods = set()
    for results in [incomplete_results, unaligned_results]:
        if results:
            for rate_results in results.values():
                all_methods.update(rate_results.keys())
    methods = sorted(all_methods)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot configurations
    plot_configs = [
        (axes[0, 0], incomplete_results, 'ACC', 'Missing Rate (η)', 'Incomplete Data'),
        (axes[0, 1], incomplete_results, 'NMI', 'Missing Rate (η)', 'Incomplete Data'),
        (axes[1, 0], unaligned_results, 'ACC', 'Unaligned Rate (p)', 'Unaligned Data'),
        (axes[1, 1], unaligned_results, 'NMI', 'Unaligned Rate (p)', 'Unaligned Data'),
    ]

    def _compute_ylim(values: List[float]) -> Tuple[float, float]:
        if not values:
            return (0.0, 1.0)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        span = max(vmax - vmin, 1e-6)
        margin = max(0.1 * span, 0.02)
        lower = max(0.0, vmin - margin)
        upper = min(1.0, vmax + margin)
        if upper - lower < 0.1:
            center = (vmin + vmax) / 2
            lower = max(0.0, center - 0.05)
            upper = min(1.0, center + 0.05)
        return (lower, upper)

    for ax, results, metric, x_label, title_prefix in plot_configs:
        if results is None:
            ax.set_visible(False)
            continue
            
        rates = sorted([float(r) for r in results.keys()])
        
        for m_idx, method in enumerate(methods):
            metric_values = []
            metric_stds = []
            valid_rates = []
            
            for rate in rates:
                # Try both float and string keys
                rate_key = rate if rate in results else str(rate)
                if rate_key in results and method in results[rate_key]:
                    metric_values.append(results[rate_key][method][metric.lower()])
                    metric_stds.append(results[rate_key][method].get(f'{metric.lower()}_std', 0))
                    valid_rates.append(rate)
            
            if metric_values:
                color = colors[m_idx % len(colors)]
                marker = markers[m_idx % len(markers)]
                linewidth = 2.5 if method == 'OT-CFM' else 1.5
                markersize = 10 if method == 'OT-CFM' else 6
                
                ax.errorbar(
                    valid_rates, metric_values,
                    yerr=metric_stds if max(metric_stds) > 0 else None,
                    label=method,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=linewidth,
                    capsize=3
                )
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{title_prefix} - {metric}', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        # Dynamic y-limits based on visible results for this subplot
        all_vals = []
        if results:
            for rate_results in results.values():
                for method_results in rate_results.values():
                    val = method_results.get(metric.lower())
                    if val is not None:
                        try:
                            all_vals.append(float(val))
                        except Exception:
                            pass
        y_lower, y_upper = _compute_ylim(all_vals)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xticks(rates)
        ax.set_xticklabels([f'{r:.0%}' for r in rates])
    
    fig.suptitle(f'Robustness Test on {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, f"{dataset_name}_robustness_combined_{timestamp}.png")
    pdf_path = os.path.join(save_dir, f"{dataset_name}_robustness_combined_{timestamp}.pdf")
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Combined plot saved to:")
    print(f"  PNG: {fig_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Robustness Test for OT-CFM')
    
    parser.add_argument('--test_type', type=str, default='both',
                        choices=['incomplete', 'unaligned', 'both'],
                        help='Type of robustness test to run')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per setting')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detect if not specified)')
    parser.add_argument('--include_internal', action='store_true',
                        help='Include internal baseline methods (disabled by default)')
    parser.add_argument('--no_external', action='store_true',
                        help='Exclude external baseline methods')
    parser.add_argument('--save_dir', type=str, default='results/robustness',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively')
    parser.add_argument('--missing_rates', type=float, nargs='+',
                        default=None, help='Custom missing rates')
    parser.add_argument('--unaligned_rates', type=float, nargs='+',
                        default=None, help='Custom unaligned rates')
    parser.add_argument('--use_tuned', action='store_true',
                        help='Use Optuna-tuned hyperparameters for OT-CFM')
    parser.add_argument('--tuned_key', type=str, default=None,
                        help='Specific key for tuned params (e.g., scene15_robust_incomplete)')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create robustness tester
    tester = RobustnessTest(
        dataset_name=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        num_runs=args.num_runs,
        include_external=not args.no_external,
        include_internal=args.include_internal,
        verbose=args.verbose,
        seed=args.seed,
        use_tuned=args.use_tuned,
        tuned_key=args.tuned_key
    )
    
    incomplete_results = None
    unaligned_results = None
    
    # Run tests
    if args.test_type in ['incomplete', 'both']:
        incomplete_results = tester.run_incomplete_test(args.missing_rates)
        save_robustness_results(
            incomplete_results, 'incomplete', args.dataset, args.save_dir
        )
        plot_robustness_results(
            incomplete_results, 'incomplete', args.dataset, args.save_dir, args.show_plots
        )
    
    if args.test_type in ['unaligned', 'both']:
        unaligned_results = tester.run_unaligned_test(args.unaligned_rates)
        save_robustness_results(
            unaligned_results, 'unaligned', args.dataset, args.save_dir
        )
        plot_robustness_results(
            unaligned_results, 'unaligned', args.dataset, args.save_dir, args.show_plots
        )
    
    # Create combined plot if both tests were run
    if args.test_type == 'both':
        plot_combined_results(
            incomplete_results, unaligned_results,
            args.dataset, args.save_dir, args.show_plots
        )
    
    # Print summary
    print("\n" + "="*60)
    print("Robustness Test Summary")
    print("="*60)
    
    if incomplete_results:
        print("\nIncomplete Data Test:")
        print("-"*40)
        print(f"{'Rate':<10} {'OT-CFM ACC':<15} {'Best Baseline':<20}")
        for rate in sorted(incomplete_results.keys(), key=float):
            otcfm_acc = incomplete_results[rate]['OT-CFM']['acc']
            best_baseline = None
            best_acc = 0
            for method, metrics in incomplete_results[rate].items():
                if method != 'OT-CFM' and metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    best_baseline = method
            rate_str = f"{float(rate):.0%}"
            baseline_str = f"{best_baseline}: {best_acc:.4f}" if best_baseline else "N/A"
            print(f"{rate_str:<10} {otcfm_acc:.4f}±{incomplete_results[rate]['OT-CFM']['acc_std']:.4f}    {baseline_str}")
    
    if unaligned_results:
        print("\nUnaligned Data Test:")
        print("-"*40)
        print(f"{'Rate':<10} {'OT-CFM ACC':<15} {'Best Baseline':<20}")
        for rate in sorted(unaligned_results.keys(), key=float):
            otcfm_acc = unaligned_results[rate]['OT-CFM']['acc']
            best_baseline = None
            best_acc = 0
            for method, metrics in unaligned_results[rate].items():
                if method != 'OT-CFM' and metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    best_baseline = method
            rate_str = f"{float(rate):.0%}"
            baseline_str = f"{best_baseline}: {best_acc:.4f}" if best_baseline else "N/A"
            print(f"{rate_str:<10} {otcfm_acc:.4f}±{unaligned_results[rate]['OT-CFM']['acc_std']:.4f}    {baseline_str}")
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
