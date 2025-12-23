"""
Main experiment runner for OT-CFM
Provides unified interface for training, evaluation, and comparison
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import (
    ExperimentConfig, ModelConfig, TrainingConfig, DataConfig,
    get_default_config, parse_args
)
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_cub, load_reuters, load_synthetic,
    load_handwritten, load_coil20,
    create_dataloader, MultiViewDataset
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.baselines import get_baseline_methods, run_baseline_comparison
from otcfm.ablation import AblationStudy, AblationConfig
from otcfm.metrics import evaluate_clustering, compare_methods, MetricTracker


# Dataset loader mapping
DATASET_LOADERS = {
    'caltech101': load_caltech101,
    'scene15': load_scene15,
    'noisy_mnist': load_noisy_mnist,
    'bdgp': load_bdgp,
    'cub': load_cub,
    'reuters': load_reuters,
    'synthetic': load_synthetic,
    'handwritten': load_handwritten,
    'coil20': load_coil20,
}


def setup_experiment(config: ExperimentConfig) -> Path:
    """Setup experiment directory and logging"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{config.experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.json"
    config_dict = {
        'experiment_name': config.experiment_name,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'data': config.data.__dict__
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return exp_dir


def load_dataset(config: DataConfig) -> tuple:
    """
    Load dataset based on configuration
    
    Returns:
        views: List of view arrays
        labels: Ground truth labels
        view_dims: Dimensions of each view
    """
    dataset_name = config.dataset_name.lower()
    
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DATASET_LOADERS[dataset_name]
    
    if dataset_name == 'synthetic':
        data = loader(
            n_samples=config.n_samples if hasattr(config, 'n_samples') else 1000,
            n_clusters=config.num_clusters if hasattr(config, 'num_clusters') else 10
        )
        views = data['views']
        labels = data['labels']
    else:
        # Most loaders return (views, labels) tuple
        result = loader(config.data_root)
        if isinstance(result, tuple):
            views, labels = result
        else:
            views = result['views']
            labels = result['labels']
    
    view_dims = [v.shape[1] for v in views]
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Number of samples: {len(labels)}")
    print(f"  Number of views: {len(views)}")
    print(f"  View dimensions: {view_dims}")
    print(f"  Number of clusters: {len(np.unique(labels))}")
    
    return views, labels, view_dims


def run_experiment(config: ExperimentConfig) -> Dict:
    """
    Run full experiment with OT-CFM
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dictionary with experiment results
    """
    # Setup
    exp_dir = setup_experiment(config)
    device = config.training.device
    
    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load data
    views, labels, view_dims = load_dataset(config.data)
    
    # Create dataset with missing/unaligned simulation if needed
    dataset = MultiViewDataset(
        views=views,
        labels=labels,
        missing_rate=config.data.missing_rate,
        unaligned_rate=config.data.unaligned_rate
    )
    
    # Create dataloader
    train_loader = create_dataloader(dataset, config.training.batch_size, shuffle=True)
    
    # Update num_clusters if not set
    if config.model.num_clusters == 0:
        config.model.num_clusters = len(np.unique(labels))
    
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        num_clusters=config.model.num_clusters,
        flow_hidden_dim=config.model.flow_hidden_dim,
        flow_num_layers=config.model.flow_num_layers,
        time_dim=config.model.time_dim,
        ode_steps=config.model.ode_steps,
        sigma_min=config.model.sigma_min,
        kernel_type=config.model.kernel_type,
        kernel_gamma=config.model.kernel_gamma,
        lambda_gw=config.model.lambda_gw,
        lambda_cluster=config.model.lambda_cluster,
        lambda_recon=config.model.lambda_recon,
        lambda_contrastive=config.model.lambda_contrastive,
        dropout=config.model.dropout
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        experiment_dir=str(exp_dir),
        device=device
    )
    
    # Train
    print("\nTraining OT-CFM...")
    results = trainer.train(train_loader, labels)
    
    # Save results
    results_path = exp_dir / "results.json"
    save_results = {
        'final': results['final'],
        'best': results['best']
    }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    return results


def run_comparison_experiment(
    config: ExperimentConfig,
    run_baselines: bool = True,
    include_external: bool = True,
    include_internal: bool = True,
    save_dir: str = "results"
) -> Dict:
    """
    Run comparison experiment with baselines
    
    Args:
        config: Experiment configuration
        run_baselines: Whether to run baseline methods
    
    Returns:
        Dictionary with all method results
    """
    # Setup
    exp_dir = setup_experiment(config)
    device = config.training.device
    
    # Load data
    views, labels, view_dims = load_dataset(config.data)
    
    # Create dataset
    dataset = MultiViewDataset(
        views=views,
        labels=labels,
        missing_rate=config.data.missing_rate,
        unaligned_rate=config.data.unaligned_rate
    )
    
    train_loader = create_dataloader(dataset, config.training.batch_size, shuffle=True)
    
    if config.model.num_clusters == 0:
        config.model.num_clusters = len(np.unique(labels))
    
    all_results = {}
    
    # Run OT-CFM
    print("\n" + "="*60)
    print("Training OT-CFM...")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        num_clusters=config.model.num_clusters,
        flow_hidden_dim=config.model.flow_hidden_dim,
        flow_num_layers=config.model.flow_num_layers,
        time_dim=config.model.time_dim,
        ode_steps=config.model.ode_steps,
        sigma_min=config.model.sigma_min,
        kernel_type=config.model.kernel_type,
        kernel_gamma=config.model.kernel_gamma,
        lambda_gw=config.model.lambda_gw,
        lambda_cluster=config.model.lambda_cluster,
        lambda_recon=config.model.lambda_recon,
        lambda_contrastive=config.model.lambda_contrastive,
        dropout=config.model.dropout
    )
    
    trainer = Trainer(
        model=model,
        config=config.training,
        experiment_dir=str(exp_dir / "otcfm"),
        device=device
    )
    
    otcfm_results = trainer.train(train_loader, labels)
    all_results['OT-CFM'] = otcfm_results['best']
    
    # Run baselines if requested
    if run_baselines:
        print("\n" + "="*60)
        print("Running Baseline Methods...")
        print("="*60)
        
        # Convert to numpy for baselines
        views_np = [v.numpy() if isinstance(v, torch.Tensor) else v for v in views]
        mask_np = dataset.missing_mask.numpy() if hasattr(dataset, 'missing_mask') else None
        
        baseline_results = run_baseline_comparison(
            views_np, labels, config.model.num_clusters, device, mask_np,
            include_external, include_internal
        )
        
        for name, metrics in baseline_results.items():
            if 'error' not in metrics:
                all_results[name] = metrics
    
    # Print comparison
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)
    print(f"{'Method':<25} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-"*60)
    
    for method, metrics in sorted(all_results.items(), key=lambda x: -x[1].get('acc', 0)):
        if 'acc' in metrics:
            print(f"{method:<25} {metrics['acc']:<10.4f} {metrics['nmi']:<10.4f} {metrics.get('ari', 0):<10.4f}")
    
    # Save comparison results
    comparison_path = exp_dir / "comparison_results.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results to CSV in a dedicated results folder
    save_results_to_csv(
        all_results, 
        dataset_name=config.data.dataset_name,
        save_dir=save_dir
    )
    
    return all_results


def save_results_to_csv(
    results: Dict,
    dataset_name: str,
    save_dir: str = "results"
) -> str:
    """
    Save experiment results to a CSV file
    
    Args:
        results: Dictionary with method names as keys and metrics as values
        dataset_name: Name of the dataset used
        save_dir: Directory to save results
    
    Returns:
        Path to the saved CSV file
    """
    # Prepare data for DataFrame
    rows = []
    for method, metrics in results.items():
        if isinstance(metrics, dict) and 'error' not in metrics:
            row = {
                'Method': method,
                'ACC': metrics.get('acc', metrics.get('ACC', np.nan)),
                'NMI': metrics.get('nmi', metrics.get('NMI', np.nan)),
                'ARI': metrics.get('ari', metrics.get('ARI', np.nan)),
                'Purity': metrics.get('purity', metrics.get('Purity', np.nan)),
                'F1': metrics.get('f1', metrics.get('F1', np.nan)),
            }
            rows.append(row)
    
    # Create DataFrame and sort by ACC
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values('ACC', ascending=False)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f'{dataset_name}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nResults saved to: {results_path}")
    return results_path


def run_ablation_experiment(config: ExperimentConfig, save_dir: str = "results") -> Dict:
    """
    Run ablation study
    
    Args:
        config: Experiment configuration
        save_dir: Directory to save CSV results
    
    Returns:
        Dictionary with ablation results
    """
    # Setup
    exp_dir = setup_experiment(config)
    device = config.training.device
    
    # Load data
    views, labels, view_dims = load_dataset(config.data)
    
    # Create dataset
    dataset = MultiViewDataset(
        views=views,
        labels=labels,
        missing_rate=config.data.missing_rate,
        unaligned_rate=config.data.unaligned_rate
    )
    
    train_loader = create_dataloader(dataset, config.training.batch_size, shuffle=True)
    
    if config.model.num_clusters == 0:
        config.model.num_clusters = len(np.unique(labels))
    
    # Ablation configuration
    ablation_config = AblationConfig(
        modes=["full", "no_gw", "no_cluster", "no_ot", "no_flow", "no_contrastive"],
        num_runs=3,
        results_dir=str(exp_dir / "ablation")
    )
    
    # Run ablation study
    study = AblationStudy(config, ablation_config, device)
    results = study.run(train_loader, labels, view_dims)
    
    # Save ablation results to CSV
    save_ablation_results_to_csv(
        results,
        dataset_name=config.data.dataset_name,
        save_dir=save_dir
    )
    
    return results


def save_ablation_results_to_csv(
    results: Dict,
    dataset_name: str,
    save_dir: str = "results"
) -> str:
    """
    Save ablation study results to a CSV file
    
    Args:
        results: Dictionary with ablation mode results
        dataset_name: Name of the dataset used
        save_dir: Directory to save results
    
    Returns:
        Path to the saved CSV file
    """
    rows = []
    for mode, metrics in results.items():
        if isinstance(metrics, dict):
            row = {
                'Mode': mode,
                'ACC': metrics.get('acc', metrics.get('ACC', np.nan)),
                'NMI': metrics.get('nmi', metrics.get('NMI', np.nan)),
                'ARI': metrics.get('ari', metrics.get('ARI', np.nan)),
                'ACC_std': metrics.get('acc_std', np.nan),
                'NMI_std': metrics.get('nmi_std', np.nan),
            }
            rows.append(row)
    
    results_df = pd.DataFrame(rows)
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f'{dataset_name}_ablation_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nAblation results saved to: {results_path}")
    return results_path


def run_multi_dataset_experiment(
    datasets: List[str],
    base_config: ExperimentConfig,
    include_external: bool = False,
    include_internal: bool = True,
    save_dir: str = "results"
) -> Dict:
    """
    Run experiments on multiple datasets
    
    Args:
        datasets: List of dataset names
        base_config: Base configuration
        include_external: Whether to include external baseline methods
        include_internal: Whether to include internal baseline methods
        save_dir: Directory to save CSV results
    
    Returns:
        Dictionary with results for each dataset
    """
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'#'*60}")
        
        # Update config for this dataset
        config = base_config
        config.data.dataset_name = dataset_name
        config.experiment_name = f"otcfm_{dataset_name}"
        
        try:
            results = run_comparison_experiment(
                config, 
                run_baselines=True,
                include_external=include_external,
                include_internal=include_internal,
                save_dir=save_dir
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("Multi-Dataset Summary (OT-CFM Performance)")
    print("="*80)
    print(f"{'Dataset':<20} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-"*80)
    
    for dataset, results in all_results.items():
        if 'OT-CFM' in results:
            metrics = results['OT-CFM']
            print(f"{dataset:<20} {metrics['acc']:<10.4f} {metrics['nmi']:<10.4f} {metrics.get('ari', 0):<10.4f}")
        elif 'error' in results:
            print(f"{dataset:<20} Error: {results['error'][:40]}...")
    
    # Save multi-dataset summary to CSV
    save_multi_dataset_results_to_csv(all_results, save_dir=save_dir)
    
    return all_results


def save_multi_dataset_results_to_csv(
    all_results: Dict,
    save_dir: str = "results"
) -> str:
    """
    Save multi-dataset experiment results to a CSV file
    
    Args:
        all_results: Dictionary with dataset names as keys and method results as values
        save_dir: Directory to save results
    
    Returns:
        Path to the saved CSV file
    """
    rows = []
    
    for dataset_name, dataset_results in all_results.items():
        if isinstance(dataset_results, dict) and 'error' not in dataset_results:
            for method, metrics in dataset_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    row = {
                        'Dataset': dataset_name,
                        'Method': method,
                        'ACC': metrics.get('acc', metrics.get('ACC', np.nan)),
                        'NMI': metrics.get('nmi', metrics.get('NMI', np.nan)),
                        'ARI': metrics.get('ari', metrics.get('ARI', np.nan)),
                        'Purity': metrics.get('purity', metrics.get('Purity', np.nan)),
                        'F1': metrics.get('f1', metrics.get('F1', np.nan)),
                    }
                    rows.append(row)
    
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(['Dataset', 'ACC'], ascending=[True, False])
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f'multi_dataset_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nMulti-dataset results saved to: {results_path}")
    return results_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='OT-CFM Multi-View Clustering')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'compare', 'ablation', 'multi'],
                        help='Experiment mode')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help='Dataset name')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets for multi-dataset experiment')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu, auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--missing_rate', type=float, default=0.0,
                        help='Missing view rate')
    parser.add_argument('--unaligned_rate', type=float, default=0.0,
                        help='Unaligned sample rate')
    parser.add_argument('--include_external', action='store_true', default=False,
                        help='Include external baseline methods in comparison')
    parser.add_argument('--no_internal', action='store_true', default=False,
                        help='Exclude internal baseline methods in comparison')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save CSV results')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig(
            experiment_name=config_dict['experiment_name'],
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data'])
        )
    else:
        config = get_default_config()
        config.data.dataset_name = args.dataset
        config.data.data_root = args.data_root
        config.data.missing_rate = args.missing_rate
        config.data.unaligned_rate = args.unaligned_rate
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.training.device = args.device
    
    # Run experiment based on mode
    if args.mode == 'train':
        results = run_experiment(config)
        print(f"\nFinal ACC: {results['final']['acc']:.4f}")
        print(f"Best ACC: {results['best']['acc']:.4f}")
        
    elif args.mode == 'compare':
        results = run_comparison_experiment(
            config, run_baselines=True,
            include_external=args.include_external,
            include_internal=not args.no_internal,
            save_dir=args.results_dir
        )
        
    elif args.mode == 'ablation':
        results = run_ablation_experiment(config, save_dir=args.results_dir)
        
    elif args.mode == 'multi':
        datasets = args.datasets or ['synthetic', 'scene15', 'caltech101']
        results = run_multi_dataset_experiment(
            datasets, config,
            include_external=args.include_external,
            include_internal=not args.no_internal,
            save_dir=args.results_dir
        )
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
