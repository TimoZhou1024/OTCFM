"""
Multi-seed convergence analysis for OT-CFM
Runs training with multiple random seeds and saves metrics for statistical analysis.

Usage:
    uv run python scripts/run_multi_seed_convergence.py --dataset Handwritten --epochs 100 --n_seeds 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from otcfm.config import (
    ExperimentConfig, ModelConfig, TrainingConfig, DataConfig,
    get_default_config
)
from otcfm.datasets import (
    load_scene15, load_handwritten, load_coil20, load_noisy_mnist,
    load_caltech101, load_cub, load_bdgp, load_nus_wide,
    MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.utils import set_seed


DATASET_LOADERS = {
    'scene15': load_scene15,
    'handwritten': load_handwritten,
    'coil20': load_coil20,
    'noisymnist': load_noisy_mnist,
    'caltech101': load_caltech101,
    'cub': load_cub,
    'bdgp': load_bdgp,
    'nus_wide': load_nus_wide,
    'nus-wide': load_nus_wide,
    'nuswide': load_nus_wide,
}


def run_single_seed(dataset_name, seed, epochs, config, output_dir):
    """Run training with a single seed and return history"""
    
    # Set seed
    set_seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Seed {seed}: Loading dataset {dataset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    loader = DATASET_LOADERS[dataset_name.lower()]
    views, labels = loader(data_dir='./data')
    num_clusters = len(np.unique(labels))
    view_dims = [v.shape[1] for v in views]
    
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {views[0].shape[0]}")
    print(f"Views: {len(views)}")
    print(f"View dimensions: {view_dims}")
    print(f"Clusters: {num_clusters}")
    print(f"Random seed: {seed}\n")
    
    # Create dataset and dataloader
    dataset = MultiViewDataset(
        views, labels,
        missing_rate=config.data.missing_rate,
        unaligned_rate=config.data.unaligned_rate
    )
    train_loader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        num_clusters=num_clusters,
        flow_hidden_dim=config.model.flow_hidden_dim,
        flow_num_layers=config.model.flow_num_layers,
        time_dim=config.model.time_dim,
        ode_steps=config.model.ode_steps,
        lambda_gw=config.model.lambda_gw,
        lambda_cluster=config.model.lambda_cluster,
        lambda_recon=config.model.lambda_recon,
        lambda_contrastive=config.model.lambda_contrastive,
        dropout=config.model.dropout,
        is_aligned=(config.data.unaligned_rate == 0.0)
    )
    
    # Create experiment directory for this seed
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        experiment_dir=str(seed_dir),
        device=config.training.device,
        verbose=False  # Disable verbose for cleaner output
    )
    
    # Override epochs
    trainer.config.epochs = epochs
    
    # Train and get history
    print(f"Training with seed {seed}...")
    results = trainer.train(train_loader, labels)
    
    # Save history
    history_file = seed_dir / 'history.json'
    with open(history_file, 'w') as f:
        json.dump(results['history'], f, indent=2)
    
    # Get best ACC (use best if available, otherwise final)
    best_acc = results.get('best', {}).get('acc', results.get('final', {}).get('acc', 0.0))
    print(f"✅ Seed {seed} completed - Best ACC: {best_acc:.4f}")
    
    return results['history']


def aggregate_histories(histories, output_dir):
    """Aggregate histories from multiple seeds and compute statistics"""
    
    n_seeds = len(histories)
    n_epochs = len(histories[0])
    
    print(f"\n{'='*70}")
    print(f"Aggregating results from {n_seeds} seeds × {n_epochs} epochs")
    print(f"{'='*70}")
    
    # Metric and loss keys
    metric_keys = ['acc', 'nmi', 'ari', 'purity', 'f1']
    loss_keys = ['loss', 'recon', 'gw', 'cluster', 'contrastive', 'cfm']
    
    # Initialize aggregated data structure
    aggregated = {
        'n_seeds': n_seeds,
        'n_epochs': n_epochs,
        'epochs': list(range(n_epochs)),
        'metrics': {},
        'losses': {}
    }
    
    # Aggregate metrics
    for key in metric_keys:
        values = np.array([[h[epoch].get(key, 0) for epoch in range(n_epochs)] 
                          for h in histories])
        aggregated['metrics'][key] = {
            'mean': values.mean(axis=0).tolist(),
            'std': values.std(axis=0).tolist(),
            'min': values.min(axis=0).tolist(),
            'max': values.max(axis=0).tolist()
        }
    
    # Aggregate losses
    for key in loss_keys:
        values = np.array([[h[epoch].get(key, 0) for epoch in range(n_epochs)] 
                          for h in histories])
        aggregated['losses'][key] = {
            'mean': values.mean(axis=0).tolist(),
            'std': values.std(axis=0).tolist(),
            'min': values.min(axis=0).tolist(),
            'max': values.max(axis=0).tolist()
        }
    
    # Compute final statistics
    final_stats = {
        'metrics': {},
        'losses': {}
    }
    
    for key in metric_keys:
        final_values = [h[-1].get(key, 0) for h in histories]
        final_stats['metrics'][key] = {
            'mean': float(np.mean(final_values)),
            'std': float(np.std(final_values)),
            'min': float(np.min(final_values)),
            'max': float(np.max(final_values))
        }
    
    for key in loss_keys:
        final_values = [h[-1].get(key, 0) for h in histories]
        final_stats['losses'][key] = {
            'mean': float(np.mean(final_values)),
            'std': float(np.std(final_values)),
            'min': float(np.min(final_values)),
            'max': float(np.max(final_values))
        }
    
    aggregated['final_stats'] = final_stats
    
    # Save aggregated results
    agg_file = output_dir / 'aggregated_results.json'
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"✅ Aggregated results saved to: {agg_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Final Results (Mean ± Std across {n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"Metrics:")
    for key in metric_keys:
        stats = final_stats['metrics'][key]
        print(f"  {key.upper():8s}: {stats['mean']*100:6.2f}% ± {stats['std']*100:5.2f}%")
    
    print(f"\nLosses:")
    for key in loss_keys:
        stats = final_stats['losses'][key]
        print(f"  {key:12s}: {stats['mean']:7.4f} ± {stats['std']:6.4f}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Multi-seed convergence analysis')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASET_LOADERS.keys()),
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds to run (default: 5)')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: multi_seed_results/{dataset}_{timestamp})')
    parser.add_argument('--missing_rate', type=float, default=0.0,
                       help='Missing view rate (default: 0.0)')
    parser.add_argument('--unaligned_rate', type=float, default=0.0,
                       help='Unaligned rate (default: 0.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'multi_seed_results/{args.dataset}_{timestamp}')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Multi-Seed Convergence Analysis")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Seed range: {args.start_seed} to {args.start_seed + args.n_seeds - 1}")
    print(f"Output directory: {output_dir}")
    print(f"Missing rate: {args.missing_rate}")
    print(f"Unaligned rate: {args.unaligned_rate}")
    
    # Get default config
    config = get_default_config()
    config.data.dataset_name = args.dataset
    config.data.missing_rate = args.missing_rate
    config.data.unaligned_rate = args.unaligned_rate
    
    # Load tuned parameters if available
    tuned_params_file = Path('config/tuned_params.json')
    if tuned_params_file.exists():
        with open(tuned_params_file, 'r') as f:
            tuned_params = json.load(f)
        
        if args.dataset in tuned_params:
            print(f"\n✅ Loading tuned parameters for {args.dataset}")
            params = tuned_params[args.dataset]
            
            # Update model config
            for key, value in params.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
    
    # Save experiment config
    config_dict = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'missing_rate': args.missing_rate,
        'unaligned_rate': args.unaligned_rate,
        'model_config': vars(config.model),
        'training_config': vars(config.training)
    }
    
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Run training for each seed
    histories = []
    seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    
    for seed in seeds:
        try:
            history = run_single_seed(
                args.dataset, seed, args.epochs, config, output_dir
            )
            histories.append(history)
        except Exception as e:
            print(f"❌ Error with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(histories) == 0:
        print("❌ No successful runs. Exiting.")
        return
    
    if len(histories) < args.n_seeds:
        print(f"⚠️  Only {len(histories)}/{args.n_seeds} runs completed successfully")
    
    # Aggregate results
    aggregated = aggregate_histories(histories, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Multi-seed convergence analysis complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext step: Visualize results with:")
    print(f"  uv run python scripts/plot_multi_seed_convergence.py --results_dir {output_dir}")


if __name__ == '__main__':
    main()
