"""
Tune hyperparameters for all datasets using Optuna
Saves results to config/tuned_params.json
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_optuna_tuning import OptunaHyperparameterTuner

# List of datasets to tune
DATASETS = [
    "synthetic",
    "handwritten",
    "coil20",
    "scene15",
    # Add more as needed
    # "caltech101",
    # "noisy_mnist",
    # "bdgp",
]


def main():
    parser = argparse.ArgumentParser(description='Tune all datasets with Optuna')
    
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets to tune (default: all)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of Optuna trials per dataset')
    parser.add_argument('--tuning_epochs', type=int, default=50,
                        help='Number of epochs per trial')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='config',
                        help='Directory to save tuned parameters')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that already have tuned parameters')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Determine datasets to tune
    datasets = args.datasets if args.datasets else DATASETS
    
    # Check for existing tuned params if skip_existing
    if args.skip_existing:
        import json
        tuned_path = Path(args.save_dir) / "tuned_params.json"
        if tuned_path.exists():
            with open(tuned_path) as f:
                existing = set(json.load(f).keys())
            datasets = [d for d in datasets if d.lower() not in existing]
            print(f"Skipping already tuned datasets. Remaining: {datasets}")
    
    print(f"\n{'#'*60}")
    print(f"Tuning {len(datasets)} datasets")
    print(f"Trials per dataset: {args.n_trials}")
    print(f"{'#'*60}\n")
    
    results = {}
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(datasets)}] Tuning: {dataset}")
        print(f"{'='*60}\n")
        
        try:
            tuner = OptunaHyperparameterTuner(
                dataset_name=dataset,
                data_root=args.data_root,
                n_trials=args.n_trials,
                device=args.device,
                seed=args.seed,
                tuning_epochs=args.tuning_epochs,
                save_dir=args.save_dir,
            )
            
            result = tuner.run()
            results[dataset] = {
                "best_acc": result["best_value"],
                "n_trials": result["n_trials"],
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error tuning {dataset}: {e}")
            results[dataset] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Print summary
    print(f"\n{'#'*60}")
    print("Tuning Summary")
    print(f"{'#'*60}")
    print(f"{'Dataset':<20} {'Best ACC':<12} {'Status':<10}")
    print("-" * 42)
    
    for dataset, result in results.items():
        if result["status"] == "success":
            print(f"{dataset:<20} {result['best_acc']:<12.4f} {'✓':<10}")
        else:
            print(f"{dataset:<20} {'N/A':<12} {'✗ ' + result['error'][:20]:<10}")
    
    print(f"\nTuned parameters saved to: {args.save_dir}/tuned_params.json")
    print("\nTo use tuned parameters:")
    print("  uv run python scripts/run_experiment.py --dataset <name> --use_tuned")


if __name__ == "__main__":
    main()
