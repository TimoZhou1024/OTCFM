"""
Fix aggregation for the incomplete multi-seed run
Manually aggregate existing history.json files from multi_seed_results/handwritten_20260122_115845/
"""

import json
import numpy as np
from pathlib import Path

def main():
    results_dir = Path('multi_seed_results/handwritten_20260122_115845')
    
    # Find all seed directories with history.json
    histories = []
    for seed_dir in sorted(results_dir.glob('seed_*')):
        history_file = seed_dir / 'history.json'
        if history_file.exists():
            print(f"Loading {history_file}...")
            with open(history_file, 'r') as f:
                history = json.load(f)
                histories.append(history)
                print(f"  └─ {len(history)} epochs loaded")
    
    print(f"\nTotal: {len(histories)} seeds found")
    
    if len(histories) == 0:
        print("❌ No history files found")
        return
    
    # Aggregate
    n_seeds = len(histories)
    n_epochs = len(histories[0])
    
    print(f"Aggregating {n_seeds} seeds × {n_epochs} epochs...")
    
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
    
    # Save aggregated results
    agg_file = results_dir / 'aggregated_results.json'
    
    # Add final stats (last epoch statistics)
    final_stats = {
        'metrics': {},
        'losses': {}
    }
    
    for key in metric_keys:
        mean_final = aggregated['metrics'][key]['mean'][-1]
        std_final = aggregated['metrics'][key]['std'][-1]
        min_final = aggregated['metrics'][key]['min'][-1]
        max_final = aggregated['metrics'][key]['max'][-1]
        final_stats['metrics'][key] = {
            'mean': mean_final,
            'std': std_final,
            'min': min_final,
            'max': max_final
        }
    
    for key in loss_keys:
        mean_final = aggregated['losses'][key]['mean'][-1]
        std_final = aggregated['losses'][key]['std'][-1]
        min_final = aggregated['losses'][key]['min'][-1]
        max_final = aggregated['losses'][key]['max'][-1]
        final_stats['losses'][key] = {
            'mean': mean_final,
            'std': std_final,
            'min': min_final,
            'max': max_final
        }
    
    aggregated['final_stats'] = final_stats
    
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {agg_file}")
    
    # Print final statistics
    print(f"\n{'='*70}")
    print(f"Final Results (Mean ± Std across {n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"Metrics (at epoch {n_epochs}):")
    for key in metric_keys:
        mean_final = aggregated['metrics'][key]['mean'][-1]
        std_final = aggregated['metrics'][key]['std'][-1]
        print(f"  {key.upper():8s}: {mean_final*100:6.2f}% ± {std_final*100:5.2f}%")
    
    print(f"\nLosses (at epoch {n_epochs}):")
    for key in loss_keys:
        mean_final = aggregated['losses'][key]['mean'][-1]
        std_final = aggregated['losses'][key]['std'][-1]
        print(f"  {key:12s}: {mean_final:7.4f} ± {std_final:6.4f}")
    
    print(f"\n{'='*70}")
    print("Next step: Plot results with:")
    print(f"  uv run python scripts/plot_multi_seed_convergence.py --results_dir {results_dir}")

if __name__ == '__main__':
    main()
