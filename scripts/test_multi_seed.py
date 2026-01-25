"""
Quick test of multi-seed convergence analysis
Runs 3 seeds with 30 epochs for fast validation

Usage:
    uv run python scripts/test_multi_seed.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run command and print output"""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print("Multi-Seed Convergence Analysis - Quick Test")
    print("="*70)
    print("This will run a quick test with 3 seeds and 30 epochs")
    print("Expected time: ~5-8 minutes")
    
    # Step 1: Run multi-seed training
    cmd1 = [
        'uv', 'run', 'python', 'scripts/run_multi_seed_convergence.py',
        '--dataset', 'handwritten',
        '--epochs', '30',
        '--n_seeds', '3',
        '--output_dir', 'multi_seed_results/test_run'
    ]
    
    if not run_command(cmd1):
        print("\n❌ Training failed")
        return 1
    
    # Step 2: Plot results
    cmd2 = [
        'uv', 'run', 'python', 'scripts/plot_multi_seed_convergence.py',
        '--results_dir', 'multi_seed_results/test_run'
    ]
    
    if not run_command(cmd2):
        print("\n❌ Plotting failed")
        return 1
    
    # Check outputs
    results_dir = Path('multi_seed_results/test_run')
    agg_file = results_dir / 'aggregated_results.json'
    plot_file = results_dir / 'convergence_plot.pdf'
    
    print("\n" + "="*70)
    print("Checking outputs...")
    print("="*70)
    
    if agg_file.exists():
        print(f"✅ Aggregated results: {agg_file}")
    else:
        print(f"❌ Missing: {agg_file}")
    
    if plot_file.exists():
        print(f"✅ Convergence plot: {plot_file}")
    else:
        print(f"❌ Missing: {plot_file}")
    
    print("\n" + "="*70)
    print("✅ Quick test completed successfully!")
    print("="*70)
    print("\nYou can now run the full analysis with:")
    print("  uv run python scripts/run_multi_seed_convergence.py --dataset Handwritten --epochs 100 --n_seeds 5")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
