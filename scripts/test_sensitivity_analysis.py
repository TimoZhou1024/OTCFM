"""
Quick test example for sensitivity analysis
Tests the framework with minimal settings for verification
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from run_sensitivity_analysis import (
    SensitivityAnalyzer,
    plot_single_param_sweep,
    plot_two_param_heatmap,
    save_results,
    generate_statistical_report,
    PARAM_RANGES
)

def test_single_param():
    """Test single parameter sweep with minimal settings"""
    print("="*60)
    print("Testing Single Parameter Sweep")
    print("="*60)
    
    # Create analyzer with small dataset and few epochs
    analyzer = SensitivityAnalyzer(
        dataset_name='Synthetic',
        epochs=5,  # Very few epochs for testing
        batch_size=64,
        num_runs=1,  # Single run for speed
        verbose=True
    )
    
    # Test with just 3 values of lambda_gw
    test_values = [0.1, 0.2, 0.5]
    df = analyzer.single_param_sweep('lambda_gw', test_values)
    
    print("\nResults DataFrame:")
    print(df)
    
    # Save results
    csv_path, json_path = save_results(
        df, 
        'results/sensitivity_test',
        'Synthetic',
        'single_lambda_gw_test'
    )
    
    # Generate plot
    fig_path = plot_single_param_sweep(
        df,
        'results/sensitivity_test',
        'Synthetic',
        show=False
    )
    
    # Generate report
    report_path = generate_statistical_report(
        df,
        'results/sensitivity_test',
        'Synthetic',
        'single_lambda_gw_test'
    )
    
    print(f"\n✓ Single parameter test completed successfully")
    print(f"  Files created: {csv_path}, {fig_path}, {report_path}")
    
    return df

def test_grid_search():
    """Test two-parameter grid search with minimal settings"""
    print("\n" + "="*60)
    print("Testing Grid Search (2 Parameters)")
    print("="*60)
    
    analyzer = SensitivityAnalyzer(
        dataset_name='Synthetic',
        epochs=5,
        batch_size=64,
        num_runs=1,
        verbose=True
    )
    
    # Test with just 2x2 grid
    param1_values = [0.1, 0.5]
    param2_values = [0.5, 2.0]
    
    df = analyzer.two_param_grid(
        'lambda_gw',
        'lambda_cluster',
        param1_values,
        param2_values
    )
    
    print("\nResults DataFrame:")
    print(df)
    
    # Save results
    csv_path, json_path = save_results(
        df,
        'results/sensitivity_test',
        'Synthetic',
        'grid_test'
    )
    
    # Generate heatmap
    fig_path = plot_two_param_heatmap(
        df,
        'results/sensitivity_test',
        'Synthetic',
        show=False
    )
    
    # Generate report
    report_path = generate_statistical_report(
        df,
        'results/sensitivity_test',
        'Synthetic',
        'grid_test'
    )
    
    print(f"\n✓ Grid search test completed successfully")
    print(f"  Files created: {csv_path}, {fig_path}, {report_path}")
    
    return df

def main():
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS TEST SUITE")
    print("="*60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test 1: Single parameter
        df_single = test_single_param()
        
        # Test 2: Grid search
        df_grid = test_grid_search()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nYou can now run full sensitivity analysis with:")
        print("  uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --mode single --param lambda_gw")
        print("  uv run python scripts/run_sensitivity_analysis.py --dataset Scene15 --mode grid --param lambda_gw lambda_cluster")
        print("\nSee docs/sensitivity_analysis_guide.md for more details.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
