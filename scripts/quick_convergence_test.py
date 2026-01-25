"""
Quick convergence test - shorter epochs for rapid visualization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_convergence_analysis import run_convergence_analysis

if __name__ == '__main__':
    # Quick test with fewer epochs
    result, exp_dir = run_convergence_analysis(
        dataset_name='Handwritten',
        epochs=60,  # Reduced from 150
        use_tuned=True,
        output_dir=Path("convergence_results"),
        data_dir="./data"
    )
    
    print(f"\n✓ Quick test complete!")
    print(f"Results in: {exp_dir}")
    print(f"Final ACC: {result['final']['acc']:.4f}")
    print(f"Plot: {exp_dir}/Handwritten_convergence.pdf")
