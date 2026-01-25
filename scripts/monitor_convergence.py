"""
Monitor convergence analysis experiment progress in real-time
"""
import time
from pathlib import Path
import pandas as pd
import json

def monitor_progress():
    """Monitor the latest convergence experiment"""
    
    results_dir = Path("convergence_results")
    
    if not results_dir.exists():
        print("⚠️  No convergence_results directory yet. Waiting for experiment to start...")
        return
    
    # Find latest experiment
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not exp_dirs:
        print("⚠️  No experiment directories found yet.")
        return
    
    exp_dir = exp_dirs[0]
    print(f"\n{'='*70}")
    print(f"📊 Monitoring: {exp_dir.name}")
    print(f"{'='*70}\n")
    
    losses_csv = exp_dir / "losses.csv"
    metrics_csv = exp_dir / "metrics.csv"
    
    if losses_csv.exists():
        losses_df = pd.read_csv(losses_csv)
        print(f"✅ Training Progress:")
        print(f"   - Epochs logged: {len(losses_df)}")
        
        # Show phase breakdown
        for phase in ['recon', 'dec', 'full']:
            phase_data = losses_df[losses_df['phase'] == phase]
            if len(phase_data) > 0:
                print(f"   - Phase '{phase}': {len(phase_data)} epochs")
                if phase == 'full' and len(phase_data) > 1:
                    # Show recent losses
                    recent = phase_data.tail(3)
                    print(f"     Recent losses:")
                    for _, row in recent.iterrows():
                        print(f"       Epoch {row['epoch']}: total={row['total_loss']:.4f}")
    else:
        print("⏳ Waiting for losses.csv to be created...")
    
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        print(f"\n✅ Metrics Progress:")
        print(f"   - Evaluations: {len(metrics_df)}")
        
        # Show latest metrics
        if len(metrics_df) > 0:
            latest = metrics_df.iloc[-1]
            print(f"   - Latest (Epoch {latest['epoch']}, Phase {latest['phase']}):")
            print(f"     ACC: {latest['acc']*100:.2f}%")
            print(f"     NMI: {latest['nmi']*100:.2f}%")
            print(f"     ARI: {latest['ari']*100:.2f}%")
    else:
        print("\n⏳ Waiting for metrics.csv to be created...")
    
    # Check for completion
    final_json = exp_dir / "final_metrics.json"
    pdf_file = list(exp_dir.glob("*_convergence.pdf"))
    
    if final_json.exists() and pdf_file:
        print(f"\n{'='*70}")
        print(f"✅ EXPERIMENT COMPLETE!")
        print(f"{'='*70}")
        
        with open(final_json, 'r') as f:
            final = json.load(f)
        
        print(f"\n📊 Final Results:")
        print(f"   - ACC: {final['acc']*100:.2f}%")
        print(f"   - NMI: {final['nmi']*100:.2f}%")
        print(f"   - ARI: {final['ari']*100:.2f}%")
        print(f"   - Purity: {final['purity']*100:.2f}%")
        print(f"   - F1: {final['f1']*100:.2f}%")
        
        print(f"\n📁 Output Files:")
        print(f"   - Losses: {losses_csv}")
        print(f"   - Metrics: {metrics_csv}")
        print(f"   - Final: {final_json}")
        print(f"   - Figure: {pdf_file[0]}")
        
        print(f"\n{'='*70}")
        print(f"🎉 Next: Run 'uv run python scripts/check_convergence_results.py'")
        print(f"{'='*70}\n")
        
        return True
    else:
        print(f"\n⏳ Still running... (Check again in a few minutes)")
        return False


if __name__ == '__main__':
    complete = monitor_progress()
