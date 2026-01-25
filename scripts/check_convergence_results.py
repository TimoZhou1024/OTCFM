"""
Quick script to check convergence analysis results and copy best figure to paper
"""
import json
import pandas as pd
from pathlib import Path
import shutil

def check_results():
    """Check latest convergence results and prepare for paper"""
    
    results_dir = Path("convergence_results")
    
    if not results_dir.exists():
        print("❌ No convergence_results directory found!")
        return
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not exp_dirs:
        print("❌ No experiment directories found!")
        return
    
    print(f"📊 Found {len(exp_dirs)} experiment(s)\n")
    
    # Check the latest experiment
    for i, exp_dir in enumerate(exp_dirs[:3]):  # Show top 3 latest
        print(f"{'='*70}")
        print(f"Experiment {i+1}: {exp_dir.name}")
        print(f"{'='*70}")
        
        # Check for required files
        losses_csv = exp_dir / "losses.csv"
        metrics_csv = exp_dir / "metrics.csv"
        final_json = exp_dir / "final_metrics.json"
        pdf_file = exp_dir / f"{exp_dir.name.split('_')[0]}_convergence.pdf"
        
        files_ok = True
        
        if losses_csv.exists():
            print(f"✅ losses.csv found")
            losses_df = pd.read_csv(losses_csv)
            print(f"   - {len(losses_df)} epochs logged")
            print(f"   - Phases: {', '.join(losses_df['phase'].unique())}")
        else:
            print(f"❌ losses.csv missing")
            files_ok = False
        
        if metrics_csv.exists():
            print(f"✅ metrics.csv found")
            metrics_df = pd.read_csv(metrics_csv)
            print(f"   - {len(metrics_df)} epochs logged")
        else:
            print(f"❌ metrics.csv missing")
            files_ok = False
        
        if final_json.exists():
            print(f"✅ final_metrics.json found")
            with open(final_json, 'r') as f:
                final = json.load(f)
            print(f"   - ACC: {final.get('acc', 0)*100:.2f}%")
            print(f"   - NMI: {final.get('nmi', 0)*100:.2f}%")
            print(f"   - ARI: {final.get('ari', 0)*100:.2f}%")
        else:
            print(f"❌ final_metrics.json missing")
            files_ok = False
        
        if pdf_file.exists():
            print(f"✅ {pdf_file.name} found")
            print(f"   - Size: {pdf_file.stat().st_size / 1024:.1f} KB")
        else:
            print(f"❌ {pdf_file.name} missing")
            files_ok = False
        
        if files_ok and i == 0:  # Latest and complete
            print(f"\n{'='*70}")
            print(f"📋 RECOMMENDED ACTIONS FOR PAPER:")
            print(f"{'='*70}")
            
            # Suggest copying to figures
            figures_dir = Path("figures")
            figures_dir.mkdir(exist_ok=True)
            
            target_pdf = figures_dir / pdf_file.name
            
            print(f"\n1. Copy figure to paper directory:")
            print(f"   cp {pdf_file} {target_pdf}")
            
            if pdf_file.exists():
                shutil.copy2(pdf_file, target_pdf)
                print(f"   ✅ DONE! Figure copied to {target_pdf}")
            
            print(f"\n2. Update main.tex with these values:")
            if final_json.exists():
                with open(final_json, 'r') as f:
                    final = json.load(f)
                print(f"   - Final ACC: {final.get('acc', 0)*100:.2f}%")
                print(f"   - Final NMI: {final.get('nmi', 0)*100:.2f}%")
                print(f"   - Final ARI: {final.get('ari', 0)*100:.2f}%")
            
            if losses_csv.exists():
                losses_df = pd.read_csv(losses_csv)
                phase1 = losses_df[losses_df['phase'] == 'recon']
                phase2 = losses_df[losses_df['phase'] == 'dec']
                phase3 = losses_df[losses_df['phase'] == 'full']
                
                if len(phase1) > 0:
                    print(f"\n3. Phase 1 (Reconstruction) loss:")
                    print(f"   - Start: {phase1['total_loss'].iloc[0]:.4f}")
                    print(f"   - End: {phase1['total_loss'].iloc[-1]:.4f}")
                
                if len(phase2) > 0 and metrics_csv.exists():
                    metrics_df = pd.read_csv(metrics_csv)
                    phase2_metrics = metrics_df[metrics_df['phase'] == 'dec']
                    if len(phase2_metrics) > 0:
                        print(f"\n4. Phase 2 (DEC) ACC improvement:")
                        print(f"   - Start: {phase2_metrics['acc'].iloc[0]*100:.2f}%")
                        print(f"   - End: {phase2_metrics['acc'].iloc[-1]*100:.2f}%")
                
                if len(phase3) > 0:
                    print(f"\n5. Phase 3 (Full) - convergence at epoch:")
                    # Find where ACC plateaus (within 1% of final)
                    if metrics_csv.exists():
                        metrics_df = pd.read_csv(metrics_csv)
                        phase3_metrics = metrics_df[metrics_df['phase'] == 'full']
                        if len(phase3_metrics) > 0:
                            final_acc = phase3_metrics['acc'].iloc[-1]
                            plateau_mask = phase3_metrics['acc'] >= (final_acc - 0.01)
                            if plateau_mask.any():
                                plateau_epoch = phase3_metrics[plateau_mask]['epoch'].iloc[0]
                                print(f"   - Converged at epoch {plateau_epoch}")
                                print(f"   - ACC: {final_acc*100:.2f}%")
            
            print(f"\n{'='*70}")
            print(f"✅ Paper-ready! Use figure: {target_pdf}")
            print(f"{'='*70}\n")
            
            return True
        
        print()  # Blank line between experiments
    
    return False


if __name__ == '__main__':
    success = check_results()
    
    if not success:
        print("\n⚠️  No complete experiment found yet.")
        print("   Experiment may still be running...")
        print("   Run this script again when training completes.")
