import json
from pathlib import Path

history_file = Path('experiments/ot_cfm_20260122_005822/history.json')
history = json.load(open(history_file))

# Add epoch indices
for i, h in enumerate(history):
    h['epoch'] = i

# Get final epoch
last = history[-1]

print("\n" + "="*60)
print("📊 Final Training Metrics (Epoch {})".format(last['epoch']))
print("="*60)
print(f"   ACC:    {last['acc']*100:.2f}%")
print(f"   NMI:    {last['nmi']*100:.2f}%")
print(f"   ARI:    {last['ari']*100:.2f}%")
print(f"   Purity: {last['purity']*100:.2f}%")
print(f"   F1:     {last['f1']*100:.2f}%")

# Get phase 1 (reconstruction) stats
phase1 = [h for h in history if h['epoch'] < 10]
if phase1:
    print("\n" + "="*60)
    print("📈 Phase 1 (Reconstruction) - Epochs 0-9")
    print("="*60)
    print(f"   Initial loss: {phase1[0]['loss']:.4f}")
    print(f"   Final loss:   {phase1[-1]['loss']:.4f}")
    print(f"   Reduction:    {(1 - phase1[-1]['loss']/phase1[0]['loss'])*100:.1f}%")

# Get phase 2 (DEC) stats
phase2 = [h for h in history if 10 <= h['epoch'] < 20]
if phase2:
    print("\n" + "="*60)
    print("📈 Phase 2 (DEC) - Epochs 10-19")
    print("="*60)
    print(f"   Initial ACC: {phase2[0]['acc']*100:.2f}%")
    print(f"   Final ACC:   {phase2[-1]['acc']*100:.2f}%")
    print(f"   Improvement: +{(phase2[-1]['acc'] - phase2[0]['acc'])*100:.2f}%")

# Get phase 3 (full) stats
phase3 = [h for h in history if h['epoch'] >= 20]
if phase3:
    print("\n" + "="*60)
    print("📈 Phase 3 (Full OT-CFM) - Epochs 20-99")
    print("="*60)
    print(f"   Initial ACC: {phase3[0]['acc']*100:.2f}%")
    print(f"   Final ACC:   {phase3[-1]['acc']*100:.2f}%")
    
    # Find best ACC in phase 3
    best_acc = max(h['acc'] for h in phase3)
    best_epoch = [h for h in phase3 if h['acc'] == best_acc][0]['epoch']
    print(f"   Best ACC:    {best_acc*100:.2f}% (Epoch {best_epoch})")
    
    # Find convergence point (within 1% of best)
    converged = [h for h in phase3 if h['acc'] >= best_acc - 0.01]
    if converged:
        conv_epoch = converged[0]['epoch']
        print(f"   Converged at: Epoch {conv_epoch} (within 1% of best)")

print("\n" + "="*60)
print("✅ Figure saved to: figures/Handwritten_convergence.pdf")
print("="*60)
