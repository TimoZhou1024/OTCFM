"""List all convergence experiments and find the best one"""
import json
from pathlib import Path

results = []
for d in sorted(Path('convergence_results').glob('Handwritten_*')):
    f = d / 'final_metrics.json'
    if f.exists():
        m = json.load(open(f))
        results.append({
            'dir': d.name,
            'acc': m.get('acc', 0) * 100,
            'nmi': m.get('nmi', 0) * 100,
            'ari': m.get('ari', 0) * 100,
            'purity': m.get('purity', 0) * 100,
            'f1': m.get('f1', 0) * 100
        })

results.sort(key=lambda x: x['acc'], reverse=True)

print('\nAll Handwritten experiments (sorted by ACC):')
print(f"{'Directory':<35} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'Purity':>8} {'F1':>8}")
print('-' * 85)
for r in results:
    print(f"{r['dir']:<35} {r['acc']:>7.2f}% {r['nmi']:>7.2f}% {r['ari']:>7.2f}% {r['purity']:>7.2f}% {r['f1']:>7.2f}%")

if results:
    print(f"\n✅ Best result: {results[0]['dir']}")
    print(f"   ACC: {results[0]['acc']:.2f}%")
