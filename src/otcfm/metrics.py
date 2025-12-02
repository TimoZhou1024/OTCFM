"""
Clustering evaluation metrics:
- ACC (Accuracy with Hungarian matching)
- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)
- Purity
- F1-score
"""

import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional
import torch


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian matching algorithm.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
    
    Returns:
        Accuracy score
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    
    assert y_pred.size == y_true.size, "Size mismatch between predictions and labels"
    
    n_samples = y_pred.size
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    
    # Build cost matrix
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(n_samples):
        cost_matrix[y_pred[i], y_true[i]] += 1
    
    # Hungarian matching (maximize -> negate for minimize)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Compute accuracy
    correct = cost_matrix[row_ind, col_ind].sum()
    accuracy = correct / n_samples
    
    return accuracy


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute cluster purity.
    
    Purity = (1/N) * sum_k max_j |c_k ∩ t_j|
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
    
    Returns:
        Purity score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_samples = len(y_true)
    n_clusters = len(np.unique(y_pred))
    
    purity = 0.0
    for cluster_id in np.unique(y_pred):
        cluster_mask = (y_pred == cluster_id)
        cluster_labels = y_true[cluster_mask]
        
        if len(cluster_labels) > 0:
            # Count most frequent true label in this cluster
            unique, counts = np.unique(cluster_labels, return_counts=True)
            purity += counts.max()
    
    return purity / n_samples


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score after Hungarian matching.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
    
    Returns:
        Macro F1 score
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    
    n_samples = y_pred.size
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    
    # Build cost matrix for Hungarian matching
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(n_samples):
        cost_matrix[y_pred[i], y_true[i]] += 1
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Create mapping from predicted to true labels
    mapping = {pred: true for pred, true in zip(row_ind, col_ind)}
    
    # Remap predictions
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    
    # Compute F1
    return f1_score(y_true, y_pred_mapped, average='macro')


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive clustering evaluation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
        embeddings: Optional embeddings for internal metrics
    
    Returns:
        Dictionary of evaluation metrics (both lowercase and uppercase keys)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {}
    
    # External metrics (require ground truth) - with both case versions
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    f1 = f1_macro(y_true, y_pred)
    
    # Uppercase keys (for display/tables)
    metrics['ACC'] = acc
    metrics['NMI'] = nmi
    metrics['ARI'] = ari
    metrics['Purity'] = purity
    metrics['F1'] = f1
    
    # Lowercase keys (for code compatibility)
    metrics['acc'] = acc
    metrics['nmi'] = nmi
    metrics['ari'] = ari
    metrics['purity'] = purity
    metrics['f1'] = f1
    metrics['f1_macro'] = f1
    
    # Internal metrics (don't require ground truth, need embeddings)
    if embeddings is not None and len(np.unique(y_pred)) > 1:
        try:
            metrics['Silhouette'] = silhouette_score(embeddings, y_pred)
            metrics['Davies-Bouldin'] = davies_bouldin_score(embeddings, y_pred)
            metrics['Calinski-Harabasz'] = calinski_harabasz_score(embeddings, y_pred)
        except Exception as e:
            print(f"Warning: Could not compute internal metrics: {e}")
    
    return metrics


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string"""
    items = [f"{k}: {v:.{precision}f}" for k, v in metrics.items()]
    return " | ".join(items)


class MetricTracker:
    """Track metrics over training"""
    
    def __init__(self, metrics: list = None):
        self.metrics = metrics or ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
        self.history = {m: [] for m in self.metrics}
        self.best = {m: 0.0 for m in self.metrics}
        self.best_epoch = {m: 0 for m in self.metrics}
    
    def update(self, metrics_dict: Dict[str, float], epoch: int = 0):
        """Update tracker with new metrics"""
        for m in self.metrics:
            # Check for both uppercase and lowercase versions
            value = metrics_dict.get(m) or metrics_dict.get(m.lower())
            if value is not None:
                self.history[m].append(value)
                
                # Track best (higher is better for these metrics)
                if value > self.best[m]:
                    self.best[m] = value
                    self.best_epoch[m] = epoch
    
    def get_best(self) -> Dict[str, float]:
        """Get best metrics"""
        return self.best.copy()
    
    def get_summary(self) -> str:
        """Get summary string"""
        lines = []
        for m in self.metrics:
            lines.append(f"Best {m}: {self.best[m]:.4f} (epoch {self.best_epoch[m]})")
        return "\n".join(lines)


def compare_methods(
    results: Dict[str, Dict[str, float]],
    metrics: list = None
) -> str:
    """
    Create comparison table of multiple methods
    
    Args:
        results: Dict of {method_name: {metric: value}}
        metrics: List of metrics to compare
    
    Returns:
        Formatted comparison table
    """
    if metrics is None:
        metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
    
    # Header
    header = f"{'Method':<20}" + "".join([f"{m:>10}" for m in metrics])
    lines = [header, "-" * len(header)]
    
    # Results for each method
    for method, method_metrics in results.items():
        row = f"{method:<20}"
        for m in metrics:
            value = method_metrics.get(m, 0.0)
            row += f"{value:>10.4f}"
        lines.append(row)
    
    # Best values
    lines.append("-" * len(header))
    best_row = f"{'Best':<20}"
    for m in metrics:
        values = [r.get(m, 0.0) for r in results.values()]
        best_row += f"{max(values):>10.4f}"
    lines.append(best_row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    np.random.seed(42)
    n_samples = 1000
    n_clusters = 10
    
    # Generate synthetic data
    y_true = np.random.randint(0, n_clusters, n_samples)
    
    # Perfect prediction (should give 1.0 for all metrics)
    y_pred_perfect = y_true.copy()
    metrics = evaluate_clustering(y_true, y_pred_perfect)
    print("Perfect prediction:")
    print(format_metrics(metrics))
    
    # Random prediction
    y_pred_random = np.random.randint(0, n_clusters, n_samples)
    metrics = evaluate_clustering(y_true, y_pred_random)
    print("\nRandom prediction:")
    print(format_metrics(metrics))
    
    # Partially correct prediction
    y_pred_partial = y_true.copy()
    noise_idx = np.random.choice(n_samples, n_samples // 4, replace=False)
    y_pred_partial[noise_idx] = np.random.randint(0, n_clusters, len(noise_idx))
    metrics = evaluate_clustering(y_true, y_pred_partial)
    print("\n75% correct prediction:")
    print(format_metrics(metrics))
    
    # Test comparison
    print("\n\nMethod comparison:")
    results = {
        'OT-CFM (Ours)': {'ACC': 0.892, 'NMI': 0.834, 'ARI': 0.812, 'Purity': 0.905, 'F1': 0.878},
        'DCG': {'ACC': 0.845, 'NMI': 0.798, 'ARI': 0.776, 'Purity': 0.867, 'F1': 0.834},
        'IMVCDC': {'ACC': 0.823, 'NMI': 0.781, 'ARI': 0.754, 'Purity': 0.849, 'F1': 0.812},
    }
    print(compare_methods(results))
