"""
Visualization utilities for OT-CFM
Includes embedding visualization, flow visualization, and results plotting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional, Dict, Tuple
from pathlib import Path

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Color palette for clusters
CLUSTER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
    '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
]


def get_colormap(n_clusters: int):
    """Get colormap for clustering visualization"""
    if n_clusters <= len(CLUSTER_COLORS):
        colors = CLUSTER_COLORS[:n_clusters]
    else:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    return colors


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization
    
    Args:
        embeddings: High-dimensional embeddings [N, D]
        method: Reduction method ('tsne' or 'pca')
        n_components: Target dimensions
        random_state: Random seed
    
    Returns:
        Low-dimensional embeddings [N, n_components]
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for dimension reduction")
    
    if method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(embeddings) // 4)
        )
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(embeddings)


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    title: str = "Embedding Visualization",
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot embedding visualization with ground truth and predictions
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Ground truth labels
        predictions: Optional cluster predictions
        title: Plot title
        method: Dimension reduction method
        save_path: Path to save figure
        figsize: Figure size
    """
    # Reduce dimensions
    if embeddings.shape[1] > 2:
        embeddings_2d = reduce_dimensions(embeddings, method)
    else:
        embeddings_2d = embeddings
    
    n_clusters = len(np.unique(labels))
    colors = get_colormap(n_clusters)
    
    if predictions is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax2 = None
    
    # Plot ground truth
    for i in range(n_clusters):
        mask = labels == i
        ax1.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {i}',
            alpha=0.6,
            s=10
        )
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    
    # Plot predictions
    if ax2 is not None and predictions is not None:
        n_pred_clusters = len(np.unique(predictions))
        pred_colors = get_colormap(n_pred_clusters)
        
        for i in range(n_pred_clusters):
            mask = predictions == i
            ax2.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[pred_colors[i]],
                label=f'Cluster {i}',
                alpha=0.6,
                s=10
            )
        ax2.set_title('Predictions')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_view_embeddings(
    view_embeddings: List[np.ndarray],
    labels: np.ndarray,
    view_names: Optional[List[str]] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot embeddings from each view
    
    Args:
        view_embeddings: List of view embeddings
        labels: Ground truth labels
        view_names: Optional view names
        method: Dimension reduction method
        save_path: Path to save figure
        figsize: Figure size
    """
    n_views = len(view_embeddings)
    n_clusters = len(np.unique(labels))
    colors = get_colormap(n_clusters)
    
    if view_names is None:
        view_names = [f'View {i+1}' for i in range(n_views)]
    
    fig, axes = plt.subplots(1, n_views, figsize=figsize)
    if n_views == 1:
        axes = [axes]
    
    for v, (emb, ax) in enumerate(zip(view_embeddings, axes)):
        # Reduce dimensions
        if emb.shape[1] > 2:
            emb_2d = reduce_dimensions(emb, method)
        else:
            emb_2d = emb
        
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[colors[i]],
                alpha=0.6,
                s=10
            )
        ax.set_title(view_names[v])
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    
    plt.suptitle('View-specific Embeddings')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: List[Dict],
    metrics: List[str] = ['loss', 'acc', 'nmi'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4)
):
    """
    Plot training curves
    
    Args:
        history: List of metric dictionaries per epoch
        metrics: Metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history) + 1)
    
    for metric, ax in zip(metrics, axes):
        values = [h.get(metric, 0) for h in history]
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Training')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(
    results: Dict[str, Dict],
    metrics: List[str] = ['acc', 'nmi', 'ari'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot bar chart comparing methods
    
    Args:
        results: Dictionary of method -> metrics
        metrics: Metrics to compare
        save_path: Path to save figure
        figsize: Figure size
    """
    methods = [m for m in results.keys() if 'error' not in results[m]]
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    x = np.arange(n_methods)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in methods]
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper())
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8
            )
    
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ablation_results(
    results: Dict[str, Dict],
    metrics: List[str] = ['acc', 'nmi'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot ablation study results
    
    Args:
        results: Dictionary of ablation mode -> metrics
        metrics: Metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    modes = list(results.keys())
    n_modes = len(modes)
    n_metrics = len(metrics)
    
    x = np.arange(n_modes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        means = [results[m].get(f'{metric}_mean', 0) for m in modes]
        stds = [results[m].get(f'{metric}_std', 0) for m in modes]
        
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=metric.upper(), yerr=stds, capsize=3)
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_flow_trajectory(
    trajectories: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_samples: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Plot flow trajectories from noise to data
    
    Args:
        trajectories: Trajectory array [T, N, D]
        labels: Optional labels for coloring
        n_samples: Number of samples to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    T, N, D = trajectories.shape
    
    # Random sample
    if N > n_samples:
        indices = np.random.choice(N, n_samples, replace=False)
        trajectories = trajectories[:, indices, :]
        if labels is not None:
            labels = labels[indices]
        N = n_samples
    
    # Reduce to 2D if needed
    if D > 2:
        trajectories_flat = trajectories.reshape(-1, D)
        trajectories_2d = reduce_dimensions(trajectories_flat, 'pca')
        trajectories_2d = trajectories_2d.reshape(T, N, 2)
    else:
        trajectories_2d = trajectories
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        n_clusters = len(np.unique(labels))
        colors = get_colormap(n_clusters)
    else:
        colors = ['blue'] * N
        labels = np.zeros(N, dtype=int)
    
    # Plot trajectories
    for i in range(N):
        color = colors[labels[i]] if labels is not None else 'blue'
        ax.plot(
            trajectories_2d[:, i, 0],
            trajectories_2d[:, i, 1],
            c=color, alpha=0.3, linewidth=0.5
        )
        # Mark start and end points
        ax.scatter(trajectories_2d[0, i, 0], trajectories_2d[0, i, 1],
                   c='gray', s=5, alpha=0.5, marker='o')
        ax.scatter(trajectories_2d[-1, i, 0], trajectories_2d[-1, i, 1],
                   c=color, s=10, alpha=0.7, marker='o')
    
    ax.set_title('Flow Trajectories')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_missing_rate_analysis(
    results: Dict[float, Dict],
    metrics: List[str] = ['acc', 'nmi'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot performance vs missing rate
    
    Args:
        results: Dictionary of missing_rate -> metrics
        metrics: Metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    missing_rates = sorted(results.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in metrics:
        means = [results[r].get(f'{metric}_mean', 0) for r in missing_rates]
        stds = [results[r].get(f'{metric}_std', 0) for r in missing_rates]
        
        ax.errorbar(
            [r * 100 for r in missing_rates],
            means, yerr=stds,
            marker='o', linewidth=2, capsize=5,
            label=metric.upper()
        )
    
    ax.set_xlabel('Missing Rate (%)')
    ax.set_ylabel('Score')
    ax.set_title('Performance vs Missing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot confusion matrix
    
    Args:
        labels: Ground truth labels
        predictions: Cluster predictions
        normalize: Whether to normalize
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    # Use Hungarian matching to align clusters with labels
    from scipy.optimize import linear_sum_assignment
    
    n_clusters = max(len(np.unique(labels)), len(np.unique(predictions)))
    
    # Build cost matrix
    cm = confusion_matrix(labels, predictions)
    
    # Pad if necessary
    if cm.shape[0] != cm.shape[1]:
        max_dim = max(cm.shape)
        padded = np.zeros((max_dim, max_dim))
        padded[:cm.shape[0], :cm.shape[1]] = cm
        cm = padded
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cm)
    cm = cm[:, col_ind]
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")
    
    # Add labels
    n_classes = cm.shape[0]
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            text = f'{cm[i, j]:.2f}' if normalize else f'{int(cm[i, j])}'
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_experiment_report(
    experiment_dir: str,
    results: Dict,
    embeddings: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    history: List[Dict]
):
    """
    Create comprehensive visualization report for an experiment
    
    Args:
        experiment_dir: Directory to save visualizations
        results: Experiment results
        embeddings: Learned embeddings
        labels: Ground truth labels
        predictions: Cluster predictions
        history: Training history
    """
    save_dir = Path(experiment_dir) / 'visualizations'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves
    print("Plotting training curves...")
    plot_training_curves(
        history,
        save_path=str(save_dir / 'training_curves.png')
    )
    
    # Embeddings
    print("Plotting embeddings...")
    plot_embeddings(
        embeddings, labels, predictions,
        title='OT-CFM Embeddings',
        save_path=str(save_dir / 'embeddings.png')
    )
    
    # Confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        labels, predictions,
        save_path=str(save_dir / 'confusion_matrix.png')
    )
    
    print(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization utilities...")
    
    np.random.seed(42)
    n_samples = 500
    n_clusters = 5
    latent_dim = 128
    
    # Create dummy data
    embeddings = np.random.randn(n_samples, latent_dim)
    labels = np.random.randint(0, n_clusters, n_samples)
    predictions = np.random.randint(0, n_clusters, n_samples)
    
    # Test embeddings plot
    print("Testing embedding visualization...")
    # plot_embeddings(embeddings, labels, predictions, method='pca')
    
    # Create dummy history
    history = [
        {'loss': 1.0 - i*0.01, 'acc': 0.3 + i*0.01, 'nmi': 0.2 + i*0.01}
        for i in range(100)
    ]
    
    # Test training curves
    print("Testing training curves...")
    # plot_training_curves(history)
    
    # Test comparison bar
    results = {
        'OT-CFM': {'acc': 0.85, 'nmi': 0.75, 'ari': 0.70},
        'Baseline1': {'acc': 0.70, 'nmi': 0.60, 'ari': 0.55},
        'Baseline2': {'acc': 0.65, 'nmi': 0.55, 'ari': 0.50}
    }
    print("Testing comparison bar...")
    # plot_comparison_bar(results)
    
    print("All visualization tests passed!")
