"""
Utility functions for OT-CFM
Includes logging, checkpointing, and helper functions
"""

import os
import sys
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib


# ============================================================
# Logging Utilities
# ============================================================

def setup_logger(
    name: str,
    log_dir: str = 'logs',
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to add console handler
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    if console:
        console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    if console:
        logger.addHandler(console_handler)
    
    return logger


class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.experiment_dir / 'experiment.log'
        self.metrics_file = self.experiment_dir / 'metrics.json'
        self.metrics = {}
    
    def log(self, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")
    
    def log_metrics(self, epoch: int, metrics: Dict):
        """Log metrics for an epoch"""
        self.metrics[epoch] = metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_config(self, config: Dict):
        """Log configuration"""
        config_file = self.experiment_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)


# ============================================================
# Checkpointing Utilities
# ============================================================

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float('-inf')
        self.checkpoints = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        metric_name: str = 'acc',
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Metrics dictionary
            metric_name: Metric to use for best model selection
            filename: Optional filename (default: checkpoint_epoch_N.pth)
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        current_metric = metrics.get(metric_name, 0)
        
        if self.save_best_only and current_metric <= self.best_metric:
            return None
        
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': max(self.best_metric, current_metric)
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Update best metric
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            # Save as best
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Manage checkpoint history
        self.checkpoints.append(str(checkpoint_path))
        if len(self.checkpoints) > self.max_checkpoints:
            old_ckpt = self.checkpoints.pop(0)
            if Path(old_ckpt).exists() and 'best' not in old_ckpt:
                Path(old_ckpt).unlink()
        
        return str(checkpoint_path)
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = 'best_model.pth'
    ) -> Dict:
        """
        Load checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            filename: Checkpoint filename
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


# ============================================================
# Reproducibility Utilities
# ============================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """Get device for training"""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


# ============================================================
# Data Utilities
# ============================================================

def normalize_features(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize features
    
    Args:
        X: Feature array [N, D]
        method: Normalization method ('standard', 'minmax', 'l2')
    
    Returns:
        Normalized features
    """
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        return (X - mean) / std
    elif method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min + 1e-8)
    elif method == 'l2':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split_data(
    views: List[np.ndarray],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split multi-view data into train and test sets
    
    Args:
        views: List of view arrays
        labels: Label array
        train_ratio: Ratio of training data
        seed: Random seed
    
    Returns:
        Tuple of (train_views, train_labels, test_views, test_labels)
    """
    np.random.seed(seed)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_views = [v[train_idx] for v in views]
    test_views = [v[test_idx] for v in views]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    return train_views, train_labels, test_views, test_labels


def create_synthetic_data(
    n_samples: int = 1000,
    n_views: int = 3,
    n_clusters: int = 10,
    view_dims: List[int] = None,
    cluster_std: float = 1.0,
    seed: int = 42
) -> Dict:
    """
    Create synthetic multi-view clustering data
    
    Args:
        n_samples: Number of samples
        n_views: Number of views
        n_clusters: Number of clusters
        view_dims: Dimensions of each view
        cluster_std: Standard deviation within clusters
        seed: Random seed
    
    Returns:
        Dictionary with views, labels
    """
    np.random.seed(seed)
    
    if view_dims is None:
        view_dims = [100] * n_views
    
    # Generate labels
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Generate view-specific cluster centers
    views = []
    for v_dim in view_dims:
        centers = np.random.randn(n_clusters, v_dim) * 5
        view = np.zeros((n_samples, v_dim))
        
        for i, label in enumerate(labels):
            view[i] = centers[label] + np.random.randn(v_dim) * cluster_std
        
        views.append(view)
    
    return {
        'views': views,
        'labels': labels,
        'view_dims': view_dims,
        'n_clusters': n_clusters
    }


# ============================================================
# Model Utilities
# ============================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def compute_model_hash(model: torch.nn.Module) -> str:
    """Compute hash of model parameters"""
    state_dict = model.state_dict()
    params_str = str([(k, v.tolist()) for k, v in sorted(state_dict.items())])
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


# ============================================================
# IO Utilities
# ============================================================

def save_results(
    results: Dict,
    filepath: str,
    format: str = 'json'
):
    """Save results to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'npy':
        np.save(filepath, results)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(filepath: str, format: str = 'json') -> Dict:
    """Load results from file"""
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'npy':
        return np.load(filepath, allow_pickle=True).item()
    else:
        raise ValueError(f"Unknown format: {format}")


def generate_experiment_id() -> str:
    """Generate unique experiment ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"{timestamp}_{random_suffix}"


# ============================================================
# Timer Utilities
# ============================================================

class Timer:
    """Simple timer for profiling"""
    
    def __init__(self):
        self.times = {}
        self.starts = {}
    
    def start(self, name: str):
        """Start timer"""
        self.starts[name] = datetime.now()
    
    def stop(self, name: str) -> float:
        """Stop timer and return elapsed time"""
        if name not in self.starts:
            raise ValueError(f"Timer '{name}' not started")
        
        elapsed = (datetime.now() - self.starts[name]).total_seconds()
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        
        del self.starts[name]
        return elapsed
    
    def get_average(self, name: str) -> float:
        """Get average time for a named timer"""
        if name not in self.times:
            return 0.0
        return np.mean(self.times[name])
    
    def summary(self) -> Dict:
        """Get summary of all timers"""
        return {
            name: {
                'mean': np.mean(times),
                'std': np.std(times),
                'total': np.sum(times),
                'count': len(times)
            }
            for name, times in self.times.items()
        }


# ============================================================
# Progress Bar Utilities
# ============================================================

class ProgressBar:
    """Simple progress bar"""
    
    def __init__(self, total: int, prefix: str = '', width: int = 50):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
    
    def update(self, amount: int = 1):
        """Update progress"""
        self.current += amount
        self._display()
    
    def _display(self):
        """Display progress bar"""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{self.prefix} |{bar}| {percent:.1%}', end='', flush=True)
        
        if self.current >= self.total:
            print()
    
    def reset(self):
        """Reset progress bar"""
        self.current = 0


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("Seed set successfully")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test synthetic data generation
    data = create_synthetic_data(n_samples=100, n_views=3, n_clusters=5)
    print(f"Created synthetic data with {len(data['views'])} views")
    
    # Test timer
    timer = Timer()
    timer.start('test')
    import time
    time.sleep(0.1)
    elapsed = timer.stop('test')
    print(f"Timer test: {elapsed:.3f}s")
    
    # Test experiment logger
    logger = ExperimentLogger('test_experiment')
    logger.log("Test message")
    logger.log_metrics(0, {'acc': 0.5, 'nmi': 0.3})
    print("Logger test passed")
    
    # Test checkpoint manager
    import torch.nn as nn
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_manager = CheckpointManager('test_checkpoints', max_checkpoints=2)
    ckpt_manager.save(model, optimizer, 0, {'acc': 0.5})
    ckpt_manager.save(model, optimizer, 1, {'acc': 0.6})
    print("Checkpoint manager test passed")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_experiment', ignore_errors=True)
    shutil.rmtree('test_checkpoints', ignore_errors=True)
    
    print("\nAll utility tests passed!")
