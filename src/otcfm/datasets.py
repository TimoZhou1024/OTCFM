"""
Multi-View Clustering Datasets
Supports: Caltech101, Scene15, NoisyMNIST, BDGP, CUB, NUS-WIDE, Reuters
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import urllib.request
import zipfile


class MultiViewDataset(Dataset):
    """Base class for multi-view datasets"""
    
    def __init__(
        self,
        views: List[np.ndarray],
        labels: np.ndarray,
        missing_rate: float = 0.0,
        unaligned_rate: float = 0.0,
        seed: int = 42
    ):
        """
        Args:
            views: List of view matrices [N x D_v]
            labels: Ground truth labels [N]
            missing_rate: Fraction of missing views
            unaligned_rate: Fraction of unaligned samples
        """
        self.num_views = len(views)
        self.num_samples = views[0].shape[0]
        self.labels = labels
        self.num_classes = len(np.unique(labels))
        
        np.random.seed(seed)
        
        # Normalize views
        self.views = []
        self.view_dims = []
        for v in views:
            scaler = StandardScaler()
            v_normalized = scaler.fit_transform(v.astype(np.float32))
            self.views.append(torch.FloatTensor(v_normalized))
            self.view_dims.append(v.shape[1])
        
        # Create missing mask [N x V]
        self.missing_mask = torch.ones(self.num_samples, self.num_views)
        if missing_rate > 0:
            self._create_missing_mask(missing_rate, seed)
        
        # Create alignment permutation
        self.alignment_perm = [torch.arange(self.num_samples) for _ in range(self.num_views)]
        if unaligned_rate > 0:
            self._create_unalignment(unaligned_rate, seed)
    
    def _create_missing_mask(self, missing_rate: float, seed: int):
        """Create random missing pattern ensuring at least one view is available"""
        np.random.seed(seed)
        for i in range(self.num_samples):
            # Randomly select views to be missing
            num_missing = int(missing_rate * self.num_views)
            if num_missing >= self.num_views:
                num_missing = self.num_views - 1  # Keep at least one view
            
            missing_views = np.random.choice(self.num_views, num_missing, replace=False)
            self.missing_mask[i, missing_views] = 0
    
    def _create_unalignment(self, unaligned_rate: float, seed: int):
        """Create random unalignment by shuffling view indices"""
        np.random.seed(seed)
        num_unaligned = int(unaligned_rate * self.num_samples)
        unaligned_indices = np.random.choice(self.num_samples, num_unaligned, replace=False)
        
        for v in range(1, self.num_views):  # Keep first view aligned
            perm = torch.arange(self.num_samples)
            # Shuffle unaligned indices within the same class to maintain some structure
            shuffled = np.random.permutation(unaligned_indices)
            perm[unaligned_indices] = torch.LongTensor(shuffled)
            self.alignment_perm[v] = perm
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return views and metadata for a sample"""
        views = []
        for v in range(self.num_views):
            aligned_idx = self.alignment_perm[v][idx]
            views.append(self.views[v][aligned_idx])
        
        return {
            'views': views,
            'mask': self.missing_mask[idx],
            'label': self.labels[idx],
            'index': idx
        }


def load_caltech101(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Caltech101 multi-view dataset (7 views)"""
    path = os.path.join(data_dir, "Caltech101-7.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download Caltech101-7.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    for i in range(7):
        view_key = f'X{i+1}' if f'X{i+1}' in data else f'x{i+1}'
        if view_key in data:
            views.append(np.array(data[view_key], dtype=np.float32))
    
    labels = np.array(data['Y']).flatten() - 1  # 0-indexed
    return views, labels


def load_scene15(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Scene15 multi-view dataset"""
    path = os.path.join(data_dir, "Scene-15.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download Scene-15.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    for i in range(3):
        view_key = f'X{i+1}' if f'X{i+1}' in data else f'x{i+1}'
        if view_key in data:
            views.append(np.array(data[view_key], dtype=np.float32))
    
    labels = np.array(data['Y']).flatten() - 1
    return views, labels


def load_noisy_mnist(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Noisy MNIST dataset (2 views: original + noisy)"""
    path = os.path.join(data_dir, "NoisyMNIST.mat")
    if not os.path.exists(path):
        # Create synthetic noisy MNIST
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(data_dir, train=True, download=True)
        data = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        labels = mnist.targets.numpy()
        
        # Subsample
        np.random.seed(42)
        indices = np.random.choice(len(data), 10000, replace=False)
        data = data[indices]
        labels = labels[indices]
        
        # Create noisy version
        noise = np.random.normal(0, 0.3, data.shape).astype(np.float32)
        noisy_data = np.clip(data + noise, 0, 1)
        
        views = [data, noisy_data]
        return views, labels
    
    data = sio.loadmat(path)
    views = [data['X1'].astype(np.float32), data['X2'].astype(np.float32)]
    labels = data['Y'].flatten() - 1
    return views, labels


def load_bdgp(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load BDGP dataset"""
    path = os.path.join(data_dir, "BDGP.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download BDGP.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = [
        np.array(data['X1'], dtype=np.float32),
        np.array(data['X2'], dtype=np.float32)
    ]
    labels = np.array(data['Y']).flatten() - 1
    return views, labels


def load_cub(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load CUB (Caltech-UCSD Birds) dataset"""
    path = os.path.join(data_dir, "CUB.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download CUB.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    for key in ['X1', 'X2', 'X3']:
        if key in data:
            views.append(np.array(data[key], dtype=np.float32))
    labels = np.array(data['Y']).flatten() - 1
    return views, labels


def load_reuters(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Reuters multi-lingual dataset"""
    path = os.path.join(data_dir, "Reuters.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download Reuters.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    for i in range(5):  # 5 languages
        key = f'X{i+1}'
        if key in data:
            views.append(np.array(data[key], dtype=np.float32))
    labels = np.array(data['Y']).flatten() - 1
    return views, labels


def load_nus_wide(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load NUS-WIDE dataset"""
    path = os.path.join(data_dir, "NUS-WIDE.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download NUS-WIDE.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    for i in range(5):
        key = f'X{i+1}'
        if key in data:
            views.append(np.array(data[key], dtype=np.float32))
    labels = np.array(data['Y']).flatten() - 1
    return views, labels


def create_synthetic_multiview(
    n_samples: int = 1000,
    n_clusters: int = 5,
    n_views: int = 3,
    view_dims: List[int] = [100, 100, 100],
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Create synthetic multi-view data for testing"""
    np.random.seed(seed)
    
    # Generate cluster centers
    labels = np.random.randint(0, n_clusters, n_samples)
    
    views = []
    for v, dim in enumerate(view_dims):
        # Cluster centers for this view
        centers = np.random.randn(n_clusters, dim) * 3
        
        # Generate samples
        X = np.zeros((n_samples, dim), dtype=np.float32)
        for i in range(n_samples):
            X[i] = centers[labels[i]] + np.random.randn(dim) * noise_level
        
        views.append(X)
    
    return views, labels


DATASET_LOADERS = {
    'Caltech101': load_caltech101,
    'Scene15': load_scene15,
    'NoisyMNIST': load_noisy_mnist,
    'BDGP': load_bdgp,
    'CUB': load_cub,
    'Reuters': load_reuters,
    'NUS-WIDE': load_nus_wide,
    'Synthetic': lambda x: create_synthetic_multiview(),
}


def load_synthetic(
    n_samples: int = 1000,
    n_clusters: int = 10,
    n_views: int = 3,
    view_dims: List[int] = None,
    noise_level: float = 0.1,
    seed: int = 42
) -> Dict:
    """Load/create synthetic multi-view dataset"""
    if view_dims is None:
        view_dims = [100, 100, 100]
    
    views, labels = create_synthetic_multiview(
        n_samples=n_samples,
        n_clusters=n_clusters,
        n_views=n_views,
        view_dims=view_dims,
        noise_level=noise_level,
        seed=seed
    )
    
    return {
        'views': views,
        'labels': labels
    }


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    missing_rate: float = 0.0,
    unaligned_rate: float = 0.0,
    seed: int = 42
) -> MultiViewDataset:
    """Load dataset by name"""
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_LOADERS.keys())}")
    
    views, labels = DATASET_LOADERS[dataset_name](data_dir)
    
    dataset = MultiViewDataset(
        views=views,
        labels=labels,
        missing_rate=missing_rate,
        unaligned_rate=unaligned_rate,
        seed=seed
    )
    
    return dataset


def create_dataloader(
    dataset: MultiViewDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0  # Default to 0 for Windows compatibility
) -> DataLoader:
    """Create DataLoader for multi-view dataset (alias for get_dataloader)"""
    return get_dataloader(dataset, batch_size, shuffle, num_workers)


def multiview_collate_fn(batch):
    """Custom collate function for multi-view data"""
    num_views = len(batch[0]['views'])
    
    views = [torch.stack([b['views'][v] for b in batch]) for v in range(num_views)]
    masks = torch.stack([b['mask'] for b in batch])
    labels = torch.LongTensor([b['label'] for b in batch])
    indices = torch.LongTensor([b['index'] for b in batch])
    
    return {
        'views': views,
        'mask': masks,
        'labels': labels,
        'indices': indices
    }


def get_dataloader(
    dataset: MultiViewDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0  # Default to 0 for Windows compatibility
) -> DataLoader:
    """Create DataLoader for multi-view dataset"""
    
    # Disable pin_memory on CPU, use it only with CUDA
    pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multiview_collate_fn,
        pin_memory=pin_memory,
        drop_last=False
    )


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Test synthetic data
    views, labels = create_synthetic_multiview()
    print(f"Synthetic: {len(views)} views, {len(labels)} samples, {len(np.unique(labels))} classes")
    
    # Test dataset class
    dataset = MultiViewDataset(views, labels, missing_rate=0.3, unaligned_rate=0.2)
    print(f"Dataset size: {len(dataset)}")
    print(f"View dimensions: {dataset.view_dims}")
    print(f"Missing mask sum: {dataset.missing_mask.sum()}")
    
    # Test dataloader
    loader = get_dataloader(dataset, batch_size=32)
    batch = next(iter(loader))
    print(f"Batch views shapes: {[v.shape for v in batch['views']]}")
    print(f"Batch mask shape: {batch['mask'].shape}")
