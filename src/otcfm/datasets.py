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
from PIL import Image
import glob


# Dataset download URLs
DATASET_URLS = {
    # Handwritten digits (UCI) - Multiple feature dataset
    'handwritten': {
        'url': 'https://archive.ics.uci.edu/static/public/72/multiple+features.zip',
        'type': 'zip',
        'description': 'UCI Multiple Features / Handwritten Digits (6 views, 2000 samples, 10 classes)',
    },
    
    # COIL-20 (Columbia Object Image Library)
    'coil20': {
        'url': 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip',
        'type': 'zip',
        'description': 'COIL-20 Object Images (multi-view capable, 1440 samples, 20 classes)',
    },
}


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
    """Load Scene15 multi-view dataset
    
    Supports multiple file formats:
    - Scene-15.mat with X1, X2, X3 and Y keys
    - scene-15.mat with X (cell array) and gt keys
    """
    # Try multiple file names
    possible_names = ["Scene-15.mat", "scene-15.mat", "Scene15.mat", "scene15.mat"]
    path = None
    for name in possible_names:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            path = candidate
            break
    
    if path is None:
        raise FileNotFoundError(f"Please download Scene-15.mat to {data_dir}")
    
    data = sio.loadmat(path)
    views = []
    
    # Format 1: X1, X2, X3 keys (or x1, x2, x3)
    if 'X1' in data or 'x1' in data:
        for i in range(3):
            view_key = f'X{i+1}' if f'X{i+1}' in data else f'x{i+1}'
            if view_key in data:
                views.append(np.array(data[view_key], dtype=np.float32))
        labels = np.array(data['Y']).flatten() - 1
    
    # Format 2: X cell array with gt labels (common format)
    elif 'X' in data:
        X = data['X']
        n_views = X.shape[1]
        for i in range(n_views):
            view_data = X[0, i]
            # Transpose if features x samples
            if view_data.shape[0] < view_data.shape[1]:
                view_data = view_data.T
            views.append(np.array(view_data, dtype=np.float32))
        
        # Labels: try 'gt', 'Y', 'labels'
        if 'gt' in data:
            labels = np.array(data['gt']).flatten()
        elif 'Y' in data:
            labels = np.array(data['Y']).flatten()
        elif 'labels' in data:
            labels = np.array(data['labels']).flatten()
        else:
            raise KeyError(f"Cannot find labels in {path}. Keys: {list(data.keys())}")
        
        # Adjust to 0-indexed if needed
        if labels.min() == 1:
            labels = labels - 1
    else:
        raise KeyError(f"Cannot find views in {path}. Keys: {list(data.keys())}")
    
    return views, labels


def load_noisy_mnist(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Noisy MNIST dataset (2 views: original + noisy)"""
    path = os.path.join(data_dir, "NoisyMNIST.mat")
    
    # Try to load from .mat file first
    if os.path.exists(path):
        try:
            data = sio.loadmat(path)
            views = [data['X1'].astype(np.float32), data['X2'].astype(np.float32)]
            labels = data['Y'].flatten() - 1
            return views, labels
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}. Generating synthetic data instead.")
    
    # Generate synthetic noisy MNIST
    print("Generating synthetic Noisy MNIST dataset...")
    try:
        from torchvision import datasets
        # Create temporary directory for download
        temp_dir = os.path.join(data_dir, "mnist_temp")
        os.makedirs(temp_dir, exist_ok=True)
        mnist = datasets.MNIST(temp_dir, train=True, download=True)
        data = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        labels = mnist.targets.numpy()
    except Exception as e:
        print(f"Warning: Failed to download MNIST: {e}. Creating random data instead.")
        # Fallback: create random data
        np.random.seed(42)
        n_samples = 10000
        data = np.random.rand(n_samples, 784).astype(np.float32)
        labels = np.random.randint(0, 10, n_samples)
    
    # Subsample to reasonable size
    if len(data) > 10000:
        np.random.seed(42)
        indices = np.random.choice(len(data), 10000, replace=False)
        data = data[indices]
        labels = labels[indices]
    
    # Create noisy version
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, data.shape).astype(np.float32)
    noisy_data = np.clip(data + noise, 0, 1)
    
    views = [data, noisy_data]
    
    # Try to save for future use
    try:
        os.makedirs(data_dir, exist_ok=True)
        sio.savemat(path, {'X1': views[0], 'X2': views[1], 'Y': labels + 1})
        print(f"Saved synthetic Noisy MNIST to {path}")
    except Exception as e:
        print(f"Warning: Failed to save {path}: {e}")
    
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


def _extract_array(v) -> np.ndarray:
    """Extract numpy array from various MATLAB formats"""
    # Handle sparse matrices
    if hasattr(v, 'toarray'):
        v = v.toarray()
    # Handle nested object arrays (common in MATLAB cell arrays)
    if isinstance(v, np.ndarray) and v.dtype == object:
        if v.shape == (1, 1):
            return _extract_array(v[0, 0])
        elif v.ndim == 1 and len(v) == 1:
            return _extract_array(v[0])
        elif v.ndim == 2 and v.shape[0] == 1:
            return _extract_array(v[0])
        elif v.ndim == 2 and v.shape[1] == 1:
            return _extract_array(v[:, 0])
    return np.array(v, dtype=np.float32)


def load_nus_wide(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load NUS-WIDE dataset"""
    path = os.path.join(data_dir, "NUS-WIDE.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Please download NUS-WIDE.mat to {data_dir}")

    data = sio.loadmat(path)

    # Filter out MATLAB metadata keys
    keys = [k for k in data.keys() if not k.startswith('__')]

    views = []

    # Check for 'fea' key (cell array containing multiple views)
    if 'fea' in data:
        fea = data['fea']
        # Handle cell array format: fea is (1, n_views) or (n_views, 1) object array
        if isinstance(fea, np.ndarray) and fea.dtype == object:
            fea_flat = fea.flatten()
            for i in range(len(fea_flat)):
                v = fea_flat[i]
                if hasattr(v, 'toarray'):
                    v = v.toarray()
                v = np.array(v, dtype=np.float32)
                if v.ndim == 2:
                    views.append(v)

    # If 'fea' not found or didn't work, try numbered patterns
    if not views:
        import re
        for pattern in [r'X(\d+)', r'x(\d+)', r'fea(\d+)', r'view(\d+)']:
            matches = [(k, int(re.match(pattern, k).group(1))) for k in keys if re.match(pattern, k)]
            if matches:
                view_keys = [k for k, _ in sorted(matches, key=lambda x: x[1])]
                for key in view_keys:
                    try:
                        v = _extract_array(data[key])
                        if v.ndim == 2:
                            views.append(v)
                    except (ValueError, TypeError):
                        continue
                break

    # Try to find label key
    labels = None
    for label_key in ['Y', 'y', 'gt', 'gnd', 'label', 'labels', 'truth']:
        if label_key in data:
            lbl = data[label_key]
            if hasattr(lbl, 'toarray'):
                lbl = lbl.toarray()
            labels = np.array(lbl).flatten()
            # Adjust to 0-indexed if needed
            if labels.min() >= 1:
                labels = labels - 1
            break

    if labels is None:
        raise KeyError(f"Cannot find label key in {path}. Available keys: {keys}")

    if not views:
        raise KeyError(f"Cannot find view keys in {path}. Available keys: {keys}")

    return views, labels


def download_and_extract(url: str, data_dir: str, dataset_name: str) -> str:
    """Download and extract a zip file"""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")
    extract_dir = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(extract_dir):
        print(f"Downloading {dataset_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(zip_path)
            print(f"Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset_name}: {e}")
    
    return extract_dir


def load_handwritten(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load UCI Handwritten Digits (Multiple Features) dataset
    6 views: fou, fac, kar, pix, zer, mor
    2000 samples (200 per digit 0-9), 10 classes
    """
    dataset_dir = os.path.join(data_dir, "handwritten")
    
    # Check if already extracted or download
    if not os.path.exists(dataset_dir):
        extract_dir = download_and_extract(
            DATASET_URLS['handwritten']['url'],
            data_dir,
            'handwritten'
        )
        dataset_dir = extract_dir
    
    # Feature file names and their dimensions
    feature_files = {
        'mfeat-fou': 76,   # Fourier coefficients
        'mfeat-fac': 216,  # Profile correlations
        'mfeat-kar': 64,   # Karhunen-Loève coefficients
        'mfeat-pix': 240,  # Pixel averages
        'mfeat-zer': 47,   # Zernike moments
        'mfeat-mor': 6,    # Morphological features
    }
    
    views = []
    labels = None
    
    for fname, dim in feature_files.items():
        # Try different possible paths
        possible_paths = [
            os.path.join(dataset_dir, fname),
            os.path.join(dataset_dir, 'mfeat', fname),
            os.path.join(dataset_dir, 'multiple+features', fname),
        ]
        
        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break
        
        if file_path is None:
            # Search recursively
            found = glob.glob(os.path.join(dataset_dir, '**', fname), recursive=True)
            if found:
                file_path = found[0]
        
        if file_path is None:
            raise FileNotFoundError(f"Could not find {fname} in {dataset_dir}")
        
        # Load the feature file
        data = np.loadtxt(file_path, dtype=np.float32)
        views.append(data)
        
        # Generate labels (200 samples per digit, 0-9)
        if labels is None:
            n_samples = data.shape[0]
            samples_per_class = n_samples // 10
            labels = np.repeat(np.arange(10), samples_per_class)
    
    return views, labels


def load_coil20(data_dir: str, n_views: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load COIL-20 dataset as multi-view
    Creates multiple views by grouping different rotation angles
    
    Args:
        data_dir: Directory to store/load data
        n_views: Number of views to create (default 3)
                Each view contains 72/n_views images per object
    
    Returns:
        views: List of view matrices, each [n_objects * samples_per_view, img_size]
        labels: Object labels (0-19)
    """
    dataset_dir = os.path.join(data_dir, "coil20")
    
    # Check if already extracted or download
    if not os.path.exists(dataset_dir):
        extract_dir = download_and_extract(
            DATASET_URLS['coil20']['url'],
            data_dir,
            'coil20'
        )
        dataset_dir = extract_dir
    
    # Find the actual image directory
    img_dir = dataset_dir
    possible_dirs = [
        os.path.join(dataset_dir, 'coil-20-proc'),
        os.path.join(dataset_dir, 'coil20'),
        dataset_dir
    ]
    
    for d in possible_dirs:
        if os.path.exists(d) and glob.glob(os.path.join(d, '*.png')):
            img_dir = d
            break
    
    # COIL-20: 20 objects, 72 images per object (5 degree rotation steps)
    # Filename format: obj{N}__{index}.png where N is 1-20, index is 0-71
    n_objects = 20
    n_images_per_object = 72
    images_per_view = n_images_per_object // n_views
    
    # Load all images for all objects first
    all_images = {}  # {obj_id: [img0, img1, ..., img71]}
    
    for obj_id in range(1, n_objects + 1):
        all_images[obj_id] = []
        for img_idx in range(n_images_per_object):
            # Try different filename patterns
            patterns = [
                f"obj{obj_id}__{img_idx}.png",
                f"obj{obj_id:02d}__{img_idx}.png",
            ]
            
            img_path = None
            for pattern in patterns:
                p = os.path.join(img_dir, pattern)
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path is not None:
                img = Image.open(img_path).convert('L')  # Grayscale
                img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                all_images[obj_id].append(img_array)
    
    # Verify we have images
    if not all_images[1]:
        raise RuntimeError(f"No images found in {img_dir}. Please check the dataset.")
    
    # Create multi-view data
    # Each view gets a subset of rotation angles
    views = [[] for _ in range(n_views)]
    labels = []
    
    for obj_id in range(1, n_objects + 1):
        obj_images = all_images[obj_id]
        if len(obj_images) < n_images_per_object:
            print(f"Warning: Object {obj_id} has only {len(obj_images)} images")
            continue
        
        # Split images into views
        for view_idx in range(n_views):
            start_idx = view_idx * images_per_view
            end_idx = start_idx + images_per_view
            view_images = obj_images[start_idx:end_idx]
            views[view_idx].extend(view_images)
        
        # Add labels (one per sample per view)
        labels.extend([obj_id - 1] * images_per_view)
    
    # Convert to numpy arrays
    views = [np.array(v, dtype=np.float32) for v in views]
    labels = np.array(labels)
    
    # Verify shapes
    print(f"COIL-20 loaded: {len(views)} views, {views[0].shape[0]} samples, {views[0].shape[1]} features per view")
    
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
    'Handwritten': load_handwritten,
    'COIL20': load_coil20,
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
