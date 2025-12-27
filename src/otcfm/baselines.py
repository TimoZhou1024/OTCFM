"""
Baseline methods for comparison with OT-CFM
Includes both traditional and deep learning multi-view clustering methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class BaseClusteringMethod(ABC):
    """Base class for all clustering methods"""
    
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.labels_ = None
    
    @abstractmethod
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        """Fit and predict cluster assignments"""
        pass
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings if available"""
        return None


# ============================================================
# Traditional Multi-View Clustering Methods
# ============================================================

class ConcatKMeans(BaseClusteringMethod):
    """Simple baseline: concatenate all views and apply K-Means"""

    def __init__(self, num_clusters: int, n_init: int = 10):
        super().__init__(num_clusters)
        self.n_init = n_init
        self.embeddings_ = None

    def fit_predict(self, views: List[np.ndarray], mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # Normalize each view
        normalized = []
        for v in views:
            scaler = StandardScaler()
            normalized.append(scaler.fit_transform(v))

        # Concatenate
        X = np.concatenate(normalized, axis=1)

        # Apply K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=self.n_init)
        self.labels_ = kmeans.fit_predict(X)
        self.embeddings_ = X

        return self.labels_
    
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings_


class MultiViewSpectral(BaseClusteringMethod):
    """Multi-view spectral clustering with affinity fusion"""

    def __init__(self, num_clusters: int, n_neighbors: int = 10):
        super().__init__(num_clusters)
        self.n_neighbors = n_neighbors
        self.embeddings_ = None

    def _compute_affinity(self, X: np.ndarray) -> np.ndarray:
        """Compute RBF affinity matrix"""
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = 1.0 / X.shape[1]
        return rbf_kernel(X, gamma=gamma)

    def fit_predict(self, views: List[np.ndarray], mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # Compute affinity matrix for each view
        affinities = [self._compute_affinity(v) for v in views]

        # Fuse affinities (simple average)
        fused_affinity = np.mean(affinities, axis=0)

        # Apply spectral clustering
        spectral = SpectralClustering(
            n_clusters=self.num_clusters,
            affinity='precomputed',
            n_init=10
        )
        self.labels_ = spectral.fit_predict(fused_affinity)

        # Store fused affinity as embedding
        self.embeddings_ = fused_affinity

        return self.labels_


class CanonicalCorrelationAnalysis(BaseClusteringMethod):
    """CCA-based multi-view clustering"""

    def __init__(self, num_clusters: int, n_components: int = 50):
        super().__init__(num_clusters)
        self.n_components = n_components
        self.embeddings_ = None

    def fit_predict(self, views: List[np.ndarray], mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if len(views) != 2:
            # Fall back to PCA concatenation for more than 2 views
            return self._pca_fallback(views)

        from sklearn.cross_decomposition import CCA

        n_components = min(self.n_components, views[0].shape[1], views[1].shape[1])
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(views[0], views[1])

        # Average CCA projections
        embeddings = (X_c + Y_c) / 2

        # K-Means on CCA embeddings
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(embeddings)
        self.embeddings_ = embeddings

        return self.labels_
    
    def _pca_fallback(self, views: List[np.ndarray]) -> np.ndarray:
        # Apply PCA to each view and concatenate
        projections = []
        for v in views:
            pca = PCA(n_components=min(self.n_components, v.shape[1]))
            projections.append(pca.fit_transform(v))
        
        embeddings = np.concatenate(projections, axis=1)
        
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(embeddings)
        self.embeddings_ = embeddings
        
        return self.labels_


class WeightedViewClustering(BaseClusteringMethod):
    """Weighted multi-view clustering with learned view weights"""

    def __init__(self, num_clusters: int, n_init: int = 10):
        super().__init__(num_clusters)
        self.n_init = n_init
        self.weights_ = None
        self.embeddings_ = None

    def fit_predict(self, views: List[np.ndarray], mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        n_views = len(views)
        
        # Initialize weights uniformly
        weights = np.ones(n_views) / n_views
        
        # Iterative weight learning
        for _ in range(10):
            # Compute weighted combination
            normalized = []
            for v, w in zip(views, weights):
                scaler = StandardScaler()
                normalized.append(scaler.fit_transform(v) * np.sqrt(w))
            
            X = np.concatenate(normalized, axis=1)
            
            # Cluster
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=self.n_init)
            labels = kmeans.fit_predict(X)
            
            # Update weights based on clustering quality (silhouette)
            from sklearn.metrics import silhouette_samples
            new_weights = []
            for i, v in enumerate(views):
                scaler = StandardScaler()
                v_norm = scaler.fit_transform(v)
                sil = silhouette_samples(v_norm, labels).mean()
                new_weights.append(max(sil + 1, 0.1))  # Shift to positive
            
            weights = np.array(new_weights)
            weights = weights / weights.sum()
        
        self.labels_ = labels
        self.weights_ = weights
        self.embeddings_ = X
        
        return self.labels_


# ============================================================
# Deep Learning Multi-View Clustering Methods
# ============================================================

class DeepMultiViewClustering(nn.Module, BaseClusteringMethod):
    """Deep multi-view clustering with autoencoder"""
    
    def __init__(
        self,
        view_dims: List[int],
        latent_dim: int = 128,
        num_clusters: int = 10,
        hidden_dims: List[int] = None
    ):
        nn.Module.__init__(self)
        BaseClusteringMethod.__init__(self, num_clusters)
        
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Encoders for each view
        self.encoders = nn.ModuleList()
        for dim in view_dims:
            layers = []
            in_dim = dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, latent_dim))
            self.encoders.append(nn.Sequential(*layers))
        
        # Decoders for each view
        self.decoders = nn.ModuleList()
        for dim in view_dims:
            layers = []
            in_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, dim))
            self.decoders.append(nn.Sequential(*layers))
        
        # Clustering layer
        self.cluster_layer = nn.Parameter(torch.randn(num_clusters, latent_dim))
    
    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(v) for enc, v in zip(self.encoders, views)]
    
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        return [dec(z) for dec, z in zip(self.decoders, latents)]
    
    def fuse(self, latents: List[torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(latents), dim=0)
    
    def get_cluster_prob(self, z: torch.Tensor) -> torch.Tensor:
        # Student's t-distribution
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2))
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    def forward(self, views: List[torch.Tensor]) -> Dict:
        latents = self.encode(views)
        fusion = self.fuse(latents)
        recons = self.decode(latents)
        q = self.get_cluster_prob(fusion)
        
        return {
            'latents': latents,
            'fusion': fusion,
            'reconstructions': recons,
            'q': q
        }
    
    def fit_predict(
        self,
        views: List[np.ndarray],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = 'cuda',
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Train and predict"""
        self.to(device)
        
        # Convert to tensors
        views_t = [torch.FloatTensor(v).to(device) for v in views]
        n_samples = views[0].shape[0]
        
        # Create simple dataloader
        indices = np.arange(n_samples)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Pretrain autoencoders
        print("Pretraining autoencoders...")
        for epoch in range(epochs // 2):
            np.random.shuffle(indices)
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_views = [v[batch_idx] for v in views_t]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_views)
                
                # Reconstruction loss
                recon_loss = sum(
                    F.mse_loss(r, v)
                    for r, v in zip(outputs['reconstructions'], batch_views)
                )
                
                recon_loss.backward()
                optimizer.step()
                total_loss += recon_loss.item()
        
        # Initialize cluster centers
        with torch.no_grad():
            outputs = self.forward(views_t)
            fusion = outputs['fusion'].cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
            kmeans.fit(fusion)
            self.cluster_layer.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        
        # Joint training
        print("Joint training...")
        for epoch in range(epochs // 2):
            np.random.shuffle(indices)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_views = [v[batch_idx] for v in views_t]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_views)
                
                # Reconstruction loss
                recon_loss = sum(
                    F.mse_loss(r, v)
                    for r, v in zip(outputs['reconstructions'], batch_views)
                )
                
                # Clustering loss (KL divergence)
                q = outputs['q']
                p = q ** 2 / q.sum(dim=0)
                p = p / p.sum(dim=1, keepdim=True)
                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                
                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()
        
        # Get final predictions
        with torch.no_grad():
            outputs = self.forward(views_t)
            self.labels_ = outputs['q'].argmax(dim=1).cpu().numpy()
            self.embeddings_ = outputs['fusion'].cpu().numpy()
        
        return self.labels_
    
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings_


class ContrastiveMultiViewClustering(nn.Module, BaseClusteringMethod):
    """Contrastive multi-view clustering (simplified COMPLETER-like)"""
    
    def __init__(
        self,
        view_dims: List[int],
        latent_dim: int = 128,
        num_clusters: int = 10,
        hidden_dim: int = 256,
        temperature: float = 0.5
    ):
        nn.Module.__init__(self)
        BaseClusteringMethod.__init__(self, num_clusters)
        
        self.temperature = temperature
        self.latent_dim = latent_dim
        
        # Encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for dim in view_dims
        ])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def encode(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enc(v) for enc, v in zip(self.encoders, views)]
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss"""
        z1 = F.normalize(self.projector(z1), dim=1)
        z2 = F.normalize(self.projector(z2), dim=1)
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        
        # Positive pairs
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)
        
        # Loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(pos_mask * log_prob).sum() / (2 * batch_size)
        
        return loss
    
    def fit_predict(
        self,
        views: List[np.ndarray],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = 'cuda',
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        self.to(device)

        views_t = [torch.FloatTensor(v).to(device) for v in views]
        n_samples = views[0].shape[0]
        indices = np.arange(n_samples)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        print("Contrastive training...")
        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_views = [v[batch_idx] for v in views_t]
                
                optimizer.zero_grad()
                latents = self.encode(batch_views)
                
                # Pairwise contrastive loss
                loss = 0
                for j in range(len(latents)):
                    for k in range(j+1, len(latents)):
                        loss += self.contrastive_loss(latents[j], latents[k])
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Get embeddings and cluster
        with torch.no_grad():
            latents = self.encode(views_t)
            fusion = torch.mean(torch.stack(latents), dim=0).cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(fusion)
        self.embeddings_ = fusion
        
        return self.labels_
    
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings_


class IncompleteMultiViewClustering(BaseClusteringMethod):
    """
    Incomplete multi-view clustering baseline
    Handles missing views by imputing with view-specific means
    """
    
    def __init__(self, num_clusters: int):
        super().__init__(num_clusters)
        self.embeddings_ = None
    
    def fit_predict(
        self,
        views: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if mask is None:
            # Full views available
            mask = np.ones((views[0].shape[0], len(views)))
        
        # Impute missing views with mean
        imputed_views = []
        for v_idx, view in enumerate(views):
            view_mask = mask[:, v_idx].astype(bool)
            
            if view_mask.all():
                imputed_views.append(view)
            else:
                imputed = view.copy()
                view_mean = view[view_mask].mean(axis=0)
                imputed[~view_mask] = view_mean
                imputed_views.append(imputed)
        
        # Normalize and concatenate
        normalized = []
        for v in imputed_views:
            scaler = StandardScaler()
            normalized.append(scaler.fit_transform(v))
        
        X = np.concatenate(normalized, axis=1)
        
        # K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(X)
        self.embeddings_ = X
        
        return self.labels_


class UnalignedMultiViewClustering(BaseClusteringMethod):
    """
    Baseline for unaligned multi-view clustering
    Uses optimal transport to align views before clustering
    """

    def __init__(self, num_clusters: int, n_components: int = 50):
        super().__init__(num_clusters)
        self.n_components = n_components
        self.embeddings_ = None

    def fit_predict(
        self,
        views: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        try:
            import ot
        except ImportError:
            print("POT library not available, falling back to concatenation")
            return ConcatKMeans(self.num_clusters).fit_predict(views)
        
        # Reduce dimensionality
        reduced = []
        for v in views:
            pca = PCA(n_components=min(self.n_components, v.shape[1]))
            reduced.append(pca.fit_transform(v))
        
        # Use first view as reference
        reference = reduced[0]
        aligned = [reference]
        
        # Align other views using OT
        for v in reduced[1:]:
            n = reference.shape[0]
            
            # Compute cost matrix
            from scipy.spatial.distance import cdist
            C = cdist(reference, v, metric='sqeuclidean')
            
            # Solve OT
            a = np.ones(n) / n
            b = np.ones(n) / n
            T = ot.emd(a, b, C)
            
            # Apply transport
            aligned_v = n * T @ v
            aligned.append(aligned_v)
        
        # Concatenate aligned views
        X = np.concatenate(aligned, axis=1)
        
        # K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(X)
        self.embeddings_ = X
        
        return self.labels_


# ============================================================
# Baseline Manager
# ============================================================

def get_baseline_methods(
    view_dims: List[int],
    num_clusters: int,
    device: str = 'cuda',
    include_external: bool = True,
    include_internal: bool = True
) -> Dict[str, BaseClusteringMethod]:
    """
    Get dictionary of all baseline methods

    Args:
        view_dims: Dimensions of each view
        num_clusters: Number of clusters
        device: Device for deep learning methods
        include_external: Whether to include external methods from GitHub
        include_internal: Whether to include internal baseline methods

    Returns:
        Dictionary of method name to method instance
    """
    baselines = {}

    # Add internal baseline methods
    if include_internal:
        baselines.update({
            # Traditional methods
            'Concat-KMeans': ConcatKMeans(num_clusters),
            'Multi-View Spectral': MultiViewSpectral(num_clusters),
            'CCA-Clustering': CanonicalCorrelationAnalysis(num_clusters),
            'Weighted-View': WeightedViewClustering(num_clusters),

            # Deep learning methods
            'DMVC': DeepMultiViewClustering(view_dims, num_clusters=num_clusters),
            'Contrastive-MVC': ContrastiveMultiViewClustering(view_dims, num_clusters=num_clusters),

            # Incomplete/Unaligned methods
            'Incomplete-MVC': IncompleteMultiViewClustering(num_clusters),
            'Unaligned-MVC': UnalignedMultiViewClustering(num_clusters),
        })
    
    # Add external methods from GitHub repositories
    if include_external:
        try:
            from .external_baselines import get_external_baselines, list_missing_external_methods
            
            external = get_external_baselines(view_dims, num_clusters, device)
            if external:
                baselines.update(external)
                print(f"Loaded {len(external)} external methods")
            else:
                # Show hints for missing methods
                missing = list_missing_external_methods()
                if missing:
                    print(f"No external methods found. To add SOTA baselines, run:")
                    print(f"  python -m src.otcfm.external_baselines --missing")
        except ImportError as e:
            print(f"External baselines module not available: {e}")
    
    return baselines


def run_baseline_comparison(
    views: List[np.ndarray],
    labels: np.ndarray,
    num_clusters: int,
    device: str = 'cuda',
    mask: Optional[np.ndarray] = None,
    include_external: bool = True,
    include_internal: bool = True
) -> Dict:
    """
    Run all baseline methods and compare results

    Args:
        views: List of view arrays
        labels: Ground truth labels
        num_clusters: Number of clusters
        device: Device for deep methods
        mask: Optional missing view mask [N x V], where 1 = available, 0 = missing
        include_external: Whether to include external methods
        include_internal: Whether to include internal methods

    Returns:
        Dictionary of method results
    """
    from .metrics import evaluate_clustering

    view_dims = [v.shape[1] for v in views]
    baselines = get_baseline_methods(view_dims, num_clusters, device,
                                     include_external=include_external,
                                     include_internal=include_internal)

    results = {}

    for name, method in baselines.items():
        print(f"Running {name}...")
        try:
            # Pass mask to all methods that support it
            if isinstance(method, nn.Module):
                # Deep learning methods - pass mask in kwargs
                predictions = method.fit_predict(views, device=device, mask=mask)
            else:
                # Traditional methods - pass mask in kwargs
                predictions = method.fit_predict(views, mask=mask)

            embeddings = method.get_embeddings()
            metrics = evaluate_clustering(labels, predictions, embeddings)
            results[name] = metrics
            print(f"  ACC: {metrics['acc']:.4f}, NMI: {metrics['nmi']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'error': str(e)}

    return results


if __name__ == "__main__":
    # Test baseline methods
    print("Testing baseline methods...")
    
    np.random.seed(42)
    n_samples = 500
    num_clusters = 5
    view_dims = [100, 150, 80]
    
    # Create dummy data
    views = [np.random.randn(n_samples, dim) for dim in view_dims]
    labels = np.random.randint(0, num_clusters, n_samples)
    
    # Test each baseline
    results = run_baseline_comparison(views, labels, num_clusters, device='cpu')
    
    print("\n" + "="*60)
    print("Baseline Comparison Results:")
    print("="*60)
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name:25s} | ACC: {metrics['acc']:.4f} | NMI: {metrics['nmi']:.4f}")
    print("="*60)
