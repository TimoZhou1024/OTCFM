"""
OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans

from .models import (
    MultiViewEncoderDecoder,
    VectorFieldNetwork,
    ClusteringModule,
    ODESolver
)
from .losses import OTCFMLoss


class OTCFM(nn.Module):
    """
    Optimal Transport Coupled Flow Matching for Multi-View Clustering
    
    This model combines:
    1. Multi-view encoder-decoder for latent space construction
    2. Flow matching for generative modeling and imputation
    3. Gromov-Wasserstein alignment for unaligned multi-view learning
    4. Clustering module for end-to-end clustering
    """
    
    def __init__(
        self,
        view_dims: List[int],
        latent_dim: int = 128,
        hidden_dims: List[int] = None,
        num_clusters: int = 10,
        flow_hidden_dim: int = 256,
        flow_num_layers: int = 4,
        time_dim: int = 64,
        ode_steps: int = 10,
        sigma_min: float = 1e-4,
        kernel_type: str = "rbf",
        kernel_gamma: float = 1.0,
        lambda_gw: float = 0.2,        # 增加 GW 对齐损失
        lambda_cluster: float = 1.0,   # 显著增加聚类损失权重
        lambda_recon: float = 0.5,     # 减少重建损失
        lambda_contrastive: float = 0.3,  # 增加对比学习
        dropout: float = 0.1,
        use_cross_view_flow: bool = True  # 使用跨视图流匹配
    ):
        super().__init__()
        
        self.num_views = len(view_dims)
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.ode_steps = ode_steps
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Multi-view encoder-decoder
        self.encoder_decoder = MultiViewEncoderDecoder(
            view_dims=view_dims,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Vector field network for flow matching
        self.vector_field = VectorFieldNetwork(
            latent_dim=latent_dim,
            hidden_dim=flow_hidden_dim,
            time_dim=time_dim,
            num_layers=flow_num_layers,
            condition_dim=latent_dim,  # Use available views as condition
            dropout=dropout
        )
        
        # ODE solver
        self.ode_solver = ODESolver(self.vector_field, num_steps=ode_steps)
        
        # Clustering module
        self.clustering = ClusteringModule(latent_dim, num_clusters)
        
        # Loss function with optimized weights
        self.loss_fn = OTCFMLoss(
            sigma_min=sigma_min,
            kernel_type=kernel_type,
            kernel_gamma=kernel_gamma,
            lambda_gw=lambda_gw,
            lambda_cluster=lambda_cluster,
            lambda_recon=lambda_recon,
            lambda_contrastive=lambda_contrastive,
            use_cross_view_flow=use_cross_view_flow
        )
        
        # Store hyperparameters
        self.sigma_min = sigma_min
    
    def encode(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encode views to latent space and compute consensus
        
        Args:
            views: List of view tensors [B, D_v]
            mask: View availability mask [B, V]
        
        Returns:
            latents: List of view latents
            consensus: Fused consensus representation
        """
        latents = self.encoder_decoder.encode(views, mask)
        consensus = self.encoder_decoder.fuse_latents(latents, mask)
        return latents, consensus
    
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode latents to views"""
        return self.encoder_decoder.decode(latents)
    
    def impute_missing(
        self,
        views: List[torch.Tensor],
        mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Impute missing views using conditional flow matching
        
        Args:
            views: List of view tensors (missing views can be zeros)
            mask: View availability mask [B, V]
        
        Returns:
            imputed_views: List with missing views imputed
        """
        batch_size = views[0].shape[0]
        device = views[0].device
        
        # Encode available views
        latents, consensus = self.encode(views, mask)
        
        # Use consensus as condition for imputation
        condition = consensus
        
        imputed_latents = []
        for v in range(self.num_views):
            view_mask = mask[:, v]  # [B]
            
            if view_mask.all():
                # All views available, no imputation needed
                imputed_latents.append(latents[v])
            else:
                # Some views missing, need imputation
                imputed_z = latents[v].clone()
                
                # For missing samples, generate from noise
                missing_idx = ~view_mask.bool()
                if missing_idx.any():
                    num_missing = missing_idx.sum().item()
                    z0 = torch.randn(num_missing, self.latent_dim, device=device)
                    cond = condition[missing_idx]
                    
                    # Solve ODE to generate missing latents
                    z1 = self.ode_solver.solve(z0, cond)
                    imputed_z[missing_idx] = z1
                
                imputed_latents.append(imputed_z)
        
        # Decode imputed latents
        imputed_views = self.decode(imputed_latents)
        
        return imputed_views
    
    def forward(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict:
        """
        Forward pass
        
        Args:
            views: List of view tensors
            mask: View availability mask
            return_all: Whether to return all intermediate results
        
        Returns:
            Dictionary with outputs
        """
        batch_size = views[0].shape[0]
        device = views[0].device
        
        if mask is None:
            mask = torch.ones(batch_size, self.num_views, device=device)
        
        # Encode views
        latents, consensus = self.encode(views, mask)
        
        # Compute cluster assignments
        q, p = self.clustering(consensus)
        
        outputs = {
            'latents': latents,
            'consensus': consensus,
            'q': q,
            'p': p,
            'assignments': q.argmax(dim=1)
        }
        
        if return_all:
            # Decode for reconstruction
            recons = self.decode(latents)
            outputs['reconstructions'] = recons
        
        return outputs
    
    def compute_loss(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        ablation_mode: str = "full"
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss
        
        Args:
            views: List of view tensors
            mask: View availability mask
            ablation_mode: Ablation setting
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        outputs = self.forward(views, mask, return_all=True)
        
        loss, loss_dict = self.loss_fn(
            vector_field=self.vector_field,
            latents=outputs['latents'],
            consensus=outputs['consensus'],
            q=outputs['q'],
            p=outputs['p'],
            x_original=views,
            x_recon=outputs['reconstructions'],
            mask=mask,
            ablation_mode=ablation_mode
        )
        
        return loss, loss_dict
    
    def get_cluster_assignments(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get hard cluster assignments
        
        Args:
            views: List of view tensors
            mask: View availability mask
        
        Returns:
            Cluster assignments as numpy array
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(views, mask)
            assignments = outputs['assignments'].cpu().numpy()
        return assignments
    
    def get_embeddings(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get consensus embeddings
        
        Args:
            views: List of view tensors
            mask: View availability mask
        
        Returns:
            Embeddings as numpy array
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(views, mask)
            embeddings = outputs['consensus'].cpu().numpy()
        return embeddings
    
    def init_clustering(self, dataloader, device: str = 'cuda'):
        """
        Initialize clustering centroids using K-Means
        
        Args:
            dataloader: DataLoader for the dataset
            device: Device to use
        """
        self.eval()
        all_embeddings = []
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                views = [v.to(device) for v in batch['views']]
                mask = batch['mask'].to(device)
                indices = batch['indices']
                
                outputs = self.forward(views, mask)
                all_embeddings.append(outputs['consensus'].cpu())
                all_indices.append(indices)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indices = torch.cat(all_indices, dim=0).numpy()
        
        # 按原始顺序排序
        order = np.argsort(all_indices)
        all_embeddings = all_embeddings[order]
        
        self.clustering.init_centroids(all_embeddings)
        self.clustering.centroids = self.clustering.centroids.to(device)


class OTCFMTrainer:
    """
    Trainer for OT-CFM model with alternating optimization
    """
    
    def __init__(
        self,
        model: OTCFM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        cluster_update_freq: int = 5,
        ablation_mode: str = "full"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cluster_update_freq = cluster_update_freq
        self.ablation_mode = ablation_mode
    
    def train_epoch(self, dataloader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        for batch in dataloader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, loss_dict = self.model.compute_loss(
                views, mask, self.ablation_mode
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {'loss': avg_loss, **loss_components}
    
    def update_clustering(self, dataloader):
        """Update clustering centroids (E-step of alternating optimization)"""
        self.model.eval()
        all_embeddings = []
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                views = [v.to(self.device) for v in batch['views']]
                mask = batch['mask'].to(self.device)
                indices = batch['indices']
                
                outputs = self.model(views, mask)
                all_embeddings.append(outputs['consensus'])
                all_indices.append(indices)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indices = torch.cat(all_indices, dim=0).numpy()
        
        # 按原始顺序排序
        order = np.argsort(all_indices)
        all_embeddings = all_embeddings[order]
        
        # Update centroids using current embeddings
        from sklearn.cluster import KMeans
        embeddings_np = all_embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=self.model.num_clusters, n_init=10)
        kmeans.fit(embeddings_np)
        
        self.model.clustering.centroids.data = torch.FloatTensor(
            kmeans.cluster_centers_
        ).to(self.device)
    
    @torch.no_grad()
    def evaluate(self, dataloader, labels: np.ndarray) -> Dict:
        """Evaluate clustering performance"""
        from .metrics import evaluate_clustering
        
        self.model.eval()
        all_embeddings = []
        all_predictions = []
        all_indices = []
        
        for batch in dataloader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            indices = batch['indices']
            
            outputs = self.model(views, mask)
            all_embeddings.append(outputs['consensus'].cpu())
            all_predictions.append(outputs['assignments'].cpu())
            all_indices.append(indices)
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        predictions = torch.cat(all_predictions, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()
        
        # 按原始顺序排序，确保与 labels 对应
        order = np.argsort(indices)
        embeddings = embeddings[order]
        predictions = predictions[order]
        
        metrics = evaluate_clustering(labels, predictions, embeddings)
        return metrics


if __name__ == "__main__":
    # Test OT-CFM model
    print("Testing OT-CFM model...")
    
    batch_size = 32
    view_dims = [100, 200, 150]
    latent_dim = 128
    num_clusters = 10
    
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=latent_dim,
        num_clusters=num_clusters
    )
    
    # Create dummy data
    views = [torch.randn(batch_size, dim) for dim in view_dims]
    mask = torch.ones(batch_size, len(view_dims))
    
    # Test forward pass
    outputs = model(views, mask, return_all=True)
    print(f"Consensus shape: {outputs['consensus'].shape}")
    print(f"Assignments shape: {outputs['assignments'].shape}")
    print(f"Q shape: {outputs['q'].shape}")
    
    # Test loss computation
    loss, loss_dict = model.compute_loss(views, mask)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test imputation
    mask_missing = mask.clone()
    mask_missing[:, 1] = 0  # Second view missing
    imputed = model.impute_missing(views, mask_missing)
    print(f"Imputed view shapes: {[v.shape for v in imputed]}")
    
    print("All tests passed!")
