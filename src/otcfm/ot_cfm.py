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
        lambda_contrastive: float = 0.3,  # 增加对比学习 (仅用于aligned数据)
        dropout: float = 0.1,
        use_cross_view_flow: bool = True,  # 使用跨视图流匹配
        is_aligned: bool = True  # 是否为对齐数据。UMVC场景设为False
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
            use_cross_view_flow=use_cross_view_flow,
            is_aligned=is_aligned  # 控制是否使用contrastive loss
        )
        
        # Store hyperparameters
        self.sigma_min = sigma_min
        self.is_aligned = is_aligned
    
    def encode(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode views to latent space and compute consensus
        
        Args:
            views: List of view tensors [B, D_v]
            mask: View availability mask [B, V]
        
        Returns:
            latents: List of view latents
            consensus: Fused consensus representation (None for unaligned data)
        
        Note:
            For UNALIGNED data (is_aligned=False), averaging latents across views
            is mathematically invalid since x_i^(v) and x_i^(u) are NOT the same sample.
            In this case, we return None for consensus and use cluster centroids
            as conditioning for imputation instead.
        """
        latents = self.encoder_decoder.encode(views, mask)
        
        if self.is_aligned:
            # Aligned data: safe to average across views
            consensus = self.encoder_decoder.fuse_latents(latents, mask)
        else:
            # Unaligned data: averaging is INVALID!
            # We will use cluster centroids for conditioning instead
            # For now, return mean of first view as placeholder for shape compatibility
            # The actual conditioning will be handled in forward() using cluster assignments
            consensus = None
        
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
        
        Note:
            For ALIGNED data: condition on consensus (average of available views)
            For UNALIGNED data: condition on cluster centroid (since averaging is invalid)
        """
        batch_size = views[0].shape[0]
        device = views[0].device
        
        # Encode available views
        latents, consensus = self.encode(views, mask)
        
        # Determine condition for imputation
        if self.is_aligned and consensus is not None:
            # Aligned data: use consensus (average of available views)
            condition = consensus
        else:
            # Unaligned data: use cluster centroids as condition
            # First, get cluster assignments from available views
            available_latents = []
            for v in range(self.num_views):
                if mask[:, v].any():
                    available_latents.append(latents[v])
            
            if available_latents:
                # Use mean of available view latents to find nearest cluster
                # Note: This is per-sample, not cross-sample averaging
                pseudo_z = torch.stack(available_latents, dim=1).mean(dim=1)
                q, _ = self.clustering(pseudo_z)
                assignments = q.argmax(dim=1)
                condition = self.clustering.centroids[assignments]
            else:
                # No views available: use random cluster centroid
                random_assignments = torch.randint(0, self.num_clusters, (batch_size,), device=device)
                condition = self.clustering.centroids[random_assignments]
        
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
                    
                    # Solve ODE: Noise -> Data (conditioned on consensus/centroid)
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
        
        # For unaligned data, we need an alternative to consensus
        if consensus is None:
            # UNALIGNED: Each view operates INDEPENDENTLY.
            # Index i in View A has NO correspondence to index i in View B.
            # Therefore, NO cross-view voting or averaging is allowed.
            
            # Compute per-view soft assignments and per-view conditioning
            per_view_q = []
            per_view_conditions = []  # Each view has its own condition
            
            for z_v in latents:
                q_v, _ = self.clustering(z_v)  # [B, K]
                per_view_q.append(q_v)
                
                # Each view independently selects its centroid
                assignments_v = q_v.argmax(dim=1)  # [B]
                cond_v = self.clustering.centroids[assignments_v]  # [B, D]
                per_view_conditions.append(cond_v)
            
            # For clustering evaluation, we need a single q per sample.
            # Use average of per-view q ONLY for the clustering loss computation.
            # This is a heuristic for training - at inference, each view is evaluated independently.
            q = torch.stack(per_view_q, dim=0).mean(dim=0)  # [V, B, K] -> [B, K]
            p = self.clustering._target_distribution(q)
            
            # Store per-view conditions for CFM loss (NOT a single consensus)
            # For now, use the first view's condition as a placeholder for outputs
            # The actual per-view conditions are used in compute_loss
            consensus = per_view_conditions[0]  # Placeholder for output dict
            
            # Store per-view conditions for later use
            outputs = {
                'latents': latents,
                'consensus': None,  # Mark as unaligned - no valid consensus
                'per_view_conditions': per_view_conditions,  # NEW: per-view centroids
                'q': q,
                'p': p,
                'per_view_q': per_view_q,  # NEW: per-view soft assignments
                'assignments': q.argmax(dim=1)
            }
        else:
            # Aligned data: use actual consensus for clustering
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
        
        # Get per_view_conditions if available (for unaligned case)
        per_view_conditions = outputs.get('per_view_conditions', None)
        
        loss, loss_dict = self.loss_fn(
            vector_field=self.vector_field,
            latents=outputs['latents'],
            consensus=outputs['consensus'],
            q=outputs['q'],
            p=outputs['p'],
            x_original=views,
            x_recon=outputs['reconstructions'],
            mask=mask,
            ablation_mode=ablation_mode,
            per_view_conditions=per_view_conditions  # NEW: per-view centroids for unaligned
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
        Get embeddings for clustering
        
        For aligned data: returns consensus embeddings (view average)
        For unaligned data: returns first view's latent embeddings 
                           (since no valid consensus exists)
        
        Args:
            views: List of view tensors
            mask: View availability mask
        
        Returns:
            Embeddings as numpy array
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(views, mask)
            if outputs['consensus'] is not None:
                # Aligned: use consensus
                embeddings = outputs['consensus'].cpu().numpy()
            else:
                # Unaligned: use first view's latent (no valid cross-view consensus)
                embeddings = outputs['latents'][0].cpu().numpy()
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
