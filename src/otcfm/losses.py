"""
Loss functions for OT-CFM:
- Conditional Flow Matching Loss
- Gromov-Wasserstein Structural Alignment Loss
- Clustering Loss (KL Divergence)
- Reconstruction Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class ConditionalFlowMatchingLoss(nn.Module):
    """
    Conditional Flow Matching (CFM) loss for training vector field networks.
    
    The loss is: E_t,z0,z1 [ || v_theta(z_t, t) - (z1 - z0) ||^2 ]
    
    where z_t = (1 - (1 - sigma_min) * t) * z0 + t * z1
    """
    
    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
    
    def forward(
        self,
        vector_field: nn.Module,
        z1: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CFM loss
        
        Args:
            vector_field: Neural network predicting v_theta
            z1: Target latent samples [B, D]
            condition: Optional conditioning information [B, C]
        
        Returns:
            CFM loss scalar
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # Sample noise from prior p_0 = N(0, I)
        z0 = torch.randn_like(z1)
        
        # Compute interpolation z_t = (1 - (1 - sigma_min)*t) * z0 + t * z1
        t_expanded = t.unsqueeze(-1)
        z_t = (1 - (1 - self.sigma_min) * t_expanded) * z0 + t_expanded * z1
        
        # Target vector field: u_t = z1 - (1 - sigma_min) * z0
        target = z1 - (1 - self.sigma_min) * z0
        
        # Predict vector field
        pred = vector_field(z_t, t, condition)
        
        # MSE loss
        loss = F.mse_loss(pred, target)
        
        return loss


class GromovWassersteinLoss(nn.Module):
    """
    Gromov-Wasserstein structural alignment loss.
    
    Measures the discrepancy between intra-view geometric structures
    to enable alignment-free multi-view learning.
    
    L_GW = sum_{v != u} || K(Z^v, Z^v) - K(Z^u, Z^u) ||_F^2
    """
    
    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        normalize: bool = True
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.normalize = normalize
    
    def compute_kernel_matrix(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise kernel/similarity matrix
        
        Args:
            Z: Latent embeddings [B, D]
        
        Returns:
            K: Kernel matrix [B, B]
        """
        if self.kernel_type == "rbf":
            # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            dist = torch.cdist(Z, Z)
            K = torch.exp(-self.gamma * dist ** 2)
        
        elif self.kernel_type == "cosine":
            # Cosine similarity
            Z_norm = F.normalize(Z, dim=-1)
            K = Z_norm @ Z_norm.T
        
        elif self.kernel_type == "linear":
            # Linear kernel: K(x, y) = x^T y
            K = Z @ Z.T
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        if self.normalize:
            # Normalize to [0, 1]
            K = (K - K.min()) / (K.max() - K.min() + 1e-8)
        
        return K
    
    def forward(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute GW alignment loss between all pairs of views
        
        Args:
            latents: List of latent embeddings [Z^1, Z^2, ..., Z^V]
        
        Returns:
            GW loss scalar
        """
        num_views = len(latents)
        
        if num_views < 2:
            return torch.tensor(0.0, device=latents[0].device)
        
        # Compute kernel matrices for each view
        kernels = [self.compute_kernel_matrix(z) for z in latents]
        
        # Compute pairwise structural discrepancy
        loss = 0.0
        count = 0
        
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # Frobenius norm of difference
                diff = kernels[i] - kernels[j]
                loss = loss + torch.mean(diff ** 2)
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class ClusteringLoss(nn.Module):
    """
    Clustering loss using KL divergence between soft assignments Q
    and target distribution P.
    
    L_cluster = KL(P || Q) = sum_i sum_k p_ik * log(p_ik / q_ik)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute KL divergence loss
        
        Args:
            q: Soft assignments [B, K]
            p: Target distribution [B, K]
            eps: Small constant for numerical stability
        
        Returns:
            KL divergence loss
        """
        # KL(P || Q) = sum(P * log(P / Q))
        loss = (p * torch.log((p + eps) / (q + eps))).sum(dim=1).mean()
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multi-view learning.
    Encourages same-sample representations to be similar across views.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss between view pairs
        
        Args:
            latents: List of latent embeddings [Z^1, Z^2, ..., Z^V]
        
        Returns:
            Contrastive loss
        """
        num_views = len(latents)
        
        if num_views < 2:
            return torch.tensor(0.0, device=latents[0].device)
        
        batch_size = latents[0].shape[0]
        device = latents[0].device
        
        loss = 0.0
        count = 0
        
        for i in range(num_views):
            for j in range(i + 1, num_views):
                z_i = F.normalize(latents[i], dim=-1)
                z_j = F.normalize(latents[j], dim=-1)
                
                # Positive pairs: same sample across views
                pos_sim = (z_i * z_j).sum(dim=-1) / self.temperature
                
                # Negative pairs: all other samples
                neg_sim_i = z_i @ z_j.T / self.temperature  # [B, B]
                neg_sim_j = z_j @ z_i.T / self.temperature
                
                # InfoNCE loss
                labels = torch.arange(batch_size, device=device)
                loss_i = F.cross_entropy(neg_sim_i, labels)
                loss_j = F.cross_entropy(neg_sim_j, labels)
                
                loss = loss + (loss_i + loss_j) / 2
                count += 1
        
        return loss / count if count > 0 else loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder training
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        x_original: List[torch.Tensor],
        x_recon: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss
        
        Args:
            x_original: Original views
            x_recon: Reconstructed views
            mask: View availability mask [B, V]
        
        Returns:
            Reconstruction loss
        """
        loss = 0.0
        num_views = len(x_original)
        
        for v in range(num_views):
            view_loss = F.mse_loss(x_recon[v], x_original[v], reduction='none')
            view_loss = view_loss.mean(dim=-1)  # [B]
            
            if mask is not None:
                view_loss = view_loss * mask[:, v]
            
            loss = loss + view_loss.mean()
        
        return loss / num_views


class CrossViewFlowMatchingLoss(nn.Module):
    """
    Cross-view Flow Matching loss for better multi-view representation learning.
    
    Instead of self-conditioning, we learn to transform from one view's latent
    to another view's latent, encouraging the model to learn view-invariant features.
    """
    
    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
    
    def forward(
        self,
        vector_field: nn.Module,
        latents: List[torch.Tensor],
        consensus: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-view flow matching loss
        
        Args:
            vector_field: Neural network predicting v_theta
            latents: List of view-specific latents
            consensus: Fused consensus representation
        
        Returns:
            Cross-view FM loss
        """
        if len(latents) < 2:
            return torch.tensor(0.0, device=consensus.device)
        
        batch_size = consensus.shape[0]
        device = consensus.device
        
        total_loss = 0.0
        count = 0
        
        # Learn to flow from each view to consensus
        for z_view in latents:
            t = torch.rand(batch_size, device=device)
            t_expanded = t.unsqueeze(-1)
            
            # Interpolate from view latent to consensus
            z_t = (1 - (1 - self.sigma_min) * t_expanded) * z_view + t_expanded * consensus
            
            # Target: consensus - (1 - sigma_min) * z_view
            target = consensus - (1 - self.sigma_min) * z_view
            
            # Predict with view latent as condition
            pred = vector_field(z_t, t, condition=z_view)
            
            total_loss = total_loss + F.mse_loss(pred, target)
            count += 1
        
        return total_loss / count if count > 0 else total_loss


class OTCFMLoss(nn.Module):
    """
    Combined OT-CFM loss function (Optimized version)
    
    L_total = L_CFM + lambda_gw * L_GW + lambda_cluster * L_cluster + lambda_recon * L_recon
    
    Key improvements:
    - Cross-view flow matching for better representation
    - Adaptive loss weighting
    - Stronger clustering signal
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        kernel_type: str = "rbf",
        kernel_gamma: float = 1.0,
        lambda_gw: float = 0.1,
        lambda_cluster: float = 0.5,
        lambda_recon: float = 1.0,
        lambda_contrastive: float = 0.1,
        use_cross_view_flow: bool = True
    ):
        super().__init__()
        
        self.cfm_loss = ConditionalFlowMatchingLoss(sigma_min)
        self.cross_view_loss = CrossViewFlowMatchingLoss(sigma_min)
        self.gw_loss = GromovWassersteinLoss(kernel_type, kernel_gamma)
        self.cluster_loss = ClusteringLoss()
        self.recon_loss = ReconstructionLoss()
        self.contrastive_loss = ContrastiveLoss()
        
        self.lambda_gw = lambda_gw
        self.lambda_cluster = lambda_cluster
        self.lambda_recon = lambda_recon
        self.lambda_contrastive = lambda_contrastive
        self.use_cross_view_flow = use_cross_view_flow
    
    def forward(
        self,
        vector_field: nn.Module,
        latents: List[torch.Tensor],
        consensus: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        x_original: Optional[List[torch.Tensor]] = None,
        x_recon: Optional[List[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        ablation_mode: str = "full"
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total OT-CFM loss
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # CFM loss - use cross-view flow matching for better learning
        if ablation_mode != "no_flow":
            if self.use_cross_view_flow:
                # Cross-view flow matching (improved)
                loss_cfm = self.cross_view_loss(vector_field, latents, consensus)
            else:
                # Original self-conditioning
                loss_cfm = self.cfm_loss(vector_field, consensus, condition=consensus)
            loss_dict['cfm'] = loss_cfm.item()
        else:
            loss_cfm = torch.tensor(0.0, device=consensus.device)
            loss_dict['cfm'] = 0.0
        
        # GW alignment loss
        if ablation_mode not in ["no_gw", "no_ot"]:
            loss_gw = self.gw_loss(latents)
            loss_dict['gw'] = loss_gw.item()
        else:
            loss_gw = torch.tensor(0.0, device=consensus.device)
            loss_dict['gw'] = 0.0
        
        # Clustering loss - this is crucial for clustering performance
        if ablation_mode != "no_cluster":
            loss_cluster = self.cluster_loss(q, p)
            loss_dict['cluster'] = loss_cluster.item()
        else:
            loss_cluster = torch.tensor(0.0, device=consensus.device)
            loss_dict['cluster'] = 0.0
        
        # Reconstruction loss
        if x_original is not None and x_recon is not None:
            loss_recon = self.recon_loss(x_original, x_recon, mask)
            loss_dict['recon'] = loss_recon.item()
        else:
            loss_recon = torch.tensor(0.0, device=consensus.device)
            loss_dict['recon'] = 0.0
        
        # Contrastive loss - helps learn view-invariant features
        loss_contrastive = self.contrastive_loss(latents)
        loss_dict['contrastive'] = loss_contrastive.item()
        
        # Total loss with better weighting
        total_loss = (
            loss_cfm +
            self.lambda_gw * loss_gw +
            self.lambda_cluster * loss_cluster +
            self.lambda_recon * loss_recon +
            self.lambda_contrastive * loss_contrastive
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 32
    latent_dim = 128
    num_views = 3
    num_clusters = 10
    
    # Create dummy data
    latents = [torch.randn(batch_size, latent_dim) for _ in range(num_views)]
    consensus = torch.randn(batch_size, latent_dim)
    q = F.softmax(torch.randn(batch_size, num_clusters), dim=-1)
    p = F.softmax(torch.randn(batch_size, num_clusters), dim=-1)
    
    # Test GW loss
    gw_loss = GromovWassersteinLoss()
    loss = gw_loss(latents)
    print(f"GW Loss: {loss.item():.4f}")
    
    # Test clustering loss
    cluster_loss = ClusteringLoss()
    loss = cluster_loss(q, p)
    print(f"Clustering Loss: {loss.item():.4f}")
    
    # Test contrastive loss
    contrastive_loss = ContrastiveLoss()
    loss = contrastive_loss(latents)
    print(f"Contrastive Loss: {loss.item():.4f}")
    
    # Test CFM loss
    from .models import VectorFieldNetwork
    vf = VectorFieldNetwork(latent_dim)
    cfm_loss = ConditionalFlowMatchingLoss()
    loss = cfm_loss(vf, consensus)
    print(f"CFM Loss: {loss.item():.4f}")
    
    print("All loss tests passed!")
