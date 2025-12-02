"""
OT-CFM Model Components:
- View-specific Encoders
- Time Embedding
- Vector Field Network
- ODE Solver
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from functools import partial


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class MLPEncoder(nn.Module):
    """MLP encoder for a single view"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_bn: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """MLP decoder for a single view"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_bn: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ResidualBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.time_mlp = nn.Linear(time_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(self.linear1(h))
        h = h + self.time_mlp(F.silu(t_emb))
        h = self.norm2(h)
        h = self.dropout(F.silu(self.linear2(h)))
        return x + h


class VectorFieldNetwork(nn.Module):
    """
    Neural network that predicts the vector field v_theta(z, t, c)
    for the flow matching objective.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 4,
        condition_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Input projection
        input_dim = latent_dim
        if condition_dim is not None:
            input_dim += condition_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: Latent state [B, D]
            t: Time [B] or [B, 1]
            condition: Optional conditioning [B, C]
        
        Returns:
            Vector field [B, D]
        """
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t.squeeze(-1))
        
        # Concatenate condition if provided
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        
        # Forward through network
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h, t_emb)
        
        return self.output_proj(h)


class ODESolver:
    """ODE solver for flow matching using Euler method"""
    
    def __init__(self, vector_field: VectorFieldNetwork, num_steps: int = 10):
        self.vector_field = vector_field
        self.num_steps = num_steps
    
    def solve(
        self,
        z0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
        t_end: float = 1.0
    ) -> torch.Tensor:
        """
        Solve ODE from t_start to t_end using Euler method
        
        Args:
            z0: Initial state [B, D]
            condition: Optional conditioning [B, C]
            t_start: Start time
            t_end: End time
        
        Returns:
            Final state z1 [B, D]
        """
        dt = (t_end - t_start) / self.num_steps
        z = z0.clone()
        
        for i in range(self.num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((z.shape[0],), t, device=z.device)
            
            v = self.vector_field(z, t_tensor, condition)
            z = z + dt * v
        
        return z
    
    def solve_trajectory(
        self,
        z0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
        t_end: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Solve ODE and return full trajectory
        """
        dt = (t_end - t_start) / self.num_steps
        z = z0.clone()
        trajectory = [z.clone()]
        
        for i in range(self.num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((z.shape[0],), t, device=z.device)
            
            v = self.vector_field(z, t_tensor, condition)
            z = z + dt * v
            trajectory.append(z.clone())
        
        return trajectory


class MultiViewEncoderDecoder(nn.Module):
    """
    Multi-view encoder-decoder with shared latent space
    """
    
    def __init__(
        self,
        view_dims: List[int],
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_views = len(view_dims)
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        
        # View-specific encoders
        self.encoders = nn.ModuleList([
            MLPEncoder(dim, hidden_dims, latent_dim, dropout)
            for dim in view_dims
        ])
        
        # View-specific decoders
        self.decoders = nn.ModuleList([
            MLPDecoder(latent_dim, hidden_dims[::-1], dim, dropout)
            for dim in view_dims
        ])
    
    def encode(self, views: List[torch.Tensor], mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Encode each view to latent space"""
        latents = []
        for v, (encoder, view) in enumerate(zip(self.encoders, views)):
            z = encoder(view)
            latents.append(z)
        return latents
    
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode latent to each view"""
        recons = []
        for decoder, z in zip(self.decoders, latents):
            x = decoder(z)
            recons.append(x)
        return recons
    
    def fuse_latents(
        self,
        latents: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse multiple view latents into consensus representation"""
        if mask is None:
            # Simple average
            stacked = torch.stack(latents, dim=1)  # [B, V, D]
            return stacked.mean(dim=1)
        else:
            # Weighted average based on availability
            stacked = torch.stack(latents, dim=1)  # [B, V, D]
            mask = mask.unsqueeze(-1)  # [B, V, 1]
            weighted = (stacked * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return weighted / counts


class ClusteringModule(nn.Module):
    """
    Clustering module using Student's t-distribution
    """
    
    def __init__(self, latent_dim: int, num_clusters: int, alpha: float = 1.0):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.alpha = alpha
        
        # Learnable cluster centroids
        self.centroids = nn.Parameter(torch.randn(num_clusters, latent_dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft assignments using Student's t-distribution
        
        Args:
            z: Latent embeddings [B, D]
        
        Returns:
            q: Soft assignments [B, K]
            p: Target distribution [B, K]
        """
        # Compute distances to centroids
        # z: [B, D], centroids: [K, D]
        dist = torch.cdist(z, self.centroids)  # [B, K]
        
        # Student's t-distribution
        q = 1.0 / (1.0 + (dist ** 2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        
        # Compute target distribution P
        p = self._target_distribution(q)
        
        return q, p
    
    def _target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary target distribution P that emphasizes
        high confidence assignments
        """
        # p_ik = q_ik^2 / sum_i(q_ik)
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p
    
    def init_centroids(self, z: torch.Tensor, method: str = "kmeans"):
        """Initialize centroids from data"""
        from sklearn.cluster import KMeans
        
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=42)
        kmeans.fit(z_np)
        
        self.centroids.data = torch.FloatTensor(kmeans.cluster_centers_).to(z.device)
    
    def get_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments"""
        q, _ = self.forward(z)
        return q.argmax(dim=1)


if __name__ == "__main__":
    # Test model components
    print("Testing model components...")
    
    batch_size = 32
    latent_dim = 128
    view_dims = [100, 200, 150]
    
    # Test encoder-decoder
    model = MultiViewEncoderDecoder(view_dims, [256, 128], latent_dim)
    views = [torch.randn(batch_size, dim) for dim in view_dims]
    latents = model.encode(views)
    print(f"Encoded latent shapes: {[z.shape for z in latents]}")
    
    recons = model.decode(latents)
    print(f"Reconstructed shapes: {[x.shape for x in recons]}")
    
    consensus = model.fuse_latents(latents)
    print(f"Consensus shape: {consensus.shape}")
    
    # Test vector field
    vf = VectorFieldNetwork(latent_dim, hidden_dim=256, time_dim=64)
    z = torch.randn(batch_size, latent_dim)
    t = torch.rand(batch_size)
    v = vf(z, t)
    print(f"Vector field output shape: {v.shape}")
    
    # Test ODE solver
    solver = ODESolver(vf, num_steps=10)
    z0 = torch.randn(batch_size, latent_dim)
    z1 = solver.solve(z0)
    print(f"ODE solution shape: {z1.shape}")
    
    # Test clustering
    cluster = ClusteringModule(latent_dim, num_clusters=10)
    q, p = cluster(consensus)
    print(f"Soft assignments shape: {q.shape}")
    print(f"Target distribution shape: {p.shape}")
