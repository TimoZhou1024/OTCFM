"""
Adapters for external multi-view clustering methods from GitHub

This module provides wrapper classes that integrate external MVC methods
into the OT-CFM comparison framework.

Usage:
    1. Clone the external method to external_methods/<method_name>/
    2. Create a wrapper class inheriting from BaseClusteringMethod
    3. Register it in get_external_baselines()

Example:
    git clone https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git external_methods/MFLVC
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .baselines import BaseClusteringMethod

# Path to external methods directory
EXTERNAL_PATH = Path(__file__).parent.parent.parent / "external_methods"


# ============================================================
# Helper Functions
# ============================================================

def safe_import_external(method_path: Path, module_name: str):
    """Safely import an external module"""
    if not method_path.exists():
        return None
    
    sys.path.insert(0, str(method_path))
    try:
        module = __import__(module_name)
        return module
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")
        return None
    finally:
        if str(method_path) in sys.path:
            sys.path.remove(str(method_path))


def fallback_kmeans(views: List[np.ndarray], num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback to simple KMeans when external method fails"""
    # Normalize and concatenate
    normalized = []
    for v in views:
        scaler = StandardScaler()
        normalized.append(scaler.fit_transform(v))
    X = np.concatenate(normalized, axis=1)
    
    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    return labels, X


# ============================================================
# MFLVC: Multi-level Feature Learning for Contrastive MVC
# Paper: CVPR 2022
# GitHub: https://github.com/SubmissionsIn/MFLVC
# ============================================================

class MFLVCWrapper(BaseClusteringMethod):
    """
    MFLVC: Multi-level Feature Learning for Contrastive Multi-View Clustering
    
    Reference:
        Xu et al., "Multi-level Feature Learning for Contrastive 
        Multi-View Clustering", CVPR 2022
    """
    
    def __init__(self, num_clusters: int, feature_dim: int = 512, 
                 high_feature_dim: int = 128, mse_epochs: int = 200,
                 con_epochs: int = 50, batch_size: int = 256, 
                 lr: float = 0.0003, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.mse_epochs = mse_epochs
        self.con_epochs = con_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        mflvc_path = EXTERNAL_PATH / "MFLVC"
        
        if not mflvc_path.exists():
            print(f"MFLVC not found. Clone it with:")
            print(f"  git clone https://github.com/SubmissionsIn/MFLVC.git {mflvc_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        sys.path.insert(0, str(mflvc_path))
        try:
            from network import Network
            from loss import Loss
            
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            # Create custom dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, views_data):
                    self.views = [torch.FloatTensor(v) for v in views_data]
                    self.n_samples = views_data[0].shape[0]
                    
                def __len__(self):
                    return self.n_samples
                    
                def __getitem__(self, idx):
                    xs = [v[idx] for v in self.views]
                    return xs, 0, idx  # (views, label_placeholder, index)
            
            dataset = CustomDataset(views)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
            
            # Initialize model
            model = Network(
                view=n_views,
                input_size=view_dims,
                feature_dim=self.feature_dim,
                high_feature_dim=self.high_feature_dim,
                class_num=self.num_clusters,
                device=self.device
            ).to(self.device)
            
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=0.
            )
            criterion = Loss(
                self.batch_size, self.num_clusters, 
                temperature_f=0.5, temperature_l=1.0, device=self.device
            )
            
            # Phase 1: Pretrain with reconstruction
            print(f"  MFLVC Phase 1: Reconstruction pretraining ({self.mse_epochs} epochs)...")
            mse_loss_fn = torch.nn.MSELoss()
            for epoch in range(self.mse_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(self.device) for x in xs]
                    optimizer.zero_grad()
                    _, _, xrs, _ = model(xs)
                    loss = sum(mse_loss_fn(xs[v], xrs[v]) for v in range(n_views))
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{self.mse_epochs}")
            
            # Phase 2: Contrastive training
            print(f"  MFLVC Phase 2: Contrastive training ({self.con_epochs} epochs)...")
            for epoch in range(self.con_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(self.device) for x in xs]
                    optimizer.zero_grad()
                    hs, qs, xrs, zs = model(xs)
                    
                    loss_list = []
                    for v in range(n_views):
                        for w in range(v + 1, n_views):
                            loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                            loss_list.append(criterion.forward_label(qs[v], qs[w]))
                        loss_list.append(mse_loss_fn(xs[v], xrs[v]))
                    loss = sum(loss_list)
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{self.con_epochs}")
            
            # Get final predictions
            model.eval()
            full_loader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=False
            )
            
            with torch.no_grad():
                for xs, _, _ in full_loader:
                    xs = [x.to(self.device) for x in xs]
                    qs, preds = model.forward_cluster(xs)
                    
                    # Average predictions across views
                    q_avg = torch.stack(qs, dim=0).mean(dim=0)
                    self.labels_ = torch.argmax(q_avg, dim=1).cpu().numpy()
                    
                    # Get embeddings
                    hs, _, _, zs = model(xs)
                    self.embeddings_ = torch.stack(zs, dim=0).mean(dim=0).cpu().numpy()
            
            print(f"  MFLVC completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"MFLVC execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
            
        finally:
            if str(mflvc_path) in sys.path:
                sys.path.remove(str(mflvc_path))
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# SURE: Self-supervised Uncorrelated Representation for MVC
# Paper: TPAMI 2022
# GitHub: https://github.com/XLearning-SCU/2022-TPAMI-SURE
# ============================================================

class SUREWrapper(BaseClusteringMethod):
    """
    SURE: Robust Multi-View Clustering with Incomplete Information
    
    Uses contrastive learning with noisy negative pairs and robust loss.
    Two-view Siamese network architecture with cross-reconstruction.
    
    Reference:
        Yang et al., "Robust Multi-View Clustering with Incomplete Information", 
        TPAMI 2022
    """
    
    def __init__(self, num_clusters: int, epochs: int = 80, 
                 batch_size: int = 256, lr: float = 1e-3,
                 lam: float = 0.5, margin: int = 5, neg_prop: int = 30,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam  # Weight between contrastive and reconstruction loss
        self.margin = margin  # Margin for contrastive loss
        self.neg_prop = neg_prop  # Ratio of negative to positive pairs
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        sure_path = EXTERNAL_PATH / "SURE"
        
        if not sure_path.exists():
            print(f"SURE not found. Clone it with:")
            print(f"  git clone https://github.com/XLearning-SCU/2022-TPAMI-SURE.git {sure_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        device = kwargs.get('device', self.device)
        
        try:
            # Build a flexible SURE model for any 2-view input dimensions
            view_dims = [v.shape[1] for v in views]
            n_samples = views[0].shape[0]
            
            # Define generic SURE network for arbitrary dimensions
            class SURENetwork(nn.Module):
                """Siamese network for SURE with cross-reconstruction"""
                def __init__(self, dims, hidden_dim=1024, latent_dim=10):
                    super().__init__()
                    self.encoder0 = nn.Sequential(
                        nn.Linear(dims[0], hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, latent_dim),
                        nn.BatchNorm1d(latent_dim),
                        nn.ReLU(True)
                    )
                    self.encoder1 = nn.Sequential(
                        nn.Linear(dims[1], hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(True),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, latent_dim),
                        nn.BatchNorm1d(latent_dim),
                        nn.ReLU(True)
                    )
                    # Decoders take concatenated latent representations
                    self.decoder0 = nn.Sequential(
                        nn.Linear(latent_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, dims[0])
                    )
                    self.decoder1 = nn.Sequential(
                        nn.Linear(latent_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, dims[1])
                    )
                    
                def forward(self, x0, x1):
                    h0 = self.encoder0(x0.view(x0.size(0), -1))
                    h1 = self.encoder1(x1.view(x1.size(0), -1))
                    union = torch.cat([h0, h1], dim=1)
                    z0 = self.decoder0(union)
                    z1 = self.decoder1(union)
                    return h0, h1, z0, z1
            
            # Create pair dataset for contrastive learning
            class PairDataset(torch.utils.data.Dataset):
                """Dataset that creates positive and negative pairs"""
                def __init__(self, v0, v1, neg_prop=30):
                    self.v0 = torch.FloatTensor(v0)
                    self.v1 = torch.FloatTensor(v1)
                    self.n_samples = len(v0)
                    self.neg_prop = neg_prop
                    
                def __len__(self):
                    return self.n_samples * (1 + self.neg_prop)
                
                def __getitem__(self, idx):
                    if idx < self.n_samples:
                        # Positive pair: same index from both views
                        i = idx
                        return self.v0[i], self.v1[i], 1, 1  # (x0, x1, label, real_label)
                    else:
                        # Negative pair: random different indices
                        pair_idx = idx - self.n_samples
                        i = pair_idx % self.n_samples
                        j = np.random.randint(0, self.n_samples)
                        while j == i:
                            j = np.random.randint(0, self.n_samples)
                        return self.v0[i], self.v1[j], 0, 0  # Negative pair
            
            # Noise-robust contrastive loss
            class NoiseRobustLoss(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                def forward(self, pair_dist, labels, margin, start_fine=False):
                    dist_sq = pair_dist * pair_dist
                    labels = labels.float()
                    N = len(labels)
                    
                    if start_fine:
                        loss = labels * dist_sq + (1 - labels) * (1 / margin) * torch.pow(
                            torch.clamp(torch.pow(pair_dist, 0.5) * (margin - pair_dist), min=0.0), 2)
                    else:
                        loss = labels * dist_sq + (1 - labels) * torch.pow(
                            torch.clamp(margin - pair_dist, min=0.0), 2)
                    return torch.sum(loss) / (2.0 * N)
            
            # Initialize model
            model = SURENetwork(view_dims, hidden_dim=512, latent_dim=self.num_clusters).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            ncl_criterion = NoiseRobustLoss()
            mse_criterion = nn.MSELoss()
            
            # Create dataloader
            dataset = PairDataset(views[0], views[1], neg_prop=self.neg_prop)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )
            
            # Training
            print(f"  SURE Training ({epochs} epochs)...")
            margin = self.margin
            start_fine = False
            
            for epoch in range(epochs):
                model.train()
                total_ncl_loss = 0
                total_ver_loss = 0
                pos_dist_sum = 0
                neg_dist_sum = 0
                pos_count = 0
                neg_count = 0
                
                for x0, x1, labels, real_labels in dataloader:
                    x0 = x0.to(device)
                    x1 = x1.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    h0, h1, z0, z1 = model(x0, x1)
                    
                    # Pairwise distance
                    pair_dist = F.pairwise_distance(h0, h1)
                    
                    # Track distances for margin adaptation
                    pos_dist_sum += pair_dist[labels == 1].sum().item()
                    neg_dist_sum += pair_dist[labels == 0].sum().item()
                    pos_count += (labels == 1).sum().item()
                    neg_count += (labels == 0).sum().item()
                    
                    # Losses
                    ncl_loss = ncl_criterion(pair_dist, labels, margin, start_fine)
                    ver_loss = mse_criterion(x0, z0) + mse_criterion(x1, z1)
                    loss = ncl_loss + self.lam * ver_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_ncl_loss += ncl_loss.item()
                    total_ver_loss += ver_loss.item()
                
                # Compute mean distances
                pos_dist_mean = pos_dist_sum / max(pos_count, 1)
                neg_dist_mean = neg_dist_sum / max(neg_count, 1)
                
                # Adapt margin on first epoch
                if epoch == 0 and self.margin != 1.0:
                    margin = max(1, round(pos_dist_mean + neg_dist_mean))
                
                # Switch to fine loss when neg_dist >= margin
                if not start_fine and neg_dist_mean >= margin:
                    start_fine = True
                    print(f"    Epoch {epoch+1}: Switching to fine loss")
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}, NCL: {total_ncl_loss/len(dataloader):.4f}, "
                          f"VER: {total_ver_loss/len(dataloader):.4f}")
            
            # Get embeddings for clustering
            model.eval()
            with torch.no_grad():
                x0 = torch.FloatTensor(views[0]).to(device)
                x1 = torch.FloatTensor(views[1]).to(device)
                h0, h1, _, _ = model(x0, x1)
                
                # Concatenate embeddings from both views
                embeddings = torch.cat([h0, h1], dim=1).cpu().numpy()
                self.embeddings_ = embeddings
            
            # Clustering using KMeans on concatenated embeddings
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10, 
                                  random_state=42).fit_predict(embeddings)
            
            print(f"  SURE completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"SURE execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# DealMVC: Dual Contrastive Prediction for IMVC
# Paper: CVPR 2023
# GitHub: https://github.com/SubmissionsIn/DealMVC
# ============================================================

class DealMVCWrapper(BaseClusteringMethod):
    """
    DealMVC: Dual Contrastive Prediction for Incomplete Multi-View Clustering
    
    Reference:
        "DealMVC: Dual Contrastive Calibration for Multi-View Clustering", 
        CVPR 2023
    """
    
    def __init__(self, num_clusters: int, mse_epochs: int = 200, 
                 con_epochs: int = 50, feature_dim: int = 512,
                 high_feature_dim: int = 128, batch_size: int = 256,
                 lr: float = 0.0003, threshold: float = 0.5,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.mse_epochs = mse_epochs
        self.con_epochs = con_epochs
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        dealmvc_path = EXTERNAL_PATH / "DealMVC"
        
        if not dealmvc_path.exists():
            print(f"DealMVC not found. Clone it with:")
            print(f"  git clone https://github.com/xihongyang1999/DealMVC.git {dealmvc_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        mse_epochs = kwargs.get('mse_epochs', self.mse_epochs)
        con_epochs = kwargs.get('con_epochs', self.con_epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        threshold = kwargs.get('threshold', self.threshold)
        device = kwargs.get('device', self.device)
        
        try:
            from torch.nn.functional import normalize
            from sklearn.preprocessing import MinMaxScaler
            
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            # Define Network (same as DealMVC)
            class Encoder(nn.Module):
                def __init__(self, input_dim, feature_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, feature_dim),
                    )
                def forward(self, x):
                    return self.encoder(x)
            
            class Decoder(nn.Module):
                def __init__(self, input_dim, feature_dim):
                    super().__init__()
                    self.decoder = nn.Sequential(
                        nn.Linear(feature_dim, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, input_dim)
                    )
                def forward(self, x):
                    return self.decoder(x)
            
            class DealMVCNet(nn.Module):
                def __init__(self, view, input_size, feature_dim, high_dim, class_num, device):
                    super().__init__()
                    self.encoders = nn.ModuleList([
                        Encoder(input_size[v], feature_dim).to(device) for v in range(view)
                    ])
                    self.decoders = nn.ModuleList([
                        Decoder(input_size[v], feature_dim).to(device) for v in range(view)
                    ])
                    self.feature_module = nn.Sequential(nn.Linear(feature_dim, high_dim))
                    self.label_module = nn.Sequential(
                        nn.Linear(feature_dim, class_num),
                        nn.Softmax(dim=1)
                    )
                    self.view = view
                    
                def forward(self, xs):
                    hs, qs, xrs, zs = [], [], [], []
                    for v in range(self.view):
                        z = self.encoders[v](xs[v])
                        h = normalize(self.feature_module(z), dim=1)
                        q = self.label_module(z)
                        xr = self.decoders[v](z)
                        hs.append(h)
                        zs.append(z)
                        qs.append(q)
                        xrs.append(xr)
                    return hs, qs, xrs, zs
            
            # Create dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, views_data):
                    self.views = [torch.FloatTensor(v) for v in views_data]
                    self.n_samples = views_data[0].shape[0]
                def __len__(self):
                    return self.n_samples
                def __getitem__(self, idx):
                    return [v[idx] for v in self.views], 0, idx
            
            dataset = CustomDataset(views)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, n_samples // 2),
                shuffle=True, drop_last=True
            )
            
            # Initialize model
            model = DealMVCNet(n_views, view_dims, self.feature_dim,
                               self.high_feature_dim, self.num_clusters, device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mse_loss = nn.MSELoss()
            
            # Phase 1: Reconstruction pretraining
            print(f"  DealMVC Phase 1: Reconstruction ({mse_epochs} epochs)...")
            for epoch in range(mse_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    _, _, xrs, _ = model(xs)
                    loss = sum(mse_loss(xs[v], xrs[v]) for v in range(n_views))
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{mse_epochs}")
            
            # Phase 2: Contrastive training with local and global calibration
            print(f"  DealMVC Phase 2: Contrastive calibration ({con_epochs} epochs)...")
            for epoch in range(con_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    hs, qs, xrs, zs = model(xs)
                    
                    loss_list = []
                    
                    # Local contrastive calibration between view pairs
                    for v in range(n_views):
                        for w in range(v + 1, n_views):
                            # Similarity matrix
                            sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                            sim_probs = sim / sim.sum(1, keepdim=True)
                            
                            # Pseudo label matrix
                            Q = torch.mm(qs[v], qs[w].t())
                            Q.fill_diagonal_(1)
                            pos_mask = (Q >= threshold).float()
                            Q = Q * pos_mask
                            Q = Q / (Q.sum(1, keepdim=True) + 1e-7)
                            
                            # Local contrastive loss
                            loss_local = -(torch.log(sim_probs + 1e-7) * Q).sum(1).mean()
                            loss_list.append(loss_local)
                        
                        loss_list.append(mse_loss(xs[v], xrs[v]))
                    
                    # Global contrastive calibration
                    fusion_h = sum(hs) / n_views  # Simple averaging
                    sim_fusion = torch.exp(torch.mm(fusion_h, fusion_h.t()))
                    sim_fusion_probs = sim_fusion / sim_fusion.sum(1, keepdim=True)
                    
                    # Fusion pseudo labels
                    fusion_z = sum(zs) / n_views
                    pse_fusion = model.label_module(fusion_z)
                    Q_global = torch.mm(pse_fusion, pse_fusion.t())
                    Q_global.fill_diagonal_(1)
                    pos_mask_global = (Q_global >= threshold).float()
                    Q_global = Q_global * pos_mask_global
                    Q_global = Q_global / (Q_global.sum(1, keepdim=True) + 1e-7)
                    
                    loss_global = -(torch.log(sim_fusion_probs + 1e-7) * Q_global).sum(1).mean()
                    loss_list.append(loss_global)
                    
                    loss = sum(loss_list)
                    loss.backward()
                    optimizer.step()
                    
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{con_epochs}")
            
            # Get final embeddings
            model.eval()
            full_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
            
            with torch.no_grad():
                for xs, _, _ in full_loader:
                    xs = [x.to(device) for x in xs]
                    hs, _, _, zs = model(xs)
                    # Concatenate high-level features
                    self.embeddings_ = torch.cat(hs, dim=1).cpu().numpy()
            
            # Clustering
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10,
                                  random_state=42).fit_predict(self.embeddings_)
            
            print(f"  DealMVC completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"DealMVC execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# COMPLETER: Incomplete Multi-View Clustering via Contrastive Prediction
# Paper: CVPR 2021
# GitHub: https://github.com/XLearning-SCU/2021-CVPR-Completer
# ============================================================

class COMPLETERWrapper(BaseClusteringMethod):
    """
    COMPLETER: Incomplete Multi-View Clustering via Contrastive Prediction
    
    Reference:
        Lin et al., "COMPLETER: Incomplete Multi-view Clustering via 
        Contrastive Prediction", CVPR 2021
    """
    
    def __init__(self, num_clusters: int, epochs: int = 200, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray],
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        completer_path = EXTERNAL_PATH / "COMPLETER"
        
        if not completer_path.exists():
            print(f"COMPLETER not found. Clone it with:")
            print(f"  git clone https://github.com/XLearning-SCU/2021-CVPR-Completer.git {completer_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        sys.path.insert(0, str(completer_path))
        try:
            from model import COMPLETER
            
            model = COMPLETER(
                n_clusters=self.num_clusters,
                dims=[v.shape[1] for v in views],
                device=self.device
            )
            
            self.labels_, self.embeddings_ = model.fit_predict(
                views, mask=mask, epochs=self.epochs
            )
            
            return self.labels_
            
        except Exception as e:
            print(f"COMPLETER execution failed: {e}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
            
        finally:
            if str(completer_path) in sys.path:
                sys.path.remove(str(completer_path))
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# GCFAggMVC: Graph-based Consensus Fusion for MVC
# Paper: CVPR 2023
# GitHub: https://github.com/Galaxy922/GCFAggMVC
# ============================================================

class GCFAggMVCWrapper(BaseClusteringMethod):
    """
    GCFAggMVC: Graph Contrastive Fusion Aggregation for Multi-View Clustering
    
    Uses Transformer encoder for cross-view aggregation and structure-guided
    contrastive loss for multi-view representation learning.
    
    Reference:
        Yan et al., "GCFAgg: Global and Cross-view Feature Aggregation for 
        Multi-View Clustering", CVPR 2023
    """
    
    def __init__(self, num_clusters: int, rec_epochs: int = 200, 
                 fine_tune_epochs: int = 100, low_feature_dim: int = 512,
                 high_feature_dim: int = 128, batch_size: int = 256,
                 lr: float = 0.0003, temperature_f: float = 0.5,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.rec_epochs = rec_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.low_feature_dim = low_feature_dim
        self.high_feature_dim = high_feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.temperature_f = temperature_f
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        gcf_path = EXTERNAL_PATH / "GCFAggMVC"
        
        if not gcf_path.exists():
            print(f"GCFAggMVC not found. Clone it with:")
            print(f"  git clone https://github.com/Galaxy922/GCFAggMVC.git {gcf_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        rec_epochs = kwargs.get('rec_epochs', self.rec_epochs)
        fine_tune_epochs = kwargs.get('fine_tune_epochs', self.fine_tune_epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        device = kwargs.get('device', self.device)
        
        try:
            from torch.nn.functional import normalize
            
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            # Define GCFAggMVC network
            class Encoder(nn.Module):
                def __init__(self, input_dim, feature_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, feature_dim),
                    )
                def forward(self, x):
                    return self.encoder(x)
            
            class Decoder(nn.Module):
                def __init__(self, input_dim, feature_dim):
                    super().__init__()
                    self.decoder = nn.Sequential(
                        nn.Linear(feature_dim, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, input_dim)
                    )
                def forward(self, x):
                    return self.decoder(x)
            
            class GCFAggNet(nn.Module):
                def __init__(self, view, input_size, low_dim, high_dim, device):
                    super().__init__()
                    self.encoders = nn.ModuleList([
                        Encoder(input_size[v], low_dim).to(device) for v in range(view)
                    ])
                    self.decoders = nn.ModuleList([
                        Decoder(input_size[v], low_dim).to(device) for v in range(view)
                    ])
                    self.Specific_view = nn.Sequential(nn.Linear(low_dim, high_dim))
                    self.Common_view = nn.Sequential(nn.Linear(low_dim * view, high_dim))
                    self.view = view
                    self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
                        d_model=low_dim * view, nhead=1, dim_feedforward=256, batch_first=True
                    )
                    
                def forward(self, xs):
                    xrs, zs, hs = [], [], []
                    for v in range(self.view):
                        z = self.encoders[v](xs[v])
                        h = normalize(self.Specific_view(z), dim=1)
                        xr = self.decoders[v](z)
                        hs.append(h)
                        zs.append(z)
                        xrs.append(xr)
                    return xrs, zs, hs
                
                def GCFAgg(self, xs):
                    zs = [self.encoders[v](xs[v]) for v in range(self.view)]
                    commonz = torch.cat(zs, dim=1)
                    # TransformerEncoderLayer returns (output, attention) with return_weights
                    commonz_out = self.TransformerEncoderLayer(commonz)
                    commonz = normalize(self.Common_view(commonz_out), dim=1)
                    return commonz, None  # S not needed for our simplified version
            
            # Structure-guided contrastive loss
            class StructureLoss(nn.Module):
                def __init__(self, batch_size, temperature_f, device):
                    super().__init__()
                    self.batch_size = batch_size
                    self.temperature_f = temperature_f
                    self.device = device
                    self.criterion = nn.CrossEntropyLoss(reduction="sum")
                    
                def forward(self, h_i, h_j):
                    N = h_i.size(0)
                    h = torch.cat((h_i, h_j), dim=0)
                    sim = torch.matmul(h, h.T) / self.temperature_f
                    
                    # Positive pairs: (i, i+N) and (i+N, i)
                    sim_i_j = torch.diag(sim, N)
                    sim_j_i = torch.diag(sim, -N)
                    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*N, 1)
                    
                    # Negative samples: all except self and positive pair
                    mask = torch.ones((2*N, 2*N), device=self.device, dtype=torch.bool)
                    mask.fill_diagonal_(False)
                    for i in range(N):
                        mask[i, N + i] = False
                        mask[N + i, i] = False
                    negative_samples = sim[mask].reshape(2*N, -1)
                    
                    labels = torch.zeros(2*N, device=self.device, dtype=torch.long)
                    logits = torch.cat((positive_samples, negative_samples), dim=1)
                    loss = self.criterion(logits, labels)
                    return loss / (2 * N)
            
            # Create dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, views_data):
                    self.views = [torch.FloatTensor(v) for v in views_data]
                    self.n_samples = views_data[0].shape[0]
                def __len__(self):
                    return self.n_samples
                def __getitem__(self, idx):
                    return [v[idx] for v in self.views], 0, idx
            
            dataset = CustomDataset(views)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, n_samples // 2),
                shuffle=True, drop_last=True
            )
            
            # Initialize model
            model = GCFAggNet(n_views, view_dims, self.low_feature_dim, 
                              self.high_feature_dim, device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mse_loss = nn.MSELoss()
            contrastive_loss = StructureLoss(batch_size, self.temperature_f, device)
            
            # Phase 1: Reconstruction pretraining
            print(f"  GCFAggMVC Phase 1: Reconstruction ({rec_epochs} epochs)...")
            for epoch in range(rec_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    xrs, _, _ = model(xs)
                    loss = sum(mse_loss(xs[v], xrs[v]) for v in range(n_views))
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{rec_epochs}")
            
            # Phase 2: Fine-tuning with contrastive loss
            print(f"  GCFAggMVC Phase 2: Fine-tuning ({fine_tune_epochs} epochs)...")
            for epoch in range(fine_tune_epochs):
                model.train()
                for xs, _, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    xrs, _, hs = model(xs)
                    commonz, _ = model.GCFAgg(xs)
                    
                    loss_list = []
                    for v in range(n_views):
                        loss_list.append(contrastive_loss(hs[v], commonz))
                        loss_list.append(mse_loss(xs[v], xrs[v]))
                    loss = sum(loss_list)
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 25 == 0:
                    print(f"    Epoch {epoch+1}/{fine_tune_epochs}")
            
            # Get final embeddings
            model.eval()
            full_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
            
            with torch.no_grad():
                for xs, _, _ in full_loader:
                    xs = [x.to(device) for x in xs]
                    commonz, _ = model.GCFAgg(xs)
                    self.embeddings_ = commonz.cpu().numpy()
            
            # Clustering
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10, 
                                  random_state=42).fit_predict(self.embeddings_)
            
            print(f"  GCFAggMVC completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"GCFAggMVC execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# DCG: Diffusion-based Cross-view Generation for Incomplete MVC
# Paper: AAAI 2025
# GitHub: https://github.com/zhangyuanyang21/2025-AAAI-DCG
# ============================================================

class DCGWrapper(BaseClusteringMethod):
    """
    DCG: Diffusion-based Cross-view Generation for Incomplete Multi-View Clustering
    
    Uses diffusion models to generate missing views and performs clustering
    with attention-based fusion and contrastive learning.
    
    Reference:
        Zhang et al., "Diffusion-based Cross-view Generation for 
        Incomplete Multi-View Clustering", AAAI 2025
    """
    
    def __init__(self, num_clusters: int, epochs: int = 200, 
                 latent_dim: int = 128, batch_size: int = 256,
                 lr: float = 0.0003, num_timesteps: int = 1000,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_timesteps = num_timesteps
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        dcg_path = EXTERNAL_PATH / "2025-AAAI-DCG"
        
        if not dcg_path.exists():
            print(f"DCG not found. Clone it with:")
            print(f"  git clone https://github.com/zhangyuanyang21/2025-AAAI-DCG.git {dcg_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        device = kwargs.get('device', self.device)
        
        try:
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            if n_views != 2:
                print(f"  DCG only supports 2 views, got {n_views}. Using fallback.")
                self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
                return self.labels_
            
            # Define components similar to DCG
            class Autoencoder(nn.Module):
                def __init__(self, input_dim, latent_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, latent_dim)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, input_dim)
                    )
                def forward(self, x):
                    z = self.encoder(x)
                    xr = self.decoder(z)
                    return z, xr
            
            class DiffusionUNet(nn.Module):
                """Simple diffusion network for noise prediction"""
                def __init__(self, latent_dim, emb_size=128):
                    super().__init__()
                    self.time_emb = nn.Sequential(
                        nn.Linear(1, emb_size),
                        nn.SiLU(),
                        nn.Linear(emb_size, emb_size)
                    )
                    self.net = nn.Sequential(
                        nn.Linear(latent_dim + emb_size, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, latent_dim)
                    )
                    
                def forward(self, x, t):
                    t_emb = self.time_emb(t.float().unsqueeze(-1))
                    h = torch.cat([x, t_emb], dim=-1)
                    return self.net(h)
            
            class AttentionFusion(nn.Module):
                """Attention-based view fusion"""
                def __init__(self, latent_dim):
                    super().__init__()
                    self.query = nn.Linear(latent_dim, latent_dim)
                    self.key = nn.Linear(latent_dim, latent_dim)
                    self.value = nn.Linear(latent_dim, latent_dim)
                    
                def forward(self, z1, z2):
                    # Simple attention fusion
                    z_stack = torch.stack([z1, z2], dim=1)  # [N, 2, D]
                    q = self.query(z_stack.mean(dim=1, keepdim=True))  # [N, 1, D]
                    k = self.key(z_stack)  # [N, 2, D]
                    v = self.value(z_stack)  # [N, 2, D]
                    
                    attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (z1.size(-1) ** 0.5), dim=-1)
                    out = torch.bmm(attn, v).squeeze(1)  # [N, D]
                    return out
            
            class ClusterLayer(nn.Module):
                """Soft clustering assignment"""
                def __init__(self, latent_dim, n_clusters):
                    super().__init__()
                    self.fc = nn.Linear(latent_dim, n_clusters)
                    
                def forward(self, z):
                    logits = self.fc(z)
                    q = F.softmax(logits, dim=1)
                    return q, logits
            
            # Initialize models
            ae1 = Autoencoder(view_dims[0], self.latent_dim).to(device)
            ae2 = Autoencoder(view_dims[1], self.latent_dim).to(device)
            df1 = DiffusionUNet(self.latent_dim).to(device)
            df2 = DiffusionUNet(self.latent_dim).to(device)
            attention = AttentionFusion(self.latent_dim).to(device)
            cluster_layer = ClusterLayer(self.latent_dim, self.num_clusters).to(device)
            
            optimizer = torch.optim.Adam(
                list(ae1.parameters()) + list(ae2.parameters()) + 
                list(df1.parameters()) + list(df2.parameters()) +
                list(attention.parameters()) + list(cluster_layer.parameters()),
                lr=lr
            )
            
            # Create dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, v1, v2):
                    self.v1 = torch.FloatTensor(v1)
                    self.v2 = torch.FloatTensor(v2)
                def __len__(self):
                    return len(self.v1)
                def __getitem__(self, idx):
                    return self.v1[idx], self.v2[idx], idx
            
            dataset = CustomDataset(views[0], views[1])
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, n_samples // 2),
                shuffle=True, drop_last=True
            )
            
            # Noise scheduler (simplified)
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
            
            def add_noise(x, noise, t):
                sqrt_alpha = alphas_cumprod[t].sqrt().view(-1, 1)
                sqrt_one_minus_alpha = (1 - alphas_cumprod[t]).sqrt().view(-1, 1)
                return sqrt_alpha * x + sqrt_one_minus_alpha * noise
            
            # Training
            print(f"  DCG Training ({epochs} epochs)...")
            mse_loss = nn.MSELoss()
            
            for epoch in range(epochs):
                ae1.train(); ae2.train()
                df1.train(); df2.train()
                attention.train(); cluster_layer.train()
                
                total_loss = 0
                for x1, x2, _ in data_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    optimizer.zero_grad()
                    
                    # Encode
                    z1, xr1 = ae1(x1)
                    z2, xr2 = ae2(x2)
                    
                    # Reconstruction loss
                    loss_rec = mse_loss(xr1, x1) + mse_loss(xr2, x2)
                    
                    # Diffusion loss
                    t = torch.randint(0, self.num_timesteps, (z1.size(0),), device=device)
                    noise1 = torch.randn_like(z1)
                    noise2 = torch.randn_like(z2)
                    noisy_z1 = add_noise(z1, noise1, t)
                    noisy_z2 = add_noise(z2, noise2, t)
                    pred_noise1 = df1(noisy_z1, t)
                    pred_noise2 = df2(noisy_z2, t)
                    loss_diff = mse_loss(pred_noise1, noise1) + mse_loss(pred_noise2, noise2)
                    
                    # Attention fusion
                    h = attention(z1, z2)
                    
                    # Clustering loss
                    q1, _ = cluster_layer(z1)
                    q2, _ = cluster_layer(z2)
                    qh, _ = cluster_layer(h)
                    
                    # Cross-entropy between views
                    loss_cluster = -torch.mean(torch.sum(q1 * torch.log(q2 + 1e-7), dim=1))
                    loss_cluster += -torch.mean(torch.sum(qh * torch.log(q1 + 1e-7), dim=1))
                    
                    loss = loss_rec + 0.5 * loss_diff + 0.5 * loss_cluster
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
            
            # Get final embeddings
            ae1.eval(); ae2.eval(); attention.eval()
            full_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
            
            with torch.no_grad():
                for x1, x2, _ in full_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    z1, _ = ae1(x1)
                    z2, _ = ae2(x2)
                    h = attention(z1, z2)
                    self.embeddings_ = h.cpu().numpy()
            
            # Clustering
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10,
                                  random_state=42).fit_predict(self.embeddings_)
            
            print(f"  DCG completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"DCG execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# MRG-UMC: Multi-level Reliable Guidance for Unpaired MVC
# Paper: IEEE TNNLS 2025
# GitHub: https://github.com/LikeXin94/MRG-UMC
# ============================================================

class MRGUMCWrapper(BaseClusteringMethod):
    """
    MRG-UMC: Multi-level Reliable Guidance for Unpaired Multi-view Clustering
    
    This method uses multi-level reliable guidance to handle unpaired/unaligned
    multi-view data through autoencoders with cross-view prediction and
    contrastive learning.
    
    Reference:
        Li et al., "Multi-level Reliable Guidance for Unpaired Multi-view 
        Clustering", IEEE TNNLS 2025
    """
    
    def __init__(self, num_clusters: int, epochs: int = 200,
                 latent_dim: int = 128, batch_size: int = 256,
                 lr: float = 1e-4, lambda_z_norm: float = 1e-4,
                 lambda_inter: float = 1e-4, lambda_cross: float = 1e-4,
                 lambda_guidance: float = 1e4, tau: float = 0.1,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_z_norm = lambda_z_norm
        self.lambda_inter = lambda_inter
        self.lambda_cross = lambda_cross
        self.lambda_guidance = lambda_guidance
        self.tau = tau
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        mrgumc_path = EXTERNAL_PATH / "MRG-UMC"
        
        if not mrgumc_path.exists():
            print(f"MRG-UMC not found. Clone it with:")
            print(f"  git clone https://github.com/LikeXin94/MRG-UMC.git {mrgumc_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        device = kwargs.get('device', self.device)
        
        try:
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            # Define Autoencoder (similar to MRG-UMC)
            class Autoencoder(nn.Module):
                def __init__(self, encoder_dim, activation='relu', batchnorm=True):
                    super().__init__()
                    depth = len(encoder_dim) - 1
                    
                    # Encoder
                    encoder_layers = []
                    for i in range(depth):
                        encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
                        if i < depth - 1:
                            if batchnorm:
                                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                            encoder_layers.append(nn.ReLU())
                    encoder_layers.append(nn.Softmax(dim=1))
                    self.encoder = nn.Sequential(*encoder_layers)
                    
                    # Decoder
                    decoder_dim = list(reversed(encoder_dim))
                    decoder_layers = []
                    for i in range(depth):
                        decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
                        if batchnorm:
                            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                        decoder_layers.append(nn.ReLU())
                    self.decoder = nn.Sequential(*decoder_layers)
                    
                def forward(self, x):
                    z = self.encoder(x)
                    xr = self.decoder(z)
                    return xr, z
            
            class Prediction(nn.Module):
                """Cross-view prediction module"""
                def __init__(self, dim_list, activation='relu', batchnorm=True):
                    super().__init__()
                    depth = len(dim_list) - 1
                    layers = []
                    for i in range(depth):
                        layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                        if batchnorm:
                            layers.append(nn.BatchNorm1d(dim_list[i + 1]))
                        if i < depth - 1:
                            layers.append(nn.ReLU())
                    layers.append(nn.Softmax(dim=1))
                    self.net = nn.Sequential(*layers)
                    
                def forward(self, x):
                    return self.net(x)
            
            # Determine architecture based on view dimensions
            def get_encoder_arch(input_dim, latent_dim):
                if input_dim > 1000:
                    return [input_dim, 1024, 1024, 1024, latent_dim]
                elif input_dim > 500:
                    return [input_dim, 512, 512, latent_dim]
                else:
                    return [input_dim, 256, 256, latent_dim]
            
            # Create autoencoders for each view
            autoencoders = []
            for v in range(n_views):
                arch = get_encoder_arch(view_dims[v], self.latent_dim)
                ae = Autoencoder(arch).to(device)
                autoencoders.append(ae)
            
            # Create prediction modules (for cross-view prediction)
            predictions = nn.ModuleList()
            for v in range(n_views):
                pred = Prediction([self.latent_dim, 256, self.latent_dim]).to(device)
                predictions.append(pred)
            predictions = predictions.to(device)
            
            # Create dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, views_data):
                    self.views = [torch.FloatTensor(StandardScaler().fit_transform(v)) 
                                  for v in views_data]
                    self.n_samples = views_data[0].shape[0]
                def __len__(self):
                    return self.n_samples
                def __getitem__(self, idx):
                    return [v[idx] for v in self.views], idx
            
            dataset = CustomDataset(views)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, n_samples // 2),
                shuffle=True, drop_last=True
            )
            
            # Optimizer
            all_params = []
            for ae in autoencoders:
                all_params.extend(ae.parameters())
            all_params.extend(predictions.parameters())
            optimizer = torch.optim.Adam(all_params, lr=lr)
            
            mse_loss = nn.MSELoss()
            
            # Contrastive loss
            def contrastive_loss(z1, z2, tau=self.tau):
                z1_norm = F.normalize(z1, dim=1)
                z2_norm = F.normalize(z2, dim=1)
                N = z1.size(0)
                
                logits = torch.mm(z1_norm, z2_norm.t()) / tau
                labels = torch.arange(N, device=device)
                loss = F.cross_entropy(logits, labels)
                return loss
            
            print(f"  MRG-UMC Training ({epochs} epochs)...")
            
            # Phase 1: Reconstruction pretraining
            pretrain_epochs = min(50, epochs // 4)
            print(f"    Phase 1: Pretraining ({pretrain_epochs} epochs)...")
            for epoch in range(pretrain_epochs):
                for ae in autoencoders:
                    ae.train()
                
                for xs, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    
                    loss = 0
                    for v in range(n_views):
                        xr, z = autoencoders[v](xs[v])
                        loss += mse_loss(xr, xs[v])
                    
                    loss.backward()
                    optimizer.step()
            
            # Phase 2: Joint training with multi-level guidance
            main_epochs = epochs - pretrain_epochs
            print(f"    Phase 2: Joint training ({main_epochs} epochs)...")
            for epoch in range(main_epochs):
                for ae in autoencoders:
                    ae.train()
                predictions.train()
                
                total_loss = 0
                for xs, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    
                    # Get latent representations
                    zs = []
                    xrs = []
                    for v in range(n_views):
                        xr, z = autoencoders[v](xs[v])
                        zs.append(z)
                        xrs.append(xr)
                    
                    # Reconstruction loss
                    loss_rec = sum(mse_loss(xrs[v], xs[v]) for v in range(n_views))
                    
                    # Z-norm orthogonal constraint
                    loss_z_norm = 0
                    for z in zs:
                        gram = torch.mm(z.t(), z)
                        eye = torch.eye(z.size(1), device=device)
                        loss_z_norm += torch.norm(gram - eye)
                    loss_z_norm = self.lambda_z_norm * loss_z_norm / n_views
                    
                    # Inner-view contrastive (multi-level clustering)
                    loss_inter = 0
                    for v in range(n_views):
                        # Predict cross-view representation
                        z_pred = predictions[v](zs[v])
                        # Contrastive between original and predicted
                        loss_inter += contrastive_loss(zs[v], z_pred)
                    loss_inter = self.lambda_inter * loss_inter / n_views
                    
                    # Cross-view contrastive guidance
                    loss_cross = 0
                    for v1 in range(n_views):
                        for v2 in range(v1 + 1, n_views):
                            # Cross-view prediction
                            z1_pred = predictions[v2](zs[v1])
                            z2_pred = predictions[v1](zs[v2])
                            # Alignment loss
                            loss_cross += contrastive_loss(zs[v1], z2_pred)
                            loss_cross += contrastive_loss(zs[v2], z1_pred)
                    num_pairs = n_views * (n_views - 1) / 2
                    loss_cross = self.lambda_cross * loss_cross / max(num_pairs, 1)
                    
                    # Multi-level reliable guidance (synthesized view alignment)
                    loss_guidance = 0
                    z_avg = sum(zs) / n_views
                    for v in range(n_views):
                        loss_guidance += mse_loss(zs[v], z_avg.detach())
                    loss_guidance = self.lambda_guidance * loss_guidance / n_views
                    
                    loss = loss_rec + loss_z_norm + loss_inter + loss_cross + loss_guidance
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{main_epochs}, Loss: {total_loss/len(data_loader):.4f}")
            
            # Get final embeddings (concatenate all view latents)
            for ae in autoencoders:
                ae.eval()
            full_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
            
            with torch.no_grad():
                for xs, _ in full_loader:
                    xs = [x.to(device) for x in xs]
                    zs = []
                    for v in range(n_views):
                        _, z = autoencoders[v](xs[v])
                        zs.append(z)
                    # Concatenate all view representations
                    self.embeddings_ = torch.cat(zs, dim=1).cpu().numpy()
            
            # Clustering
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10,
                                  random_state=42).fit_predict(self.embeddings_)
            
            print(f"  MRG-UMC completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"MRG-UMC execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# CANDY: Contextually-spectral based correspondence refinery
# Paper: NeurIPS 2024
# GitHub: https://github.com/XLearning-SCU/2024-NeurIPS-CANDY
# ============================================================

class CANDYWrapper(BaseClusteringMethod):
    """
    CANDY: Robust Contrastive Multi-view Clustering against Dual Noisy Correspondence
    
    This method handles noisy correspondences (both false positives and false negatives)
    using contextually-spectral based correspondence refinery with robust affinity
    learning and denoising contrastive loss.
    
    Reference:
        Guo et al., "Robust Contrastive Multi-view Clustering against Dual Noisy 
        Correspondence", NeurIPS 2024
    """
    
    def __init__(self, num_clusters: int, epochs: int = 200,
                 feature_dim: int = 128, batch_size: int = 256,
                 lr: float = 1e-3, temperature: float = 0.07,
                 momentum: float = 0.99, warmup_epochs: int = 20,
                 singular_thresh: float = 0.2, drop_rate: float = 0.5,
                 device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.singular_thresh = singular_thresh
        self.drop_rate = drop_rate
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        candy_path = EXTERNAL_PATH / "2024-NeurIPS-CANDY"
        
        if not candy_path.exists():
            print(f"CANDY not found. Clone it with:")
            print(f"  git clone https://github.com/XLearning-SCU/2024-NeurIPS-CANDY.git {candy_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        # Override with kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        lr = kwargs.get('lr', self.lr)
        device = kwargs.get('device', self.device)
        
        try:
            import copy
            
            # Prepare data
            view_dims = [v.shape[1] for v in views]
            n_views = len(views)
            n_samples = views[0].shape[0]
            
            # CANDY is designed for 2 views, but we can adapt for more
            if n_views < 2:
                print(f"  CANDY requires at least 2 views, got {n_views}. Using fallback.")
                self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
                return self.labels_
            
            # Define FCN encoder (similar to CANDY)
            class FCN(nn.Module):
                def __init__(self, dim_layer, drop_out=0.5):
                    super().__init__()
                    layers = []
                    for i in range(1, len(dim_layer) - 1):
                        layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
                        layers.append(nn.BatchNorm1d(dim_layer[i]))
                        layers.append(nn.ReLU())
                        if drop_out != 0.0 and i != len(dim_layer) - 2:
                            layers.append(nn.Dropout(drop_out))
                    layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
                    layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
                    self.ffn = nn.Sequential(*layers)
                    
                def forward(self, x):
                    return self.ffn(x)
            
            class MLP(nn.Module):
                def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0):
                    super().__init__()
                    dim_out = dim_out or dim_in
                    dim_hidden = int(dim_in * hidden_ratio)
                    self.mlp = nn.Sequential(
                        nn.Linear(dim_in, dim_hidden),
                        nn.ReLU(),
                        nn.Linear(dim_hidden, dim_out)
                    )
                def forward(self, x):
                    return self.mlp(x)
            
            # Build layer dimensions for each view
            layer_dims = []
            for v_dim in view_dims:
                if v_dim > 1000:
                    dims = [v_dim, 1024, 512, self.feature_dim]
                elif v_dim > 500:
                    dims = [v_dim, 512, 256, self.feature_dim]
                else:
                    dims = [v_dim, 256, 128, self.feature_dim]
                layer_dims.append(dims)
            
            # CANDY Model
            class CANDYModel(nn.Module):
                def __init__(self, n_views, layer_dims, temperature, drop_rate):
                    super().__init__()
                    self.n_views = n_views
                    
                    self.online_encoder = nn.ModuleList(
                        [FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)]
                    )
                    self.target_encoder = copy.deepcopy(self.online_encoder)
                    
                    for param_q, param_k in zip(
                        self.online_encoder.parameters(), self.target_encoder.parameters()
                    ):
                        param_k.data.copy_(param_q.data)
                        param_k.requires_grad = False
                    
                    self.cross_view_decoder = nn.ModuleList(
                        [MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)]
                    )
                    
                    self.temperature = temperature
                    self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]
                
                @torch.no_grad()
                def update_target(self, momentum):
                    for i in range(self.n_views):
                        for param_o, param_t in zip(
                            self.online_encoder[i].parameters(),
                            self.target_encoder[i].parameters()
                        ):
                            param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)
                
                @torch.no_grad()
                def robust_affinity(self, z1, z2, t=0.07):
                    """Compute robust affinity matrices"""
                    G_intra, G_inter = [], []
                    z1 = [F.normalize(z1[i], dim=1) for i in range(len(z1))]
                    z2 = [F.normalize(z2[i], dim=1) for i in range(len(z2))]
                    
                    for i in range(len(z1)):
                        for j in range(len(z2)):
                            if i == j:
                                G = (2 - 2 * (z2[i] @ z2[j].t())).clamp(min=0.0)
                                G = torch.exp(-G / t)
                                G[torch.eye(G.shape[0], device=G.device) > 0] = 1.0
                                G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                                G_intra.append(G)
                            else:
                                G = (2 - 2 * (z1[i] @ z2[j].t())).clamp(min=0.0)
                                G = torch.exp(-G / t)
                                diag_mask = torch.eye(G.shape[0], device=G.device) > 0
                                G[diag_mask] = G[diag_mask] / G.diag().max().clamp_min(1e-7).detach()
                                G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                                G_inter.append(G)
                    
                    return G_intra, G_inter
                
                def forward(self, data, momentum, warm_up, singular_thresh):
                    self.update_target(momentum)
                    
                    z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
                    p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
                    z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
                    
                    if warm_up:
                        N = z[0].shape[0]
                        mp_intra = [torch.eye(N, device=z[0].device) for _ in range(self.n_views)]
                        mp_inter = mp_intra
                    else:
                        mp_intra, mp_inter = self.robust_affinity(p, z_t, self.temperature)
                    
                    # Contrastive losses
                    cc_loss, id_loss = 0.0, 0.0
                    
                    for i in range(self.n_views):
                        for j in range(self.n_views):
                            if i == j:
                                # Intra-view identity loss
                                id_loss += self._denoise_contrastive(
                                    z[i], z_t[i],
                                    mp_intra[i].mm(mp_intra[j].t()),
                                    singular_thresh, enable_denoise=False
                                )
                            else:
                                # Cross-view correspondence loss
                                pos_mask = mp_inter[i % len(mp_inter)].mm(mp_intra[j].t())
                                pos_mask = pos_mask + 0.2 * torch.eye(
                                    pos_mask.shape[0], device=pos_mask.device
                                )
                                cc_loss += self._denoise_contrastive(
                                    p[i], z_t[j], pos_mask, singular_thresh, enable_denoise=True
                                )
                    
                    cc_loss = cc_loss / self.n_views
                    id_loss = id_loss / self.n_views
                    
                    return cc_loss + id_loss
                
                def _denoise_contrastive(self, query, key, mask_pos, singular_thresh, enable_denoise=True):
                    """Denoising contrastive loss"""
                    query = F.normalize(query, dim=1)
                    key = F.normalize(key, dim=1)
                    
                    similarity = (query @ key.t() / self.temperature).softmax(1)
                    logp = -similarity.log()
                    
                    L = mask_pos
                    if enable_denoise:
                        # SVD-based denoising
                        try:
                            U, S, Vh = torch.linalg.svd(L)
                            S[S < singular_thresh] = 0
                            L = U @ torch.diag(S) @ Vh
                        except:
                            pass  # Keep L unchanged if SVD fails
                    
                    L = L / L.sum(dim=1, keepdim=True).clamp_min(1e-7)
                    loss = (L * logp).mean()
                    return loss
                
                @torch.no_grad()
                def extract_feature(self, data):
                    """Extract features for clustering"""
                    z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
                    z = [F.normalize(z[i], dim=1) for i in range(self.n_views)]
                    return z
            
            # Create dataset
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, views_data):
                    self.views = [torch.FloatTensor(StandardScaler().fit_transform(v)) 
                                  for v in views_data]
                    self.n_samples = views_data[0].shape[0]
                def __len__(self):
                    return self.n_samples
                def __getitem__(self, idx):
                    return [v[idx] for v in self.views], idx
            
            dataset = CustomDataset(views)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, n_samples // 2),
                shuffle=True, drop_last=True
            )
            
            # Initialize model
            model = CANDYModel(n_views, layer_dims, self.temperature, self.drop_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            print(f"  CANDY Training ({epochs} epochs)...")
            
            for epoch in range(epochs):
                model.train()
                warm_up = epoch < self.warmup_epochs
                
                total_loss = 0
                for xs, _ in data_loader:
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    
                    loss = model(xs, self.momentum, warm_up, self.singular_thresh)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
            
            # Get final embeddings
            model.eval()
            full_loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
            
            with torch.no_grad():
                for xs, _ in full_loader:
                    xs = [x.to(device) for x in xs]
                    zs = model.extract_feature(xs)
                    # Concatenate all view features
                    self.embeddings_ = torch.cat(zs, dim=1).cpu().numpy()
            
            # Clustering
            self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10,
                                  random_state=42).fit_predict(self.embeddings_)
            
            print(f"  CANDY completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
            
        except Exception as e:
            import traceback
            print(f"CANDY execution failed: {e}")
            traceback.print_exc()
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


# ============================================================
# Registry: Get all available external methods
# ============================================================

def get_external_baselines(
    view_dims: List[int],
    num_clusters: int,
    device: str = 'cuda'
) -> Dict[str, BaseClusteringMethod]:
    """
    Get all available external baseline methods.
    
    Only methods whose code exists in external_methods/ will be included.
    
    Args:
        view_dims: Dimensions of each view
        num_clusters: Number of clusters
        device: PyTorch device
    
    Returns:
        Dictionary mapping method names to wrapper instances
    """
    external_methods = {}
    
    # Define all supported external methods
    method_configs = [
        ("MFLVC", "MFLVC (CVPR22)", MFLVCWrapper),
        ("SURE", "SURE (TPAMI22)", SUREWrapper),
        ("DealMVC", "DealMVC (CVPR23)", DealMVCWrapper),
        ("COMPLETER", "COMPLETER (CVPR21)", COMPLETERWrapper),
        ("GCFAggMVC", "GCFAggMVC (CVPR23)", GCFAggMVCWrapper),
        ("2025-AAAI-DCG", "DCG (AAAI25)", DCGWrapper),
        ("MRG-UMC", "MRG-UMC (TNNLS25)", MRGUMCWrapper),
        ("2024-NeurIPS-CANDY", "CANDY (NeurIPS24)", CANDYWrapper),
    ]
    
    for folder_name, display_name, wrapper_class in method_configs:
        method_path = EXTERNAL_PATH / folder_name
        if method_path.exists():
            external_methods[display_name] = wrapper_class(
                num_clusters=num_clusters, 
                device=device
            )
            print(f"  Found external method: {display_name}")
    
    return external_methods


def list_available_external_methods() -> List[str]:
    """List all external methods that are available (code exists)"""
    available = []
    
    method_folders = ["MFLVC", "SURE", "DealMVC", "COMPLETER", "GCFAggMVC", "2025-AAAI-DCG", "MRG-UMC", "2024-NeurIPS-CANDY"]
    
    for folder in method_folders:
        if (EXTERNAL_PATH / folder).exists():
            available.append(folder)
    
    return available


def list_missing_external_methods() -> Dict[str, str]:
    """List external methods that are not installed, with clone commands"""
    missing = {}
    
    method_repos = {
        "MFLVC": "https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git",
        "SURE": "https://github.com/XLearning-SCU/2022-NeurIPS-SURE.git",
        "DealMVC": "https://github.com/SubmissionsIn/DealMVC.git",
        "COMPLETER": "https://github.com/XLearning-SCU/2021-CVPR-Completer.git",
        "GCFAggMVC": "https://github.com/Galaxy922/GCFAggMVC.git",
        "2025-AAAI-DCG": "https://github.com/zhangyuanyang21/2025-AAAI-DCG.git",
        "MRG-UMC": "https://github.com/LikeXin94/MRG-UMC.git",
        "2024-NeurIPS-CANDY": "https://github.com/XLearning-SCU/2024-NeurIPS-CANDY.git",
    }
    
    for name, repo in method_repos.items():
        path = EXTERNAL_PATH / name
        if not path.exists():
            missing[name] = f"git clone {repo} {path}"
    
    return missing


# ============================================================
# CLI for managing external methods
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage external MVC methods")
    parser.add_argument("--list", action="store_true", help="List available methods")
    parser.add_argument("--missing", action="store_true", help="Show missing methods with clone commands")
    parser.add_argument("--clone-all", action="store_true", help="Clone all missing methods")
    
    args = parser.parse_args()
    
    if args.list:
        available = list_available_external_methods()
        print("Available external methods:")
        for m in available:
            print(f"  ✓ {m}")
        if not available:
            print("  (none)")
    
    if args.missing:
        missing = list_missing_external_methods()
        print("\nMissing external methods (run these commands to install):")
        for name, cmd in missing.items():
            print(f"\n  # {name}")
            print(f"  {cmd}")
    
    if args.clone_all:
        import subprocess
        missing = list_missing_external_methods()
        EXTERNAL_PATH.mkdir(parents=True, exist_ok=True)
        
        for name, cmd in missing.items():
            print(f"\nCloning {name}...")
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"  ✓ {name} cloned successfully")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to clone {name}: {e}")
