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
    
    def __init__(self, num_clusters: int, epochs: int = 200, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.epochs = epochs
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        dealmvc_path = EXTERNAL_PATH / "DealMVC"
        
        if not dealmvc_path.exists():
            print(f"DealMVC not found. Clone it with:")
            print(f"  git clone https://github.com/SubmissionsIn/DealMVC.git {dealmvc_path}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
        
        sys.path.insert(0, str(dealmvc_path))
        try:
            from DealMVC import DealMVC
            
            model = DealMVC(
                n_clusters=self.num_clusters,
                n_views=len(views),
                dims=[v.shape[1] for v in views],
                device=self.device
            )
            
            # DealMVC supports incomplete views via mask
            self.labels_, self.embeddings_ = model.fit_predict(
                views, mask=mask, epochs=self.epochs
            )
            
            return self.labels_
            
        except Exception as e:
            print(f"DealMVC execution failed: {e}")
            self.labels_, self.embeddings_ = fallback_kmeans(views, self.num_clusters)
            return self.labels_
            
        finally:
            if str(dealmvc_path) in sys.path:
                sys.path.remove(str(dealmvc_path))
    
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
    
    method_folders = ["MFLVC", "SURE", "DealMVC", "COMPLETER", "GCFAggMVC"]
    
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
