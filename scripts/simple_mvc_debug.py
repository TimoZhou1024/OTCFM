"""
简化版 OT-CFM 用于调试
核心思想：先确保编码器保留聚类结构，再添加复杂组件
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def cluster_acc(y_true, y_pred):
    """计算聚类准确率"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(D)]) * 1.0 / y_pred.size


class SimpleEncoder(nn.Module):
    """简单编码器，保持信息"""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    """简单解码器"""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)


class SimpleMVC(nn.Module):
    """
    简化版多视图聚类模型
    关键：使用 DEC 风格的聚类损失
    """
    def __init__(self, view_dims: List[int], latent_dim: int, num_clusters: int):
        super().__init__()
        
        self.num_views = len(view_dims)
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        
        # 编码器和解码器
        self.encoders = nn.ModuleList([
            SimpleEncoder(dim, latent_dim) for dim in view_dims
        ])
        self.decoders = nn.ModuleList([
            SimpleDecoder(latent_dim, dim) for dim in view_dims
        ])
        
        # 聚类中心
        self.centroids = nn.Parameter(torch.randn(num_clusters, latent_dim) * 0.1)
        
        # DEC alpha 参数
        self.alpha = 1.0
    
    def encode(self, views: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """编码并融合"""
        latents = [enc(v) for enc, v in zip(self.encoders, views)]
        # 简单平均融合
        consensus = torch.stack(latents, dim=0).mean(dim=0)
        return latents, consensus
    
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """解码"""
        return [dec(z) for dec, z in zip(self.decoders, latents)]
    
    def get_cluster_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        使用 Student's t 分布计算软分配 (DEC 风格)
        """
        # 计算到聚类中心的距离
        # z: [B, D], centroids: [K, D]
        dist = torch.cdist(z, self.centroids)  # [B, K]
        
        # Student's t 分布
        q = 1.0 / (1.0 + dist ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """计算目标分布 P"""
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p
    
    def forward(self, views: List[torch.Tensor]) -> Dict:
        latents, consensus = self.encode(views)
        recons = self.decode(latents)
        q = self.get_cluster_prob(consensus)
        
        return {
            'latents': latents,
            'consensus': consensus,
            'reconstructions': recons,
            'q': q,
            'assignments': q.argmax(dim=1)
        }
    
    def init_centroids(self, dataloader, device):
        """使用 KMeans 初始化聚类中心"""
        self.eval()
        all_z = []
        
        with torch.no_grad():
            for batch in dataloader:
                views = [v.to(device) for v in batch['views']]
                _, consensus = self.encode(views)
                all_z.append(consensus.cpu())
        
        all_z = torch.cat(all_z, dim=0).numpy()
        
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=42)
        kmeans.fit(all_z)
        
        self.centroids.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        
        return all_z


def train_simple_mvc(views: List[np.ndarray], labels: np.ndarray, device: str = 'cpu'):
    """训练简化版模型"""
    from otcfm.datasets import MultiViewDataset, create_dataloader
    
    # 创建数据集
    dataset = MultiViewDataset(views=views, labels=labels)
    dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    view_dims = [v.shape[1] for v in views]
    num_clusters = len(np.unique(labels))
    
    # 创建模型
    model = SimpleMVC(view_dims, latent_dim=128, num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("="*60)
    print("Phase 1: 预训练编码器-解码器")
    print("="*60)
    
    # Phase 1: 预训练 (只用重建损失)
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch_views = [v.to(device) for v in batch['views']]
            
            optimizer.zero_grad()
            outputs = model(batch_views)
            
            # 重建损失
            recon_loss = sum(
                F.mse_loss(r, v) 
                for r, v in zip(outputs['reconstructions'], batch_views)
            ) / len(batch_views)
            
            recon_loss.backward()
            optimizer.step()
            total_loss += recon_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Recon Loss = {total_loss/len(dataloader):.4f}")
    
    # 检查预训练后的表示质量
    print("\n检查预训练后的表示:")
    all_z = model.init_centroids(dataloader, device)
    
    # 在嵌入上做 KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    pred = kmeans.fit_predict(all_z)
    acc = cluster_acc(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    print(f"  预训练嵌入 KMeans: ACC={acc:.4f}, NMI={nmi:.4f}")
    
    print("\n" + "="*60)
    print("Phase 2: 联合训练 (重建 + 聚类)")
    print("="*60)
    
    # Phase 2: 联合训练
    best_acc = 0
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        # 每 10 个 epoch 更新目标分布
        if epoch % 10 == 0:
            model.eval()
            all_q = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_views = [v.to(device) for v in batch['views']]
                    outputs = model(batch_views)
                    all_q.append(outputs['q'])
            all_q = torch.cat(all_q, dim=0)
            target_p = model.target_distribution(all_q)
            model.train()
        
        batch_idx = 0
        for batch in dataloader:
            batch_views = [v.to(device) for v in batch['views']]
            batch_size = batch_views[0].shape[0]
            
            optimizer.zero_grad()
            outputs = model(batch_views)
            
            # 重建损失
            recon_loss = sum(
                F.mse_loss(r, v) 
                for r, v in zip(outputs['reconstructions'], batch_views)
            ) / len(batch_views)
            
            # KL 聚类损失
            start_idx = batch_idx * 256
            end_idx = min(start_idx + batch_size, len(target_p))
            p_batch = target_p[start_idx:end_idx].to(device)
            q = outputs['q']
            
            # 确保尺寸匹配
            if q.shape[0] != p_batch.shape[0]:
                p_batch = p_batch[:q.shape[0]]
            
            kl_loss = F.kl_div(q.log(), p_batch, reduction='batchmean')
            
            # 对比损失 (鼓励同一样本的不同视图表示接近)
            latents = outputs['latents']
            contrastive_loss = 0
            for i in range(len(latents)):
                for j in range(i + 1, len(latents)):
                    z_i = F.normalize(latents[i], dim=-1)
                    z_j = F.normalize(latents[j], dim=-1)
                    contrastive_loss += 1 - (z_i * z_j).sum(dim=-1).mean()
            contrastive_loss /= (len(latents) * (len(latents) - 1) / 2)
            
            # 总损失
            loss = 0.1 * recon_loss + 1.0 * kl_loss + 0.5 * contrastive_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_idx += 1
        
        # 评估
        model.eval()
        all_pred = []
        with torch.no_grad():
            for batch in dataloader:
                batch_views = [v.to(device) for v in batch['views']]
                outputs = model(batch_views)
                all_pred.append(outputs['assignments'].cpu())
        
        all_pred = torch.cat(all_pred, dim=0).numpy()
        acc = cluster_acc(labels, all_pred)
        nmi = normalized_mutual_info_score(labels, all_pred)
        
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
                  f"ACC={acc:.4f}, NMI={nmi:.4f}, Best ACC={best_acc:.4f}")
    
    print(f"\n最终结果: Best ACC = {best_acc:.4f}")
    return best_acc


def main():
    from otcfm.datasets import create_synthetic_multiview
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    views, labels = create_synthetic_multiview(
        n_samples=1000,
        n_clusters=10,
        n_views=3,
        view_dims=[100, 100, 100],
        noise_level=0.1,
        seed=42
    )
    
    print(f"数据: {len(labels)} 样本, {len(views)} 视图")
    
    # 先测试原始数据
    X_concat = np.concatenate(views, axis=1)
    kmeans = KMeans(n_clusters=10, n_init=20, random_state=42)
    pred = kmeans.fit_predict(X_concat)
    acc = cluster_acc(labels, pred)
    print(f"原始特征 KMeans: ACC={acc:.4f}")
    
    # 训练简化模型
    train_simple_mvc(views, labels, device)


if __name__ == "__main__":
    main()
