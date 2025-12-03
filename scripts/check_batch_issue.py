"""
检查是否是批处理导致的问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from otcfm.datasets import load_synthetic, MultiViewDataset, create_dataloader
from otcfm.metrics import evaluate_clustering

torch.manual_seed(42)
np.random.seed(42)

# 加载数据
data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=0.1)
views_np = data['views']
labels = data['labels']
view_dims = [v.shape[1] for v in views_np]
n_clusters = len(np.unique(labels))

# 创建 tensor 版本（不使用 DataLoader）
views_tensor = [torch.FloatTensor(v) for v in views_np]

print("="*60)
print("检查批处理是否是问题根源")
print("="*60)


def test_full_batch_training():
    """测试全批次训练（不使用 DataLoader）"""
    print("\n1. 全批次训练（所有数据一起）:")
    
    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    # 对每个视图训练一个编码器
    encoders = [SimpleEncoder(dim, 64) for dim in view_dims]
    optimizers = [torch.optim.Adam(enc.parameters(), lr=1e-3) for enc in encoders]
    
    for epoch in range(100):
        total_loss = 0
        latents_all = []
        
        for enc, opt, v in zip(encoders, optimizers, views_tensor):
            enc.train()
            opt.zero_grad()
            
            z, recon = enc(v)
            loss = F.mse_loss(recon, v)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            latents_all.append(z.detach())
        
        if (epoch + 1) % 25 == 0:
            # 融合潜在表示
            consensus = torch.stack(latents_all, dim=0).mean(dim=0).numpy()
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(consensus)
            acc = evaluate_clustering(labels, pred, consensus)['acc']
            print(f"   Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}")
    
    # 最终使用单个视图的潜在表示
    for i, enc in enumerate(encoders):
        enc.eval()
        z, _ = enc(views_tensor[i])
        z = z.detach().numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   单独 View {i} 潜在表示: ACC = {acc:.4f}")


def test_untrained_encoder():
    """测试未训练的编码器"""
    print("\n2. 未训练的编码器（随机初始化）:")
    
    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
        
        def forward(self, x):
            return self.encoder(x)
    
    for latent_dim in [64, 100, 200, 300]:
        enc = SimpleEncoder(view_dims[0], latent_dim)
        enc.eval()
        
        with torch.no_grad():
            z = enc(views_tensor[0]).numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   latent_dim={latent_dim}: ACC = {acc:.4f}")


def test_linear_projection():
    """测试只用线性投影"""
    print("\n3. 只用线性投影（无激活函数）:")
    
    for latent_dim in [64, 100, 200]:
        proj = nn.Linear(view_dims[0], latent_dim)
        proj.eval()
        
        with torch.no_grad():
            z = proj(views_tensor[0]).numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   latent_dim={latent_dim}: ACC = {acc:.4f}")


def test_overfit_single_batch():
    """测试过拟合单个批次"""
    print("\n4. 过拟合单个批次:")
    
    # 使用整个数据作为单个批次
    class MultiViewEncoder(nn.Module):
        def __init__(self, view_dims, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim)
                ) for dim in view_dims
            ])
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, dim)
                ) for dim in view_dims
            ])
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    model = MultiViewEncoder(view_dims, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        latents, consensus, recons = model(views_tensor)
        
        loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_tensor)) / len(views_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                _, consensus, _ = model(views_tensor)
            
            z = consensus.numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(z)
            acc = evaluate_clustering(labels, pred, z)['acc']
            print(f"   Epoch {epoch+1}: loss={loss.item():.6f}, ACC={acc:.4f}")


def test_pca_baseline():
    """测试 PCA 作为基线"""
    print("\n5. PCA 基线:")
    from sklearn.decomposition import PCA
    
    # 对合并数据做 PCA
    X = np.concatenate(views_np, axis=1)
    
    for n_components in [32, 64, 128]:
        pca = PCA(n_components=n_components)
        z = pca.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   n_components={n_components}: ACC = {acc:.4f}")
    
    # 对每个视图分别做 PCA
    print("\n   每个视图分别 PCA 后融合:")
    for n_components in [32, 64]:
        latents = []
        for v in views_np:
            pca = PCA(n_components=n_components)
            z = pca.fit_transform(v)
            latents.append(z)
        
        consensus = np.stack(latents, axis=0).mean(axis=0)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(consensus)
        acc = evaluate_clustering(labels, pred, consensus)['acc']
        print(f"      n_components={n_components}: ACC = {acc:.4f}")


def analyze_embedding_distribution():
    """分析嵌入分布"""
    print("\n6. 分析嵌入分布:")
    
    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    model = SimpleEncoder(view_dims[0], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练前
    model.eval()
    with torch.no_grad():
        z_before, _ = model(views_tensor[0])
    z_before = z_before.numpy()
    
    print(f"   训练前: mean={z_before.mean():.4f}, std={z_before.std():.4f}")
    
    # 检查簇心距离
    centers_before = np.array([z_before[labels==i].mean(axis=0) for i in range(n_clusters)])
    from scipy.spatial.distance import cdist
    dist_before = cdist(centers_before, centers_before)
    print(f"   训练前簇心间距: {dist_before[dist_before>0].mean():.4f}")
    
    # 训练
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        z, recon = model(views_tensor[0])
        loss = F.mse_loss(recon, views_tensor[0])
        loss.backward()
        optimizer.step()
    
    # 训练后
    model.eval()
    with torch.no_grad():
        z_after, _ = model(views_tensor[0])
    z_after = z_after.numpy()
    
    print(f"   训练后: mean={z_after.mean():.4f}, std={z_after.std():.4f}")
    
    centers_after = np.array([z_after[labels==i].mean(axis=0) for i in range(n_clusters)])
    dist_after = cdist(centers_after, centers_after)
    print(f"   训练后簇心间距: {dist_after[dist_after>0].mean():.4f}")
    
    # 簇内方差
    var_before = np.mean([z_before[labels==i].std() for i in range(n_clusters)])
    var_after = np.mean([z_after[labels==i].std() for i in range(n_clusters)])
    print(f"   训练前簇内标准差: {var_before:.4f}")
    print(f"   训练后簇内标准差: {var_after:.4f}")


if __name__ == "__main__":
    test_full_batch_training()
    test_untrained_encoder()
    test_linear_projection()
    test_overfit_single_batch()
    test_pca_baseline()
    analyze_embedding_distribution()
