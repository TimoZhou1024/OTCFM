"""
对比之前失败的代码 vs 现在成功的代码
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
views = data['views']
labels = data['labels']
view_dims = [v.shape[1] for v in views]
n_clusters = len(np.unique(labels))

dataset = MultiViewDataset(views, labels)
dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)

print("="*60)
print("对比不同训练设置")
print("="*60)


def test_with_shuffle_true():
    """之前失败的设置：shuffle=True"""
    print("\n1. shuffle=True:")
    
    class MultiViewAE(nn.Module):
        def __init__(self, view_dims, hidden_dim, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                ) for dim in view_dims
            ])
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                ) for dim in view_dims
            ])
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    torch.manual_seed(42)  # 重置种子
    np.random.seed(42)
    
    model = MultiViewAE(view_dims, 256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 重新创建 dataloader 以重置其状态
    dataloader_new = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(100):
        model.train()
        all_embeddings = []
        all_indices = []
        
        for batch in dataloader_new:
            views_batch = batch['views']
            indices = batch['indices'].numpy()
            
            optimizer.zero_grad()
            latents, consensus, recons = model(views_batch)
            
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            all_embeddings.append(consensus.detach())
            all_indices.extend(indices.tolist())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            
            # 按 indices 排序回原始顺序
            order = np.argsort(all_indices)
            embeddings_sorted = embeddings[order]
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings_sorted)
            acc = evaluate_clustering(labels, pred, embeddings_sorted)['acc']
            print(f"   Epoch {epoch+1}: ACC = {acc:.4f}")
    
    return model


def test_original_failing_code():
    """复现原来失败的代码"""
    print("\n2. 复现原来失败的代码（来自 analyze_multiview_ae.py）:")
    
    class MultiViewAE(nn.Module):
        def __init__(self, view_dims, hidden_dims, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            
            for dim in view_dims:
                # Encoder
                enc = nn.Sequential(
                    nn.Linear(dim, hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], latent_dim)
                )
                self.encoders.append(enc)
                
                # Decoder
                dec = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[0], dim)
                )
                self.decoders.append(dec)
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            # 平均融合
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = MultiViewAE(view_dims, [256, 128], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader_new = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader_new:
            views_batch = batch['views']
            
            optimizer.zero_grad()
            latents, consensus, recons = model(views_batch)
            
            # 重建损失
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_embeddings.append(consensus.detach())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"   Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}")


def test_deeper_network():
    """测试更深的网络（与原来一样 hidden_dims=[256,128]）"""
    print("\n3. 更深的网络 [256,128] + 无 shuffle:")
    
    class MultiViewAE(nn.Module):
        def __init__(self, view_dims, hidden_dims, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            
            for dim in view_dims:
                enc = nn.Sequential(
                    nn.Linear(dim, hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], latent_dim)
                )
                self.encoders.append(enc)
                
                dec = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[0], dim)
                )
                self.decoders.append(dec)
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = MultiViewAE(view_dims, [256, 128], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader_no_shuffle = create_dataloader(dataset, batch_size=256, shuffle=False)
    
    for epoch in range(100):
        model.train()
        all_embeddings = []
        
        for batch in dataloader_no_shuffle:
            views_batch = batch['views']
            
            optimizer.zero_grad()
            latents, consensus, recons = model(views_batch)
            
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            all_embeddings.append(consensus.detach())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"   Epoch {epoch+1}: ACC = {acc:.4f}")


def test_evaluation_method():
    """关键差异：评估时收集 embeddings 的方式"""
    print("\n4. 关键差异：评估时 embeddings 的顺序")
    print("   当使用 shuffle=True 时，每个 epoch 的 embeddings 顺序不同！")
    print("   如果不按照 indices 排序回原始顺序，ACC 计算会出错！")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    class SimpleAE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    model = SimpleAE(view_dims[0], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader_shuffle = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(10):
        model.train()
        all_embeddings = []
        all_indices = []
        
        for batch in dataloader_shuffle:
            v = batch['views'][0]  # 只用第一个视图
            indices = batch['indices'].numpy()
            
            optimizer.zero_grad()
            z, recon = model(v)
            loss = F.mse_loss(recon, v)
            loss.backward()
            optimizer.step()
            
            all_embeddings.append(z.detach())
            all_indices.extend(indices.tolist())
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        # 错误方式：不排序
        kmeans1 = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred1 = kmeans1.fit_predict(embeddings)
        acc_wrong = evaluate_clustering(labels, pred1, embeddings)['acc']
        
        # 正确方式：按 indices 排序
        order = np.argsort(all_indices)
        embeddings_sorted = embeddings[order]
        kmeans2 = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred2 = kmeans2.fit_predict(embeddings_sorted)
        acc_correct = evaluate_clustering(labels, pred2, embeddings_sorted)['acc']
        
        print(f"   Epoch {epoch+1}: ACC(未排序)={acc_wrong:.4f}, ACC(已排序)={acc_correct:.4f}")


if __name__ == "__main__":
    test_with_shuffle_true()
    test_original_failing_code()
    test_deeper_network()
    test_evaluation_method()
