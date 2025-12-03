"""
分析多视图 AE 为什么失败
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
print("分析多视图 AE 为什么失败")
print("="*60)


def test_view_specific_clustering():
    """测试每个视图单独的聚类能力"""
    print("\n1. 每个视图单独聚类:")
    for i, v in enumerate(dataset.views):
        X = v.numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(X)
        acc = evaluate_clustering(labels, pred, X)['acc']
        print(f"   视图 {i}: ACC = {acc:.4f}")


def test_shared_encoder():
    """测试使用共享编码器"""
    print("\n2. 测试共享编码器 vs 独立编码器:")
    
    class SharedEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        
        def forward(self, views):
            latents = [self.encoder(v) for v in views]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [self.decoder(z) for z in latents]
            return latents, consensus, recons
    
    # 所有视图维度相同，使用共享编码器
    model = SharedEncoder(100, 256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader:
            views_batch = batch['views']
            
            optimizer.zero_grad()
            latents, consensus, recons = model(views_batch)
            
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
            print(f"   共享编码器 Epoch {epoch+1}: ACC = {acc:.4f}")


def test_independent_encoders_single_output():
    """测试独立编码器但只使用单个视图的潜在表示"""
    print("\n3. 独立编码器，但只使用 view 0 的潜在表示:")
    
    class IndependentEncoders(nn.Module):
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
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, recons
    
    model = IndependentEncoders(view_dims, 256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_latent0 = []
        
        for batch in dataloader:
            views_batch = batch['views']
            
            optimizer.zero_grad()
            latents, recons = model(views_batch)
            
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_latent0.append(latents[0].detach())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_latent0, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"   只用 view 0 Epoch {epoch+1}: ACC = {acc:.4f}")


def test_mean_vs_concat():
    """测试平均融合 vs 拼接融合"""
    print("\n4. 测试平均融合 vs 拼接融合:")
    
    class MultiViewAE(nn.Module):
        def __init__(self, view_dims, hidden_dim, latent_dim, fusion='mean'):
            super().__init__()
            self.fusion = fusion
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
            
            if fusion == 'concat':
                self.fusion_layer = nn.Linear(latent_dim * len(view_dims), latent_dim)
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            
            if self.fusion == 'mean':
                consensus = torch.stack(latents, dim=0).mean(dim=0)
            elif self.fusion == 'concat':
                consensus = self.fusion_layer(torch.cat(latents, dim=-1))
            
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    for fusion in ['mean', 'concat']:
        model = MultiViewAE(view_dims, 256, 64, fusion=fusion)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            model.train()
            all_embeddings = []
            
            for batch in dataloader:
                views_batch = batch['views']
                
                optimizer.zero_grad()
                latents, consensus, recons = model(views_batch)
                
                recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
                
                loss = recon_loss
                loss.backward()
                optimizer.step()
                
                all_embeddings.append(consensus.detach())
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(embeddings)
        acc = evaluate_clustering(labels, pred, embeddings)['acc']
        print(f"   {fusion} 融合: ACC = {acc:.4f}")


def test_wider_network():
    """测试更宽的网络"""
    print("\n5. 测试更宽的网络:")
    
    class WiderAE(nn.Module):
        def __init__(self, view_dims, hidden_dim, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                ) for dim in view_dims
            ])
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                ) for dim in view_dims
            ])
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    for hidden_dim in [256, 512, 1024]:
        for latent_dim in [64, 128, 256]:
            model = WiderAE(view_dims, hidden_dim, latent_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            for epoch in range(100):
                model.train()
                all_embeddings = []
                
                for batch in dataloader:
                    views_batch = batch['views']
                    
                    optimizer.zero_grad()
                    latents, consensus, recons = model(views_batch)
                    
                    recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
                    
                    loss = recon_loss
                    loss.backward()
                    optimizer.step()
                    
                    all_embeddings.append(consensus.detach())
            
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"   hidden={hidden_dim}, latent={latent_dim}: ACC = {acc:.4f}")


def test_latent_alignment():
    """测试添加潜在空间对齐损失"""
    print("\n6. 测试添加潜在空间对齐损失:")
    
    class AlignedAE(nn.Module):
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
    
    model = AlignedAE(view_dims, 256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        all_embeddings = []
        
        for batch in dataloader:
            views_batch = batch['views']
            
            optimizer.zero_grad()
            latents, consensus, recons = model(views_batch)
            
            # 重建损失
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            # 对齐损失：使所有视图的潜在表示尽可能接近
            align_loss = 0
            for z in latents:
                align_loss += F.mse_loss(z, consensus.detach())  # 使每个视图靠近 consensus
            align_loss /= len(latents)
            
            loss = recon_loss + 0.5 * align_loss
            loss.backward()
            optimizer.step()
            
            all_embeddings.append(consensus.detach())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"   带对齐损失 Epoch {epoch+1}: ACC = {acc:.4f}")


def test_bottleneck_dimensions():
    """测试不同的瓶颈维度"""
    print("\n7. 测试更大的潜在维度（接近原始维度）:")
    
    class LargeLatentAE(nn.Module):
        def __init__(self, view_dims, latent_dim):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, latent_dim),
                    nn.ReLU()
                ) for dim in view_dims
            ])
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, dim)
                ) for dim in view_dims
            ])
        
        def forward(self, views):
            latents = [enc(v) for enc, v in zip(self.encoders, views)]
            consensus = torch.stack(latents, dim=0).mean(dim=0)
            recons = [dec(z) for dec, z in zip(self.decoders, latents)]
            return latents, consensus, recons
    
    for latent_dim in [100, 200, 300]:
        model = LargeLatentAE(view_dims, latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            model.train()
            all_embeddings = []
            
            for batch in dataloader:
                views_batch = batch['views']
                
                optimizer.zero_grad()
                latents, consensus, recons = model(views_batch)
                
                recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
                
                loss = recon_loss
                loss.backward()
                optimizer.step()
                
                all_embeddings.append(consensus.detach())
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(embeddings)
        acc = evaluate_clustering(labels, pred, embeddings)['acc']
        print(f"   latent_dim={latent_dim}: ACC = {acc:.4f}")


if __name__ == "__main__":
    test_view_specific_clustering()
    test_shared_encoder()
    test_independent_encoders_single_output()
    test_mean_vs_concat()
    # test_wider_network()  # 耗时较长
    test_latent_alignment()
    test_bottleneck_dimensions()
