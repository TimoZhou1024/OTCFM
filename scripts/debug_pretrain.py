"""
深入调试预训练阶段
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
from otcfm.ot_cfm import OTCFM
from otcfm.metrics import evaluate_clustering

torch.manual_seed(42)
np.random.seed(42)

# 加载数据
data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=0.1)
views = data['views']
labels = data['labels']
view_dims = [v.shape[1] for v in views]
n_clusters = len(np.unique(labels))

# 创建 dataset
dataset = MultiViewDataset(views, labels)
dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)

device = 'cpu'

print("="*60)
print("深入调试预训练阶段")
print("="*60)

# 获取一个 batch 检查数据
batch = next(iter(dataloader))
print(f"\n数据检查:")
for i, v in enumerate(batch['views']):
    print(f"  视图 {i}: shape={v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")

# 直接在标准化后的数据上聚类
print(f"\n在 Dataset 中的数据上聚类:")
X_dataset = torch.cat([dataset.views[i] for i in range(len(dataset.views))], dim=1).numpy()
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pred = kmeans.fit_predict(X_dataset)
acc = evaluate_clustering(labels, pred, X_dataset)['acc']
print(f"  ACC = {acc:.4f}")


def test_simple_autoencoder():
    """测试一个简单的自编码器"""
    print("\n" + "="*60)
    print("测试简单自编码器")
    print("="*60)
    
    class SimpleAE(nn.Module):
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
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    # 合并所有视图
    X = torch.cat([dataset.views[i] for i in range(len(dataset.views))], dim=1)
    
    model = SimpleAE(X.shape[1], 256, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        z, recon = model(X)
        loss = F.mse_loss(recon, X)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                z, _ = model(X)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(z.numpy())
            acc = evaluate_clustering(labels, pred, z.numpy())['acc']
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, ACC={acc:.4f}")
    
    return acc


def test_multiview_ae():
    """测试多视图自编码器（类似 OT-CFM 的 encoder）"""
    print("\n" + "="*60)
    print("测试多视图自编码器")
    print("="*60)
    
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
    
    model = MultiViewAE(view_dims, [256, 128], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader:
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
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}")
    
    return acc


def test_otcfm_encoder_only():
    """测试 OT-CFM 的 encoder 部分"""
    print("\n" + "="*60)
    print("测试 OT-CFM encoder 部分")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters
    ).to(device)
    
    optimizer = torch.optim.Adam(model.encoder_decoder.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # 只用 encoder-decoder
            latents = model.encoder_decoder.encode(views_batch, mask)
            recons = model.encoder_decoder.decode(latents)
            consensus = model.encoder_decoder.fuse_latents(latents, mask)
            
            # 重建损失
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_embeddings.append(consensus.detach().cpu())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}")
    
    return acc


def test_otcfm_with_contrastive():
    """测试 OT-CFM encoder + 对比损失"""
    print("\n" + "="*60)
    print("测试 OT-CFM encoder + 对比损失")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters
    ).to(device)
    
    optimizer = torch.optim.Adam(model.encoder_decoder.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # 只用 encoder-decoder
            latents = model.encoder_decoder.encode(views_batch, mask)
            recons = model.encoder_decoder.decode(latents)
            consensus = model.encoder_decoder.fuse_latents(latents, mask)
            
            # 重建损失
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            # 对比损失 - 这里可能有问题！
            contrastive_loss = 0
            count = 0
            for i in range(len(latents)):
                for j in range(i + 1, len(latents)):
                    z_i = F.normalize(latents[i], dim=-1)
                    z_j = F.normalize(latents[j], dim=-1)
                    # 正样本相似度
                    pos_sim = (z_i * z_j).sum(dim=-1).mean()
                    contrastive_loss -= pos_sim  # 最大化相似度 = 最小化负相似度
                    count += 1
            if count > 0:
                contrastive_loss = contrastive_loss / count
            
            loss = recon_loss + 0.5 * contrastive_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_embeddings.append(consensus.detach().cpu())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            emb_std = embeddings.std()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}, emb_std={emb_std:.4f}")
    
    return acc


def test_ntxent_contrastive():
    """测试使用 NT-Xent 对比损失"""
    print("\n" + "="*60)
    print("测试使用 NT-Xent 对比损失")
    print("="*60)
    
    def nt_xent_loss(z_i, z_j, temperature=0.5):
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss"""
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
        
        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters
    ).to(device)
    
    optimizer = torch.optim.Adam(model.encoder_decoder.parameters(), lr=1e-3)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_embeddings = []
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            latents = model.encoder_decoder.encode(views_batch, mask)
            recons = model.encoder_decoder.decode(latents)
            consensus = model.encoder_decoder.fuse_latents(latents, mask)
            
            # 重建损失
            recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
            
            # NT-Xent 对比损失 - 使用 view 0 和 view 1
            contrastive_loss = nt_xent_loss(latents[0], latents[1], temperature=0.5)
            
            loss = recon_loss + 0.1 * contrastive_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_embeddings.append(consensus.detach().cpu())
        
        if (epoch + 1) % 25 == 0:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred = kmeans.fit_predict(embeddings)
            acc = evaluate_clustering(labels, pred, embeddings)['acc']
            print(f"  Epoch {epoch+1}: loss={total_loss:.4f}, ACC={acc:.4f}")
    
    return acc


if __name__ == "__main__":
    acc1 = test_simple_autoencoder()
    acc2 = test_multiview_ae()
    acc3 = test_otcfm_encoder_only()
    acc4 = test_otcfm_with_contrastive()
    acc5 = test_ntxent_contrastive()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"简单 AE:               ACC = {acc1:.4f}")
    print(f"多视图 AE:             ACC = {acc2:.4f}")
    print(f"OT-CFM encoder:        ACC = {acc3:.4f}")
    print(f"OT-CFM + 对比损失:     ACC = {acc4:.4f}")
    print(f"OT-CFM + NT-Xent:      ACC = {acc5:.4f}")
