"""
深入诊断 dataloader 问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from otcfm.datasets import create_synthetic_multiview, MultiViewDataset, create_dataloader


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(D)]) * 1.0 / y_pred.size


class MultiViewAE(nn.Module):
    def __init__(self, view_dims, latent_dim=128):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            ) for dim in view_dims
        ])
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, dim)
            ) for dim in view_dims
        ])
    
    def encode(self, views):
        return [enc(v) for enc, v in zip(self.encoders, views)]
    
    def fuse(self, latents):
        return torch.stack(latents, dim=0).mean(dim=0)
    
    def decode(self, latents):
        return [dec(z) for dec, z in zip(self.decoders, latents)]


def diagnose():
    np.random.seed(42)
    torch.manual_seed(42)
    
    views, labels = create_synthetic_multiview(
        n_samples=1000, n_clusters=10, n_views=3,
        view_dims=[100, 100, 100], noise_level=0.1, seed=42
    )
    
    dataset = MultiViewDataset(views=views, labels=labels)
    
    print("="*70)
    print("诊断 DataLoader 问题")
    print("="*70)
    
    # 检查 batch 顺序
    print("\n1. 检查 batch['views'] 的结构")
    loader = create_dataloader(dataset, batch_size=256, shuffle=False)
    
    for batch in loader:
        print(f"   batch['views'] 类型: {type(batch['views'])}")
        print(f"   batch['views'] 长度: {len(batch['views'])}")
        print(f"   batch['views'][0] 形状: {batch['views'][0].shape}")
        print(f"   batch['mask'] 形状: {batch['mask'].shape}")
        break
    
    # 核心测试：使用不同的方式遍历数据
    print("\n2. 测试不同的数据遍历方式")
    
    # 方式 A: 手动 batch
    views_norm = [StandardScaler().fit_transform(v) for v in views]
    views_tensor = [torch.FloatTensor(v) for v in views_norm]
    
    model_a = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model_a.train()
        # 手动分 batch
        indices = np.arange(1000)
        np.random.shuffle(indices)
        
        for i in range(0, 1000, 256):
            batch_idx = indices[i:i+256]
            batch_views = [v[batch_idx] for v in views_tensor]
            
            optimizer_a.zero_grad()
            latents = model_a.encode(batch_views)
            recons = model_a.decode(latents)
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch_views)) / 3
            loss.backward()
            optimizer_a.step()
    
    model_a.eval()
    with torch.no_grad():
        latents = model_a.encode(views_tensor)
        z = model_a.fuse(latents).numpy()
    
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   方式 A (手动 batch): ACC = {acc:.4f}")
    
    # 方式 B: 使用我们的 dataloader
    model_b = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    
    loader_train = create_dataloader(dataset, batch_size=256, shuffle=True)
    loader_eval = create_dataloader(dataset, batch_size=1000, shuffle=False)
    
    for epoch in range(30):
        model_b.train()
        for batch in loader_train:
            optimizer_b.zero_grad()
            latents = model_b.encode(batch['views'])
            recons = model_b.decode(latents)
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch['views'])) / 3
            loss.backward()
            optimizer_b.step()
    
    model_b.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            latents = model_b.encode(batch['views'])
            z = model_b.fuse(latents)
            all_z.append(z)
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   方式 B (dataloader): ACC = {acc:.4f}")
    
    # 方式 C: PyTorch 默认 DataLoader
    print("\n3. 使用 PyTorch 默认 DataLoader")
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, views, labels):
            self.views = [torch.FloatTensor(StandardScaler().fit_transform(v)) for v in views]
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return [v[idx] for v in self.views], self.labels[idx]
    
    def simple_collate(batch):
        views = [[b[0][v] for b in batch] for v in range(3)]
        views = [torch.stack(v) for v in views]
        labels = torch.LongTensor([b[1] for b in batch])
        return views, labels
    
    simple_dataset = SimpleDataset(views, labels)
    simple_loader = torch.utils.data.DataLoader(
        simple_dataset, batch_size=256, shuffle=True, collate_fn=simple_collate
    )
    
    model_c = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model_c.train()
        for batch_views, _ in simple_loader:
            optimizer_c.zero_grad()
            latents = model_c.encode(batch_views)
            recons = model_c.decode(latents)
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch_views)) / 3
            loss.backward()
            optimizer_c.step()
    
    model_c.eval()
    all_z = []
    eval_loader = torch.utils.data.DataLoader(
        simple_dataset, batch_size=1000, shuffle=False, collate_fn=simple_collate
    )
    with torch.no_grad():
        for batch_views, _ in eval_loader:
            latents = model_c.encode(batch_views)
            z = model_c.fuse(latents)
            all_z.append(z)
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   方式 C (简单 DataLoader): ACC = {acc:.4f}")
    
    # 检查 MultiViewDataset 的标准化
    print("\n4. 检查 MultiViewDataset 内部的数据")
    
    # 原始数据统计
    for i in range(3):
        print(f"   原始视图 {i}: mean={views[i].mean():.4f}, std={views[i].std():.4f}")
    
    # dataset 内部数据统计
    for i in range(3):
        v = dataset.views[i]
        print(f"   Dataset 视图 {i}: mean={v.mean():.4f}, std={v.std():.4f}")
    
    # 通过索引获取
    sample = dataset[0]
    for i in range(3):
        print(f"   Sample[0] 视图 {i}: mean={sample['views'][i].mean():.4f}")


if __name__ == "__main__":
    diagnose()
