"""
诊断 OT-CFM 训练过程中的问题
逐步添加组件，找到导致性能下降的原因
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
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
    """多视图自编码器"""
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


def test_step_by_step():
    """逐步测试每个组件"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建数据
    views, labels = create_synthetic_multiview(
        n_samples=1000, n_clusters=10, n_views=3,
        view_dims=[100, 100, 100], noise_level=0.1, seed=42
    )
    
    # 标准化
    views_norm = [StandardScaler().fit_transform(v) for v in views]
    views_tensor = [torch.FloatTensor(v) for v in views_norm]
    
    print("="*70)
    print("逐步测试 OT-CFM 组件")
    print("="*70)
    
    # 1. 单视图编码器
    print("\n1. 测试单视图编码器")
    for i, v in enumerate(views_tensor):
        enc = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        dec = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )
        
        optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        
        for epoch in range(30):
            enc.train(); dec.train()
            optimizer.zero_grad()
            z = enc(v)
            recon = dec(z)
            loss = F.mse_loss(recon, v)
            loss.backward()
            optimizer.step()
        
        enc.eval()
        with torch.no_grad():
            z = enc(v).numpy()
        
        pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
        acc = cluster_acc(labels, pred)
        print(f"   视图 {i}: ACC = {acc:.4f}")
    
    # 2. 多视图编码器 + 平均融合
    print("\n2. 测试多视图编码器 + 平均融合")
    
    model = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = MultiViewDataset(views=views, labels=labels)
    loader = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(30):
        model.train()
        for batch in loader:
            batch_views = batch['views']
            optimizer.zero_grad()
            
            latents = model.encode(batch_views)
            recons = model.decode(latents)
            
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch_views)) / 3
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader:
            latents = model.encode(batch['views'])
            z = model.fuse(latents)
            all_z.append(z)
    
    all_z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(all_z)
    acc = cluster_acc(labels, pred)
    print(f"   多视图融合后: ACC = {acc:.4f}")
    
    # 3. 检查是否是数据集类的问题
    print("\n3. 检查 MultiViewDataset 类")
    
    # 直接用 tensor 而不是 dataset
    model2 = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model2.train()
        optimizer.zero_grad()
        
        latents = model2.encode(views_tensor)
        recons = model2.decode(latents)
        
        loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_tensor)) / 3
        loss.backward()
        optimizer.step()
    
    model2.eval()
    with torch.no_grad():
        latents = model2.encode(views_tensor)
        z = model2.fuse(latents).numpy()
    
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   直接用 tensor（无 dataset）: ACC = {acc:.4f}")
    
    # 4. 测试 dataset 类是否改变了数据
    print("\n4. 检查 dataset 是否影响数据")
    
    dataset = MultiViewDataset(views=views, labels=labels)
    
    # 收集所有数据
    all_views = [[] for _ in range(3)]
    for i in range(len(dataset)):
        sample = dataset[i]
        for v_idx in range(3):
            all_views[v_idx].append(sample['views'][v_idx])
    
    views_from_dataset = [torch.stack(v, dim=0) for v in all_views]
    
    # 比较
    for i in range(3):
        diff = (views_from_dataset[i] - views_tensor[i]).abs().max()
        print(f"   视图 {i} 最大差异: {diff:.6f}")
    
    # 5. 检查 collate_fn 是否有问题
    print("\n5. 检查 dataloader + collate_fn")
    
    loader = create_dataloader(dataset, batch_size=1000, shuffle=False)
    batch = next(iter(loader))
    
    for i in range(3):
        diff = (batch['views'][i] - views_tensor[i]).abs().max()
        print(f"   视图 {i} 最大差异（通过loader）: {diff:.6f}")
    
    # 6. 使用 loader 训练但禁用 shuffle
    print("\n6. 使用 loader（无 shuffle）训练")
    
    model3 = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer = torch.optim.Adam(model3.parameters(), lr=1e-3)
    loader = create_dataloader(dataset, batch_size=256, shuffle=False)
    
    for epoch in range(30):
        model3.train()
        for batch in loader:
            optimizer.zero_grad()
            latents = model3.encode(batch['views'])
            recons = model3.decode(latents)
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch['views'])) / 3
            loss.backward()
            optimizer.step()
    
    model3.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader:
            latents = model3.encode(batch['views'])
            z = model3.fuse(latents)
            all_z.append(z)
    
    all_z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(all_z)
    acc = cluster_acc(labels, pred)
    print(f"   无 shuffle loader: ACC = {acc:.4f}")
    
    # 7. 使用 loader 并 shuffle
    print("\n7. 使用 loader（有 shuffle）训练")
    
    model4 = MultiViewAE([100, 100, 100], latent_dim=128)
    optimizer = torch.optim.Adam(model4.parameters(), lr=1e-3)
    loader_train = create_dataloader(dataset, batch_size=256, shuffle=True)
    loader_eval = create_dataloader(dataset, batch_size=256, shuffle=False)
    
    for epoch in range(30):
        model4.train()
        for batch in loader_train:
            optimizer.zero_grad()
            latents = model4.encode(batch['views'])
            recons = model4.decode(latents)
            loss = sum(F.mse_loss(r, v) for r, v in zip(recons, batch['views'])) / 3
            loss.backward()
            optimizer.step()
    
    model4.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            latents = model4.encode(batch['views'])
            z = model4.fuse(latents)
            all_z.append(z)
    
    all_z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(all_z)
    acc = cluster_acc(labels, pred)
    print(f"   有 shuffle loader: ACC = {acc:.4f}")


if __name__ == "__main__":
    test_step_by_step()
