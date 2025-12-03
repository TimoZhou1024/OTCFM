"""
最终定位：对比 Dataset 的数据和原始数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from otcfm.datasets import load_synthetic, MultiViewDataset, create_dataloader
from otcfm.metrics import evaluate_clustering

torch.manual_seed(42)
np.random.seed(42)

# 加载原始数据
data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=0.1)
views_np = data['views']
labels = data['labels']
n_clusters = len(np.unique(labels))

print("="*60)
print("对比 Dataset 的数据和原始数据")
print("="*60)

# 创建 Dataset
dataset = MultiViewDataset(views_np, labels)

print("\n1. 检查数据是否一致:")

# 获取所有数据通过 Dataset
views_from_dataset = [[] for _ in range(len(views_np))]
labels_from_dataset = []

for i in range(len(dataset)):
    item = dataset[i]
    for v_idx, v in enumerate(item['views']):
        views_from_dataset[v_idx].append(v.numpy())
    labels_from_dataset.append(item['label'])

views_from_dataset = [np.stack(v, axis=0) for v in views_from_dataset]
labels_from_dataset = np.array(labels_from_dataset)

# 对原始数据进行相同的标准化
views_standardized = []
for v in views_np:
    scaler = StandardScaler()
    views_standardized.append(scaler.fit_transform(v))

print(f"   原始标签前10: {labels[:10]}")
print(f"   Dataset标签前10: {labels_from_dataset[:10]}")

# 检查视图数据
for v_idx in range(len(views_np)):
    diff = np.abs(views_from_dataset[v_idx] - views_standardized[v_idx]).max()
    print(f"   视图 {v_idx} 最大差异: {diff:.6f}")

print("\n2. 检查标签顺序是否正确:")
print(f"   labels 和 labels_from_dataset 是否相等: {np.array_equal(labels, labels_from_dataset)}")

print("\n3. 直接在 Dataset 视图上聚类:")
for v_idx in range(len(views_np)):
    X = views_from_dataset[v_idx]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = kmeans.fit_predict(X)
    
    # 使用原始标签
    acc_orig = evaluate_clustering(labels, pred, X)['acc']
    # 使用 dataset 标签
    acc_ds = evaluate_clustering(labels_from_dataset, pred, X)['acc']
    
    print(f"   视图 {v_idx}: ACC(原始标签)={acc_orig:.4f}, ACC(Dataset标签)={acc_ds:.4f}")

print("\n4. 检查 DataLoader 输出顺序:")
dataloader = create_dataloader(dataset, batch_size=1000, shuffle=False)

batch = next(iter(dataloader))
views_from_loader = [v.numpy() for v in batch['views']]
labels_from_loader = batch['labels'].numpy() if 'labels' in batch else None

for v_idx in range(len(views_np)):
    diff = np.abs(views_from_loader[v_idx] - views_from_dataset[v_idx]).max()
    print(f"   视图 {v_idx} (loader vs dataset) 差异: {diff:.6f}")

if labels_from_loader is not None:
    print(f"   标签是否一致: {np.array_equal(labels, labels_from_loader)}")

print("\n5. 使用 shuffle=True 的 DataLoader:")
dataloader_shuffle = create_dataloader(dataset, batch_size=1000, shuffle=True)

batch = next(iter(dataloader_shuffle))
indices = batch['indices'].numpy() if 'indices' in batch else None

if indices is not None:
    print(f"   batch indices 前10: {indices[:10]}")
    print(f"   indices 排序后与 range(1000) 相同: {np.array_equal(np.sort(indices), np.arange(1000))}")

print("\n6. 使用 DataLoader 训练（与之前的多视图 AE 相同的设置）:")

# 这次使用 shuffle=False
dataloader_no_shuffle = create_dataloader(dataset, batch_size=256, shuffle=False)

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

view_dims = [v.shape[1] for v in views_np]
model = MultiViewAE(view_dims, 256, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    all_embeddings = []
    all_indices = []
    
    for batch in dataloader_no_shuffle:
        views_batch = batch['views']
        indices = batch['indices'].numpy() if 'indices' in batch else None
        
        optimizer.zero_grad()
        latents, consensus, recons = model(views_batch)
        
        recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
        
        loss = recon_loss
        loss.backward()
        optimizer.step()
        
        all_embeddings.append(consensus.detach())
        if indices is not None:
            all_indices.extend(indices.tolist())
    
    if (epoch + 1) % 25 == 0:
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        # 按 indices 排序
        if all_indices:
            order = np.argsort(all_indices)
            embeddings_sorted = embeddings[order]
        else:
            embeddings_sorted = embeddings
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(embeddings_sorted)
        acc = evaluate_clustering(labels, pred, embeddings_sorted)['acc']
        print(f"   shuffle=False, Epoch {epoch+1}: ACC = {acc:.4f}")

print("\n7. 对比：使用原始 tensor 直接训练（确认全批次正常）:")

# 直接使用 tensor
views_tensor = [torch.FloatTensor(v) for v in views_standardized]

model2 = MultiViewAE(view_dims, 256, 64)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

for epoch in range(100):
    model2.train()
    optimizer2.zero_grad()
    
    latents, consensus, recons = model2(views_tensor)
    
    recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_tensor)) / len(views_tensor)
    
    loss = recon_loss
    loss.backward()
    optimizer2.step()
    
    if (epoch + 1) % 25 == 0:
        model2.eval()
        with torch.no_grad():
            _, consensus, _ = model2(views_tensor)
        
        z = consensus.numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   全批次（tensor）, Epoch {epoch+1}: ACC = {acc:.4f}")

print("\n8. 检查小批次问题:")
# 手动分批次训练（不使用 DataLoader）
batch_size = 256
n_batches = (len(views_standardized[0]) + batch_size - 1) // batch_size

model3 = MultiViewAE(view_dims, 256, 64)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)

for epoch in range(100):
    model3.train()
    
    # 打乱顺序
    perm = np.random.permutation(len(views_standardized[0]))
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(views_standardized[0]))
        batch_indices = perm[start:end]
        
        views_batch = [torch.FloatTensor(v[batch_indices]) for v in views_standardized]
        
        optimizer3.zero_grad()
        latents, consensus, recons = model3(views_batch)
        
        recon_loss = sum(F.mse_loss(r, v) for r, v in zip(recons, views_batch)) / len(views_batch)
        
        loss = recon_loss
        loss.backward()
        optimizer3.step()
    
    if (epoch + 1) % 25 == 0:
        model3.eval()
        with torch.no_grad():
            _, consensus, _ = model3(views_tensor)
        
        z = consensus.numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(z)
        acc = evaluate_clustering(labels, pred, z)['acc']
        print(f"   手动分批次 + shuffle, Epoch {epoch+1}: ACC = {acc:.4f}")
