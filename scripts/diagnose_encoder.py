"""
诊断脚本：确定问题是在编码器还是其他地方
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from otcfm.datasets import create_synthetic_multiview


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(D)]) * 1.0 / y_pred.size


def test_representation_methods():
    """测试不同的表示方法"""
    np.random.seed(42)
    
    # 创建数据
    views, labels = create_synthetic_multiview(
        n_samples=1000,
        n_clusters=10,
        n_views=3,
        view_dims=[100, 100, 100],
        noise_level=0.1,
        seed=42
    )
    
    print("="*70)
    print("测试不同表示方法的聚类性能")
    print("="*70)
    
    results = {}
    
    # 1. 原始拼接特征
    X_concat = np.concatenate(views, axis=1)
    kmeans = KMeans(n_clusters=10, n_init=20, random_state=42)
    pred = kmeans.fit_predict(X_concat)
    acc = cluster_acc(labels, pred)
    results['原始拼接'] = acc
    print(f"1. 原始拼接特征: ACC = {acc:.4f}")
    
    # 2. 标准化后拼接
    views_scaled = [StandardScaler().fit_transform(v) for v in views]
    X_scaled = np.concatenate(views_scaled, axis=1)
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_scaled)
    acc = cluster_acc(labels, pred)
    results['标准化拼接'] = acc
    print(f"2. 标准化后拼接: ACC = {acc:.4f}")
    
    # 3. PCA 降维到 128 维
    pca = PCA(n_components=128)
    X_pca = pca.fit_transform(X_scaled)
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_pca)
    acc = cluster_acc(labels, pred)
    results['PCA(128)'] = acc
    print(f"3. PCA(128) 降维: ACC = {acc:.4f}")
    
    # 4. PCA 降维到 64 维
    pca = PCA(n_components=64)
    X_pca = pca.fit_transform(X_scaled)
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_pca)
    acc = cluster_acc(labels, pred)
    results['PCA(64)'] = acc
    print(f"4. PCA(64) 降维: ACC = {acc:.4f}")
    
    # 5. 随机投影
    np.random.seed(42)
    W = np.random.randn(300, 128) / np.sqrt(300)
    X_random = X_scaled @ W
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_random)
    acc = cluster_acc(labels, pred)
    results['随机投影(128)'] = acc
    print(f"5. 随机投影(128): ACC = {acc:.4f}")
    
    # 6. 神经网络随机权重（未训练）
    torch.manual_seed(42)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # 简单 MLP
    net = nn.Sequential(
        nn.Linear(300, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    net.eval()
    with torch.no_grad():
        X_nn = net(X_tensor).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_nn)
    acc = cluster_acc(labels, pred)
    results['随机MLP(128)'] = acc
    print(f"6. 随机MLP(128): ACC = {acc:.4f}")
    
    # 7. 训练一个自编码器（只用重建损失）
    print("\n7. 训练自编码器...")
    
    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(300, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.decoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 300)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    ae = SimpleAE()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(50):
        ae.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            z, recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
    
    ae.eval()
    with torch.no_grad():
        X_ae, _ = ae(X_tensor)
        X_ae = X_ae.numpy()
    
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_ae)
    acc = cluster_acc(labels, pred)
    results['AE(128)'] = acc
    print(f"   自编码器(128): ACC = {acc:.4f}")
    
    # 8. 自编码器 + L2 正则化
    print("\n8. 训练带正则化的自编码器...")
    
    ae2 = SimpleAE()
    optimizer = torch.optim.Adam(ae2.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(50):
        ae2.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            z, recon = ae2(batch)
            # 添加 L2 正则化到嵌入
            loss = F.mse_loss(recon, batch) + 0.01 * (z ** 2).mean()
            loss.backward()
            optimizer.step()
    
    ae2.eval()
    with torch.no_grad():
        X_ae2, _ = ae2(X_tensor)
        X_ae2 = X_ae2.numpy()
    
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_ae2)
    acc = cluster_acc(labels, pred)
    results['AE+正则化'] = acc
    print(f"   自编码器+正则化: ACC = {acc:.4f}")
    
    # 9. 浅层编码器（单层）
    print("\n9. 训练浅层编码器...")
    
    class ShallowAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(300, 128)
            self.decoder = nn.Linear(128, 300)
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
    
    sae = ShallowAE()
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    for epoch in range(50):
        sae.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            z, recon = sae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
    
    sae.eval()
    with torch.no_grad():
        X_sae, _ = sae(X_tensor)
        X_sae = X_sae.numpy()
    
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(X_sae)
    acc = cluster_acc(labels, pred)
    results['浅层AE'] = acc
    print(f"   浅层编码器: ACC = {acc:.4f}")
    
    print("\n" + "="*70)
    print("结果总结")
    print("="*70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20}: ACC = {acc:.4f}")
    
    print("\n" + "="*70)
    print("分析")
    print("="*70)
    print("""
问题诊断:
- 原始特征和简单降维方法都能达到接近100%准确率
- 神经网络自编码器表现很差

可能原因:
1. 自编码器重建损失不保证保留聚类结构
2. BatchNorm 可能破坏聚类信息
3. 非线性变换丢失了判别信息

解决方案:
1. 使用线性或浅层编码器
2. 添加判别性损失（如对比学习）
3. 避免过度压缩（使用更大的潜在空间）
""")


if __name__ == "__main__":
    test_representation_methods()
