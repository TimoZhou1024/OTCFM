"""
验证 StandardScaler 是否是问题根源
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from otcfm.datasets import load_synthetic, create_synthetic_multiview
from otcfm.metrics import evaluate_clustering

# 加载原始数据
data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=0.1, seed=42)
views = data['views']
labels = data['labels']
n_clusters = len(np.unique(labels))

print("="*60)
print("验证 StandardScaler 是否是问题根源")
print("="*60)

# 测试1：原始数据（无标准化）
print("\n1. 原始数据（无标准化）:")
X_raw = np.concatenate(views, axis=1)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pred = kmeans.fit_predict(X_raw)
metrics = evaluate_clustering(labels, pred, X_raw)
print(f"   ACC = {metrics['acc']:.4f}")

# 测试2：StandardScaler 标准化
print("\n2. StandardScaler 标准化:")
views_std = []
for v in views:
    scaler = StandardScaler()
    views_std.append(scaler.fit_transform(v))
X_std = np.concatenate(views_std, axis=1)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pred = kmeans.fit_predict(X_std)
metrics = evaluate_clustering(labels, pred, X_std)
print(f"   ACC = {metrics['acc']:.4f}")

# 测试3：MinMaxScaler 归一化
print("\n3. MinMaxScaler 归一化:")
views_mm = []
for v in views:
    scaler = MinMaxScaler()
    views_mm.append(scaler.fit_transform(v))
X_mm = np.concatenate(views_mm, axis=1)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pred = kmeans.fit_predict(X_mm)
metrics = evaluate_clustering(labels, pred, X_mm)
print(f"   ACC = {metrics['acc']:.4f}")

# 测试4：全局 StandardScaler（所有视图一起）
print("\n4. 全局 StandardScaler（所有视图一起标准化）:")
X_concat = np.concatenate(views, axis=1)
scaler = StandardScaler()
X_global_std = scaler.fit_transform(X_concat)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pred = kmeans.fit_predict(X_global_std)
metrics = evaluate_clustering(labels, pred, X_global_std)
print(f"   ACC = {metrics['acc']:.4f}")

# 测试5：检查标准化后的数据分布
print("\n5. 数据分布分析:")
print(f"   原始数据 - 均值: {X_raw.mean():.4f}, 标准差: {X_raw.std():.4f}")
print(f"   标准化后 - 均值: {X_std.mean():.4f}, 标准差: {X_std.std():.4f}")

# 检查簇心距离
from scipy.spatial.distance import cdist
centers_raw = np.array([X_raw[labels==i].mean(axis=0) for i in range(n_clusters)])
centers_std = np.array([X_std[labels==i].mean(axis=0) for i in range(n_clusters)])

dist_raw = cdist(centers_raw, centers_raw)
dist_std = cdist(centers_std, centers_std)

print(f"\n   原始数据簇心间平均距离: {dist_raw[dist_raw>0].mean():.4f}")
print(f"   标准化后簇心间平均距离: {dist_std[dist_std>0].mean():.4f}")

# 测试6：不同噪声水平下的影响
print("\n6. 不同噪声水平下的标准化影响:")
for noise in [0.1, 0.5, 1.0, 2.0]:
    data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=noise, seed=42)
    views = data['views']
    labels = data['labels']
    
    # 原始
    X_raw = np.concatenate(views, axis=1)
    pred_raw = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_raw)
    acc_raw = evaluate_clustering(labels, pred_raw, X_raw)['acc']
    
    # 标准化
    views_std = [StandardScaler().fit_transform(v) for v in views]
    X_std = np.concatenate(views_std, axis=1)
    pred_std = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_std)
    acc_std = evaluate_clustering(labels, pred_std, X_std)['acc']
    
    print(f"   noise={noise}: 原始 ACC={acc_raw:.4f}, 标准化 ACC={acc_std:.4f}")

print("\n" + "="*60)
print("结论: 如果标准化后 ACC 显著下降，则 StandardScaler 是问题根源")
print("="*60)
