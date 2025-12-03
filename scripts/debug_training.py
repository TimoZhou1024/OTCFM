"""
深度调试脚本 - 分析 OT-CFM 训练过程中的问题
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import get_default_config, TrainingConfig
from otcfm.datasets import create_synthetic_multiview, MultiViewDataset, create_dataloader
from otcfm.ot_cfm import OTCFM
from otcfm.metrics import evaluate_clustering


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def debug_training():
    """深度调试 OT-CFM 训练"""
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建简单的合成数据
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
    
    print(f"数据集: {len(labels)} 样本, {len(views)} 视图")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 首先测试：直接用 KMeans 在原始特征上聚类
    print("\n" + "="*60)
    print("Step 1: 在原始特征上直接 KMeans")
    print("="*60)
    
    X_concat = np.concatenate([v for v in views], axis=1)
    kmeans = KMeans(n_clusters=10, n_init=20, random_state=42)
    pred_raw = kmeans.fit_predict(X_concat)
    
    from scipy.optimize import linear_sum_assignment
    def cluster_acc(y_true, y_pred):
        """计算聚类准确率"""
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return sum([w[row_ind[i], col_ind[i]] for i in range(D)]) * 1.0 / y_pred.size
    
    acc_raw = cluster_acc(labels, pred_raw)
    nmi_raw = normalized_mutual_info_score(labels, pred_raw)
    print(f"原始特征 KMeans: ACC={acc_raw:.4f}, NMI={nmi_raw:.4f}")
    
    # 创建数据集和模型
    dataset = MultiViewDataset(views=views, labels=labels)
    dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    view_dims = [v.shape[1] for v in views]
    
    # 创建模型 - 使用非常简单的配置
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,  # 更小的潜在空间
        hidden_dims=[256, 128],
        num_clusters=10,
        flow_hidden_dim=128,
        flow_num_layers=2,
        lambda_gw=0.0,        # 先禁用 GW
        lambda_cluster=2.0,   # 强调聚类
        lambda_recon=0.1,     # 最小化重建
        lambda_contrastive=0.5,
        dropout=0.0,
        use_cross_view_flow=False  # 先禁用复杂的 flow
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n" + "="*60)
    print("Step 2: 训练模型并监控")
    print("="*60)
    
    # 初始化聚类中心
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch_views = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(batch_views, mask)
            all_embeddings.append(outputs['consensus'].cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 检查初始嵌入
    print(f"\n初始嵌入统计:")
    print(f"  均值: {all_embeddings.mean().item():.4f}")
    print(f"  标准差: {all_embeddings.std().item():.4f}")
    print(f"  最小值: {all_embeddings.min().item():.4f}")
    print(f"  最大值: {all_embeddings.max().item():.4f}")
    
    # 在初始嵌入上 KMeans
    kmeans_init = KMeans(n_clusters=10, n_init=20, random_state=42)
    pred_init = kmeans_init.fit_predict(all_embeddings.numpy())
    acc_init = cluster_acc(labels, pred_init)
    print(f"初始嵌入 KMeans: ACC={acc_init:.4f}")
    
    # 初始化模型的聚类中心
    model.clustering.init_centroids(all_embeddings)
    model.clustering.centroids = model.clustering.centroids.to(device)
    
    # 训练循环
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        loss_details = {}
        
        for batch in dataloader:
            batch_views = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(batch_views, mask)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            for k, v in loss_dict.items():
                loss_details[k] = loss_details.get(k, 0) + v
        
        # 评估
        model.eval()
        all_embeddings = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch_views = [v.to(device) for v in batch['views']]
                mask = batch['mask'].to(device)
                outputs = model(batch_views, mask)
                all_embeddings.append(outputs['consensus'].cpu())
                all_preds.append(outputs['assignments'].cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        
        # 使用模型预测
        acc_model = cluster_acc(labels, all_preds)
        nmi_model = normalized_mutual_info_score(labels, all_preds)
        
        # 使用 KMeans 在嵌入上
        kmeans_emb = KMeans(n_clusters=10, n_init=10, random_state=42)
        pred_emb = kmeans_emb.fit_predict(all_embeddings)
        acc_emb = cluster_acc(labels, pred_emb)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Loss: {avg_loss:.4f} (cluster={loss_details.get('cluster',0)/len(dataloader):.4f})")
            print(f"  Model预测 ACC: {acc_model:.4f}, NMI: {nmi_model:.4f}")
            print(f"  KMeans嵌入 ACC: {acc_emb:.4f}")
            print(f"  嵌入std: {all_embeddings.std():.4f}")
        
        # 每10个epoch更新聚类中心
        if (epoch + 1) % 10 == 0:
            embeddings_tensor = torch.FloatTensor(all_embeddings)
            model.clustering.init_centroids(embeddings_tensor)
            model.clustering.centroids = model.clustering.centroids.to(device)
    
    print("\n" + "="*60)
    print("Step 3: 问题诊断")
    print("="*60)
    
    # 检查嵌入是否有判别性
    print("\n检查各聚类的嵌入分布:")
    for c in range(10):
        mask_c = labels == c
        emb_c = all_embeddings[mask_c]
        print(f"  Cluster {c}: n={mask_c.sum()}, mean_norm={np.linalg.norm(emb_c.mean(axis=0)):.4f}, "
              f"std={emb_c.std():.4f}")
    
    # 检查聚类中心
    print("\n模型聚类中心:")
    centroids = model.clustering.centroids.cpu().detach().numpy()
    for i in range(10):
        print(f"  Center {i}: norm={np.linalg.norm(centroids[i]):.4f}")
    
    # 检查软分配
    print("\n检查软分配 Q 的熵:")
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch_views = [v.to(device) for v in batch['views']]
        mask = batch['mask'].to(device)
        outputs = model(batch_views, mask)
        q = outputs['q'].cpu().numpy()
        
        # 计算熵
        entropy = -np.sum(q * np.log(q + 1e-8), axis=1)
        print(f"  平均熵: {entropy.mean():.4f} (最大可能: {np.log(10):.4f})")
        print(f"  熵范围: [{entropy.min():.4f}, {entropy.max():.4f}]")
        print(f"  Q 最大值分布: {q.max(axis=1).mean():.4f} (理想应接近1)")


if __name__ == "__main__":
    debug_training()
