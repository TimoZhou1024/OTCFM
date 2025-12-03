"""
最终诊断：完整模拟 Trainer 流程来找出问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from otcfm.datasets import load_synthetic, MultiViewDataset, create_dataloader
from otcfm.ot_cfm import OTCFM
from otcfm.metrics import evaluate_clustering

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 加载数据
data = load_synthetic(n_samples=1000, n_clusters=10, noise_level=0.1)
views = data['views']
labels = data['labels']
view_dims = [v.shape[1] for v in views]
n_clusters = len(np.unique(labels))

print(f"数据: {len(labels)} 样本, {len(views)} 视图, {n_clusters} 簇")

# 创建 dataset 和 dataloader
dataset = MultiViewDataset(views, labels)
dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)

device = 'cpu'


def test_direct_clustering():
    """测试1：直接在原始数据上聚类"""
    print("\n" + "="*60)
    print("测试1: 直接在原始数据上聚类")
    print("="*60)
    
    # 拼接所有视图
    X = np.concatenate(views, axis=1)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = kmeans.fit_predict(X)
    
    metrics = evaluate_clustering(labels, pred, X)
    print(f"   ACC = {metrics['acc']:.4f}")
    return metrics['acc']


def test_trainer_pretraining():
    """测试2：模拟 Trainer 预训练阶段"""
    print("\n" + "="*60)
    print("测试2: 模拟 Trainer 预训练阶段")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters,
        lambda_gw=0.1,
        lambda_cluster=1.0,
        lambda_recon=0.5,
        lambda_contrastive=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 预训练：只用重建和对比损失
    print("   预训练阶段...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(views_batch, mask, return_all=True)
            
            # 只算重建损失
            recon_loss = 0
            for v_idx in range(len(views_batch)):
                recon_loss += F.mse_loss(outputs['reconstructions'][v_idx], views_batch[v_idx])
            recon_loss /= len(views_batch)
            
            # 加对比损失
            latents = outputs['latents']
            contrastive_loss = 0
            count = 0
            for i in range(len(latents)):
                for j in range(i+1, len(latents)):
                    z_i = F.normalize(latents[i], dim=-1)
                    z_j = F.normalize(latents[j], dim=-1)
                    contrastive_loss -= (z_i * z_j).sum(dim=-1).mean()
                    count += 1
            if count > 0:
                contrastive_loss /= count
            
            loss = recon_loss + 0.5 * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
    print(f"   预训练损失: {total_loss:.4f}")
    
    # 评估预训练后的嵌入
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = kmeans.fit_predict(embeddings)
    metrics = evaluate_clustering(labels, pred, embeddings)
    print(f"   预训练后 ACC = {metrics['acc']:.4f}")
    
    return model, optimizer, metrics['acc']


def test_init_clustering(model, optimizer):
    """测试3：聚类中心初始化"""
    print("\n" + "="*60)
    print("测试3: 聚类中心初始化")
    print("="*60)
    
    # 模拟 init_clustering
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 初始化聚类中心
    model.clustering.init_centroids(all_embeddings)
    model.clustering.centroids = model.clustering.centroids.to(device)
    
    # 检查聚类效果
    model.eval()
    with torch.no_grad():
        all_preds = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_preds.append(outputs['assignments'].cpu())
        predictions = torch.cat(all_preds, dim=0).numpy()
    
    metrics = evaluate_clustering(labels, predictions, all_embeddings.numpy())
    print(f"   初始化聚类后 ACC = {metrics['acc']:.4f}")
    
    return metrics['acc']


def test_main_training(model, optimizer):
    """测试4：主训练阶段（使用完整损失）"""
    print("\n" + "="*60)
    print("测试4: 主训练阶段")
    print("="*60)
    
    # 先记录训练前的性能
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        all_preds = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
            all_preds.append(outputs['assignments'].cpu())
        embeddings_before = torch.cat(all_embeddings, dim=0).numpy()
        preds_before = torch.cat(all_preds, dim=0).numpy()
    
    metrics_before = evaluate_clustering(labels, preds_before, embeddings_before)
    print(f"   训练前 ACC = {metrics_before['acc']:.4f}")
    
    # 计算 embeddings 的统计信息
    emb_mean_before = np.mean(embeddings_before)
    emb_std_before = np.std(embeddings_before)
    print(f"   训练前 embedding: mean={emb_mean_before:.4f}, std={emb_std_before:.4f}")
    
    # 主训练循环
    for epoch in range(50):
        model.train()
        total_loss = 0
        loss_details = {}
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(views_batch, mask, ablation_mode="full")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_details[k] = loss_details.get(k, 0) + v
        
        if (epoch + 1) % 10 == 0:
            # 评估
            model.eval()
            with torch.no_grad():
                all_embeddings = []
                all_preds = []
                for batch in dataloader:
                    views_batch = [v.to(device) for v in batch['views']]
                    mask = batch['mask'].to(device)
                    outputs = model(views_batch, mask)
                    all_embeddings.append(outputs['consensus'].cpu())
                    all_preds.append(outputs['assignments'].cpu())
                embeddings = torch.cat(all_embeddings, dim=0).numpy()
                predictions = torch.cat(all_preds, dim=0).numpy()
            
            metrics = evaluate_clustering(labels, predictions, embeddings)
            emb_mean = np.mean(embeddings)
            emb_std = np.std(embeddings)
            
            print(f"   Epoch {epoch+1}: loss={total_loss:.4f}, ACC={metrics['acc']:.4f}, "
                  f"emb_mean={emb_mean:.4f}, emb_std={emb_std:.4f}")
            for k, v in loss_details.items():
                print(f"      {k}: {v:.4f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        all_preds = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
            all_preds.append(outputs['assignments'].cpu())
        embeddings_after = torch.cat(all_embeddings, dim=0).numpy()
        preds_after = torch.cat(all_preds, dim=0).numpy()
    
    metrics_after = evaluate_clustering(labels, preds_after, embeddings_after)
    print(f"\n   最终 ACC = {metrics_after['acc']:.4f}")
    
    return metrics_after['acc']


def test_train_without_cfm():
    """测试5：不使用 CFM 损失训练"""
    print("\n" + "="*60)
    print("测试5: 不使用 CFM 损失 (ablation_mode='no_flow')")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters,
        lambda_gw=0.1,
        lambda_cluster=1.0,
        lambda_recon=0.5,
        lambda_contrastive=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 初始化聚类
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
    model.clustering.init_centroids(all_embeddings)
    model.clustering.centroids = model.clustering.centroids.to(device)
    
    # 训练 - 不使用 flow
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(views_batch, mask, ablation_mode="no_flow")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                all_embeddings = []
                all_preds = []
                for batch in dataloader:
                    views_batch = [v.to(device) for v in batch['views']]
                    mask = batch['mask'].to(device)
                    outputs = model(views_batch, mask)
                    all_embeddings.append(outputs['consensus'].cpu())
                    all_preds.append(outputs['assignments'].cpu())
                embeddings = torch.cat(all_embeddings, dim=0).numpy()
                predictions = torch.cat(all_preds, dim=0).numpy()
            
            metrics = evaluate_clustering(labels, predictions, embeddings)
            print(f"   Epoch {epoch+1}: loss={total_loss:.4f}, ACC={metrics['acc']:.4f}")
    
    return metrics['acc']


def test_simple_ae_clustering():
    """测试6：简单 AE + K-Means（不用 OT-CFM 的复杂损失）"""
    print("\n" + "="*60)
    print("测试6: 简单 AE + K-Means")
    print("="*60)
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=64,
        hidden_dims=[256, 128],
        num_clusters=n_clusters
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 只训练重建
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(views_batch, mask, return_all=True)
            
            # 只用重建损失
            recon_loss = 0
            for v_idx in range(len(views_batch)):
                recon_loss += F.mse_loss(outputs['reconstructions'][v_idx], views_batch[v_idx])
            recon_loss /= len(views_batch)
            
            recon_loss.backward()
            optimizer.step()
            
            total_loss += recon_loss.item()
        
        if (epoch + 1) % 25 == 0:
            print(f"   Epoch {epoch+1}: recon_loss={total_loss:.4f}")
    
    # 用 K-Means 聚类
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in dataloader:
            views_batch = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            outputs = model(views_batch, mask)
            all_embeddings.append(outputs['consensus'].cpu())
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    predictions = kmeans.fit_predict(embeddings)
    
    metrics = evaluate_clustering(labels, predictions, embeddings)
    print(f"   最终 ACC = {metrics['acc']:.4f}")
    
    return metrics['acc']


if __name__ == "__main__":
    print("="*60)
    print("最终诊断：找出 OT-CFM 训练失败的根本原因")
    print("="*60)
    
    # 测试1：直接聚类
    acc1 = test_direct_clustering()
    
    # 测试2：预训练
    model, optimizer, acc2 = test_trainer_pretraining()
    
    # 测试3：初始化聚类
    acc3 = test_init_clustering(model, optimizer)
    
    # 测试4：主训练（完整损失）
    acc4 = test_main_training(model, optimizer)
    
    # 测试5：不用 CFM 损失
    acc5 = test_train_without_cfm()
    
    # 测试6：简单 AE + K-Means
    acc6 = test_simple_ae_clustering()
    
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print(f"测试1 (直接聚类):         ACC = {acc1:.4f}")
    print(f"测试2 (预训练后):         ACC = {acc2:.4f}")
    print(f"测试3 (聚类初始化后):     ACC = {acc3:.4f}")
    print(f"测试4 (主训练后):         ACC = {acc4:.4f}")
    print(f"测试5 (不用CFM损失):      ACC = {acc5:.4f}")
    print(f"测试6 (简单AE+K-Means):   ACC = {acc6:.4f}")
    
    print("\n诊断结论：")
    if acc4 < 0.5 and acc5 > 0.8:
        print("  → CFM 损失导致了性能下降！")
    elif acc4 < 0.5 and acc3 > 0.8:
        print("  → 主训练阶段破坏了预训练学到的表示！")
    elif acc3 < 0.5 and acc2 > 0.8:
        print("  → 聚类初始化后出现问题！")
    elif acc2 < 0.5:
        print("  → 预训练阶段就已经失败！")
    else:
        print("  → 需要进一步分析...")
