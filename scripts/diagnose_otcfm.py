"""
最终诊断：确定 OT-CFM 中的问题组件
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from otcfm.datasets import create_synthetic_multiview, MultiViewDataset, create_dataloader
from otcfm.ot_cfm import OTCFM


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(D)]) * 1.0 / y_pred.size


def test_otcfm_components():
    np.random.seed(42)
    torch.manual_seed(42)
    
    views, labels = create_synthetic_multiview(
        n_samples=1000, n_clusters=10, n_views=3,
        view_dims=[100, 100, 100], noise_level=0.1, seed=42
    )
    
    dataset = MultiViewDataset(views=views, labels=labels)
    loader = create_dataloader(dataset, batch_size=256, shuffle=True)
    loader_eval = create_dataloader(dataset, batch_size=1000, shuffle=False)
    
    view_dims = [100, 100, 100]
    
    print("="*70)
    print("测试 OT-CFM 各组件")
    print("="*70)
    
    # 1. 测试只用重建损失
    print("\n1. 只用重建损失训练 OT-CFM")
    
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=128,
        num_clusters=10,
        lambda_gw=0.0,
        lambda_cluster=0.0,
        lambda_contrastive=0.0,
        lambda_recon=1.0,
        use_cross_view_flow=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            
            outputs = model(batch['views'], batch['mask'], return_all=True)
            
            # 只用重建损失
            loss = sum(
                F.mse_loss(r, v) 
                for r, v in zip(outputs['reconstructions'], batch['views'])
            ) / 3
            
            loss.backward()
            optimizer.step()
    
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            outputs = model(batch['views'], batch['mask'])
            all_z.append(outputs['consensus'])
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   ACC = {acc:.4f}")
    
    # 2. 测试重建 + 对比损失
    print("\n2. 重建 + 对比损失")
    
    model2 = OTCFM(
        view_dims=view_dims,
        latent_dim=128,
        num_clusters=10,
        lambda_gw=0.0,
        lambda_cluster=0.0,
        lambda_contrastive=0.5,
        lambda_recon=1.0,
        use_cross_view_flow=False
    )
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model2.train()
        for batch in loader:
            optimizer2.zero_grad()
            
            outputs = model2(batch['views'], batch['mask'], return_all=True)
            
            # 重建损失
            recon_loss = sum(
                F.mse_loss(r, v) 
                for r, v in zip(outputs['reconstructions'], batch['views'])
            ) / 3
            
            # 对比损失
            latents = outputs['latents']
            contrastive_loss = 0
            for i in range(len(latents)):
                for j in range(i + 1, len(latents)):
                    z_i = F.normalize(latents[i], dim=-1)
                    z_j = F.normalize(latents[j], dim=-1)
                    contrastive_loss += 1 - (z_i * z_j).sum(dim=-1).mean()
            contrastive_loss /= 3
            
            loss = recon_loss + 0.5 * contrastive_loss
            loss.backward()
            optimizer2.step()
    
    model2.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            outputs = model2(batch['views'], batch['mask'])
            all_z.append(outputs['consensus'])
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   ACC = {acc:.4f}")
    
    # 3. 测试使用 compute_loss
    print("\n3. 使用 model.compute_loss() 训练")
    
    model3 = OTCFM(
        view_dims=view_dims,
        latent_dim=128,
        num_clusters=10,
        lambda_gw=0.0,
        lambda_cluster=0.0,
        lambda_contrastive=0.0,
        lambda_recon=1.0,
        use_cross_view_flow=False
    )
    
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model3.train()
        for batch in loader:
            optimizer3.zero_grad()
            
            loss, loss_dict = model3.compute_loss(batch['views'], batch['mask'])
            
            loss.backward()
            optimizer3.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"recon={loss_dict.get('recon', 0):.4f}, "
                  f"cfm={loss_dict.get('cfm', 0):.4f}")
    
    model3.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            outputs = model3(batch['views'], batch['mask'])
            all_z.append(outputs['consensus'])
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   ACC = {acc:.4f}")
    
    # 4. 测试禁用 CFM
    print("\n4. 使用 ablation_mode='no_flow'")
    
    model4 = OTCFM(
        view_dims=view_dims,
        latent_dim=128,
        num_clusters=10,
        lambda_gw=0.0,
        lambda_cluster=0.0,
        lambda_contrastive=0.0,
        lambda_recon=1.0,
        use_cross_view_flow=False
    )
    
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model4.train()
        for batch in loader:
            optimizer4.zero_grad()
            
            loss, loss_dict = model4.compute_loss(
                batch['views'], batch['mask'], ablation_mode='no_flow'
            )
            
            loss.backward()
            optimizer4.step()
    
    model4.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader_eval:
            outputs = model4(batch['views'], batch['mask'])
            all_z.append(outputs['consensus'])
    
    z = torch.cat(all_z, dim=0).numpy()
    pred = KMeans(n_clusters=10, n_init=20, random_state=42).fit_predict(z)
    acc = cluster_acc(labels, pred)
    print(f"   ACC = {acc:.4f}")
    
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    print("""
如果使用 compute_loss 的结果很差，但手动计算损失结果很好，
那么问题在 OTCFMLoss 类中。

让我们检查 CFM 损失是否破坏了表示学习。
""")


if __name__ == "__main__":
    test_otcfm_components()
