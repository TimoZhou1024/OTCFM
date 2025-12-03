"""
优化和对比实验脚本
分析 OT-CFM 性能问题并进行调优
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import get_default_config, ModelConfig, TrainingConfig
from otcfm.datasets import (
    create_synthetic_multiview, MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.baselines import run_baseline_comparison
from otcfm.metrics import evaluate_clustering


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def create_harder_synthetic_data(
    n_samples: int = 2000,
    n_clusters: int = 10,
    n_views: int = 3,
    view_dims: List[int] = None,
    noise_level: float = 1.0,  # 更高的噪声
    cluster_std: float = 1.5,   # 聚类内方差
    seed: int = 42
) -> Dict:
    """
    创建更难的合成数据集
    - 更高的噪声水平
    - 聚类之间有更多重叠
    - 不同视图有不同的聚类结构
    """
    np.random.seed(seed)
    
    if view_dims is None:
        view_dims = [100, 100, 100]
    
    # 生成标签
    labels = np.random.randint(0, n_clusters, n_samples)
    
    views = []
    for v, dim in enumerate(view_dims):
        # 每个视图有不同的聚类中心（部分共享结构）
        base_centers = np.random.randn(n_clusters, dim) * 2
        
        # 添加视图特定的偏移
        view_offset = np.random.randn(n_clusters, dim) * 0.5
        centers = base_centers + view_offset
        
        # 生成样本
        X = np.zeros((n_samples, dim), dtype=np.float32)
        for i in range(n_samples):
            # 聚类内的高斯噪声
            X[i] = centers[labels[i]] + np.random.randn(dim) * cluster_std
            # 额外的随机噪声
            X[i] += np.random.randn(dim) * noise_level
        
        views.append(X)
    
    return {
        'views': views,
        'labels': labels
    }


def train_otcfm_with_config(
    views: List[np.ndarray],
    labels: np.ndarray,
    config_overrides: Dict = None,
    epochs: int = 100,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    使用指定配置训练 OT-CFM
    """
    view_dims = [v.shape[1] for v in views]
    num_clusters = len(np.unique(labels))
    
    # 创建数据集
    dataset = MultiViewDataset(views=views, labels=labels)
    dataloader = create_dataloader(dataset, batch_size=256, shuffle=True)
    
    # 默认配置
    model_config = {
        'latent_dim': 128,
        'hidden_dims': [512, 256],
        'num_clusters': num_clusters,
        'flow_hidden_dim': 256,
        'flow_num_layers': 4,
        'time_dim': 64,
        'ode_steps': 10,
        'sigma_min': 1e-4,
        'kernel_type': 'rbf',
        'kernel_gamma': 1.0,
        'lambda_gw': 0.1,
        'lambda_cluster': 0.5,
        'lambda_recon': 1.0,
        'lambda_contrastive': 0.1,
        'dropout': 0.1
    }
    
    # 覆盖配置
    if config_overrides:
        model_config.update(config_overrides)
    
    # 创建模型
    model = OTCFM(view_dims=view_dims, **model_config)
    
    # 创建训练器
    training_config = TrainingConfig()
    training_config.epochs = epochs
    training_config.learning_rate = 1e-3
    training_config.device = device
    
    trainer = Trainer(
        model=model,
        config=training_config,
        experiment_dir=f"experiments/optimize_test",
        device=device
    )
    
    # 训练
    if verbose:
        print(f"Training with config: {config_overrides}")
    
    results = trainer.train(dataloader, labels)
    
    return {
        'final': results['final'],
        'best': results['best'],
        'config': model_config
    }


def hyperparameter_search(
    views: List[np.ndarray],
    labels: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """
    超参数搜索
    """
    results = {}
    
    # 参数组合
    param_grid = [
        # 1. 增加聚类损失权重
        {'name': 'high_cluster', 'lambda_cluster': 2.0, 'lambda_recon': 0.5},
        
        # 2. 减少 CFM 的影响（通过增加其他损失）
        {'name': 'low_flow', 'lambda_cluster': 1.0, 'lambda_gw': 0.5, 'lambda_contrastive': 0.5},
        
        # 3. 使用更大的潜在空间
        {'name': 'larger_latent', 'latent_dim': 256, 'lambda_cluster': 1.0},
        
        # 4. 更深的网络
        {'name': 'deeper', 'hidden_dims': [512, 256, 128], 'flow_num_layers': 6},
        
        # 5. 强调对比学习
        {'name': 'contrastive', 'lambda_contrastive': 1.0, 'lambda_cluster': 0.5},
        
        # 6. 平衡配置
        {'name': 'balanced', 'lambda_cluster': 1.0, 'lambda_gw': 0.2, 
         'lambda_recon': 0.5, 'lambda_contrastive': 0.3},
    ]
    
    for params in param_grid:
        name = params.pop('name')
        print(f"\n{'='*60}")
        print(f"Testing configuration: {name}")
        print(f"{'='*60}")
        
        try:
            result = train_otcfm_with_config(
                views, labels, 
                config_overrides=params,
                epochs=50,  # 快速测试
                device=device,
                verbose=True
            )
            results[name] = {
                'acc': result['best']['acc'],
                'nmi': result['best']['nmi'],
                'ari': result['best'].get('ari', 0),
                'config': params
            }
            print(f"Best ACC: {result['best']['acc']:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            results[name] = {'error': str(e)}
    
    return results


def run_comparison_on_harder_data():
    """
    在更难的数据集上运行对比实验
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建更难的数据集
    print("\n创建更难的合成数据集...")
    data = create_harder_synthetic_data(
        n_samples=2000,
        n_clusters=10,
        noise_level=1.0,
        cluster_std=1.5,
        seed=42
    )
    views = data['views']
    labels = data['labels']
    
    print(f"数据集信息:")
    print(f"  样本数: {len(labels)}")
    print(f"  视图数: {len(views)}")
    print(f"  视图维度: {[v.shape[1] for v in views]}")
    print(f"  聚类数: {len(np.unique(labels))}")
    
    # 运行 baseline
    print("\n" + "="*60)
    print("运行 Baseline 方法...")
    print("="*60)
    
    baseline_results = run_baseline_comparison(
        views, labels, 
        num_clusters=10, 
        device=device
    )
    
    # 运行超参数搜索
    print("\n" + "="*60)
    print("运行 OT-CFM 超参数搜索...")
    print("="*60)
    
    otcfm_results = hyperparameter_search(views, labels, device=device)
    
    # 汇总结果
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    print("\nBaseline 方法:")
    print(f"{'方法':<25} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-"*55)
    for name, metrics in sorted(baseline_results.items(), key=lambda x: -x[1].get('acc', 0)):
        if 'acc' in metrics:
            print(f"{name:<25} {metrics['acc']:<10.4f} {metrics['nmi']:<10.4f} {metrics.get('ari', 0):<10.4f}")
    
    print("\nOT-CFM 配置:")
    print(f"{'配置':<25} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-"*55)
    for name, metrics in sorted(otcfm_results.items(), key=lambda x: -x[1].get('acc', 0) if 'acc' in x[1] else 0):
        if 'acc' in metrics:
            print(f"{name:<25} {metrics['acc']:<10.4f} {metrics['nmi']:<10.4f} {metrics.get('ari', 0):<10.4f}")
    
    # 保存结果
    all_results = {
        'baselines': baseline_results,
        'otcfm_configs': otcfm_results
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/optimization_results.json', 'w') as f:
        # 转换 numpy 类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(all_results), f, indent=2)
    
    print("\n结果已保存到 experiments/optimization_results.json")
    
    return all_results


def analyze_and_fix():
    """
    分析问题并提出修复建议
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OT-CFM 性能分析报告                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

问题诊断:
─────────────────────────────────────────────────────────────────────────────────
1. 合成数据集太简单
   - 原始 noise_level=0.1 导致聚类结构过于明显
   - 所有 baseline 方法都能达到 100% 准确率
   
2. CFM 损失函数设计问题
   - 当前使用 consensus 作为条件和目标，可能导致梯度信号不足
   - 建议: 使用交叉视图预测，增强表示学习
   
3. 损失权重不平衡
   - lambda_cluster=0.5 可能太小
   - 重建损失(lambda_recon=1.0)可能过度主导训练
   
4. 训练策略
   - 缺少预训练阶段
   - 聚类中心更新频率可能不合适

建议的优化方向:
─────────────────────────────────────────────────────────────────────────────────
1. 增加聚类损失权重: lambda_cluster >= 1.0
2. 添加预训练阶段: 先训练编码器-解码器
3. 使用对比学习增强特征: lambda_contrastive >= 0.3  
4. 减少重建损失: lambda_recon <= 0.5
5. 使用更大的潜在空间: latent_dim >= 256 对于复杂数据

建议运行:
    python scripts/optimize_and_compare.py

这将在更难的数据集上测试不同配置。
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'quick', 'analyze'])
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_and_fix()
    elif args.mode == 'quick':
        # 快速测试
        device = get_device()
        data = create_harder_synthetic_data(n_samples=1000, n_clusters=5)
        result = train_otcfm_with_config(
            data['views'], data['labels'],
            config_overrides={'lambda_cluster': 1.0},
            epochs=30,
            device=device
        )
        print(f"\nQuick test result: ACC={result['best']['acc']:.4f}")
    else:
        run_comparison_on_harder_data()
