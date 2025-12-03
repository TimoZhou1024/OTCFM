# OT-CFM 调试报告

## 问题描述

OT-CFM 模型在合成数据集上的聚类性能极差（ACC ≈ 14-16%），而所有基线方法都达到了 100% 的准确率。

## 调试过程

### 排除的可能原因

1. **DataLoader 数据损坏** - 验证数据通过 DataLoader 后保持一致
2. **StandardScaler 破坏聚类结构** - 标准化后仍能达到 100% ACC
3. **神经网络结构问题** - 单独测试各组件都正常工作
4. **损失函数问题** - CFM、GW、聚类损失等单独测试都正常
5. **多视图融合问题** - 使用全批次训练时多视图 AE 也能达到 100%

### 根本原因

**评估时样本顺序不匹配！**

当使用 `shuffle=True` 的 DataLoader 时：
- 每个 epoch 的样本顺序被随机打乱
- 评估代码收集 embeddings/predictions 时没有记录原始索引
- 导致评估时 embeddings 的顺序与 labels 不对应
- ACC 计算结果错误（约等于随机猜测的 10%）

### 验证实验

```python
# 未排序的评估（错误）
ACC = 0.14  # 接近随机

# 按 indices 排序后的评估（正确）
ACC = 1.00  # 完美分类
```

## 修复方案

修改 `trainer.py` 和 `ot_cfm.py` 中的评估相关方法，收集 `batch['indices']` 并按原始顺序排序：

### 修改的方法

1. `Trainer._evaluate()` - 添加 indices 收集和排序
2. `Trainer._update_clustering()` - 添加 indices 收集和排序
3. `OTCFM.init_clustering()` - 添加 indices 收集和排序
4. `OTCFMTrainer.update_clustering()` - 添加 indices 收集和排序
5. `OTCFMTrainer.evaluate()` - 添加 indices 收集和排序

### 关键代码改动

```python
@torch.no_grad()
def _evaluate(self, dataloader: DataLoader, labels: np.ndarray) -> Dict:
    self.model.eval()
    all_embeddings = []
    all_predictions = []
    all_indices = []
    
    for batch in dataloader:
        views = [v.to(self.device) for v in batch['views']]
        mask = batch['mask'].to(self.device)
        indices = batch['indices']  # 获取样本索引
        
        outputs = self.model(views, mask)
        all_embeddings.append(outputs['consensus'].cpu())
        all_predictions.append(outputs['assignments'].cpu())
        all_indices.append(indices)
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    indices = torch.cat(all_indices, dim=0).numpy()
    
    # 按原始顺序排序，确保与 labels 对应
    order = np.argsort(indices)
    embeddings = embeddings[order]
    predictions = predictions[order]
    
    metrics = evaluate_clustering(labels, predictions, embeddings)
    return metrics
```

## 修复后结果

```
============================================================
Method Comparison
============================================================
Method                    ACC        NMI        ARI
------------------------------------------------------------
OT-CFM                    1.0000     1.0000     1.0000
Concat-KMeans             1.0000     1.0000     1.0000
Multi-View Spectral       1.0000     1.0000     1.0000
CCA-Clustering            1.0000     1.0000     1.0000
Weighted-View             1.0000     1.0000     1.0000
DMVC                      1.0000     1.0000     1.0000
Contrastive-MVC           1.0000     1.0000     1.0000
Incomplete-MVC            1.0000     1.0000     1.0000
Unaligned-MVC             0.9960     0.9918     0.9919
```

## 教训

1. 使用 `shuffle=True` 的 DataLoader 时，评估必须记录原始索引
2. 调试时应该同时检查数据流和评估流程
3. 当性能接近随机基线时，首先检查标签对应关系
