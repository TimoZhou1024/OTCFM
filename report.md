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

---

## 新增功能：Handwritten 和 COIL-20 数据集

### 新增数据集概述

| 数据集 | 样本数 | 视图数 | 类别数 | 来源 |
|--------|--------|--------|--------|------|
| Handwritten (UCI) | 2,000 | 6 | 10 (数字0-9) | UCI ML Repository |
| COIL-20 | 1,440 | 4 | 20 (物体) | Columbia University |

### Handwritten 数据集

**来源**: UCI Machine Learning Repository - Multiple Features Dataset

**6个视图**:
1. `mfeat-fou` - 76维傅里叶系数
2. `mfeat-fac` - 216维 profile correlations
3. `mfeat-kar` - 64维 Karhunen-Loève 系数
4. `mfeat-pix` - 240维像素平均值 (2x3窗口)
5. `mfeat-zer` - 47维 Zernike moments
6. `mfeat-mor` - 6维形态学特征

**数据规模**: 2000个样本 (每类200个)，10个类别 (数字0-9)

### COIL-20 数据集

**来源**: Columbia Object Image Library

**多视图构建方式**: 将72张旋转图像 (每5°一张) 按角度分成4组
- 视图1: 0°, 20°, 40°, ... (18张/物体)
- 视图2: 5°, 25°, 45°, ... (18张/物体)
- 视图3: 10°, 30°, 50°, ... (18张/物体)
- 视图4: 15°, 35°, 55°, ... (18张/物体)

**数据规模**: 1440个样本 (20物体 × 72张), 每视图360个样本

### 实现细节

#### 自动下载功能

```python
DATASET_URLS = {
    'handwritten': 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-{}.txt',
    'coil20': 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
}

def download_and_extract(url: str, save_dir: str, extract: bool = False) -> str:
    """下载文件，可选解压"""
    # 自动处理下载和解压
```

#### 数据集加载器

```python
# 使用方式
from otcfm.datasets import load_handwritten, load_coil20

# 加载 Handwritten
views, labels = load_handwritten(data_root='./data/handwritten')
# views: List[np.ndarray], 6个视图
# labels: np.ndarray, shape (2000,)

# 加载 COIL-20  
views, labels = load_coil20(data_root='./data/coil20')
# views: List[np.ndarray], 4个视图
# labels: np.ndarray, shape (1440,)
```

#### 集成到 DATASET_LOADERS

```python
DATASET_LOADERS = {
    'Synthetic': load_synthetic_dataset,
    'SyntheticIncomplete': load_synthetic_incomplete,
    'NUS-WIDE': load_nus_wide,
    'Handwritten': load_handwritten,  # 新增
    'COIL20': load_coil20,            # 新增
}
```

### 使用示例

```python
# 在配置中指定数据集
config = {
    'data': {
        'name': 'Handwritten',  # 或 'COIL20'
        'root': './data',
        'num_views': 6,  # Handwritten: 6, COIL20: 4
        'num_clusters': 10,  # Handwritten: 10, COIL20: 20
    },
    # ... 其他配置
}
```

### 验证结果

```
>>> from otcfm.datasets import load_handwritten, load_coil20, DATASET_URLS
>>> print('Dataset URLs:', list(DATASET_URLS.keys()))
Dataset URLs: ['handwritten', 'coil20']
>>> print('Imports successful!')
Imports successful!
```

---

## 总结

本次调试工作完成了以下任务：

1. **修复核心 Bug**: 解决了评估时样本顺序不匹配的问题，使 OT-CFM 在合成数据上达到 ACC=1.0
2. **新增数据集支持**: 添加了 Handwritten (UCI) 和 COIL-20 两个真实世界多视图数据集
3. **自动下载功能**: 实现了数据集自动下载和解压功能

OT-CFM 框架现在可以在更多真实数据集上进行评估和测试。

---

## 关键修复：UMVC 中的跨视图操作问题

### 问题发现

在 UMVC（Unaligned Multi-View Clustering，无对齐多视图聚类）场景中，发现了一个**关键逻辑错误**：

> **样本索引 i 在不同视图中没有对应关系！**

这意味着：
- 视图 A 中的样本 i 和视图 B 中的样本 i 是**完全不相关**的
- 任何基于样本索引进行的跨视图操作都是**无意义的**

### 错误的实现（旧代码）

```python
# WRONG: 这假设样本索引 i 在所有视图中对应同一样本
q = torch.stack(per_view_q, dim=0).mean(dim=0)  # [V, B, K] -> [B, K]
assignments = q.argmax(dim=1)
consensus = self.clustering.centroids[assignments]
```

**问题分析**：
- `per_view_q[0][i]` 是视图 0 中样本 i 的软分配
- `per_view_q[1][i]` 是视图 1 中样本 i 的软分配
- 但在 UMVC 中，这两个样本 i **不是同一个样本**！
- 因此平均 `(per_view_q[0][i] + per_view_q[1][i]) / 2` 是**无意义的**

### 正确的实现（新代码）

```python
# CORRECT: 每个视图独立操作，不进行任何跨视图操作
per_view_conditions = []
for z_v in latents:
    q_v, _ = self.clustering(z_v)  # [B, K]
    assignments_v = q_v.argmax(dim=1)  # [B]
    cond_v = self.clustering.centroids[assignments_v]  # [B, D]
    per_view_conditions.append(cond_v)  # 每个视图有自己的条件
```

**关键改动**：
1. 每个视图**独立**计算其聚类分配
2. 每个视图**独立**选择其质心作为条件
3. CFM 损失按**每视图**计算，然后取平均

### 对比：Aligned vs Unaligned

| 方面 | Aligned (IMVC) | Unaligned (UMVC) |
|------|----------------|------------------|
| 样本对应 | 索引 i 跨视图对应 | 索引 i 无对应关系 |
| 共识表示 | `consensus = mean(latents)` | 不存在有效共识 |
| 聚类条件 | 单一共识 → 质心 | 每视图独立 → 质心 |
| 对比损失 | 有效（拉近同一样本的不同视图） | **无效**（被禁用） |
| 软投票 | 可以跨视图平均 | **不可以**跨视图平均 |

### 修改的文件

1. **`ot_cfm.py`** - `forward()` 方法：
   - Unaligned 模式下，`consensus` 设为 `None`
   - 添加 `per_view_conditions`：每视图独立的质心条件
   - 修改 `get_embeddings()` 处理 `consensus=None` 的情况

2. **`losses.py`** - `OTCFMLoss.forward()` 方法：
   - 添加 `per_view_conditions` 参数
   - Unaligned 模式下，CFM 损失按每视图计算
   - 对比损失在 Unaligned 模式下被禁用

3. **`main.tex`** - 论文：
   - 修改 Algorithm 1：明确区分 Aligned 和 Unaligned 的处理
   - 修改 Section 4.1：强调每视图独立操作

### 测试验证

```
=== Testing is_aligned=True ===
  consensus is None: False ✅
  has per_view_conditions: False ✅
  Contrastive loss: 2.1570 ✅ (启用)

=== Testing is_aligned=False ===
  consensus is None: True ✅
  has per_view_conditions: True ✅
  Contrastive loss: 0.0000 ✅ (禁用)

All tests passed!
```

### 教训

1. **UMVC 的本质**：样本在不同视图中没有对应关系，这是 UMVC 与 IMVC 的根本区别
2. **任何跨视图操作都需要样本对应**：平均、对比、投票等操作都隐含假设了样本对应
3. **每视图独立是唯一正确的做法**：在 UMVC 中，每个视图必须作为独立的数据集处理
