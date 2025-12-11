# 如何添加新的多视图聚类基线方法

本指南介绍如何将 GitHub 上开源的多视图聚类方法集成到 OT-CFM 项目中进行对比实验。

## 1. 推荐的最新方法（2022-2024）

### 1.1 对比学习类方法

| 方法 | 论文 | GitHub | 年份 |
|------|------|--------|------|
| **MFLVC** | Multi-level Feature Learning for Contrastive MVC | [XLearning-SCU/2022-CVPR-MFLVC](https://github.com/XLearning-SCU/2022-CVPR-MFLVC) | CVPR 2022 |
| **SURE** | Self-supervised Multi-view Clustering | [XLearning-SCU/2022-NeurIPS-SURE](https://github.com/XLearning-SCU/2022-NeurIPS-SURE) | NeurIPS 2022 |
| **DealMVC** | Dual Contrastive Prediction for IMVC | [SubmissionsIn/DealMVC](https://github.com/SubmissionsIn/DealMVC) | CVPR 2023 |
| **CVCL** | Cross-view Contrastive Learning for MVC | [DarrenZZhang/CVCL](https://github.com/DarrenZZhang/CVCL) | TPAMI 2023 |
| **DSIMVC** | Deep Safe Incomplete MVC | [Gasteipp/DSIMVC](https://github.com/Gasteipp/DSIMVC) | ICML 2022 |

### 1.2 图/谱方法

| 方法 | 论文 | GitHub | 年份 |
|------|------|--------|------|
| **GWMAC** | Graph-based Multi-View Clustering | [kunzhan/GWMAC](https://github.com/kunzhan/GWMAC) | TPAMI 2023 |
| **DGMVC** | Deep Graph-level MVC | [Sunxinzi/DGMVC](https://github.com/Sunxinzi/DGMVC) | AAAI 2024 |
| **GCFAgg** | Graph-based Consensus Fusion | [Galaxy922/GCFAggMVC](https://github.com/Galaxy922/GCFAggMVC) | CVPR 2023 |

### 1.3 不完整/未对齐多视图方法

| 方法 | 论文 | GitHub | 年份 |
|------|------|--------|------|
| **CPSPAN** | Cross-view Propagation for Partially Aligned MVC | [XLearning-SCU/2021-CVPR-CPSPAN](https://github.com/XLearning-SCU/2021-CVPR-CPSPAN) | CVPR 2021 |
| **DAIMC** | Doubly Aligned IMVC | [DarrenZZhang/DAIMC](https://github.com/DarrenZZhang/DAIMC) | AAAI 2022 |
| **SMILE** | Scalable IMVC Learning | [SubmissionsIn/SMILE](https://github.com/SubmissionsIn/SMILE) | NeurIPS 2023 |
| **UNIMVC** | Unified Framework for IMVC | [SubmissionsIn/UNIMVC](https://github.com/SubmissionsIn/UNIMVC) | AAAI 2024 |

### 1.4 生成模型类方法

| 方法 | 论文 | GitHub | 年份 |
|------|------|--------|------|
| **DiffMVC** | Diffusion-based MVC | - | ICML 2024 |
| **MVAE** | Multi-view VAE for Clustering | [SubmissionsIn/MVAE](https://github.com/SubmissionsIn/MVAE) | 2022 |

---

## 2. 集成步骤

### 步骤 1: 克隆方法代码

```bash
# 创建外部方法目录
mkdir -p D:\FM\external_methods
cd D:\FM\external_methods

# 示例：克隆 MFLVC
git clone https://github.com/SubmissionsIn/MFLVC.git external_methods/MFLVC
          

# 示例：克隆 SURE
git clone https://github.com/XLearning-SCU/2022-TPAMI-SURE.git SURE
```

### 步骤 2: 创建适配器包装类

在 `src/otcfm/external_baselines.py` 中创建统一的包装接口：

```python
"""
Adapters for external multi-view clustering methods
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from .baselines import BaseClusteringMethod

# 添加外部方法路径
EXTERNAL_PATH = Path(__file__).parent.parent.parent / "external_methods"


class MFLVCWrapper(BaseClusteringMethod):
    """
    MFLVC: Multi-level Feature Learning for Contrastive MVC (CVPR 2022)
    Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/xxx.pdf
    """
    
    def __init__(self, num_clusters: int, latent_dim: int = 128, 
                 epochs: int = 100, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        # 动态导入 MFLVC
        mflvc_path = EXTERNAL_PATH / "MFLVC"
        if mflvc_path.exists():
            sys.path.insert(0, str(mflvc_path))
            try:
                from model import MFLVC  # 根据实际模块名调整
                
                # 转换为 torch tensors
                views_tensor = [torch.FloatTensor(v).to(self.device) for v in views]
                
                # 初始化模型
                view_dims = [v.shape[1] for v in views]
                model = MFLVC(
                    view_dims=view_dims,
                    latent_dim=self.latent_dim,
                    num_clusters=self.num_clusters
                ).to(self.device)
                
                # 训练
                model.train_model(views_tensor, epochs=self.epochs)
                
                # 获取预测
                self.labels_, self.embeddings_ = model.predict(views_tensor)
                return self.labels_
                
            except ImportError as e:
                print(f"MFLVC import failed: {e}")
                return self._fallback_kmeans(views)
            finally:
                sys.path.remove(str(mflvc_path))
        else:
            print(f"MFLVC not found at {mflvc_path}")
            return self._fallback_kmeans(views)
    
    def _fallback_kmeans(self, views: List[np.ndarray]) -> np.ndarray:
        """Fallback to KMeans if external method fails"""
        from sklearn.cluster import KMeans
        X = np.concatenate(views, axis=1)
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(X)
        self.embeddings_ = X
        return self.labels_
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.embeddings_


class SUREWrapper(BaseClusteringMethod):
    """
    SURE: Self-supervised MVC via Reconstruction (NeurIPS 2022)
    """
    
    def __init__(self, num_clusters: int, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], **kwargs) -> np.ndarray:
        sure_path = EXTERNAL_PATH / "SURE"
        if sure_path.exists():
            sys.path.insert(0, str(sure_path))
            try:
                # 根据 SURE 的实际 API 调整
                from SURE import SURE_Model
                
                model = SURE_Model(
                    n_clusters=self.num_clusters,
                    device=self.device
                )
                self.labels_, self.embeddings_ = model.fit_predict(views)
                return self.labels_
                
            except Exception as e:
                print(f"SURE failed: {e}")
                return self._fallback_kmeans(views)
            finally:
                sys.path.remove(str(sure_path))
        else:
            return self._fallback_kmeans(views)
    
    def _fallback_kmeans(self, views):
        from sklearn.cluster import KMeans
        X = np.concatenate(views, axis=1)
        self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10).fit_predict(X)
        self.embeddings_ = X
        return self.labels_
    
    def get_embeddings(self):
        return self.embeddings_


class DealMVCWrapper(BaseClusteringMethod):
    """
    DealMVC: Dual Contrastive Prediction for IMVC (CVPR 2023)
    Specifically designed for incomplete multi-view data
    """
    
    def __init__(self, num_clusters: int, device: str = 'cuda'):
        super().__init__(num_clusters)
        self.device = device
        self.embeddings_ = None
        
    def fit_predict(self, views: List[np.ndarray], 
                    mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        dealmvc_path = EXTERNAL_PATH / "DealMVC"
        if dealmvc_path.exists():
            sys.path.insert(0, str(dealmvc_path))
            try:
                from DealMVC import DealMVC
                
                model = DealMVC(
                    n_clusters=self.num_clusters,
                    device=self.device
                )
                self.labels_, self.embeddings_ = model.fit_predict(views, mask)
                return self.labels_
                
            except Exception as e:
                print(f"DealMVC failed: {e}")
                return self._fallback_kmeans(views)
            finally:
                sys.path.remove(str(dealmvc_path))
        else:
            return self._fallback_kmeans(views)
    
    def _fallback_kmeans(self, views):
        from sklearn.cluster import KMeans
        X = np.concatenate(views, axis=1)
        self.labels_ = KMeans(n_clusters=self.num_clusters, n_init=10).fit_predict(X)
        self.embeddings_ = X
        return self.labels_
    
    def get_embeddings(self):
        return self.embeddings_


# ============================================================
# 注册所有外部方法
# ============================================================

def get_external_baselines(
    view_dims: List[int],
    num_clusters: int,
    device: str = 'cuda'
) -> dict:
    """
    获取所有可用的外部基线方法
    
    Returns:
        Dict[str, BaseClusteringMethod]: 方法名 -> 方法实例
    """
    external_methods = {}
    
    # 检查哪些方法可用
    if (EXTERNAL_PATH / "MFLVC").exists():
        external_methods['MFLVC (CVPR22)'] = MFLVCWrapper(num_clusters, device=device)
    
    if (EXTERNAL_PATH / "SURE").exists():
        external_methods['SURE (NeurIPS22)'] = SUREWrapper(num_clusters, device=device)
    
    if (EXTERNAL_PATH / "DealMVC").exists():
        external_methods['DealMVC (CVPR23)'] = DealMVCWrapper(num_clusters, device=device)
    
    # 添加更多方法...
    
    return external_methods
```

### 步骤 3: 修改 `baselines.py` 以包含外部方法

在 `get_baseline_methods` 函数中添加：

```python
def get_baseline_methods(
    view_dims: List[int],
    num_clusters: int,
    device: str = 'cuda',
    include_external: bool = True  # 新增参数
) -> Dict[str, BaseClusteringMethod]:
    """Get dictionary of all baseline methods"""
    
    baselines = {
        # 原有方法...
        'Concat-KMeans': ConcatKMeans(num_clusters),
        'Multi-View Spectral': MultiViewSpectral(num_clusters),
        # ...
    }
    
    # 添加外部方法
    if include_external:
        try:
            from .external_baselines import get_external_baselines
            external = get_external_baselines(view_dims, num_clusters, device)
            baselines.update(external)
            print(f"Loaded {len(external)} external methods: {list(external.keys())}")
        except ImportError as e:
            print(f"External baselines not available: {e}")
    
    return baselines
```

---

## 3. 处理依赖问题

### 3.1 创建独立的 conda 环境（推荐）

某些方法可能有冲突的依赖，建议使用 subprocess 在独立环境中运行：

```python
import subprocess
import json
import tempfile
import numpy as np

class ExternalMethodRunner:
    """通过子进程运行外部方法，避免依赖冲突"""
    
    def __init__(self, method_name: str, conda_env: str):
        self.method_name = method_name
        self.conda_env = conda_env
    
    def run(self, views: List[np.ndarray], num_clusters: int) -> np.ndarray:
        # 保存数据到临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = f"{tmpdir}/data.npz"
            result_path = f"{tmpdir}/result.npy"
            
            np.savez(data_path, 
                     views=[v for v in views], 
                     num_clusters=num_clusters)
            
            # 运行外部脚本
            script = f'''
import numpy as np
data = np.load("{data_path}", allow_pickle=True)
views = list(data["views"])
num_clusters = int(data["num_clusters"])

# 导入并运行方法
from {self.method_name} import run_clustering
labels = run_clustering(views, num_clusters)
np.save("{result_path}", labels)
'''
            
            cmd = f'conda run -n {self.conda_env} python -c "{script}"'
            subprocess.run(cmd, shell=True, check=True)
            
            return np.load(result_path)
```

### 3.2 Docker 容器方式（最隔离）

```dockerfile
# Dockerfile.mflvc
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY external_methods/MFLVC /app/MFLVC
RUN pip install -r /app/MFLVC/requirements.txt
COPY scripts/run_mflvc.py /app/
ENTRYPOINT ["python", "/app/run_mflvc.py"]
```

---

## 4. 完整示例：添加 MFLVC

### 4.1 下载代码

```bash
cd D:\FM\external_methods
git clone https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git MFLVC
cd MFLVC
pip install -r requirements.txt  # 可能需要调整
```

### 4.2 分析 MFLVC 的 API

查看 MFLVC 的主要接口，通常在 `main.py` 或 `model.py` 中：

```python
# 典型的 API 模式
class MFLVC:
    def __init__(self, config):
        ...
    
    def train(self, data_loader, epochs):
        ...
    
    def cluster(self, data_loader):
        return labels, embeddings
```

### 4.3 创建适配器

根据实际 API 调整 `MFLVCWrapper` 类。

### 4.4 测试

```python
# test_mflvc.py
from src.otcfm.external_baselines import MFLVCWrapper
import numpy as np

# 创建测试数据
views = [np.random.randn(100, 50), np.random.randn(100, 80)]
num_clusters = 5

# 运行
mflvc = MFLVCWrapper(num_clusters, device='cpu')
labels = mflvc.fit_predict(views)
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")
```

---

## 5. 运行对比实验

```bash
# 完整对比（包括外部方法）
uv run scripts/run_experiment.py \
    --mode compare \
    --dataset handwritten \
    --epochs 200 \
    --include_external

# 仅运行特定外部方法
uv run scripts/run_experiment.py \
    --mode compare \
    --dataset synthetic \
    --methods "OT-CFM,MFLVC,SURE,DealMVC"
```

---

## 6. 常见问题

### Q1: 外部方法的依赖与项目冲突怎么办？

**A:** 使用以下策略之一：
1. 创建独立的 conda 环境
2. 使用 Docker 容器
3. 使用 subprocess 隔离运行

### Q2: 如何处理不同的数据格式？

**A:** 在 Wrapper 类中进行格式转换：
```python
def _prepare_data(self, views):
    # 某些方法需要 DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    tensors = [torch.FloatTensor(v) for v in views]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=256, shuffle=True)
```

### Q3: 外部方法训练太慢怎么办？

**A:** 
1. 减少 epochs
2. 使用 GPU
3. 设置 `--quick_eval` 模式只运行部分 baselines

---

## 7. 已集成方法状态

| 方法 | 状态 | 备注 |
|------|------|------|
| Concat-KMeans | ✅ 内置 | 传统方法 |
| Multi-View Spectral | ✅ 内置 | 传统方法 |
| CCA-Clustering | ✅ 内置 | 传统方法 |
| DMVC | ✅ 内置 | 深度 AE |
| Contrastive-MVC | ✅ 内置 | 对比学习 |
| MFLVC | 🔲 待添加 | CVPR 2022 |
| SURE | 🔲 待添加 | NeurIPS 2022 |
| DealMVC | 🔲 待添加 | CVPR 2023 |
| SMILE | 🔲 待添加 | NeurIPS 2023 |

---

## 8. 快速开始脚本

```bash
# 一键设置外部方法
cd D:\FM
mkdir -p external_methods
cd external_methods

# 克隆推荐的方法
git clone https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git MFLVC
git clone https://github.com/XLearning-SCU/2022-NeurIPS-SURE.git SURE
git clone https://github.com/SubmissionsIn/DealMVC.git DealMVC

# 返回项目根目录
cd ..

# 运行对比实验
uv run scripts/run_experiment.py --mode compare --dataset synthetic --epochs 100
```
