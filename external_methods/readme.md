# External Multi-View Clustering Methods

本项目集成了多个SOTA多视图聚类方法用于对比实验。根据各方法所处理的核心问题，将其分为三类：

## 方法分类

### 🔴 专注于 Incomplete Data (处理缺失视图)

| Method | Venue | Paper | 特点 |
|--------|-------|-------|------|
| **COMPLETER** | CVPR 2021 | Incomplete Multi-view Clustering via Contrastive Prediction | 对比预测补全缺失视图 |
| **SURE** | TPAMI 2022 | Robust Multi-View Clustering with Incomplete Information | 鲁棒损失处理不完整信息 |
| **DealMVC** | CVPR 2023 | Dual Contrastive Prediction for Incomplete MVC | 双对比预测机制 |
| **DCG** | AAAI 2025 | Diffusion-based Cross-view Generation for Incomplete MVC | 扩散模型生成缺失视图 |

### 🔵 专注于 Unaligned Data (处理跨视图对齐)

| Method | Venue | Paper | 特点 |
|--------|-------|-------|------|
| **MRG-UMC** | TNNLS 2025 | Multi-level Reliable Guidance for Unpaired MVC | 多层可靠引导处理无配对数据 |
| **CANDY** | NeurIPS 2024 | Robust Contrastive MVC against Dual Noisy Correspondence | 处理双重噪声对应（FP+FN） |

### 🟢 通用方法 (General Methods)

| Method | Venue | Paper | 特点 |
|--------|-------|-------|------|
| **MFLVC** | CVPR 2022 | Multi-level Feature Learning for Contrastive MVC | 多层特征对比学习 |
| **GCFAggMVC** | CVPR 2023 | Global and Cross-view Feature Aggregation | 全局与跨视图特征聚合 |

## 克隆方法

```bash
cd external_methods

# === Incomplete Data Methods ===
# COMPLETER (CVPR 2021)
git clone https://github.com/XLearning-SCU/2021-CVPR-Completer.git COMPLETER

# SURE (TPAMI 2022)
git clone https://github.com/XLearning-SCU/2022-NeurIPS-SURE.git SURE

# DealMVC (CVPR 2023)
git clone https://github.com/SubmissionsIn/DealMVC.git DealMVC

# DCG (AAAI 2025)
git clone https://github.com/zhangyuanyang21/2025-AAAI-DCG.git 2025-AAAI-DCG

# === Unaligned Data Methods ===
# MRG-UMC (TNNLS 2025)
git clone https://github.com/LikeXin94/MRG-UMC.git MRG-UMC

# CANDY (NeurIPS 2024)
git clone https://github.com/XLearning-SCU/2024-NeurIPS-CANDY.git 2024-NeurIPS-CANDY

# === General Methods ===
# MFLVC (CVPR 2022)
git clone https://github.com/XLearning-SCU/2022-CVPR-MFLVC.git MFLVC

# GCFAggMVC (CVPR 2023)
git clone https://github.com/Galaxy922/GCFAggMVC.git GCFAggMVC
```

## 鲁棒性测试说明

根据方法分类，鲁棒性测试会自动选择适当的对比方法：

- **Incomplete Data Test**: OT-CFM vs Incomplete Methods + General Methods
- **Unaligned Data Test**: OT-CFM vs Unaligned Methods + General Methods

这样可以更公平地评估各方法在其专长领域的表现。
