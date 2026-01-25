# Multi-Seed Convergence Analysis

完整的多次运行收敛分析系统，支持统计显著性分析和专业的可视化。

## 📋 功能特性

- ✅ 多个随机种子训练（默认5次）
- ✅ 自动记录所有epoch的指标和损失
- ✅ 计算均值和标准差
- ✅ 专业科学出版风格可视化
- ✅ 均值曲线 + 标准差阴影区域
- ✅ 2个子图垂直布局（指标 + 损失）

## 🚀 快速开始

### Step 1: 运行多次训练

```bash
# 基础用法（5个种子，100 epochs）
uv run python scripts/run_multi_seed_convergence.py --dataset Handwritten --epochs 100 --n_seeds 5

# 快速测试（3个种子，50 epochs）
uv run python scripts/run_multi_seed_convergence.py --dataset Handwritten --epochs 50 --n_seeds 3

# 高精度分析（10个种子，150 epochs）
uv run python scripts/run_multi_seed_convergence.py --dataset Scene15 --epochs 150 --n_seeds 10

# 指定输出目录
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Handwritten \
    --epochs 100 \
    --n_seeds 5 \
    --output_dir my_convergence_analysis
```

**输出**：
- `multi_seed_results/{dataset}_{timestamp}/`
  - `seed_{i}/` - 每个种子的完整训练历史
  - `aggregated_results.json` - 聚合的统计数据
  - `experiment_config.json` - 实验配置

### Step 2: 可视化结果

```bash
# 基础用法（自动保存到结果目录）
uv run python scripts/plot_multi_seed_convergence.py \
    --results_dir multi_seed_results/Handwritten_20260122_120000

# 指定输出路径
uv run python scripts/plot_multi_seed_convergence.py \
    --results_dir multi_seed_results/Handwritten_20260122_120000 \
    --output my_convergence_figure.pdf
```

**输出**：
- 结果目录中的 `convergence_plot.pdf`
- `figures/{dataset}_multi_seed_convergence.pdf`（论文用图）
- 控制台打印统计表格

## 📊 输出示例

### 控制台输出

```
================================================================================
                          Statistical Summary                          
================================================================================

                    Clustering Metrics (%)                    
--------------------------------------------------------------------------------
Metric       Mean         Std          Min          Max         
--------------------------------------------------------------------------------
ACC              95.24%       1.32%       93.50%       96.80%
NMI              88.15%       1.08%       86.75%       89.45%
ARI              82.36%       1.95%       79.80%       84.90%
PURITY           95.18%       1.25%       93.60%       96.70%
F1               94.52%       1.41%       92.80%       96.10%

                         Loss Components                         
--------------------------------------------------------------------------------
Loss            Mean            Std             Min             Max        
--------------------------------------------------------------------------------
Loss                0.4856         0.0234         0.4582         0.5145
Recon               0.1524         0.0089         0.1420         0.1635
Gw                  0.0523         0.0045         0.0465         0.0589
Cluster             0.2345         0.0156         0.2145         0.2567
Contrastive         0.0312         0.0028         0.0275         0.0348
Cfm                 0.0152         0.0015         0.0132         0.0172
================================================================================
```

### 生成的图表

- **上图**：ACC, NMI, ARI曲线（带标准差阴影）
- **下图**：Total Loss, Reconstruction, GW, Clustering曲线（带标准差阴影）
- **风格**：白色背景，网格线，粗体标签，seaborn deep配色
- **格式**：PDF，300 DPI，适合论文发表

## 🔧 高级选项

### 自定义种子范围

```bash
# 使用种子 100-104
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Handwritten \
    --n_seeds 5 \
    --start_seed 100
```

### 模拟缺失和非对齐数据

```bash
# 30%缺失，50%非对齐
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Handwritten \
    --epochs 100 \
    --n_seeds 5 \
    --missing_rate 0.3 \
    --unaligned_rate 0.5
```

## 📁 数据结构

### aggregated_results.json

```json
{
  "n_seeds": 5,
  "n_epochs": 100,
  "epochs": [0, 1, 2, ..., 99],
  "metrics": {
    "acc": {
      "mean": [0.748, 0.856, 0.912, ..., 0.952],
      "std": [0.023, 0.018, 0.015, ..., 0.013],
      "min": [0.720, 0.835, 0.895, ..., 0.935],
      "max": [0.775, 0.880, 0.930, ..., 0.968]
    },
    "nmi": {...},
    "ari": {...}
  },
  "losses": {
    "loss": {
      "mean": [1.226, 0.856, 0.645, ..., 0.486],
      "std": [0.045, 0.032, 0.028, ..., 0.023],
      ...
    },
    "recon": {...},
    "gw": {...},
    "cluster": {...}
  },
  "final_stats": {
    "metrics": {
      "acc": {"mean": 0.9524, "std": 0.0132, ...}
    },
    "losses": {...}
  }
}
```

## 📝 批处理脚本

### Windows (PowerShell)

创建 `run_multi_seed_batch.ps1`:

```powershell
# 批量运行多个数据集
$datasets = @("Handwritten", "Scene15", "Coil20")

foreach ($dataset in $datasets) {
    Write-Host "`n========== Running $dataset =========="
    uv run python scripts/run_multi_seed_convergence.py `
        --dataset $dataset `
        --epochs 100 `
        --n_seeds 5
    
    # 找到最新的结果目录
    $latest = Get-ChildItem "multi_seed_results/${dataset}_*" | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    Write-Host "`n========== Plotting $dataset =========="
    uv run python scripts/plot_multi_seed_convergence.py `
        --results_dir $latest.FullName
}

Write-Host "`n✅ All datasets completed!"
```

运行：
```powershell
.\run_multi_seed_batch.ps1
```

### Linux/Mac (Bash)

创建 `run_multi_seed_batch.sh`:

```bash
#!/bin/bash

datasets=("Handwritten" "Scene15" "Coil20")

for dataset in "${datasets[@]}"; do
    echo "========== Running $dataset =========="
    uv run python scripts/run_multi_seed_convergence.py \
        --dataset $dataset \
        --epochs 100 \
        --n_seeds 5
    
    # 找到最新的结果目录
    latest=$(ls -td multi_seed_results/${dataset}_* | head -1)
    
    echo "========== Plotting $dataset =========="
    uv run python scripts/plot_multi_seed_convergence.py \
        --results_dir "$latest"
done

echo "✅ All datasets completed!"
```

运行：
```bash
chmod +x run_multi_seed_batch.sh
./run_multi_seed_batch.sh
```

## 🎯 论文应用

### 在LaTeX中引用

```latex
\subsection{Statistical Analysis}

To ensure the reliability of our results, we conduct experiments with 
5 different random seeds and report the mean and standard deviation.
Figure~\ref{fig:multi_seed_convergence} shows the convergence behavior 
across multiple runs on the Handwritten dataset.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/Handwritten_multi_seed_convergence.pdf}
\caption{Convergence analysis across 5 random seeds. Solid lines represent 
mean values and shaded regions indicate ±1 standard deviation. 
(Top) Clustering metrics show consistent improvement with low variance. 
(Bottom) Loss components decrease smoothly across all runs.}
\label{fig:multi_seed_convergence}
\end{figure}
```

### 报告结果

```latex
\begin{table}[t]
\centering
\caption{Performance on Handwritten dataset (Mean ± Std over 5 runs)}
\begin{tabular}{lccc}
\toprule
Method & ACC (\%) & NMI (\%) & ARI (\%) \\
\midrule
OT-CFM & $95.24 \pm 1.32$ & $88.15 \pm 1.08$ & $82.36 \pm 1.95$ \\
\bottomrule
\end{tabular}
\end{table}
```

## ⚡ 性能考虑

### 运行时间估算

| 配置 | 预计时间 |
|------|---------|
| 3 seeds × 50 epochs | ~5-10分钟 |
| 5 seeds × 100 epochs | ~20-30分钟 |
| 10 seeds × 150 epochs | ~1-1.5小时 |

*时间取决于数据集大小、硬件配置和是否使用GPU*

### 并行化（可选）

修改脚本在不同进程中运行不同种子可以显著加速：

```python
# 在 run_multi_seed_convergence.py 中
from multiprocessing import Pool

def run_parallel(args):
    seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    
    with Pool(processes=min(args.n_seeds, 4)) as pool:
        histories = pool.map(
            lambda seed: run_single_seed(args.dataset, seed, ...),
            seeds
        )
```

## 🔍 故障排查

### 问题：某个种子训练失败
**解决**：脚本会跳过失败的种子并继续其他种子，最终使用成功运行的结果

### 问题：内存不足
**解决**：
1. 减少批次大小（在config中修改）
2. 减少epochs数量
3. 使用更小的数据集进行测试

### 问题：结果波动太大
**解决**：
1. 增加种子数量（--n_seeds 10）
2. 检查超参数是否调优
3. 确认数据预处理的随机性

## 📚 相关工具

- `scripts/run_experiment.py` - 单次训练
- `scripts/plot_convergence_from_history.py` - 单次运行可视化
- `scripts/run_ablation.py` - 消融实验
- `scripts/run_sensitivity_analysis.py` - 敏感性分析

---

**创建日期**: 2026-01-22  
**支持的数据集**: Handwritten, Scene15, Coil20, NoisyMNIST, Caltech101, CUB  
**Python版本**: 3.10+  
**依赖**: PyTorch, NumPy, Matplotlib, Seaborn
