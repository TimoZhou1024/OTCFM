# Multi-Seed Convergence Analysis System

**完整的统计分析工具，用于生成publication-quality收敛曲线（均值±标准差）**

## 🎯 核心功能

✅ **多随机种子训练** - 运行5次（可配置）不同随机种子的完整训练  
✅ **自动指标记录** - 每个epoch记录ACC, NMI, ARI  
✅ **自动损失记录** - 每个epoch记录Total_loss, recon_loss, gw_loss, cluster_loss  
✅ **统计分析** - 自动计算均值、标准差、最小值、最大值  
✅ **专业可视化** - 均值曲线+标准差阴影，publication-ready  

## 📦 新增文件

```
scripts/
├── run_multi_seed_convergence.py      # 多种子训练主脚本
├── plot_multi_seed_convergence.py     # 可视化脚本（均值±std）
└── test_multi_seed.py                 # 快速测试脚本

docs/
└── multi_seed_convergence_guide.md    # 完整使用指南

multi_seed_results/                     # 输出目录（自动创建）
└── {dataset}_{timestamp}/
    ├── seed_42/                        # 种子42的训练历史
    ├── seed_43/                        # 种子43的训练历史
    ├── ...
    ├── aggregated_results.json         # 聚合统计数据
    ├── experiment_config.json          # 实验配置
    └── convergence_plot.pdf            # 可视化图表
```

## 🚀 快速开始

### 1. 快速测试（3 seeds, 30 epochs, ~5分钟）

```bash
uv run python scripts/test_multi_seed.py
```

### 2. 完整实验（5 seeds, 100 epochs, ~25分钟）

```bash
# Step 1: 运行多次训练
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Handwritten \
    --epochs 100 \
    --n_seeds 5

# Step 2: 可视化结果
uv run python scripts/plot_multi_seed_convergence.py \
    --results_dir multi_seed_results/Handwritten_YYYYMMDD_HHMMSS
```

### 3. 查看结果

输出文件：
- **统计表格** - 控制台输出
- **可视化图表** - `multi_seed_results/{dataset}_{timestamp}/convergence_plot.pdf`
- **论文用图** - `figures/{dataset}_multi_seed_convergence.pdf`

## 📊 示例输出

### 控制台统计表格

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
```

### 可视化图表特性

**布局**: 2个子图垂直排列
- **上图**: ACC, NMI, ARI曲线（3条线+阴影）
- **下图**: Total Loss + 各损失组件曲线（4条线+阴影）

**样式**: 
- ✅ 白色背景
- ✅ 网格线（虚线，透明度30%）
- ✅ 粗体标签
- ✅ Seaborn 'deep'配色
- ✅ 图例有阴影框
- ✅ 300 DPI高分辨率

**阴影区域**: 均值 ± 1倍标准差

## 🔧 高级用法

### 自定义配置

```bash
# 更多种子以获得更稳定结果
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Scene15 \
    --epochs 150 \
    --n_seeds 10 \
    --start_seed 100

# 测试鲁棒性（缺失+非对齐数据）
uv run python scripts/run_multi_seed_convergence.py \
    --dataset Handwritten \
    --epochs 100 \
    --n_seeds 5 \
    --missing_rate 0.3 \
    --unaligned_rate 0.5
```

### 批量处理多个数据集

Windows PowerShell:
```powershell
$datasets = @("Handwritten", "Scene15", "Coil20")

foreach ($dataset in $datasets) {
    uv run python scripts/run_multi_seed_convergence.py `
        --dataset $dataset --epochs 100 --n_seeds 5
    
    $latest = Get-ChildItem "multi_seed_results/${dataset}_*" | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    uv run python scripts/plot_multi_seed_convergence.py `
        --results_dir $latest.FullName
}
```

## 📈 论文应用

### LaTeX引用示例

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/Handwritten_multi_seed_convergence.pdf}
\caption{Convergence analysis across 5 random seeds. 
Solid lines represent mean values and shaded regions indicate ±1 standard deviation.}
\label{fig:multi_seed_convergence}
\end{figure}
```

### 结果报告示例

```latex
\begin{table}[t]
\caption{Performance on Handwritten (Mean ± Std over 5 runs)}
\begin{tabular}{lccc}
\toprule
Method & ACC (\%) & NMI (\%) & ARI (\%) \\
\midrule
OT-CFM & $95.24 \pm 1.32$ & $88.15 \pm 1.08$ & $82.36 \pm 1.95$ \\
\bottomrule
\end{tabular}
\end{table}
```

## 🔍 数据格式

### aggregated_results.json 结构

```json
{
  "n_seeds": 5,
  "n_epochs": 100,
  "epochs": [0, 1, 2, ..., 99],
  "metrics": {
    "acc": {
      "mean": [...],  // 100个值（每个epoch的均值）
      "std": [...],   // 100个值（每个epoch的标准差）
      "min": [...],   // 100个值（每个epoch的最小值）
      "max": [...]    // 100个值（每个epoch的最大值）
    },
    "nmi": {...},
    "ari": {...},
    "purity": {...},
    "f1": {...}
  },
  "losses": {
    "loss": {...},       // Total loss
    "recon": {...},      // rec_loss
    "gw": {...},         // gc_loss (GW alignment)
    "cluster": {...},    // cc_loss (clustering)
    "contrastive": {...},
    "cfm": {...}
  },
  "final_stats": {
    "metrics": {
      "acc": {"mean": 0.9524, "std": 0.0132, "min": 0.9350, "max": 0.9680}
    },
    "losses": {...}
  }
}
```

## ⚡ 性能优化

### 预计运行时间

| 配置 | CPU时间 | GPU时间 |
|------|---------|---------|
| 3 seeds × 30 epochs | ~5-8分钟 | ~2-3分钟 |
| 5 seeds × 50 epochs | ~10-15分钟 | ~4-6分钟 |
| 5 seeds × 100 epochs | ~20-30分钟 | ~8-12分钟 |
| 10 seeds × 150 epochs | ~1-1.5小时 | ~30-45分钟 |

### 加速技巧

1. **使用GPU**: 自动检测并使用CUDA/MPS
2. **减少epochs**: 对于快速测试使用50 epochs
3. **减少seeds**: 3个种子足以验证概念
4. **使用tuned参数**: 脚本自动加载`config/tuned_params.json`

## 🐛 故障排查

### 问题：某个种子失败
**解决**: 脚本会自动跳过失败的种子，使用成功的种子计算统计

### 问题：内存不足
**解决**: 
```bash
# 减少批次大小（编辑config或代码）
# 或使用更小的数据集测试
uv run python scripts/test_multi_seed.py
```

### 问题：结果差异很大
**解决**:
- 增加种子数量：`--n_seeds 10`
- 检查超参数调优
- 确认数据加载的随机性设置

## 📚 完整文档

详细使用指南：[docs/multi_seed_convergence_guide.md](docs/multi_seed_convergence_guide.md)

## 🔗 相关工具

- `scripts/run_experiment.py` - 单次训练
- `scripts/plot_convergence_from_history.py` - 单次运行可视化
- `scripts/run_sensitivity_analysis.py` - 参数敏感性分析
- `scripts/run_ablation.py` - 消融实验

## ✅ 验证检查清单

运行完成后，确认以下文件存在：

- [ ] `multi_seed_results/{dataset}_{timestamp}/aggregated_results.json`
- [ ] `multi_seed_results/{dataset}_{timestamp}/convergence_plot.pdf`
- [ ] `figures/{dataset}_multi_seed_convergence.pdf`
- [ ] 每个种子的 `seed_{i}/history.json`

## 💡 最佳实践

1. **论文实验**: 使用5-10个种子，报告均值±标准差
2. **快速验证**: 使用3个种子，50 epochs
3. **鲁棒性测试**: 同时测试完整数据和缺失/非对齐数据
4. **对比实验**: 在相同种子集上运行所有方法

---

**创建日期**: 2026-01-22  
**版本**: 1.0  
**作者**: OT-CFM Team  
**许可**: MIT
