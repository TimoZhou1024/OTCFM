# 收敛分析完整工作流程

## 📋 当前状态 (2026-01-25)

**实验运行中** ✅
- 数据集: Handwritten
- 总轮数: 100 epochs
- 当前进度: Phase 3 (Full Training) - 第4/80轮
- 预计完成时间: ~15-20分钟

---

## 🎯 目标

在论文的实验部分添加收敛分析章节，通过损失曲线和指标变化展示OT-CFM方法的收敛特性。

---

## 🔧 已实现的工具

### 1. 主实验脚本: `scripts/run_convergence_analysis.py`
**功能**:
- 三阶段训练跟踪（重构→DEC→完整）
- 6个损失组件：total, recon, gw, cluster, contrastive, cfm
- 5个评估指标：ACC, NMI, ARI, Purity, F1
- 生成publication-quality PDF图表（2×2子图，300 DPI）

**用法**:
```bash
# 完整实验（100轮）
uv run python scripts/run_convergence_analysis.py --datasets Handwritten --epochs 100 --use_tuned

# 快速测试（60轮）
uv run python scripts/quick_convergence_test.py

# 多数据集对比
uv run python scripts/run_convergence_analysis.py --datasets Handwritten Scene15 Coil20 --epochs 100 --use_tuned
```

**输出** (保存在 `convergence_results/Handwritten_YYYYMMDD_HHMMSS/`):
- `losses.csv` - 损失历史（每轮记录）
- `metrics.csv` - 指标历史（每轮评估）
- `final_metrics.json` - 最终指标
- `Handwritten_convergence.pdf` - 收敛曲线图

---

### 2. 进度监控: `scripts/monitor_convergence.py`
**功能**: 实时显示训练进度、最新指标、完成状态

**用法**:
```bash
uv run python scripts/monitor_convergence.py
```

**示例输出**:
```
======================================================================
📊 Monitoring: Handwritten_20260125_143022
======================================================================

✅ Training Progress:
   - Epochs logged: 23
   - Phase 'recon': 10 epochs
   - Phase 'dec': 10 epochs
   - Phase 'full': 3 epochs
     Recent losses:
       Epoch 21: total=0.4582
       Epoch 22: total=0.4531
       Epoch 23: total=0.4489

✅ Metrics Progress:
   - Evaluations: 23
   - Latest (Epoch 23, Phase full):
     ACC: 72.45%
     NMI: 65.31%
     ARI: 58.92%

⏳ Still running... (Check again in a few minutes)
```

---

### 3. 结果检查与论文准备: `scripts/check_convergence_results.py`
**功能**: 检查所有实验结果，自动复制最佳图表到论文目录，提供main.tex更新建议

**用法**:
```bash
uv run python scripts/check_convergence_results.py
```

**自动操作**:
1. 检查最新实验的完整性
2. 复制PDF图表到 `figures/Handwritten_convergence.pdf`
3. 提取数值用于更新论文

**示例输出**:
```
======================================================================
📋 RECOMMENDED ACTIONS FOR PAPER:
======================================================================

1. Copy figure to paper directory:
   cp convergence_results/Handwritten_20260125_143022/Handwritten_convergence.pdf figures/Handwritten_convergence.pdf
   ✅ DONE! Figure copied to figures/Handwritten_convergence.pdf

2. Update main.tex with these values:
   - Final ACC: 94.15%
   - Final NMI: 88.52%
   - Final ARI: 86.73%

3. Phase 1 (Reconstruction) loss:
   - Start: 0.8234
   - End: 0.3156

4. Phase 2 (DEC) ACC improvement:
   - Start: 12.35%
   - End: 68.72%

5. Phase 3 (Full) - convergence at epoch:
   - Converged at epoch 85
   - ACC: 94.15%

======================================================================
✅ Paper-ready! Use figure: figures/Handwritten_convergence.pdf
======================================================================
```

---

## 📊 论文集成

### 已添加到 `main.tex` (lines 736-760)

**章节**: Convergence Analysis

**核心内容**:
1. 三阶段训练动态分析
2. 损失组件平衡讨论
3. 与扩散模型的收敛速度对比

**待更新数值** (运行 `check_convergence_results.py` 后替换):
- Phase 1 损失范围: `from $\sim$0.8 to $\sim$0.3` → 替换为实际值
- Phase 2 ACC提升: `ACC from 10% to $\sim$70%` → 替换为实际值
- Phase 3 最终指标: `ACC=94.1%, NMI=88.5%` → 替换为实际值
- 收敛轮数: `$\sim$80--100 epochs` → 替换为实际收敛点

**图表引用**: `\ref{fig:convergence}` → `figures/Handwritten_convergence.pdf`

---

## 🔄 完整工作流程

### Step 1: 运行实验
```bash
uv run python scripts/run_convergence_analysis.py --datasets Handwritten --epochs 100 --use_tuned
```

### Step 2: 监控进度（可选）
```bash
# 在另一个终端窗口运行
uv run python scripts/monitor_convergence.py
```

### Step 3: 检查结果并准备论文
```bash
# 实验完成后运行
uv run python scripts/check_convergence_results.py
```

### Step 4: 更新论文
1. 打开 `main.tex`
2. 跳转到 **Convergence Analysis** 章节 (line 736-760)
3. 根据 `check_convergence_results.py` 的输出替换占位数值
4. 验证图表引用: `\includegraphics{figures/Handwritten_convergence.pdf}`

### Step 5: 编译论文
```bash
# LaTeX编译
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📈 生成的图表说明

**图表结构** (2×2子图):
- **左上**: 总损失曲线（三阶段彩色区分）
- **右上**: 重构损失 vs GW损失对比
- **左下**: 聚类损失 vs 对比损失对比
- **右下**: ACC/NMI/ARI指标曲线

**视觉设计**:
- 阶段背景色: Phase 1 (淡红), Phase 2 (淡绿), Phase 3 (淡蓝)
- 竖线标记阶段转换点
- 300 DPI高分辨率
- Publication-ready格式

---

## 🐛 已修复的Bug

### Bug 1: 数据加载缺失参数
**问题**: `load_handwritten() missing 1 required positional argument: 'data_dir'`  
**修复**: 在 `run_convergence_analysis()` 中添加 `data_dir="./data"` 参数

### Bug 2: 损失字典解包错误
**问题**: `tuple indices must be integers or slices, not str`  
**原因**: `compute_loss()` 返回 `(loss, loss_dict)` 元组，而非直接返回字典  
**修复**: 改为 `loss, loss_dict = self.model.compute_loss(...)`

### Bug 3: 输出字典访问错误
**问题**: `_evaluate()` 中 `outputs['latents'][0]` 访问失败  
**原因**: `consensus` 可能为None或嵌套在字典中  
**修复**: 添加正确的None检查和字典访问逻辑

### Bug 4: 损失值类型安全
**问题**: loss_dict值的类型不确定  
**修复**: 所有 `loss_dict.get()` 调用外包装 `float()` 确保类型安全

---

## ✅ 验证检查清单

实验完成后，确保以下文件存在：

- [ ] `convergence_results/Handwritten_*/losses.csv` (100行数据)
- [ ] `convergence_results/Handwritten_*/metrics.csv` (100行数据)
- [ ] `convergence_results/Handwritten_*/final_metrics.json` (ACC > 90%)
- [ ] `convergence_results/Handwritten_*/Handwritten_convergence.pdf` (>100KB)
- [ ] `figures/Handwritten_convergence.pdf` (已复制)
- [ ] `main.tex` lines 736-760 中的数值已更新

---

## 🔧 故障排查

### 问题: 实验卡住不动
**解决**: 
```bash
# 检查是否真的卡住
uv run python scripts/monitor_convergence.py

# 如果确实卡住，终止并重新运行
Ctrl+C
uv run python scripts/run_convergence_analysis.py --datasets Handwritten --epochs 100 --use_tuned
```

### 问题: 内存不足（OOM）
**解决**: 减小批次大小或使用更短的实验
```bash
# 编辑 scripts/run_convergence_analysis.py
# 修改: batch_size = 128  # 默认256
# 或使用快速测试
uv run python scripts/quick_convergence_test.py
```

### 问题: 指标结果不理想
**解决**: 
1. 检查tuned_params是否加载: 日志应显示 "Loading tuned parameters for Handwritten"
2. 尝试增加训练轮数: `--epochs 150`
3. 或选择其他数据集对比: Scene15, Coil20

---

## 📚 相关文档

- **项目说明**: [README.md](README.md)
- **消融实验修复**: [docs/ABLATION_FIX_REPORT.md](docs/ABLATION_FIX_REPORT.md)
- **敏感性分析**: [docs/sensitivity_analysis_quick_ref.md](docs/sensitivity_analysis_quick_ref.md)
- **基线添加指南**: [docs/add_new_baselines_guide.md](docs/add_new_baselines_guide.md)

---

## 🎓 引用格式（论文中）

```latex
\subsection{Convergence Analysis}

Figure~\ref{fig:convergence} illustrates the training dynamics of OT-CFM 
across three sequential phases on the Handwritten dataset...

\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/Handwritten_convergence.pdf}
\caption{Convergence analysis of OT-CFM training on Handwritten dataset...}
\label{fig:convergence}
\end{figure}
```

---

**最后更新**: 2026-01-25  
**状态**: 实验运行中，预计15分钟完成  
**下一步**: 等待实验完成后运行 `check_convergence_results.py`
