#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感性分析使用演示

展示如何使用sensitivity analysis框架进行实验并生成发表级别的结果。
适用于ICML/NeurIPS/CVPR等顶级会议。
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def main():
    print_section("OT-CFM 敏感性分析框架 - 使用演示")
    
    # 1. Single Parameter Sweep
    print_section("1. 单参数扫描 (Single Parameter Sweep)")
    print("用途：分析单个超参数对性能的影响")
    print("\n命令示例：")
    print("  uv run python scripts/run_sensitivity_analysis.py \\")
    print("    --dataset Scene15 \\")
    print("    --param lambda_gw \\")
    print("    --num_runs 5 \\")
    print("    --epochs 200 \\")
    print("    --output_dir sensitivity_results/single_param")
    
    print("\n预期输出：")
    print("  📁 sensitivity_results/single_param/")
    print("     ├── results.csv              # 原始数据（所有运行）")
    print("     ├── summary_stats.json       # 统计摘要（均值±标准差）")
    print("     ├── single_param_sweep.pdf   # 2×2折线图（适用于论文）")
    print("     └── statistical_report.txt   # 详细统计分析")
    
    print("\n用于论文：")
    print("  - Figure: single_param_sweep.pdf → 'Effect of λ_GW on Clustering Performance'")
    print("  - Table: summary_stats.json → 'Sensitivity Analysis Results'")
    
    # 2. Two-Parameter Grid Search
    print_section("2. 双参数网格搜索 (Grid Search)")
    print("用途：分析两个超参数的交互作用")
    print("\n命令示例：")
    print("  uv run python scripts/run_sensitivity_analysis.py \\")
    print("    --dataset Handwritten \\")
    print("    --param lambda_gw lambda_cluster \\")
    print("    --num_runs 3 \\")
    print("    --epochs 150 \\")
    print("    --plot_3d \\")
    print("    --output_dir sensitivity_results/grid_search")
    
    print("\n预期输出：")
    print("  📁 sensitivity_results/grid_search/")
    print("     ├── results.csv              # 完整网格结果")
    print("     ├── summary_stats.json       # 每个配置的统计")
    print("     ├── grid_heatmap.pdf         # 2×2热图（适用于论文）")
    print("     ├── 3d_surface_ACC.pdf       # 3D表面图（可选）")
    print("     ├── 3d_surface_NMI.pdf")
    print("     ├── 3d_surface_ARI.pdf")
    print("     ├── 3d_surface_F1.pdf")
    print("     └── statistical_report.txt   # 相关性分析")
    
    print("\n用于论文：")
    print("  - Figure: grid_heatmap.pdf → 'Interaction between λ_GW and λ_cluster'")
    print("  - Figure: 3d_surface_ACC.pdf → 'Performance Surface Analysis'")
    print("  - Text: statistical_report.txt → 'Correlation coefficients: r=0.85, p<0.001'")
    
    # 3. Full Analysis
    print_section("3. 完整分析 (Full Analysis)")
    print("用途：系统评估所有关键超参数（发表级别）")
    print("\n命令示例：")
    print("  uv run python scripts/run_sensitivity_analysis.py \\")
    print("    --dataset Scene15 \\")
    print("    --mode full \\")
    print("    --num_runs 10 \\")
    print("    --epochs 200 \\")
    print("    --output_dir sensitivity_results/full_analysis")
    
    print("\n分析的参数（9个）：")
    print("  Loss Weights:")
    print("    - lambda_gw: [0.0, 0.1, 0.2, 0.3, 0.4]")
    print("    - lambda_cluster: [0.5, 1.0, 1.5, 2.0]")
    print("    - lambda_recon: [0.1, 0.5, 1.0]")
    print("    - lambda_contrastive: [0.0, 0.1, 0.3, 0.5]")
    print("  Architecture:")
    print("    - latent_dim: [64, 128, 256]")
    print("    - flow_hidden_dim: [128, 256, 512]")
    print("    - ode_steps: [5, 10, 20]")
    print("  Training:")
    print("    - learning_rate: [1e-4, 3e-4, 1e-3]")
    print("    - dropout: [0.0, 0.1, 0.3]")
    
    print("\n预期输出：")
    print("  📁 sensitivity_results/full_analysis/")
    print("     ├── results.csv                      # 所有实验结果")
    print("     ├── summary_stats.json               # 统计摘要")
    print("     ├── single_param_sweep_lambda_gw.pdf")
    print("     ├── single_param_sweep_lambda_cluster.pdf")
    print("     ├── ... (9个参数的图)")
    print("     └── statistical_report.txt           # 综合统计分析")
    
    print("\n计算成本：")
    print("  - 总实验数: 9参数 × 各自范围数 × 10 runs = ~300-500 experiments")
    print("  - 估计时间: 8-12小时（取决于硬件）")
    print("  - 推荐: 使用GPU加速，或先用--epochs 50快速测试")
    
    # 4. Batch Processing
    print_section("4. 批量处理多个数据集")
    print("用于全面评估框架的鲁棒性")
    print("\nBash脚本示例 (run_all_sensitivity.sh):")
    print("""
#!/bin/bash
DATASETS="Scene15 Handwritten NoisyMNIST"
for DATASET in $DATASETS; do
    echo "Running sensitivity analysis on $DATASET..."
    uv run python scripts/run_sensitivity_analysis.py \\
        --dataset $DATASET \\
        --param lambda_gw \\
        --num_runs 5 \\
        --epochs 200 \\
        --output_dir sensitivity_results/${DATASET}_lambda_gw
done
    """)
    
    # 5. Best Practices
    print_section("5. 最佳实践（Best Practices）")
    print("✓ 快速原型测试:")
    print("    --epochs 50 --n_runs 1  # 验证脚本工作")
    print("\n✓ 中等规模实验:")
    print("    --epochs 100 --n_runs 3  # 初步结果")
    print("\n✓ 发表级别实验:")
    print("    --epochs 200 --n_runs 10  # 最终论文结果")
    print("\n✓ 参数选择策略:")
    print("    1. 先做单参数扫描识别重要参数")
    print("    2. 对重要参数做双参数网格搜索")
    print("    3. 最后做完整分析验证结果")
    print("\n✓ 可视化建议:")
    print("    - 2D图用于主要结果（更清晰）")
    print("    - 3D图用于补充材料（展示交互）")
    print("    - PDF格式适合LaTeX论文")
    
    # 6. Publication Templates
    print_section("6. 论文写作模板")
    print("Section 4.5: Sensitivity Analysis")
    print("""
We conducted comprehensive sensitivity analysis to evaluate the robustness 
of OT-CFM with respect to key hyperparameters. Figure X shows the effect of 
the Gromov-Wasserstein weight λ_GW on clustering performance across four 
metrics (ACC, NMI, ARI, F1). The model achieves peak performance at λ_GW=0.2, 
demonstrating moderate sensitivity to this parameter.

Table Y presents the mean±std results across 10 independent runs for each 
parameter configuration. Statistical analysis reveals that λ_cluster has 
the highest sensitivity score (0.85), indicating its critical role in 
determining clustering quality. Correlation analysis (Figure Z) shows 
strong positive correlation (r=0.89, p<0.001) between λ_GW and λ_cluster, 
suggesting they work synergistically to balance structural alignment and 
cluster separation.

Our full parameter sweep over 9 hyperparameters (see Appendix) confirms 
that OT-CFM maintains consistent performance across a wide range of 
configurations, with performance degradation <5% within ±50% of default 
values, demonstrating the model's robustness.
    """)
    
    # 7. Output Interpretation
    print_section("7. 结果解读指南")
    print("📊 statistical_report.txt 包含:")
    print("  1. Best Configurations: 每个指标的最优参数值")
    print("  2. Sensitivity Scores: 参数重要性排序（0-1分数）")
    print("  3. Correlation Matrix: 参数间相关性（Pearson r）")
    print("  4. P-values: 统计显著性检验")
    
    print("\n💡 解读技巧:")
    print("  - Sensitivity Score > 0.7: 高度敏感，需仔细调优")
    print("  - Sensitivity Score 0.3-0.7: 中等敏感，使用默认值即可")
    print("  - Sensitivity Score < 0.3: 低敏感，对结果影响小")
    print("  - Correlation |r| > 0.7: 参数间存在强相关，需联合调优")
    
    # Quick Start Summary
    print_section("快速开始（Quick Start）")
    print("Step 1: 快速测试（5分钟）")
    print("  uv run python scripts/test_sensitivity_analysis.py")
    
    print("\nStep 2: 单参数分析（1-2小时）")
    print("  uv run python scripts/run_sensitivity_analysis.py \\")
    print("    --dataset Scene15 --param lambda_gw \\")
    print("    --num_runs 5 --epochs 200")
    
    print("\nStep 3: 查看结果")
    print("  ls sensitivity_results/*/")
    print("  cat sensitivity_results/*/statistical_report.txt")
    
    print("\nStep 4: 集成到论文")
    print("  - 复制 *.pdf 到论文的 figures/ 目录")
    print("  - 从 statistical_report.txt 提取关键数值")
    print("  - 使用 summary_stats.json 生成LaTeX表格")
    
    print("\n" + "="*80)
    print(" 更多信息请参考: docs/sensitivity_analysis_guide.md")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
