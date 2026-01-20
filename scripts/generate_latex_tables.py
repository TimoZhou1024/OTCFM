#!/usr/bin/env python3
"""
Generate LaTeX tables from sensitivity analysis results

Usage:
    python scripts/generate_latex_tables.py --input sensitivity_results/Scene15_lambda_gw --output paper_tables.tex
"""

import argparse
import json
import pandas as pd
from pathlib import Path

def load_summary_stats(results_dir: Path):
    """Load summary statistics from JSON"""
    summary_file = results_dir / "summary_stats.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    return data

def format_mean_std(mean, std):
    """Format mean ± std for LaTeX"""
    return f"${mean:.3f} \\pm {std:.3f}$"

def generate_single_param_table(data: dict, param_name: str):
    """Generate LaTeX table for single parameter sweep"""
    
    # Start table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Sensitivity Analysis: Effect of $\\lambda_{\\text{GW}}$ on Clustering Performance}\n"
    latex += "\\label{tab:sensitivity_lambda_gw}\n"
    latex += "\\begin{tabular}{l|cccc}\n"
    latex += "\\toprule\n"
    latex += f"{param_name} & ACC & NMI & ARI & F1 \\\\\n"
    latex += "\\midrule\n"
    
    # Add rows
    for config_key, config_data in data.items():
        if isinstance(config_data, dict) and "mean" in config_data:
            param_value = config_key.split('=')[1] if '=' in config_key else config_key
            
            acc = format_mean_std(config_data["mean"]["ACC"], config_data["std"]["ACC"])
            nmi = format_mean_std(config_data["mean"]["NMI"], config_data["std"]["NMI"])
            ari = format_mean_std(config_data["mean"]["ARI"], config_data["std"]["ARI"])
            f1 = format_mean_std(config_data["mean"]["F1"], config_data["std"]["F1"])
            
            latex += f"{param_value} & {acc} & {nmi} & {ari} & {f1} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def generate_grid_search_table(data: dict, param1: str, param2: str, metric: str = "ACC"):
    """Generate LaTeX table for grid search (best values only)"""
    
    # Extract best configurations
    best_configs = []
    for config_key, config_data in data.items():
        if isinstance(config_data, dict) and "mean" in config_data:
            best_configs.append({
                'config': config_key,
                'value': config_data["mean"][metric],
                'std': config_data["std"][metric]
            })
    
    # Sort by metric value (descending)
    best_configs.sort(key=lambda x: x['value'], reverse=True)
    
    # Take top 10
    best_configs = best_configs[:10]
    
    # Start table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Top 10 Configurations for Grid Search: {param1} vs {param2}}}\n"
    latex += "\\label{tab:grid_search_top10}\n"
    latex += "\\begin{tabular}{l|l|c}\n"
    latex += "\\toprule\n"
    latex += f"{param1} & {param2} & {metric} \\\\\n"
    latex += "\\midrule\n"
    
    # Add rows
    for i, config in enumerate(best_configs, 1):
        # Parse config string
        parts = config['config'].split(',')
        p1_val = parts[0].split('=')[1] if '=' in parts[0] else parts[0]
        p2_val = parts[1].split('=')[1] if '=' in parts[1] else parts[1]
        
        value_str = format_mean_std(config['value'], config['std'])
        
        # Bold the best result
        if i == 1:
            latex += f"\\textbf{{{p1_val}}} & \\textbf{{{p2_val}}} & \\textbf{{{value_str}}} \\\\\n"
        else:
            latex += f"{p1_val} & {p2_val} & {value_str} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def generate_full_analysis_table(results_dirs: list):
    """Generate comprehensive table comparing all parameters"""
    
    all_results = []
    for results_dir in results_dirs:
        data = load_summary_stats(results_dir)
        param_name = results_dir.name.split('_')[-1]  # Extract param name
        
        # Find best config
        best_acc = 0
        best_config = None
        for config_key, config_data in data.items():
            if isinstance(config_data, dict) and "mean" in config_data:
                if config_data["mean"]["ACC"] > best_acc:
                    best_acc = config_data["mean"]["ACC"]
                    best_config = config_data
        
        if best_config:
            all_results.append({
                'parameter': param_name,
                'ACC': best_config["mean"]["ACC"],
                'NMI': best_config["mean"]["NMI"],
                'ARI': best_config["mean"]["ARI"],
                'F1': best_config["mean"]["F1"],
                'ACC_std': best_config["std"]["ACC"],
                'NMI_std': best_config["std"]["NMI"],
                'ARI_std': best_config["std"]["ARI"],
                'F1_std': best_config["std"]["F1"]
            })
    
    # Start table
    latex = "\\begin{table*}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comprehensive Parameter Sensitivity Analysis: Best Performance for Each Parameter}\n"
    latex += "\\label{tab:comprehensive_sensitivity}\n"
    latex += "\\begin{tabular}{l|cccc}\n"
    latex += "\\toprule\n"
    latex += "Parameter & ACC & NMI & ARI & F1 \\\\\n"
    latex += "\\midrule\n"
    
    # Add rows
    for result in all_results:
        param = result['parameter'].replace('_', '\\_')
        acc = format_mean_std(result['ACC'], result['ACC_std'])
        nmi = format_mean_std(result['NMI'], result['NMI_std'])
        ari = format_mean_std(result['ARI'], result['ARI_std'])
        f1 = format_mean_std(result['F1'], result['F1_std'])
        
        latex += f"{param} & {acc} & {nmi} & {ari} & {f1} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table*}\n"
    
    return latex

def generate_sensitivity_score_table(report_file: Path):
    """Generate table from statistical report"""
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Parse sensitivity scores section
    if "Parameter Sensitivity Scores" not in content:
        return "% No sensitivity scores found in report\n"
    
    # Extract scores (this is a simplified parser)
    lines = content.split('\n')
    scores = []
    in_scores_section = False
    
    for line in lines:
        if "Parameter Sensitivity Scores" in line:
            in_scores_section = True
            continue
        if in_scores_section and ':' in line and line.strip():
            parts = line.strip().split(':')
            if len(parts) == 2:
                param = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    scores.append((param, score))
                except ValueError:
                    pass
        if in_scores_section and line.strip() == "":
            break
    
    if not scores:
        return "% No sensitivity scores parsed from report\n"
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Generate table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Parameter Sensitivity Scores (Higher indicates greater sensitivity)}\n"
    latex += "\\label{tab:sensitivity_scores}\n"
    latex += "\\begin{tabular}{lc}\n"
    latex += "\\toprule\n"
    latex += "Parameter & Sensitivity Score \\\\\n"
    latex += "\\midrule\n"
    
    for param, score in scores:
        param_latex = param.replace('_', '\\_')
        latex += f"{param_latex} & {score:.3f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from sensitivity analysis")
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with sensitivity results')
    parser.add_argument('--output', type=str, default='latex_tables.tex',
                       help='Output LaTeX file')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'single', 'grid', 'full'],
                       help='Table generation mode')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Load data
    print(f"Loading results from {input_dir}...")
    
    try:
        # Check what type of analysis
        summary_file = input_dir / "summary_stats.json"
        report_file = input_dir / "statistical_report.txt"
        
        latex_content = "% LaTeX tables generated from OT-CFM sensitivity analysis\n"
        latex_content += "% Requires packages: booktabs\n\n"
        
        if summary_file.exists():
            data = load_summary_stats(input_dir)
            
            # Auto-detect mode
            if args.mode == 'auto':
                # Check first key to determine format
                first_key = list(data.keys())[0]
                if ',' in first_key:
                    args.mode = 'grid'
                else:
                    args.mode = 'single'
            
            # Generate appropriate table
            if args.mode == 'single':
                print("Generating single parameter table...")
                param_name = input_dir.name.split('_')[-1]
                latex_content += generate_single_param_table(data, param_name)
                
            elif args.mode == 'grid':
                print("Generating grid search table...")
                # Extract param names from directory
                parts = input_dir.name.split('_')
                param1 = parts[-2] if len(parts) >= 2 else "param1"
                param2 = parts[-1] if len(parts) >= 1 else "param2"
                
                for metric in ["ACC", "NMI", "ARI", "F1"]:
                    latex_content += generate_grid_search_table(data, param1, param2, metric)
                    latex_content += "\n"
        
        if report_file.exists():
            print("Generating sensitivity scores table...")
            latex_content += generate_sensitivity_score_table(report_file)
        
        # Write output
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            f.write(latex_content)
        
        print(f"\n✓ LaTeX tables saved to: {output_file}")
        print("\nTo use in your paper:")
        print("  1. Add to preamble: \\usepackage{booktabs}")
        print(f"  2. Include tables: \\input{{{output_file}}}")
        print("  3. Adjust captions and labels as needed")
        
    except Exception as e:
        print(f"Error generating tables: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
