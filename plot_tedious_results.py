#!/usr/bin/env python3
"""
Plot results from tedious tasks experiments with granular completion metrics.
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Configuration
RESULTS_DIR = "/data/austin_meek/emergent-values-multimodal/tedious_tasks/"
OUTPUT_DIR = RESULTS_DIR

def extract_completion_rate(detail_text):
    """Extract completion percentage from detail text."""
    # Pattern 1: "X/Y" format (e.g., "2048/2500", "539/600")
    match = re.search(r'(\d+)/(\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # If no pattern found, return 0 for FAIL, 100 for PASS
    return None

def parse_result_file(filepath):
    """Parse a single result file and extract task outcomes with details."""
    results = {}
    image_name = Path(filepath).stem.replace("result_", "")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match task line
        task_match = re.match(r"Task (\d+) \((.*?)\): (PASS|FAIL)", line)
        if task_match:
            task_id = task_match.group(1)
            task_name = task_match.group(2)
            status = task_match.group(3)
            
            # Look for detail line
            completion_rate = 100.0 if status == "PASS" else 0.0
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("- Detail:"):
                detail = lines[i + 1].strip()
                extracted_rate = extract_completion_rate(detail)
                if extracted_rate is not None:
                    completion_rate = extracted_rate
            
            results[f"Task {task_id}"] = {
                'status': 1 if status == "PASS" else 0,
                'completion': completion_rate,
                'name': task_name
            }
        
        i += 1
    
    return image_name, results

def collect_all_results():
    """Collect results from all result_*.txt files."""
    result_files = list(Path(RESULTS_DIR).glob("result_*.txt"))
    
    pass_fail_data = []
    completion_data = []
    
    for filepath in result_files:
        try:
            image_name, results = parse_result_file(filepath)
            
            pf_row = {'Image': image_name}
            comp_row = {'Image': image_name}
            
            for task, metrics in results.items():
                pf_row[task] = metrics['status']
                comp_row[task] = metrics['completion']
            
            pass_fail_data.append(pf_row)
            completion_data.append(comp_row)
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    df_pf = pd.DataFrame(pass_fail_data)
    df_comp = pd.DataFrame(completion_data)
    
    # Reorder columns
    task_cols = sorted([col for col in df_pf.columns if col.startswith('Task')], 
                       key=lambda x: int(x.split()[1]))
    df_pf = df_pf[['Image'] + task_cols]
    df_comp = df_comp[['Image'] + task_cols]
    
    return df_pf, df_comp

def plot_results(df_pf, df_comp):
    """Create visualizations with granular completion metrics."""
    
    task_cols = [col for col in df_pf.columns if col.startswith('Task')]
    
    # 1. Heatmap of completion percentages (more granular than pass/fail)
    plt.figure(figsize=(12, len(df_comp) * 0.5 + 2))
    
    sns.heatmap(
        df_comp[task_cols].values,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Completion %'},
        yticklabels=df_comp['Image'].values,
        xticklabels=[f'T{i+1}' for i in range(len(task_cols))],
        vmin=0,
        vmax=100
    )
    plt.title('Task Completion Percentage Across Images', fontsize=14, pad=20)
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Image/Condition', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_completion_percentage.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/heatmap_completion_percentage.png")
    plt.close()
    
    # 2. Box plot: Distribution of completion rates per task
    plt.figure(figsize=(12, 6))
    
    completion_data = []
    for task in task_cols:
        for val in df_comp[task].values:
            completion_data.append({'Task': task.replace('Task ', 'T'), 'Completion %': val})
    
    df_plot = pd.DataFrame(completion_data)
    
    sns.boxplot(data=df_plot, x='Task', y='Completion %', palette='Set2')
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% (Perfect)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
    plt.ylim(-5, 105)
    plt.title('Distribution of Task Completion Rates', fontsize=14, pad=20)
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Completion Percentage', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_completion_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/boxplot_completion_distribution.png")
    plt.close()
    
    # 3. Average completion rate per task (with error bars)
    plt.figure(figsize=(10, 6))
    
    means = df_comp[task_cols].mean()
    stds = df_comp[task_cols].std()
    
    x = np.arange(len(task_cols))
    bars = plt.bar(x, means.values, yerr=stds.values, capsize=5, alpha=0.7)
    
    plt.xticks(x, [f'T{i+1}' for i in range(len(task_cols))], rotation=0)
    plt.ylabel('Average Completion %', fontsize=12)
    plt.xlabel('Task', fontsize=12)
    plt.title('Average Task Completion (with std dev)', fontsize=14, pad=20)
    plt.ylim(0, 110)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50%')
    
    # Color bars based on mean completion
    for i, bar in enumerate(bars):
        if means.values[i] >= 90:
            bar.set_color('green')
        elif means.values[i] >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add value labels
    for i, v in enumerate(means.values):
        plt.text(i, v + stds.values[i] + 3, f'{v:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bar_avg_completion_with_std.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/bar_avg_completion_with_std.png")
    plt.close()
    
    # 4. Comparison: Pass/Fail vs Average Completion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Pass/fail rate
    pass_rate = df_pf[task_cols].mean() * 100
    ax1.bar(range(len(pass_rate)), pass_rate.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(task_cols)))
    ax1.set_xticklabels([f'T{i+1}' for i in range(len(task_cols))])
    ax1.set_ylabel('Pass Rate (%)', fontsize=11)
    ax1.set_xlabel('Task', fontsize=11)
    ax1.set_title('Binary Pass Rate', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    
    # Right: Completion percentage
    comp_rate = df_comp[task_cols].mean()
    ax2.bar(range(len(comp_rate)), comp_rate.values, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(task_cols)))
    ax2.set_xticklabels([f'T{i+1}' for i in range(len(task_cols))])
    ax2.set_ylabel('Average Completion %', fontsize=11)
    ax2.set_xlabel('Task', fontsize=11)
    ax2.set_title('Granular Completion %', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    
    plt.suptitle('Binary vs Granular Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_binary_vs_granular.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/comparison_binary_vs_granular.png")
    plt.close()
    
    # 5. Per-image average completion
    plt.figure(figsize=(12, 6))
    
    image_avg_completion = df_comp[task_cols].mean(axis=1)
    
    bars = plt.bar(range(len(df_comp)), image_avg_completion.values)
    plt.xticks(range(len(df_comp)), df_comp['Image'].values, rotation=45, ha='right')
    plt.ylabel('Average Completion %', fontsize=12)
    plt.xlabel('Image/Condition', fontsize=12)
    plt.title('Average Completion Percentage by Image', fontsize=14, pad=20)
    plt.ylim(0, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50%')
    
    # Color bars
    for i, bar in enumerate(bars):
        if image_avg_completion.values[i] >= 90:
            bar.set_color('green')
        elif image_avg_completion.values[i] >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add labels
    for i, v in enumerate(image_avg_completion.values):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bar_image_avg_completion.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/bar_image_avg_completion.png")
    plt.close()

def print_summary(df_pf, df_comp):
    """Print detailed summary with granular metrics."""
    print("\n" + "="*70)
    print("DETAILED SUMMARY STATISTICS")
    print("="*70)
    
    task_cols = [col for col in df_pf.columns if col.startswith('Task')]
    
    print(f"\nTotal images/conditions tested: {len(df_pf)}")
    print(f"Total tasks per image: {len(task_cols)}")
    
    print("\n" + "-"*70)
    print("PER-TASK METRICS")
    print("-"*70)
    print(f"{'Task':<8} {'Pass Rate':<12} {'Avg Completion':<18} {'Std Dev':<10}")
    print("-"*70)
    
    for task in task_cols:
        pass_rate = df_pf[task].mean() * 100
        avg_comp = df_comp[task].mean()
        std_comp = df_comp[task].std()
        
        status = "✓" if pass_rate > 50 else "✗"
        print(f"{status} {task:<6} {pass_rate:>5.1f}%      {avg_comp:>6.1f}%           ±{std_comp:>5.1f}%")
    
    print("\n" + "-"*70)
    print("PER-IMAGE METRICS")
    print("-"*70)
    print(f"{'Image':<45} {'Pass Rate':<12} {'Avg Completion':<15}")
    print("-"*70)
    
    for idx in range(len(df_pf)):
        image = df_pf['Image'].iloc[idx]
        pass_rate = df_pf[task_cols].iloc[idx].mean() * 100
        avg_comp = df_comp[task_cols].iloc[idx].mean()
        
        status = "✓" if pass_rate > 50 else "✗"
        # Truncate long image names
        image_short = image[:42] + "..." if len(image) > 45 else image
        print(f"{status} {image_short:<44} {pass_rate:>5.1f}%      {avg_comp:>6.1f}%")
    
    overall_pass = df_pf[task_cols].values.mean() * 100
    overall_comp = df_comp[task_cols].values.mean()
    
    print(f"\n{'='*70}")
    print(f"OVERALL PASS RATE: {overall_pass:.1f}%")
    print(f"OVERALL AVG COMPLETION: {overall_comp:.1f}%")
    print(f"{'='*70}\n")

def main():
    print("Collecting results from:", RESULTS_DIR)
    df_pf, df_comp = collect_all_results()
    
    if df_pf.empty:
        print("No result files found!")
        return
    
    print(f"Found {len(df_pf)} result files")
    
    # Save raw data
    df_pf.to_csv(os.path.join(OUTPUT_DIR, 'results_pass_fail.csv'), index=False)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, 'results_completion.csv'), index=False)
    print(f"Saved: {OUTPUT_DIR}/results_pass_fail.csv")
    print(f"Saved: {OUTPUT_DIR}/results_completion.csv")
    
    # Print summary
    print_summary(df_pf, df_comp)
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(df_pf, df_comp)
    
    print("\nDone! Check the tedious_tasks/ directory for plots and CSVs.")

if __name__ == "__main__":
    main()