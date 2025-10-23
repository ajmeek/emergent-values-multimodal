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
RESULTS_DIR = "tedious_tasks/"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")

def extract_completion_rate(detail_text):
    """Extract completion percentage from detail text."""
    # Pattern 1: "X/Y" format (e.g., "2048/2500", "539/600", "occurrences=204/1000")
    match = re.search(r'(\d+)/(\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # Pattern 2: "Found X numbers; expected Y"
    match = re.search(r'Found (\d+) numbers?; expected (\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # Pattern 3: "Found X non-empty lines; expected Y"
    match = re.search(r'Found (\d+) non-empty lines; expected (\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # Pattern 4: "Only X ... detected; expected Y"
    match = re.search(r'Only (\d+) .+?detected; expected (\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # Pattern 5: "Only X ... found; expected Y"
    match = re.search(r'Only (\d+) .+?found; expected (\d+)', detail_text)
    if match:
        actual, expected = float(match.group(1)), float(match.group(2))
        return (actual / expected) * 100 if expected > 0 else 0.0
    
    # If no pattern found, return None (will use default 0 for FAIL, 100 for PASS)
    return None

def extract_raw_count(detail_text):
    """Extract the raw actual count from detail text (e.g., 10 from 10/5040)."""
    match = re.search(r'(\d+)/(\d+)', detail_text)
    if match:
        return int(match.group(1))
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
            raw_count = None
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("- Detail:"):
                detail = lines[i + 1].strip()
                extracted_rate = extract_completion_rate(detail)
                if extracted_rate is not None:
                    completion_rate = extracted_rate
                raw_count = extract_raw_count(detail)
            
            results[f"Task {task_id}"] = {
                'status': 1 if status == "PASS" else 0,
                'completion': completion_rate,
                'name': task_name,
                'raw_count': raw_count
            }
        
        i += 1
    
    return image_name, results

def collect_all_results():
    """Collect results from all result_*.txt files."""
    result_files = list(Path(RESULTS_DIR).glob("result_*.txt"))
    
    pass_fail_data = []
    completion_data = []
    raw_count_data = []
    
    for filepath in result_files:
        try:
            image_name, results = parse_result_file(filepath)
            
            pf_row = {'Image': image_name}
            comp_row = {'Image': image_name}
            raw_row = {'Image': image_name}
            
            for task, metrics in results.items():
                pf_row[task] = metrics['status']
                comp_row[task] = metrics['completion']
                raw_row[task] = metrics.get('raw_count', None)
            
            pass_fail_data.append(pf_row)
            completion_data.append(comp_row)
            raw_count_data.append(raw_row)
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    df_pf = pd.DataFrame(pass_fail_data)
    df_comp = pd.DataFrame(completion_data)
    df_raw = pd.DataFrame(raw_count_data)
    
    # Reorder columns
    task_cols = sorted([col for col in df_pf.columns if col.startswith('Task')], 
                       key=lambda x: int(x.split()[1]))
    df_pf = df_pf[['Image'] + task_cols]
    df_comp = df_comp[['Image'] + task_cols]
    df_raw = df_raw[['Image'] + task_cols]
    
    return df_pf, df_comp, df_raw

def add_experiment_comparison_plots(df_comp):
    """Create visualizations comparing different experiments/conditions."""
    
    task_cols = [col for col in df_comp.columns if col.startswith('Task')]
    
    # Calculate overall average completion per experiment
    df_comp['Overall_Avg'] = df_comp[task_cols].mean(axis=1)
    
    # Sort by overall average for better visualization
    df_sorted = df_comp.sort_values('Overall_Avg', ascending=True)
    
    # 1. Experiment comparison: Overall average completion
    plt.figure(figsize=(14, max(6, len(df_comp) * 0.4)))
    
    # Color scheme: Red (<50%), Orange (50-75%), Green (75-110%), Blue (>110%)
    colors = ['cornflowerblue' if x > 110 else 'green' if x >= 75 else 'orange' if x >= 50 else 'red' 
              for x in df_sorted['Overall_Avg'].values]
    
    bars = plt.barh(range(len(df_sorted)), df_sorted['Overall_Avg'].values, color=colors, alpha=0.7)
    plt.yticks(range(len(df_sorted)), df_sorted['Image'].values, fontsize=9)
    plt.xlabel('Overall Average Completion %', fontsize=12)
    plt.title('Experiment Comparison: Overall Task Completion', fontsize=14, pad=20)
    
    # Adjust x-axis for potential over-completion
    max_overall = df_sorted['Overall_Avg'].max()
    xlim_max = min(max(105, max_overall + 10), 300)
    plt.xlim(0, xlim_max)
    
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='50%')
    plt.axvline(x=75, color='orange', linestyle='--', alpha=0.3, label='75%')
    plt.axvline(x=100, color='green', linestyle='--', alpha=0.3, label='100% (Target)')
    
    # Add value labels
    for i, v in enumerate(df_sorted['Overall_Avg'].values):
        label_x = min(v + 1, xlim_max - 5)
        plt.text(label_x, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'experiment_comparison_overall.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/experiment_comparison_overall.png")
    plt.close()
    
    # 2. Experiment comparison: Heatmap with experiments as rows
    plt.figure(figsize=(12, max(6, len(df_comp) * 0.5)))
    
    # Use sorted order
    heatmap_data = df_sorted[task_cols].values
    
    # Use same custom colormap as main heatmap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#d73027', '#fee08b', '#d9ef8b', '#1a9850', '#4575b4']  # Red->Yellow->Green->Blue
    cmap = LinearSegmentedColormap.from_list('completion', colors_list, N=100)
    
    max_val = heatmap_data.max()
    vmax_display = min(200, max_val) if max_val > 150 else 100
    
    # Create annotation with indicators for extreme over-completion
    annot_labels = np.array([[f'{val:.0f}' if val <= 200 else f'{val:.0f}!' 
                              for val in row] for row in heatmap_data])
    
    sns.heatmap(
        heatmap_data,
        annot=annot_labels,
        fmt='',
        cmap=cmap,
        cbar_kws={'label': 'Completion %'},
        yticklabels=df_sorted['Image'].values,
        xticklabels=[f'T{i+1}' for i in range(len(task_cols))],
        vmin=0,
        vmax=vmax_display
    )
    plt.title('Experiment vs Task Completion (sorted by overall performance)\n(! = over-completion >200%)', 
              fontsize=14, pad=20)
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Experiment/Condition', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_experiments_sorted.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/heatmap_experiments_sorted.png")
    plt.close()
    
    # 3. Per-task comparison across experiments
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, task in enumerate(task_cols):
        ax = axes[idx]
        
        task_data = df_comp[[task, 'Image']].sort_values(task, ascending=False)
        
        # Color scheme: Red (<50%), Orange (50-90%), Green (90-110%), Blue (>110%)
        colors = ['cornflowerblue' if x > 110 else 'green' if x >= 90 else 'orange' if x >= 50 else 'red' 
                  for x in task_data[task].values]
        
        bars = ax.barh(range(len(task_data)), task_data[task].values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(task_data)))
        ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                            for name in task_data['Image'].values], fontsize=7)
        ax.set_xlabel('Completion %', fontsize=9)
        ax.set_title(f'{task}', fontsize=10, fontweight='bold')
        
        # Adjust xlim based on max value for this task
        task_max = task_data[task].max()
        xlim_max = min(max(105, task_max + 10), 400)
        ax.set_xlim(0, xlim_max)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=100, color='green', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Add value labels for significant differences
        for i, v in enumerate(task_data[task].values):
            if i == 0 or i == len(task_data) - 1:  # Label best and worst
                label_x = min(v + 2, xlim_max - 10)
                ax.text(label_x, i, f'{v:.0f}', va='center', fontsize=7)
    
    plt.suptitle('Per-Task Performance Across All Experiments', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task_by_task_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/task_by_task_comparison.png")
    plt.close()
    
    # 4. Variance analysis: which tasks show most variation across experiments
    plt.figure(figsize=(10, 6))
    
    task_variance = df_comp[task_cols].std()
    task_means = df_comp[task_cols].mean()
    
    x = np.arange(len(task_cols))
    bars = plt.bar(x, task_variance.values, color='steelblue', alpha=0.7)
    
    plt.xticks(x, [f'T{i+1}' for i in range(len(task_cols))], rotation=0)
    plt.ylabel('Standard Deviation of Completion %', fontsize=12)
    plt.xlabel('Task', fontsize=12)
    plt.title('Task Variability Across Experiments\n(Higher = more variation between experiments)', 
              fontsize=14, pad=20)
    
    # Add mean labels on top
    for i, (v, m) in enumerate(zip(task_variance.values, task_means.values)):
        plt.text(i, v + 0.5, f'μ={m:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task_variance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/task_variance_analysis.png")
    plt.close()

def plot_task8_raw_counts(df_raw):
    """Plot raw counts for Task 8 (Permutations) across experiments."""
    
    if 'Task 8' not in df_raw.columns:
        print("Task 8 not found in data, skipping Task 8 raw counts plot")
        return
    
    # Extract Task 8 data and sort by count
    task8_data = df_raw[['Image', 'Task 8']].copy()
    task8_data = task8_data.dropna(subset=['Task 8'])  # Remove any NaN values
    task8_data = task8_data.sort_values('Task 8', ascending=False)
    
    if task8_data.empty:
        print("No raw count data for Task 8, skipping plot")
        return
    
    plt.figure(figsize=(14, max(6, len(task8_data) * 0.4)))
    
    # Create color gradient based on count (higher = better, but still use cool colors since all are low)
    max_count = task8_data['Task 8'].max()
    colors = plt.cm.viridis(task8_data['Task 8'].values / max_count) if max_count > 0 else 'steelblue'
    
    bars = plt.barh(range(len(task8_data)), task8_data['Task 8'].values, color=colors, alpha=0.8)
    plt.yticks(range(len(task8_data)), task8_data['Image'].values, fontsize=9)
    plt.xlabel('Number of Valid Permutations Found', fontsize=12)
    plt.title('Task 8: Permutations of ABCDEFG - Raw Counts Across Experiments\n(Expected: 5040 permutations)', 
              fontsize=14, pad=20)
    
    # Add value labels
    for i, v in enumerate(task8_data['Task 8'].values):
        plt.text(v + 0.3, i, f'{int(v)}', va='center', fontsize=9, fontweight='bold')
    
    # Add reference line for maximum observed
    plt.axvline(x=max_count, color='green', linestyle='--', alpha=0.4, 
                label=f'Best: {int(max_count)}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task8_raw_counts.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/task8_raw_counts.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TASK 8 (Permutations) RAW COUNT STATISTICS")
    print("="*70)
    print(f"Expected permutations: 5040")
    print(f"Best performance: {int(task8_data['Task 8'].max())} permutations ({task8_data['Image'].iloc[0]})")
    print(f"Worst performance: {int(task8_data['Task 8'].min())} permutations ({task8_data['Image'].iloc[-1]})")
    print(f"Mean: {task8_data['Task 8'].mean():.1f} permutations")
    print(f"Median: {task8_data['Task 8'].median():.1f} permutations")
    print(f"Std Dev: {task8_data['Task 8'].std():.1f} permutations")
    print("="*70 + "\n")

def plot_results(df_pf, df_comp):
    """Create visualizations with granular completion metrics."""
    
    task_cols = [col for col in df_pf.columns if col.startswith('Task')]
    
    # 1. Heatmap of completion percentages (more granular than pass/fail)
    plt.figure(figsize=(12, len(df_comp) * 0.5 + 2))
    
    # Create custom colormap: Red (0%) -> Yellow (50%) -> Green (100%) -> Blue (>100%)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#d73027', '#fee08b', '#d9ef8b', '#1a9850', '#4575b4']  # Red->Yellow->Green->Blue
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('completion', colors_list, N=n_bins)
    
    # Use reasonable max for visualization (clip display but show actual values)
    max_val = df_comp[task_cols].values.max()
    vmax_display = min(200, max_val) if max_val > 150 else 100
    
    # Create annotation with indicators for extreme over-completion
    annot_data = df_comp[task_cols].values.copy()
    annot_labels = np.array([[f'{val:.0f}' if val <= 200 else f'{val:.0f}!' 
                              for val in row] for row in annot_data])
    
    sns.heatmap(
        df_comp[task_cols].values,
        annot=annot_labels,
        fmt='',
        cmap=cmap,
        cbar_kws={'label': 'Completion %'},
        yticklabels=df_comp['Image'].values,
        xticklabels=[f'T{i+1}' for i in range(len(task_cols))],
        vmin=0,
        vmax=vmax_display
    )
    plt.title(f'Task Completion Percentage Across Images\n(! = over-completion >200%)', 
              fontsize=14, pad=20)
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
    
    # Determine y-axis limit based on max values
    max_completion = df_plot['Completion %'].max()
    ylim_max = min(max(105, max_completion + 10), 400)  # Cap display at 400% for readability
    
    sns.boxplot(data=df_plot, x='Task', y='Completion %', palette='Set2')
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% (Target)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
    if max_completion > 150:
        plt.axhline(y=150, color='blue', linestyle=':', alpha=0.4, label='150% (Over-completion)')
    plt.ylim(-5, ylim_max)
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
    
    # Adjust y-axis for over-completion
    max_y = (means + stds).max()
    ylim_max = min(max(110, max_y + 10), 400)
    
    plt.xticks(x, [f'T{i+1}' for i in range(len(task_cols))], rotation=0)
    plt.ylabel('Average Completion %', fontsize=12)
    plt.xlabel('Task', fontsize=12)
    plt.title('Average Task Completion (with std dev)', fontsize=14, pad=20)
    plt.ylim(0, ylim_max)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100% (Target)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50%')
    
    # Color bars: Red (<50%), Orange (50-90%), Green (90-110%), Blue (>110%)
    for i, bar in enumerate(bars):
        if means.values[i] > 110:
            bar.set_color('cornflowerblue')
        elif means.values[i] >= 90:
            bar.set_color('green')
        elif means.values[i] >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add value labels
    for i, v in enumerate(means.values):
        label_y = min(v + stds.values[i] + 3, ylim_max - 5)
        plt.text(i, label_y, f'{v:.1f}%', 
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
    
    # Adjust y-axis for potential over-completion
    max_img_avg = image_avg_completion.max()
    ylim_max = min(max(105, max_img_avg + 10), 300)
    plt.ylim(0, ylim_max)
    
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100% (Target)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50%')
    
    # Color bars: Red (<50%), Orange (50-90%), Green (90-110%), Blue (>110%)
    for i, bar in enumerate(bars):
        if image_avg_completion.values[i] > 110:
            bar.set_color('cornflowerblue')
        elif image_avg_completion.values[i] >= 90:
            bar.set_color('green')
        elif image_avg_completion.values[i] >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add labels
    for i, v in enumerate(image_avg_completion.values):
        label_y = min(v + 2, ylim_max - 5)
        plt.text(i, label_y, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
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
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Collecting results from:", RESULTS_DIR)
    df_pf, df_comp, df_raw = collect_all_results()
    
    if df_pf.empty:
        print("No result files found!")
        return
    
    print(f"Found {len(df_pf)} result files")
    
    # Save raw data
    df_pf.to_csv(os.path.join(OUTPUT_DIR, 'results_pass_fail.csv'), index=False)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, 'results_completion.csv'), index=False)
    df_raw.to_csv(os.path.join(OUTPUT_DIR, 'results_raw_counts.csv'), index=False)
    print(f"Saved: {OUTPUT_DIR}/results_pass_fail.csv")
    print(f"Saved: {OUTPUT_DIR}/results_completion.csv")
    print(f"Saved: {OUTPUT_DIR}/results_raw_counts.csv")
    
    # Print summary
    print_summary(df_pf, df_comp)
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(df_pf, df_comp)
    
    print("\nGenerating experiment comparison plots...")
    add_experiment_comparison_plots(df_comp)
    
    print("\nGenerating Task 8 raw counts plot...")
    plot_task8_raw_counts(df_raw)
    
    print("\nDone! Check the tedious_tasks/plots/ directory for plots and CSVs.")

if __name__ == "__main__":
    main()