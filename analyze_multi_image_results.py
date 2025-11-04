#!/usr/bin/env python3
"""
Analyze task ordering results across multiple superstimuli images.
Aggregates and compares effectiveness of different images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results_for_image(image_dir: str) -> dict:
    """Load all results for a single image."""
    results = {
        'task_2': [],
        'task_6': [],
        'task_7': [],
        'task_8': [],
    }

    for condition in results.keys():
        condition_dir = os.path.join(image_dir, condition)
        jsonl_path = os.path.join(condition_dir, "all_results.jsonl")

        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        results[condition].append(result)
        else:
            # Try individual JSON files
            json_files = list(Path(condition_dir).glob("result_*.json"))
            for file in json_files:
                with open(file, 'r') as f:
                    result = json.load(f)
                    results[condition].append(result)

    return results


def load_baseline(baseline_dir: str) -> List[dict]:
    """Load baseline results from new baselines/ structure."""
    baseline_results = []

    # Try new structure: baselines/baseline/all_results.jsonl
    jsonl_path = os.path.join(baseline_dir, "baselines/baseline/all_results.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    baseline_results.append(json.loads(line))
        return baseline_results

    # Try old structure: baseline/all_results.jsonl
    jsonl_path = os.path.join(baseline_dir, "baseline/all_results.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    baseline_results.append(json.loads(line))
        return baseline_results

    # Try even older structure: baseline_no_image/baseline/all_results.jsonl
    jsonl_path = os.path.join(baseline_dir, "baseline_no_image/baseline/all_results.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    baseline_results.append(json.loads(line))

    return baseline_results


def calculate_image_effectiveness(image_results: dict) -> dict:
    """Calculate success rates for each task for a given image."""
    effectiveness = {}

    for task, results in image_results.items():
        if results:
            task_id = int(task.split('_')[1])
            successes = sum(1 for r in results if r.get('first_task') == task_id)
            effectiveness[task] = {
                'success_rate': successes / len(results) if results else 0,
                'n_trials': len(results),
                'n_successes': successes,
            }
        else:
            effectiveness[task] = {
                'success_rate': 0,
                'n_trials': 0,
                'n_successes': 0,
            }

    # Calculate overall effectiveness
    total_trials = sum(e['n_trials'] for e in effectiveness.values())
    total_successes = sum(e['n_successes'] for e in effectiveness.values())
    effectiveness['overall'] = {
        'success_rate': total_successes / total_trials if total_trials > 0 else 0,
        'n_trials': total_trials,
        'n_successes': total_successes,
    }

    return effectiveness


def print_summary(all_results: dict, baseline_results: List[dict]):
    """Print comprehensive summary of multi-image results."""

    print("\n" + "="*80)
    print("MULTI-IMAGE TASK ORDERING RESULTS")
    print("="*80)

    # Baseline analysis
    if baseline_results:
        print(f"\nBASELINE (n={len(baseline_results)}):")
        baseline_first_tasks = [r.get('first_task') for r in baseline_results]
        for task_id in [2, 6, 7, 8]:
            count = baseline_first_tasks.count(task_id)
            print(f"  Task {task_id} first: {count}/{len(baseline_results)} ({count/len(baseline_results)*100:.1f}%)")

    # Per-image results
    print("\n" + "-"*60)
    print("RESULTS BY IMAGE")
    print("-"*60)

    summary_data = []

    for image_name, image_results in sorted(all_results.items()):
        effectiveness = calculate_image_effectiveness(image_results)

        print(f"\n{image_name}:")
        print(f"  Overall success: {effectiveness['overall']['success_rate']*100:.1f}% " +
              f"({effectiveness['overall']['n_successes']}/{effectiveness['overall']['n_trials']})")

        for task in ['task_2', 'task_6', 'task_7', 'task_8']:
            if task in effectiveness:
                e = effectiveness[task]
                print(f"  {task}: {e['success_rate']*100:.1f}% ({e['n_successes']}/{e['n_trials']})")

        summary_data.append({
            'image': image_name,
            'overall_success': effectiveness['overall']['success_rate'],
            'task_2_success': effectiveness.get('task_2', {}).get('success_rate', 0),
            'task_6_success': effectiveness.get('task_6', {}).get('success_rate', 0),
            'task_7_success': effectiveness.get('task_7', {}).get('success_rate', 0),
            'task_8_success': effectiveness.get('task_8', {}).get('success_rate', 0),
        })

    # Create DataFrame for analysis
    df = pd.DataFrame(summary_data)

    if len(df) > 0:
        print("\n" + "-"*60)
        print("AGGREGATE STATISTICS")
        print("-"*60)

        print(f"\nImages tested: {len(df)}")
        print(f"Average overall success: {df['overall_success'].mean()*100:.1f}% ± {df['overall_success'].std()*100:.1f}%")
        print(f"Best performing image: {df.loc[df['overall_success'].idxmax(), 'image']} " +
              f"({df['overall_success'].max()*100:.1f}%)")
        print(f"Worst performing image: {df.loc[df['overall_success'].idxmin(), 'image']} " +
              f"({df['overall_success'].min()*100:.1f}%)")

        print("\nBy task (across all images):")
        for task in ['task_2', 'task_6', 'task_7', 'task_8']:
            col = f'{task}_success'
            print(f"  {task}: {df[col].mean()*100:.1f}% ± {df[col].std()*100:.1f}%")

    return df


def create_visualizations(df: pd.DataFrame, baseline_results: List[dict], all_results: dict, output_dir: str):
    """Create comparison visualizations across images."""

    plots_dir = os.path.join(output_dir, 'multi_image_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Calculate baseline success rates (which task was chosen first in baseline)
    baseline_first_tasks = [r.get('first_task') for r in baseline_results]
    baseline_row = {
        'task_2_success': baseline_first_tasks.count(2) / len(baseline_results) if baseline_results else 0,
        'task_6_success': baseline_first_tasks.count(6) / len(baseline_results) if baseline_results else 0,
        'task_7_success': baseline_first_tasks.count(7) / len(baseline_results) if baseline_results else 0,
        'task_8_success': baseline_first_tasks.count(8) / len(baseline_results) if baseline_results else 0,
    }

    # 1. Comprehensive Heatmap of success rates
    # Calculate appropriate figure size based on number of images + baseline
    fig_height = max(10, (len(df) + 1) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Prepare data for heatmap - add baseline row at top
    heatmap_data = df.set_index('image')[['task_2_success', 'task_6_success',
                                           'task_7_success', 'task_8_success']]

    # Add baseline as first row
    baseline_df = pd.DataFrame([baseline_row], index=['BASELINE'])
    heatmap_data = pd.concat([baseline_df, heatmap_data])

    # Shorter column names for cleaner display
    heatmap_data.columns = ['Numbers', 'Banana', 'Translation', 'Permutations']

    # Create heatmap with percentage annotations
    sns.heatmap(heatmap_data, annot=True, fmt='.0%', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Task Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Image', fontsize=13, fontweight='bold')
    ax.set_title('Task Prioritization Success Rates Across All Images\n(BASELINE = Natural task preference without incentive)',
                 fontsize=15, fontweight='bold', pad=20)

    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comprehensive_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comprehensive heatmap saved")

    # 2. Bar chart of overall effectiveness
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    df_sorted = df.sort_values('overall_success', ascending=True)
    y_pos = np.arange(len(df_sorted))

    bars = ax.barh(y_pos, df_sorted['overall_success'], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['image'])
    ax.set_xlabel('Overall Success Rate', fontsize=12)
    ax.set_title('Image Effectiveness at Inducing Goal-Directed Behavior', fontsize=14)
    ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')

    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['overall_success'] + 0.01, i, f"{row['overall_success']*100:.1f}%",
                va='center', fontsize=9)

    ax.legend()
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Overall effectiveness bar chart saved")

    # 3. First-task choice distribution (1x4 subplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))

    task_names = {2: 'Numbers', 6: 'Banana', 7: 'Translation', 8: 'Permutations'}
    task_colors = {2: '#FF6B6B', 6: '#4ECDC4', 7: '#45B7D1', 8: '#96CEB4'}

    # Calculate baseline distribution
    baseline_dist = {2: 0, 6: 0, 7: 0, 8: 0}
    for r in baseline_results:
        first = r.get('first_task')
        if first in baseline_dist:
            baseline_dist[first] += 1
    baseline_total = len(baseline_results)
    baseline_props = {k: v/baseline_total if baseline_total > 0 else 0 for k, v in baseline_dist.items()}

    for idx, task_num in enumerate([2, 6, 7, 8]):
        ax = axes[idx]

        # Collect all results for this task across all images
        all_first_tasks = []
        for image_name, image_results in all_results.items():
            task_key = f'task_{task_num}'
            if task_key in image_results:
                for r in image_results[task_key]:
                    first = r.get('first_task')
                    if first:
                        all_first_tasks.append(first)

        # Calculate distribution
        dist = {2: 0, 6: 0, 7: 0, 8: 0}
        for first in all_first_tasks:
            if first in dist:
                dist[first] += 1

        total = len(all_first_tasks)
        props = {k: v/total if total > 0 else 0 for k, v in dist.items()}

        # Create stacked bar for treatment
        y_pos = [1, 0]  # BASELINE at top, treatment below

        # Plot baseline
        left = 0
        for task_id in [2, 6, 7, 8]:
            width = baseline_props[task_id]
            ax.barh(1, width, left=left, color=task_colors[task_id],
                   alpha=0.7, edgecolor='black', linewidth=1)
            if width > 0.05:  # Only label if > 5%
                ax.text(left + width/2, 1, f'{width:.0%}',
                       ha='center', va='center', fontsize=9, fontweight='bold')
            left += width

        # Plot treatment
        left = 0
        for task_id in [2, 6, 7, 8]:
            width = props[task_id]
            alpha = 1.0 if task_id == task_num else 0.7
            linewidth = 2 if task_id == task_num else 1
            ax.barh(0, width, left=left, color=task_colors[task_id],
                   alpha=alpha, edgecolor='black', linewidth=linewidth)
            if width > 0.05:  # Only label if > 5%
                ax.text(left + width/2, 0, f'{width:.0%}',
                       ha='center', va='center', fontsize=9,
                       fontweight='bold' if task_id == task_num else 'normal')
            left += width

        ax.set_yticks([0, 1])
        ax.set_yticklabels([f'Incentivize\n{task_names[task_num]}\n(n={total})',
                           f'BASELINE\n(n={baseline_total})'], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Proportion Chosen First', fontsize=10)
        ax.set_title(f'{task_names[task_num]} Task', fontsize=12, fontweight='bold',
                    color=task_colors[task_num])
        ax.grid(axis='x', alpha=0.3)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=task_colors[t], alpha=0.8,
                                     edgecolor='black', label=task_names[t])
                      for t in [2, 6, 7, 8]]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Distribution of First-Task Choices\n(What task was actually chosen first?)',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'first_task_distributions_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ First-task distribution (overall) saved")

    # 4. First-task choice distribution by image (detailed view)
    # Separate images into regular and reference images
    regular_images = []
    reference_images = []
    excluded_images = []

    for image_name in sorted(all_results.keys()):
        image_results = all_results[image_name]
        task_counts = {task_num: len(image_results.get(f'task_{task_num}', []))
                      for task_num in [2, 6, 7, 8]}
        has_complete_data = all(count == 5 for count in task_counts.values())

        if image_name.startswith('reference_'):
            # Include reference images regardless of completeness
            reference_images.append(image_name)
            if not has_complete_data:
                print(f"    Note: {image_name} has incomplete data but included: {task_counts}")
        elif has_complete_data:
            regular_images.append(image_name)
        else:
            excluded_images.append((image_name, task_counts))

    # Combine: regular images first, then reference images at the bottom
    filtered_images = regular_images + reference_images

    n_regular = len(regular_images)
    n_reference = len(reference_images)
    n_images = len(filtered_images)
    print(f"    Including {n_regular} regular images + {n_reference} reference images = {n_images} total")
    if excluded_images:
        print(f"    Excluded {len(excluded_images)} non-reference images with incomplete data")

    fig_height = max(12, (n_images + 1) * 0.35)  # +1 for baseline row
    fig, axes = plt.subplots(1, 4, figsize=(16, fig_height))

    for idx, task_num in enumerate([2, 6, 7, 8]):
        ax = axes[idx]
        task_key = f'task_{task_num}'

        # Prepare data: baseline + filtered images
        rows_data = []

        # Add baseline row
        baseline_dist = {2: 0, 6: 0, 7: 0, 8: 0}
        for r in baseline_results:
            first = r.get('first_task')
            if first in baseline_dist:
                baseline_dist[first] += 1
        baseline_total = len(baseline_results)
        baseline_props = {k: v/baseline_total if baseline_total > 0 else 0 for k, v in baseline_dist.items()}
        rows_data.append(('BASELINE', baseline_props, baseline_total))

        # Add each filtered image
        for image_name in filtered_images:
            image_results = all_results[image_name]
            # Calculate distribution for this image's task condition
            dist = {2: 0, 6: 0, 7: 0, 8: 0}
            for r in image_results[task_key]:
                first = r.get('first_task')
                if first in dist:
                    dist[first] += 1
            total = len(image_results[task_key])
            props = {k: v/total if total > 0 else 0 for k, v in dist.items()}
            rows_data.append((image_name, props, total))

        # Plot stacked horizontal bars
        n_rows = len(rows_data)
        y_positions = list(range(n_rows))[::-1]  # Reverse so BASELINE is at top

        for i, (name, props, total) in enumerate(rows_data):
            y = y_positions[i]

            # Draw stacked bar
            left = 0
            for task_id in [2, 6, 7, 8]:
                width = props[task_id]
                if width > 0:
                    alpha = 1.0 if task_id == task_num and name != 'BASELINE' else 0.7
                    linewidth = 1.5 if task_id == task_num and name != 'BASELINE' else 0.5
                    ax.barh(y, width, left=left, height=0.8,
                           color=task_colors[task_id], alpha=alpha,
                           edgecolor='black', linewidth=linewidth)
                    # Add text if wide enough
                    if width > 0.08:
                        ax.text(left + width/2, y, f'{width:.0%}',
                               ha='center', va='center', fontsize=7)
                left += width

        # Set y-axis labels (only show on first subplot)
        if idx == 0:
            # Format labels: split long names into two lines at underscore or space
            labels = []
            for name, _, _ in rows_data:
                if name == 'BASELINE':
                    labels.append('BASELINE')
                else:
                    # Split long names at a reasonable point (around 30-35 chars)
                    if len(name) > 35:
                        # Try to split at underscore or space
                        split_pos = name.rfind('_', 0, 40)
                        if split_pos < 20:  # If split is too early, try later
                            split_pos = name.find('_', 30)
                        if split_pos > 0 and split_pos < len(name):
                            labels.append(name[:split_pos] + '\n' + name[split_pos+1:])
                        else:
                            labels.append(name[:40] + '\n' + name[40:])
                    else:
                        labels.append(name)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=6.5)
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([])  # No labels on other subplots

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_xlabel('Proportion Chosen First', fontsize=10)
        ax.set_title(f'Incentivize {task_names[task_num]}', fontsize=12,
                    fontweight='bold', color=task_colors[task_num])
        ax.grid(axis='x', alpha=0.3)

        # Highlight baseline row (light line)
        ax.axhline(y=y_positions[0], color='gray', linewidth=1, alpha=0.3)

        # Add separator line between regular and reference images
        if n_reference > 0:
            # Find y position between last regular and first reference
            separator_y = y_positions[n_regular]  # Just above first reference image
            ax.axhline(y=separator_y + 0.5, color='black', linewidth=1.5, alpha=0.5, linestyle='--')

    # Add legend (positioned higher to avoid overlap with titles)
    legend_elements = [plt.Rectangle((0,0),1,1, fc=task_colors[t], alpha=0.8,
                                     edgecolor='black', label=f'{task_names[t]} chosen first')
                      for t in [2, 6, 7, 8]]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'first_task_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ First-task distribution (by image) saved")

    print(f"\n✓ All visualizations saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-image task ordering results")
    parser.add_argument("--results-dir", type=str, default="task_ordering_4tasks",
                        help="Base directory containing all image results")
    parser.add_argument("--create-plots", action="store_true",
                        help="Create visualization plots")

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")

    # Load baseline from the results directory itself
    baseline_results = load_baseline(args.results_dir)
    print(f"Loaded {len(baseline_results)} baseline results")

    # Load results for each image
    # Skip special directories: baselines, logs, baseline_archive, etc.
    skip_dirs = {'baselines', 'logs', 'baseline_archive', 'baseline_no_image_archive',
                 'baseline_no_image', 'baseline', 'slurm_scripts', 'plots', 'multi_image_plots'}

    all_results = {}
    all_dirs = [d for d in os.listdir(args.results_dir)
                if os.path.isdir(os.path.join(args.results_dir, d)) and d not in skip_dirs]

    print(f"Found {len(all_dirs)} image directories")

    for image_dir_name in sorted(all_dirs):
        image_dir_path = os.path.join(args.results_dir, image_dir_name)
        image_name = image_dir_name  # Use directory name as-is (no more "image_" prefix)

        print(f"Loading results for {image_name}...")
        image_results = load_results_for_image(image_dir_path)

        # Only include if we have results
        total_results = sum(len(r) for r in image_results.values())
        if total_results > 0:
            all_results[image_name] = image_results
            print(f"  Found {total_results} results")

    if not all_results:
        print("No image results found!")
        return

    print(f"\nLoaded results for {len(all_results)} images")

    # Analyze and print summary
    summary_df = print_summary(all_results, baseline_results)

    # Create visualizations
    if args.create_plots and len(summary_df) > 0:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        create_visualizations(summary_df, baseline_results, all_results, args.results_dir)
    elif not args.create_plots:
        print("\n💡 To generate visualizations, run with --create-plots flag")

    # Save summary to file
    summary_file = os.path.join(args.results_dir, "multi_image_analysis.txt")

    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    print_summary(all_results, baseline_results)
    output_text = buffer.getvalue()
    sys.stdout = old_stdout

    with open(summary_file, 'w') as f:
        f.write(output_text)

    print(f"\nAnalysis summary saved to: {summary_file}")


if __name__ == "__main__":
    main()