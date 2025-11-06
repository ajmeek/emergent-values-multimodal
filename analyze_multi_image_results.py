#!/usr/bin/env python3
"""
Generate first-task distribution plot for task ordering experiments.
Shows which task was chosen first across multiple image conditions.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


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


def create_first_task_distribution_plot(baseline_results: List[dict], all_results: dict,
                                       output_dir: str, expected_runs: int = 10):
    """Create first-task distribution plot showing all images."""

    plots_dir = os.path.join(output_dir, 'multi_image_plots')
    os.makedirs(plots_dir, exist_ok=True)

    task_names = {2: 'Numbers', 6: 'Banana', 7: 'Translation', 8: 'Permutations'}
    task_colors = {2: '#FF6B6B', 6: '#4ECDC4', 7: '#45B7D1', 8: '#96CEB4'}

    # Separate images into regular and reference images
    regular_images = []
    reference_images = []
    excluded_images = []

    for image_name in sorted(all_results.keys()):
        image_results = all_results[image_name]
        task_counts = {task_num: len(image_results.get(f'task_{task_num}', []))
                      for task_num in [2, 6, 7, 8]}
        has_complete_data = all(count == expected_runs for count in task_counts.values())

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
    print(f"  ✓ First-task distribution plot saved")


def main():
    parser = argparse.ArgumentParser(description="Generate first-task distribution plot")
    parser.add_argument("--results-dir", type=str, default="task_ordering_4tasks",
                        help="Base directory containing all image results")
    parser.add_argument("--create-plots", action="store_true",
                        help="Create visualization plots (kept for compatibility)")
    parser.add_argument("--expected-runs", type=int, default=10,
                        help="Expected number of runs per condition for completeness check (default: 10)")

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")

    # Load baseline from the results directory itself
    baseline_results = load_baseline(args.results_dir)
    print(f"Loaded {len(baseline_results)} baseline results")

    # Skip special directories
    skip_dirs = {'baselines', 'logs', 'baseline_archive', 'baseline_no_image_archive',
                'baseline_no_image', 'baseline', 'slurm_scripts', 'plots', 'multi_image_plots'}

    all_dirs = [d for d in os.listdir(args.results_dir)
               if os.path.isdir(os.path.join(args.results_dir, d)) and d not in skip_dirs]

    print(f"Found {len(all_dirs)} image directories")

    # Load results for each image
    all_results = {}
    for image_dir_name in sorted(all_dirs):
        image_path = os.path.join(args.results_dir, image_dir_name)
        print(f"Loading results for {image_dir_name}...")

        results = load_results_for_image(image_path)
        total_results = sum(len(v) for v in results.values())
        print(f"  Found {total_results} results")

        if total_results > 0:
            all_results[image_dir_name] = results

    print(f"\nLoaded results for {len(all_results)} images")

    if len(all_results) == 0:
        print("\nNo results found to visualize!")
        return

    # Generate the plot
    print("\nGenerating first-task distribution plot...")
    create_first_task_distribution_plot(baseline_results, all_results,
                                       args.results_dir, args.expected_runs)

    print(f"\n✓ Plot saved to {args.results_dir}/multi_image_plots/first_task_distributions.png")


if __name__ == "__main__":
    main()