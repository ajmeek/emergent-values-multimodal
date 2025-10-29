#!/bin/bash

# Task Ordering Experiments - Simple Direct Execution Script
# Uses default parameters from run_task_ordering_experiment.py for reproducibility

# Test run 1: Baseline only (no image) - establish natural ordering
python run_task_ordering_experiment.py \
  --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/test_baseline_only \
  --baseline-runs 2 \
  --run-baseline-only

# Test run 2: Small complete experiment (baseline + treatments)
python run_task_ordering_experiment.py \
  --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/test_full_small \
  --baseline-runs 2 \
  --runs-per-condition 1 \
  --image-path "/data/superstimuli_group/all_superstimuli/2025-10-15 jitter0_seed20 (1).png"

# === PLACEHOLDER FOR FULL RUNS ===
# These will be filled in once test runs work

# Full run 1: Complete baseline (50 runs, no image)
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/baseline_full \
#   --baseline-runs 50 \
#   --run-baseline-only

# Full run 2: Image 1 - Complete experiment
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/image1_full \
#   --baseline-runs 30 \
#   --runs-per-condition 10 \
#   --image-path "/data/superstimuli_group/all_superstimuli/IMAGE_1.png"

# Full run 3: Image 2 - Complete experiment
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/image2_full \
#   --baseline-runs 30 \
#   --runs-per-condition 10 \
#   --image-path "/data/superstimuli_group/all_superstimuli/IMAGE_2.png"