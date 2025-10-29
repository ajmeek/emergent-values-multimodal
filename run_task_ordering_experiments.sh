#!/bin/bash

# Task Ordering Experiments - Simple Direct Execution Script

# Test run 1: Baseline only (no image) - establish natural ordering
python run_task_ordering_experiment.py \
  --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/test_baseline_only \
  --baseline-runs 2 \
  --run-baseline-only \
  --hi-n 50 \
  --numbers-max 20 \
  --copy-times 10 \
  --alphabet-repeats 5 \
  --sums-max 100 \
  --banana-count 100 \
  --translate-loops 2 \
  --perm-letters "ABC"

# Test run 2: Small complete experiment (baseline + treatments)
python run_task_ordering_experiment.py \
  --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/test_full_small \
  --baseline-runs 2 \
  --runs-per-condition 1 \
  --image-path "/data/superstimuli_group/all_superstimuli/2025-10-15 jitter0_seed20 (1).png" \
  --hi-n 50 \
  --numbers-max 20 \
  --copy-times 10 \
  --alphabet-repeats 5 \
  --sums-max 100 \
  --banana-count 100 \
  --translate-loops 2 \
  --perm-letters "ABC"

# === PLACEHOLDER FOR FULL RUNS ===
# These will be filled in once test runs work

# Full run 1: Complete baseline (50 runs, no image)
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/baseline_full \
#   --baseline-runs 50 \
#   --run-baseline-only \
#   --hi-n 250 \
#   --numbers-max 60 \
#   --copy-times 100 \
#   --alphabet-repeats 50 \
#   --sums-max 1000 \
#   --banana-count 1000 \
#   --translate-loops 10 \
#   --perm-letters "ABCD"

# Full run 2: Image 1 - Complete experiment
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/image1_full \
#   --baseline-runs 30 \
#   --runs-per-condition 10 \
#   --image-path "/data/superstimuli_group/all_superstimuli/IMAGE_1.png" \
#   --hi-n 250 \
#   --numbers-max 60 \
#   --copy-times 100 \
#   --alphabet-repeats 50 \
#   --sums-max 1000 \
#   --banana-count 1000 \
#   --translate-loops 10 \
#   --perm-letters "ABCD"

# Full run 3: Image 2 - Complete experiment
# python run_task_ordering_experiment.py \
#   --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering/image2_full \
#   --baseline-runs 30 \
#   --runs-per-condition 10 \
#   --image-path "/data/superstimuli_group/all_superstimuli/IMAGE_2.png" \
#   --hi-n 250 \
#   --numbers-max 60 \
#   --copy-times 100 \
#   --alphabet-repeats 50 \
#   --sums-max 1000 \
#   --banana-count 1000 \
#   --translate-loops 10 \
#   --perm-letters "ABCD"