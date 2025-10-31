#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=baseline_baseline_no_image
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/baseline_baseline_no_image_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/baseline_baseline_no_image_%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Load environment
source ~/.bashrc
conda activate pytorch_latest

# Navigate to project directory
cd /data/austin_meek/emergent-values-multimodal

# Run experiment
python run_task_ordering_4tasks.py \
    --condition baseline \
    --image-path "none" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/baseline_no_image \
    --seed-offset 0

echo "Job completed: baseline for baseline_no_image"
