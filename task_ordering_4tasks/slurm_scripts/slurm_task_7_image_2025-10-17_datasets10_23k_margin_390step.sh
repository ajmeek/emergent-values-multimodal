#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_7_image_2025-10-17_datasets10_23k_margin_390step
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_7_image_2025-10-17_datasets10_23k_margin_390step_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_7_image_2025-10-17_datasets10_23k_margin_390step_%j.err
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
    --condition task_7 \
    --image-path "/data/superstimuli_group/all_superstimuli/2025-10-17 datasets10_23k_margin_390step.png" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/image_2025-10-17_datasets10_23k_margin_390step \
    --seed-offset 30000

echo "Job completed: task_7 for image_2025-10-17_datasets10_23k_margin_390step"
