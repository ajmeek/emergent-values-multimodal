#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_7_image_2025-10-21_qwen_25_32b_23k_margin_499__1_
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_7_image_2025-10-21_qwen_25_32b_23k_margin_499__1__%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_7_image_2025-10-21_qwen_25_32b_23k_margin_499__1__%j.err
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
    --image-path "/data/superstimuli_group/all_superstimuli/2025-10-21 qwen_25_32b_23k_margin_499 (1).png" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/image_2025-10-21_qwen_25_32b_23k_margin_499__1_ \
    --seed-offset 30000

echo "Job completed: task_7 for image_2025-10-21_qwen_25_32b_23k_margin_499__1_"
