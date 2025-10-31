#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_8_image_2025-10-17_datasets10_23k_ce_110step
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_8_image_2025-10-17_datasets10_23k_ce_110step_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_8_image_2025-10-17_datasets10_23k_ce_110step_%j.err
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
    --condition task_8 \
    --image-path "/data/superstimuli_group/all_superstimuli/2025-10-17 datasets10_23k_ce_110step.png" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/image_2025-10-17_datasets10_23k_ce_110step \
    --seed-offset 40000

echo "Job completed: task_8 for image_2025-10-17_datasets10_23k_ce_110step"
