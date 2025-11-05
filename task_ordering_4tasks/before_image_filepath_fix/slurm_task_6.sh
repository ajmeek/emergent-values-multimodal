#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_order_task_6
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_6_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_6_%j.err
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
    --condition task_6 \
    --image-path "/data/superstimuli_group/all_superstimuli/2025-10-15 jitter0_seed20 (1).png" \
    --num-runs 10 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks \
    --seed-offset 2000

echo "Job completed: task_6"
