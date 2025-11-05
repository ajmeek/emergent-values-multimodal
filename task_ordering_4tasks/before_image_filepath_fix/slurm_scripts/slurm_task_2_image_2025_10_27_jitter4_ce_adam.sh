#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_2_image_2025_10_27_jitter4_ce_adam
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_2_image_2025_10_27_jitter4_ce_adam_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_2_image_2025_10_27_jitter4_ce_adam_%j.err
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
    --condition task_2 \
    --image-path "/data/superstimuli_group/all_superstimuli/2025_10_27_jitter4_ce_adam.png" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/image_2025_10_27_jitter4_ce_adam \
    --seed-offset 10000

echo "Job completed: task_2 for image_2025_10_27_jitter4_ce_adam"
