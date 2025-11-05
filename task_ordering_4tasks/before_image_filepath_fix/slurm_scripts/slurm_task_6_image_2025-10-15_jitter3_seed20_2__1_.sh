#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_6_image_2025-10-15_jitter3_seed20_2__1_
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_6_image_2025-10-15_jitter3_seed20_2__1__%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/task_6_image_2025-10-15_jitter3_seed20_2__1__%j.err
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
    --image-path "/data/superstimuli_group/all_superstimuli/2025-10-15 jitter3_seed20_2 (1).png" \
    --num-runs 3 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/image_2025-10-15_jitter3_seed20_2__1_ \
    --seed-offset 20000

echo "Job completed: task_6 for image_2025-10-15_jitter3_seed20_2__1_"
