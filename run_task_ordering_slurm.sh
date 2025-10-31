#!/bin/bash
#
# SLURM submission script for 4-task ordering experiments
# Runs baseline + 4 treatment conditions in parallel
#

# Configuration
SCRIPT_PATH="run_task_ordering_4tasks.py"
OUTPUT_BASE="/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks"
IMAGE_PATH="/data/superstimuli_group/all_superstimuli/2025-10-15 jitter0_seed20 (1).png"
NUM_RUNS_BASELINE=50  # More baseline runs to establish natural ordering
NUM_RUNS_TREATMENT=30  # Runs per treatment condition

# Create output directory
mkdir -p ${OUTPUT_BASE}

# Function to submit a SLURM job
submit_job() {
    local condition=$1
    local num_runs=$2
    local seed_offset=$3
    local job_name="task_order_${condition}"

    # Create SLURM script
    cat > ${OUTPUT_BASE}/slurm_${condition}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${OUTPUT_BASE}/logs/${condition}_%j.out
#SBATCH --error=${OUTPUT_BASE}/logs/${condition}_%j.err
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
python ${SCRIPT_PATH} \\
    --condition ${condition} \\
    --image-path "${IMAGE_PATH}" \\
    --num-runs ${num_runs} \\
    --output-dir ${OUTPUT_BASE} \\
    --seed-offset ${seed_offset}

echo "Job completed: ${condition}"
EOF

    # Submit the job
    echo "Submitting job for condition: ${condition} (${num_runs} runs)"
    sbatch ${OUTPUT_BASE}/slurm_${condition}.sh
}

# Create log directory
mkdir -p ${OUTPUT_BASE}/logs

echo "========================================="
echo "Submitting 4-Task Ordering Experiments"
echo "========================================="
echo "Output directory: ${OUTPUT_BASE}"
echo "Image stimulus: ${IMAGE_PATH}"
echo ""

# Submit baseline condition (no image incentive)
submit_job "baseline" ${NUM_RUNS_BASELINE} 0

# Submit treatment conditions (with image incentive for each task)
submit_job "task_2" ${NUM_RUNS_TREATMENT} 1000  # Numbers task
submit_job "task_6" ${NUM_RUNS_TREATMENT} 2000  # Banana task
submit_job "task_7" ${NUM_RUNS_TREATMENT} 3000  # Translation task
submit_job "task_8" ${NUM_RUNS_TREATMENT} 4000  # Permutation task

echo ""
echo "All jobs submitted. Use 'squeue -u $USER' to check status."
echo ""
echo "Once complete, analyze results with:"
echo "  python analyze_task_ordering_4tasks.py --results-dir ${OUTPUT_BASE}"