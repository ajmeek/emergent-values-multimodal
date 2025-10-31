#!/bin/bash
#
# SLURM submission script for 4-task ordering experiments
# Runs experiments across all superstimuli images
#

# Configuration
SCRIPT_PATH="run_task_ordering_4tasks.py"
OUTPUT_BASE="/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks"
IMAGE_DIR="/data/superstimuli_group/all_superstimuli"
NUM_RUNS_BASELINE=10  # Baseline runs (only needed once, not per image)
NUM_RUNS_TREATMENT=10  # Runs per treatment condition per image

# Create main output directory
mkdir -p ${OUTPUT_BASE}
mkdir -p ${OUTPUT_BASE}/logs

# Function to submit a SLURM job
submit_job() {
    local condition=$1
    local num_runs=$2
    local seed_offset=$3
    local image_path=$4
    local output_subdir=$5
    local job_name="${condition}_$(basename "${output_subdir}")"

    # Create SLURM script
    local script_file="${OUTPUT_BASE}/slurm_scripts/slurm_${job_name}.sh"
    mkdir -p "${OUTPUT_BASE}/slurm_scripts"

    cat > "${script_file}" << EOF
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=${job_name}
#SBATCH --output=${OUTPUT_BASE}/logs/${job_name}_%j.out
#SBATCH --error=${OUTPUT_BASE}/logs/${job_name}_%j.err
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
    --image-path "${image_path}" \\
    --num-runs ${num_runs} \\
    --output-dir ${output_subdir} \\
    --seed-offset ${seed_offset}

echo "Job completed: ${condition} for $(basename "${output_subdir}")"
EOF

    # Submit the job
    echo "Submitting: ${condition} for $(basename "${output_subdir}") (${num_runs} runs)"
    sbatch "${script_file}"
}

echo "========================================="
echo "Multi-Image Task Ordering Experiments"
echo "========================================="
echo "Image directory: ${IMAGE_DIR}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# First, run baseline ONCE (no image needed)
echo "=== BASELINE (no image) ==="
BASELINE_DIR="${OUTPUT_BASE}/baseline_no_image"
mkdir -p "${BASELINE_DIR}"
submit_job "baseline" ${NUM_RUNS_BASELINE} 0 "none" "${BASELINE_DIR}"

echo ""
echo "=== TREATMENT CONDITIONS (with images) ==="

# List all PNG images and iterate through them
for IMAGE_PATH in ${IMAGE_DIR}/*.png; do
    # Check if the file exists (in case no PNGs found)
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "No PNG images found in ${IMAGE_DIR}"
        break
    fi

    # Extract image name without extension for directory name
    IMAGE_NAME=$(basename "${IMAGE_PATH}" .png)

    # Clean up the image name (replace spaces and special chars with underscores)
    CLEAN_IMAGE_NAME=$(echo "${IMAGE_NAME}" | sed 's/[^a-zA-Z0-9-]/_/g')

    echo ""
    echo "Processing image: ${IMAGE_NAME}"
    echo "Clean name: ${CLEAN_IMAGE_NAME}"

    # Create output directory for this image
    IMAGE_OUTPUT_DIR="${OUTPUT_BASE}/image_${CLEAN_IMAGE_NAME}"
    mkdir -p "${IMAGE_OUTPUT_DIR}"

    # Submit jobs for each task incentive with this image
    submit_job "task_2" ${NUM_RUNS_TREATMENT} 10000 "${IMAGE_PATH}" "${IMAGE_OUTPUT_DIR}"
    submit_job "task_6" ${NUM_RUNS_TREATMENT} 20000 "${IMAGE_PATH}" "${IMAGE_OUTPUT_DIR}"
    submit_job "task_7" ${NUM_RUNS_TREATMENT} 30000 "${IMAGE_PATH}" "${IMAGE_OUTPUT_DIR}"
    submit_job "task_8" ${NUM_RUNS_TREATMENT} 40000 "${IMAGE_PATH}" "${IMAGE_OUTPUT_DIR}"

    # Add small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "========================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u $USER"
echo ""
echo "Results will be organized as:"
echo "  ${OUTPUT_BASE}/"
echo "    ├── baseline_no_image/"
echo "    ├── image_<name1>/"
echo "    │   ├── task_2/"
echo "    │   ├── task_6/"
echo "    │   ├── task_7/"
echo "    │   └── task_8/"
echo "    ├── image_<name2>/"
echo "    │   └── ..."
echo "    └── logs/"
echo ""
echo "To analyze a specific image's results:"
echo "  python analyze_task_ordering_4tasks.py --results-dir ${OUTPUT_BASE}/image_<name>"