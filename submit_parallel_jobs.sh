#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_PATH="run_task_ordering_4tasks.py"
OUTPUT_BASE="/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks"
IMAGE_DIR="/data/superstimuli_group/all_superstimuli"
NUM_RUNS=5

# -----------------------------------------------------------------------------
# PREFIX CONFIGURATION
# Edit this section to control which images to process
# -----------------------------------------------------------------------------

# Option 1: Process specific prefixes (default)
USE_PREFIXES=true
PREFIXES=(
    # "2025_10_31"
    "reference_BAD_armed_masked_fighters"
    "reference_MID_Caucasian_female"
    "reference_GOOD_Studio_Ghibli"
)

# Option 2: Process ALL images (set to true to override prefix filtering)
# USE_PREFIXES=false  # Uncomment to process ALL images

# Option 3: Custom pattern (for ad-hoc filtering)
# CUSTOM_PATTERN="test_*"  # Uncomment and modify to use custom glob pattern

# =============================================================================
# Main Script (no need to edit below)
# =============================================================================

# Function to count existing runs in a condition directory
count_runs() {
    local condition_dir=$1
    local results_file="${condition_dir}/all_results.jsonl"

    if [ -f "$results_file" ]; then
        wc -l < "$results_file" | tr -d ' '
    else
        echo 0
    fi
}

# Create log directory
mkdir -p ${OUTPUT_BASE}/logs

echo "============================================================"
echo "Parallel Task Ordering Experiments"
echo "============================================================"
echo "Output: ${OUTPUT_BASE}"
echo "Target: ${NUM_RUNS} runs per condition"
echo ""

# Collect images to process
IMAGES_TO_PROCESS=()

if [ ! -z "$CUSTOM_PATTERN" ]; then
    echo "Using custom pattern: ${CUSTOM_PATTERN}"
    for IMAGE_PATH in ${IMAGE_DIR}/${CUSTOM_PATTERN}.{png,PNG,jpg,JPG,jpeg,JPEG}; do
        [ -f "$IMAGE_PATH" ] && IMAGES_TO_PROCESS+=("$IMAGE_PATH")
    done
elif [ "$USE_PREFIXES" = true ]; then
    echo "Using prefixes: ${PREFIXES[@]}"
    for PREFIX in "${PREFIXES[@]}"; do
        for IMAGE_PATH in ${IMAGE_DIR}/${PREFIX}*.{png,PNG,jpg,JPG,jpeg,JPEG}; do
            [ -f "$IMAGE_PATH" ] && IMAGES_TO_PROCESS+=("$IMAGE_PATH")
        done
    done
else
    echo "Processing ALL images"
    for IMAGE_PATH in ${IMAGE_DIR}/*.{png,PNG,jpg,JPG,jpeg,JPEG}; do
        [ -f "$IMAGE_PATH" ] && IMAGES_TO_PROCESS+=("$IMAGE_PATH")
    done
fi

# Remove duplicates and count
IMAGES_TO_PROCESS=($(printf "%s\n" "${IMAGES_TO_PROCESS[@]}" | sort -u))
NUM_IMAGES=${#IMAGES_TO_PROCESS[@]}

echo "Found ${NUM_IMAGES} images to process"
echo ""

if [ ${NUM_IMAGES} -eq 0 ]; then
    echo "No images found! Check your PREFIX configuration."
    exit 1
fi

# Show first few images as confirmation
echo "First images to process:"
for i in "${!IMAGES_TO_PROCESS[@]}"; do
    [ $i -ge 3 ] && break
    echo "  - $(basename "${IMAGES_TO_PROCESS[$i]}")"
done
[ ${NUM_IMAGES} -gt 3 ] && echo "  ... and $((NUM_IMAGES - 3)) more"
echo ""

# Ask for confirmation
TOTAL_JOBS=$((NUM_IMAGES + 1))
read -p "Submit ${TOTAL_JOBS} parallel jobs (1 baseline + ${NUM_IMAGES} images)? [y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo "-------------------"

# First, submit baseline job (only needs to run once, no image)
baseline_dir="${OUTPUT_BASE}/baselines/baseline"
baseline_runs=$(count_runs "$baseline_dir")

if [ $baseline_runs -ge $NUM_RUNS ]; then
    echo "[1/${TOTAL_JOBS}] Skipping baseline - already have ${baseline_runs}/${NUM_RUNS} runs"
else
    echo "[1/${TOTAL_JOBS}] Submitting baseline job (${baseline_runs}/${NUM_RUNS} runs exist)..."
    sbatch << EOF
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_ordering
#SBATCH --output=${OUTPUT_BASE}/logs/baseline_%j.out
#SBATCH --error=${OUTPUT_BASE}/logs/baseline_%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate pytorch_latest
cd /data/austin_meek/emergent-values-multimodal

echo "Starting baseline runs at \$(date)"
echo "----------------------------------------"

python ${SCRIPT_PATH} \
    --condition baseline \
    --num-runs ${NUM_RUNS} \
    --output-dir ${OUTPUT_BASE}/baselines

echo "----------------------------------------"
echo "Completed baseline runs at \$(date)"
EOF
fi

echo ""

# Submit one job per image (with --skip-baseline)
JOB_COUNT=1  # Start at 1 since baseline is job #1
for IMAGE_PATH in "${IMAGES_TO_PROCESS[@]}"; do
    IMAGE_NAME=$(basename "${IMAGE_PATH%.*}")

    # Check if this image needs any runs
    needs_processing=false
    status_msg=""
    for task_num in 2 6 7 8; do
        task_dir="${OUTPUT_BASE}/${IMAGE_NAME}/task_${task_num}"
        existing=$(count_runs "$task_dir")
        if [ $existing -lt $NUM_RUNS ]; then
            needs_processing=true
            status_msg="${status_msg} task_${task_num}:${existing}/${NUM_RUNS}"
        fi
    done

    if [ "$needs_processing" = false ]; then
        echo "[$((++JOB_COUNT))/${TOTAL_JOBS}] Skipping ${IMAGE_NAME} - all conditions complete (${NUM_RUNS} runs each)"
    else
        echo "[$((++JOB_COUNT))/${TOTAL_JOBS}] Submitting ${IMAGE_NAME} -${status_msg}"
        sbatch << EOF
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_ordering
#SBATCH --output=${OUTPUT_BASE}/logs/${IMAGE_NAME}_%j.out
#SBATCH --error=${OUTPUT_BASE}/logs/${IMAGE_NAME}_%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate pytorch_latest
cd /data/austin_meek/emergent-values-multimodal

echo "Starting: ${IMAGE_NAME} at \$(date)"
echo "Image path: ${IMAGE_PATH}"
echo "----------------------------------------"

python ${SCRIPT_PATH} \
    --single-image-test \
    --skip-baseline \
    --image-path "${IMAGE_PATH}" \
    --num-runs ${NUM_RUNS} \
    --output-dir ${OUTPUT_BASE}

echo "----------------------------------------"
echo "Completed: ${IMAGE_NAME} at \$(date)"
EOF
        sleep 0.1  # Small delay to not overwhelm scheduler
    fi
done

echo ""
echo "============================================================"
echo "Job submission complete!"
echo "============================================================"
echo ""
echo "Target: ${NUM_RUNS} runs per condition"
echo "(Jobs with complete runs were automatically skipped)"
echo ""
echo "Results structure:"
echo "  ${OUTPUT_BASE}/"
echo "  ├── baselines/baseline/           (${NUM_RUNS} baseline runs, no image)"
echo "  └── {image_name}/                 (4 conditions × ${NUM_RUNS} runs each)"
echo "      ├── task_2/"
echo "      ├── task_6/"
echo "      ├── task_7/"
echo "      └── task_8/"