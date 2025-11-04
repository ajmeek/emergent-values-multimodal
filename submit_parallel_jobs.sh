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
read -p "Submit ${NUM_IMAGES} parallel jobs? [y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo "-------------------"

# First, submit baseline job (only needs to run once, no image)
echo "Submitting baseline job..."
sbatch << 'EOF'
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_ordering
#SBATCH --output=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/baseline_%j.out
#SBATCH --error=/data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/logs/baseline_%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate pytorch_latest
cd /data/austin_meek/emergent-values-multimodal

echo "Starting baseline runs at $(date)"
echo "----------------------------------------"

python run_task_ordering_4tasks.py \
    --condition baseline \
    --num-runs 10 \
    --output-dir /data/austin_meek/emergent-values-multimodal/task_ordering_4tasks/baselines

echo "----------------------------------------"
echo "Completed baseline runs at $(date)"
EOF

echo ""

# Submit one job per image (with --skip-baseline)
JOB_COUNT=0
for IMAGE_PATH in "${IMAGES_TO_PROCESS[@]}"; do
    IMAGE_NAME=$(basename "${IMAGE_PATH%.*}")

    echo "[$((++JOB_COUNT))/${NUM_IMAGES}] ${IMAGE_NAME}"

    sbatch << EOF
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=task_ordering
#SBATCH --output=${OUTPUT_BASE}/logs/${IMAGE_NAME}_%j.out
#SBATCH --error=${OUTPUT_BASE}/logs/${IMAGE_NAME}_%j.err
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
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
done

echo ""
echo "============================================================"
echo "All jobs submitted! (1 baseline + ${NUM_IMAGES} image jobs)"
echo "============================================================"
echo ""
echo "Results structure:"
echo "  ${OUTPUT_BASE}/"
echo "  ├── baselines/baseline/           (baseline runs, no image)"
echo "  └── {image_name}/                 (4 conditions per image)"
echo "      ├── task_2/"
echo "      ├── task_6/"
echo "      ├── task_7/"
echo "      └── task_8/"
echo ""
echo "Monitor: squeue -u \$USER"