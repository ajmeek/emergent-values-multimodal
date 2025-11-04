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

# Submit one job per image
JOB_COUNT=0
for IMAGE_PATH in "${IMAGES_TO_PROCESS[@]}"; do
    IMAGE_NAME=$(basename "${IMAGE_PATH%.*}")

    # Clean name for SLURM (max 15 chars for job name)
    CLEAN_NAME=$(echo "${IMAGE_NAME:0:15}" | tr ' ' '_')

    echo "[$((++JOB_COUNT))/${NUM_IMAGES}] ${IMAGE_NAME}"

    sbatch << EOF
#!/bin/bash
#SBATCH --partition=cais
#SBATCH --job-name=to_${CLEAN_NAME}
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
echo "All ${NUM_IMAGES} jobs submitted!"
echo ""
echo "Commands:"
echo "  Monitor queue:  squeue -u \$USER"
echo "  Cancel all:     scancel -u \$USER"
echo "  View logs:      ls -lht ${OUTPUT_BASE}/logs/ | head"
echo ""
echo "Results location:"
echo "  ${OUTPUT_BASE}/single_test_*/"
echo ""
echo "To check completion status:"
echo "  for d in ${OUTPUT_BASE}/single_test_*/; do echo \"\$(basename \"\$d\"): \$(ls \$d/*/all_results.jsonl 2>/dev/null | wc -l)/5 conditions\"; done"
echo "============================================================"