#!/bin/bash
# ARKit mesh hole filling script
# Processes multiple ARKit scenes passed as arguments

WORKER_ID="$1"
shift  # Remove first argument so $@ contains only the paths

conda deactivate
SCRIPT_DIR_HOLE_FILLING=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set up NKSR workspace (needed for hole filling)
source "$SCRIPT_DIR_HOLE_FILLING/../grand_tour/nksr/setup_dev.sh"
source $HOME/.nksr_deps/miniconda3/bin/activate nksr

echo "Worker ${WORKER_ID} received ${#} paths"
echo "All paths: $@"

echo "Logging in to Hugging Face with token $HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Unset AWS profiles
unset AWS_PROFILE AWS_DEFAULT_PROFILE

# Loop through all remaining arguments (the S3 paths)
for S3_PATH in "$@"; do
    # Extract scene ID from path (e.g., 42445441 from s3://.../42445441/)
    # Remove trailing slash and get last component
    PATH_NO_SLASH="${S3_PATH%/}"
    SCENE_ID="${PATH_NO_SLASH##*/}"

    echo "========================================"
    echo "Processing scene: $SCENE_ID from $S3_PATH"
    echo "========================================"

    # Process this ARKit scene's mesh with hole filling
    python $SCRIPT_DIR_HOLE_FILLING/../hole_filling.py --s3-path $S3_PATH --dataset arkit

    echo "Completed processing scene: $SCENE_ID"
    echo ""
done

echo "Worker ${WORKER_ID} completed all ${#} scenes"
