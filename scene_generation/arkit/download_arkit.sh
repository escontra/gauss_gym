#!/bin/bash

WORKER_ID="$1"
shift  # Remove first argument so $@ contains only the video_id:fold pairs

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Worker ${WORKER_ID} received ${#} video entries"

sudo apt-get install unzip

# Setup environment
cd ${SCRIPT_DIR}

# Loop through all remaining arguments (the video_id:fold pairs)
for ENTRY in "$@"; do
    # Split video_id and fold
    VIDEO_ID="${ENTRY%%:*}"
    FOLD="${ENTRY##*:}"

    echo "Processing video_id: $VIDEO_ID from fold: $FOLD"

    # Download data for this video_id
    python download_data.py raw \
        --split $FOLD \
        --video_id $VIDEO_ID \
        --download_dir $HOME/ARKitScenes

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded video_id: $VIDEO_ID"
    else
        echo "ERROR: Failed to download video_id: $VIDEO_ID"
    fi
done

echo "Worker ${WORKER_ID} finished processing all assigned videos"
