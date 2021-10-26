WORKER_ID="$1"
shift  # Remove first argument so $@ contains only the paths

conda deactivate
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../grand_tour/nerfstudio/source_common.sh
source $SCRIPT_DIR/../grand_tour/nerfstudio/setup_dev.sh
source $HOME/.ns_deps/miniconda3/bin/activate ns

echo "Worker ${WORKER_ID} received ${#} paths"

echo "All paths: $@"

# Loop through all remaining arguments (the paths)
for S3_PATH in "$@"; do
    # Extract scene ID from path (e.g., 42445441 from s3://.../42445441/)
    # Remove trailing slash and get last component
    PATH_NO_SLASH="${S3_PATH%/}"
    SCENE_ID="${PATH_NO_SLASH##*/}"
    # Extract base path (e.g., s3://far-falcon-assets/ARKitScenes/raw/Validation/)
    BASE_PATH="${PATH_NO_SLASH%/*}/"
    LOCAL_PATH=$HOME/ARKitScenes/raw/${SCENE_ID}
    
    echo "Processing scene: $SCENE_ID from $S3_PATH"
    echo "Base path: $BASE_PATH"
    echo "Local path: $LOCAL_PATH"

    s5cmd cp ${S3_PATH}* ${LOCAL_PATH}
    ns-train splatfacto \
        --pipeline.model.use-scale-regularization=True \
        --pipeline.model.output-depth-during-training=True \
        --pipeline.model.rasterize-mode=antialiased \
        --pipeline.model.camera-optimizer.mode=SO3xR3 \
        --pipeline.model.use-bilateral-grid=True \
        --experiment-name '' \
        --timestamp '' \
        --output-dir $LOCAL_PATH \
        --viewer.quit-on-train-completion True \
        --max-num-iterations 30000 \
        arkit-data \
        --data $LOCAL_PATH \
        --train-split-fraction 1.0
    ns-export gaussian-splat --load-config $LOCAL_PATH/splatfacto/config.yml --output-dir $LOCAL_PATH/splatfacto
    s5cmd cp ${LOCAL_PATH} ${BASE_PATH}
    rm -rf ${LOCAL_PATH}
done