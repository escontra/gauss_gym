MISSION="$1"
S3_PATH="$2"
OUTPUT_FOLDER="$HOME/grand_tour_dataset"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/source_common.sh
source $SCRIPT_DIR/setup_dev.sh

source $HOME/.ns_deps/miniconda3/bin/activate ns

echo $SCRIPT_DIR

SLICES_FOLDER=$(find ${OUTPUT_FOLDER}/${MISSION}_nerfstudio/slices -mindepth 1 -maxdepth 1 -type d)

for SLICE_FOLDER in $SLICES_FOLDER; do
    ns-train splatfacto \
    --pipeline.model.use-scale-regularization=True \
    --pipeline.model.output-depth-during-training=True \
    --pipeline.model.rasterize-mode=antialiased \
    --pipeline.model.camera-optimizer.mode=SO3xR3 \
    --pipeline.model.use-bilateral-grid=True \
    --experiment-name '' \
    --timestamp '' \
    --output-dir $SLICE_FOLDER \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --data $SLICE_FOLDER \
    --train-split-fraction=1.0 \
    --depth-unit-scale-factor=1.0
    ns-export gaussian-splat --load-config $SLICE_FOLDER/splatfacto/config.yml --output-dir $SLICE_FOLDER/splatfacto
    s5cmd cp ${SLICE_FOLDER} ${S3_PATH}/grand_tour/${MISSION}_nerfstudio/slices/
done