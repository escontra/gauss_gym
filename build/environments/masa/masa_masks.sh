MISSION="$1"
S3_PATH="$2"
OUTPUT_FOLDER="$HOME/grand_tour_dataset"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/source_common.sh
source $SCRIPT_DIR/setup_dev.sh

source $HOME/.masa_deps/miniconda3/bin/activate masaenv

echo $SCRIPT_DIR
echo $MASA_PATH

CAMERA_NAMES=("hdr_front" "hdr_left" "hdr_right")

DOWNSCALE_FACTOR=1

for CAMERA_NAME in ${CAMERA_NAMES[@]}; do
    # Convert video to mp4 for masa.
    ffmpeg -y -framerate 30 \
        -i $OUTPUT_FOLDER/$MISSION/images/${CAMERA_NAME}/%06d.jpeg \
        -vf scale=iw/$DOWNSCALE_FACTOR:ih/$DOWNSCALE_FACTOR:flags=lanczos \
        -c:v libx264 -pix_fmt yuv420p ${CAMERA_NAME}_downscaled.mp4

    SAVE_MASK_PATH=$OUTPUT_FOLDER/$MISSION/images/${CAMERA_NAME}_mask
    mkdir -p $SAVE_MASK_PATH

    # python $MASA_PATH/demo/video_demo_with_text.py \
    MASA_PATH=$MASA_PATH python $SCRIPT_DIR/masa_masks.py \
        ${CAMERA_NAME}_downscaled.mp4 \
        --out $MASA_PATH/demo_outputs/carton_kangaroo_dance_outputs.mp4 \
        --save_masks_dir $SAVE_MASK_PATH \
        --masa_config $MASA_PATH/configs/masa-gdino/masa_gdino_swinb_inference.py \
        --masa_checkpoint $MASA_PATH/saved_models/masa_models/gdino_masa.pth \
        --sam_path $MASA_PATH/saved_models/pretrain_weights/sam_vit_h_4b8939.pth \
        --texts "person . human" \
        --score-thr 0.2 \
        --unified --show_fps --sam_mask

    ffmpeg -i ${SAVE_MASK_PATH}/%06d.png \
        -vf scale=iw*$DOWNSCALE_FACTOR:ih*$DOWNSCALE_FACTOR:flags=lanczos \
        ${SAVE_MASK_PATH}/%06d.png

    s5cmd cp ${SAVE_MASK_PATH} ${S3_PATH}/grand_tour/${MISSION}/images/
done