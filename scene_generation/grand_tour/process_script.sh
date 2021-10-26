GRAND_TOUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MISSION="$1"
OUTPUT_FOLDER="$HOME/grand_tour_dataset"
S3_PATH="s3://far-falcon-assets"


# Set up NKSR workspace.
conda deactivate
CURR_DIR=$(dirname "$0")
source "$CURR_DIR/nksr/setup_dev.sh"
source $HOME/.nksr_deps/miniconda3/bin/activate nksr

pip install open3d

echo "Logging in to Hugging Face with token $HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# export AWS_PROFILE=far-sky AWS_DEFAULT_PROFILE=far-sky
unset AWS_PROFILE AWS_DEFAULT_PROFILE
# export AWS_PROFILE=far-compute AWS_DEFAULT_PROFILE=far-compute

python $CURR_DIR/process_data.py --mission $MISSION --dataset-folder $OUTPUT_FOLDER --s3-path $S3_PATH --download-only

# Compute masks with MASA.
conda deactivate
source $GRAND_TOUR_DIR/masa/masa_masks.sh $MISSION $S3_PATH

# Do the rest of the processing.
conda deactivate
source $HOME/.nksr_deps/miniconda3/bin/activate nksr
python $CURR_DIR/process_data.py --mission $MISSION --dataset-folder $OUTPUT_FOLDER --s3-path $S3_PATH --skip-masks --skip-download

conda deactivate
source $GRAND_TOUR_DIR/nerfstudio/train_splats.sh $MISSION $S3_PATH
