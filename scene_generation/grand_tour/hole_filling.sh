GRAND_TOUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MISSION="$1"
OUTPUT_FOLDER="$HOME/grand_tour_dataset"
S3_PATH="s3://far-falcon-assets"

NS_PATH=${S3_PATH}/grand_tour/${MISSION}_nerfstudio/

# Set up NKSR workspace.
conda deactivate
CURR_DIR=$(dirname "$0")
source "$CURR_DIR/nksr/setup_dev.sh"
source $HOME/.nksr_deps/miniconda3/bin/activate nksr

echo "Logging in to Hugging Face with token $HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# export AWS_PROFILE=far-sky AWS_DEFAULT_PROFILE=far-sky
unset AWS_PROFILE AWS_DEFAULT_PROFILE
# export AWS_PROFILE=far-compute AWS_DEFAULT_PROFILE=far-compute

python $CURR_DIR/../hole_filling.py --s3-path $NS_PATH --dataset grand_tour
