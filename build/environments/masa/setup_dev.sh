# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf /etc/apt/sources.list.d/cuda.list

echo "Installing masa dependencies"
echo $SCRIPT_DIR

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/masaenv
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_dev

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda env create -f $SCRIPT_DIR/environment.yml
  fi

  $CONDA_ROOT/bin/conda run -n masaenv bash $SCRIPT_DIR/conda_env/install_conda_hooks.sh

  # Fix PyTorch Intel JIT symbol issue
  echo "Fixing PyTorch Intel library conflicts..."
  $CONDA_ROOT/bin/conda run -n masaenv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  $CONDA_ROOT/bin/conda run -n masaenv pip install numpy==1.26.4 fsspec

  # Clone masa.
  rm -rf $MASA_PATH
  git clone https://github.com/siyuanliii/masa.git $MASA_PATH

  $CONDA_ROOT/bin/conda run -n masaenv sh $MASA_PATH/install_dependencies.sh
  
  # Fix NLTK resource errors.
  echo "Downloading NLTK data..."
  $CONDA_ROOT/bin/conda run -n masaenv python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

  $CONDA_ROOT/bin/conda run -n masaenv pip install numpy==1.26.4
  
  echo "Downloading MASA models..."
  MODEL_PATH=$MASA_PATH/saved_models/masa_models
  if [[ ! -d $MODEL_PATH/gdino_masa.pth ]]; then
    mkdir -p $MODEL_PATH && wget -P $MODEL_PATH https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth
  fi
  PRETRAIN_WEIGHTS_PATH=$MASA_PATH/saved_models/pretrain_weights
  if [[ ! -d $PRETRAIN_WEIGHTS_PATH/sam_vit_h_4b8939.pth ]]; then
    mkdir -p $PRETRAIN_WEIGHTS_PATH && wget -P $PRETRAIN_WEIGHTS_PATH https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  fi

  source $CONDA_ROOT/bin/activate masaenv

  touch $SENTINEL_FILE
fi