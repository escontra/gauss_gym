# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf /etc/apt/sources.list.d/cuda.list

echo "Installing NS dependencies"
echo $SCRIPT_DIR

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/ns
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

  $CONDA_ROOT/bin/conda run -n ns bash $SCRIPT_DIR/conda_env/install_conda_hooks.sh

  # Fix PyTorch Intel JIT symbol issue
  $CONDA_ROOT/bin/conda run -n ns pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
  $CONDA_ROOT/bin/conda run -n ns pip install ml_collections open3d tqdm opencv-python Pillow

  $CONDA_ROOT/bin/conda run -n ns pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

  # Install Nerfstudio.
  NS_PATH=$WORKSPACE_DIR/nerfstudio
  rm -rf $NS_PATH
  git clone https://github.com/escontra/nerfstudio-gauss-gym.git $NS_PATH
  # git clone https://github.com/nerfstudio-project/nerfstudio.git $NS_PATH
  # $CONDA_ROOT/bin/conda run -n ns pip install --upgrade pip setuptools
  $CONDA_ROOT/bin/conda run -n ns pip install -e $NS_PATH/.
  # $CONDA_ROOT/bin/conda run -n ns pip install nerfstudio

  # NS upgrades to numpy 2, which doesn't work with torch 2.1.2.
  $CONDA_ROOT/bin/conda run -n ns pip install numpy==1.26.4

  $CONDA_ROOT/bin/conda run -n ns conda config --add channels conda-forge
  $CONDA_ROOT/bin/conda run -n ns conda install s5cmd

  source $CONDA_ROOT/bin/activate ns
  
  touch $SENTINEL_FILE
fi
