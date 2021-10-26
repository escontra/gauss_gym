# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf /etc/apt/sources.list.d/cuda.list

echo "Installing nksr dependencies"
echo $SCRIPT_DIR

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/nksr
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

  # Clone NKSR.
  NKSR_PATH=$WORKSPACE_DIR/NKSR
  rm -rf $NKSR_PATH
  git clone https://github.com/nv-tlabs/NKSR.git $NKSR_PATH
  # cd $NKSR_PATH

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda env create -f $NKSR_PATH/environment.yml
  fi

  $CONDA_ROOT/bin/conda run -n nksr conda config --add channels conda-forge
  $CONDA_ROOT/bin/conda run -n nksr conda install s5cmd

  $CONDA_ROOT/bin/conda run -n nksr pip install -r $NKSR_PATH/requirements.txt
  $CONDA_ROOT/bin/conda run -n nksr pip install --no-build-isolation $NKSR_PATH/package/
  $CONDA_ROOT/bin/conda run -n nksr pip install trimesh huggingface_hub zarr transformers opencv-python pyyaml viser boto3 open3d
  $CONDA_ROOT/bin/conda run -n nksr pip install numpy==1.26.4
  $CONDA_ROOT/bin/conda run -n nksr pip uninstall -y hf_xet

  source $CONDA_ROOT/bin/activate nksr

  touch $SENTINEL_FILE
fi