# Generating Meshes Tutorial

## Dependencies:

Install uv and python 3.11:
```bash
pip3 install uv
uv install
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-distutils
```

Setup virtual_environment:

```bash
cd ~/git/gauss-gym/mesh_generation/elevation_mapping
mkdir .venv; cd .venv
python3.11 -m venv em
source em/bin/activate
cd ..; uv pip install -r pyproject.toml
```

Install torch and transformers:
```bash
# Assumes you have the grand_tour venv sourced.
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers
```

Install elevation mapping (python only installation):

```bash
# Assumes you have the grand_tour venv sourced.
cd ~/git; git clone git@github.com:leggedrobotics/elevation_mapping_cupy.git -b dev/python_library_installation
cd ~/git/elevation_mapping_cupy ; uv pip install -r requirements.txt; 
# If you have CUDA 12x - for older version adapat this
uv pip install cupy-cuda12x
# Here you can also adapt the python version if needed
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install for meshing here:
```
https://github.com/fogleman/hmm
```
Check if you have then executable here: `/usr/local/bin/hmm`


## Running:
1. download_data.py
- data will be downloaded to `~/grand_tour_dataset`

1. dynamic_points_filtering_using_images.py
- this will create binary masks for each hdr image and store it
- will also create filtered pointcloud topics with removed dynamics points


2. generate_elevation_maps.py


