# Generating Meshes Tutorial

## Dependencies:

```bash
conda env create
conda activate gt

# Install NKSR (for mesh reconstruction).
pip install nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html
```

## Download data:

```bash
python download_data.py
```


## Download data and generate masks:

```bash
python download_data.py
python generate_masks.py
```

## Create NerfStudio dataset:

```bash
python nerfstudio_convert.py
```

## Slice NerfStudio dataset to subsets and extract pointclouds:

```bash
python nerfstudio_slice.py
```


## Generate meshes

```bash
python generate_meshes.py
```
