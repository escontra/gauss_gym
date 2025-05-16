from setuptools import find_packages
from distutils.core import setup

setup(
    name="gauss_gym",
    version="1.0.0",
    author="Alejandro Escontrela",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="escontrela@berkeley.edu",
    description="Photorealistic Gym Environments for Isaac Gym",
    install_requires=[
        "isaacgym",
        "ruamel.yaml",
        "tqdm",
        "wandb==0.19.8",
        "plyfile",
        "warp-lang",
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git",
    ],
)