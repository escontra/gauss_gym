# Steps to collect PolyCam data:
# 1. Use PolyCam in LIDAR mode to collect data.
# 2. Process the data in the PolyCam app.
# 3. Export "Raw data" and "GLTF" (glb) from the app. Place the unzipped contents
#    in the <POLYCAM_PATH>. Rename the .glb file to "raw.glb".
# 4. Process the data with:
#    ns-process-data polycam --use-depth --data <POLYCAM_PATH> --output-dir <POLYCAM_PATH>
# 5. Run this script to train a splat model.
# 6. Generate meshes with:
#    python mesh_generation/generate_mesh_slices.py --config=mesh_generation/configs/polycam.py --config.load_dir=<POLYCAM_PATH>

POLYCAM_PATH=$HOME/ULI_DATA/ryokan_bridge

ns-train splatfacto \
    --experiment-name '' \
    --timestamp '' \
    --output-dir $POLYCAM_PATH \
    nerfstudio-data \
    --data $POLYCAM_PATH \
    --train-split-fraction=1.0 \
    --depth-unit-scale-factor=1.0 \
    --auto-scale-poses=False \
    --center-method='none'

ns-export gaussian-splat --load-config $POLYCAM_PATH/splatfacto/config.yml --output-dir $POLYCAM_PATH/splatfacto
