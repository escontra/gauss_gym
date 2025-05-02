# ULI_PATH=$HOME/ULI_DATA/apartment_to_grace/processed_data
ULI_PATH=$HOME/ULI_DATA/cute_bridge
# ULI_PATH=$HOME/ULI_DATA/stairwell

ns-train splatfacto \
    --experiment-name '' \
    --timestamp '' \
    --output-dir $ULI_PATH \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --data $ULI_PATH \
    --train-split-fraction=1.0 \
    --depth-unit-scale-factor=1.0 \
    --auto-scale-poses=False \
    --center-method='none'

ns-export gaussian-splat --load-config $ULI_PATH/splatfacto/config.yml --output-dir $ULI_PATH/splatfacto

python "$HOME/legged_gym/mesh_generation/generate_mesh_slices.py" \
  --config="$HOME/legged_gym/mesh_generation/configs/base.py" \
  --config.visualize=False \
  --config.load_dir="$ULI_PATH"