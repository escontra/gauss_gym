ULI_PATH=$HOME/ULI_DATA/apartment_to_grace/processed_data

ns-train splatfacto --experiment-name '' --timestamp '' --output-dir $ULI_PATH nerfstudio-data --data $ULI_PATH  --auto-scale-poses=False --train-split-fraction=1.0 --center-method='none' --depth-unit-scale-factor=1.0

ns-export gaussian-splat --load-config $ULI_PATH/splatfacto/config.yml --output-dir $ULI_PATH/splatfacto
