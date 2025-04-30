POLYCAM_PATH=$HOME/ULI_DATA/house

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
