# ARKIT_PATH=$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43895956
# ARKIT_PATH=$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43649417
ARKIT_PATH=$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43895895

ns-train splatfacto --experiment-name '' --timestamp '' --output-dir $ARKIT_PATH arkit-data --data $ARKIT_PATH  --auto-scale-poses=False --train-split-fraction=1.0 --center-method='none'

ns-export gaussian-splat --load-config $ARKIT_PATH/splatfacto/config.yml --output-dir $ARKIT_PATH/splatfacto
