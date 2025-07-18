DATA_PATH=$HOME/Downloads/grandslam_gsplat/2024-11-04-10-57-34_nerfstudio

ns-train splatfacto \
    --experiment-name 'nodownscale_ssim0p2_maxgsnum2500000_undistorted_hdronly_scene0p7to0p9' \
    --timestamp '' \
    --output-dir $DATA_PATH \
    nerfstudio-data \
    --data $DATA_PATH \
    --train-split-fraction=1.0 \
    --depth-unit-scale-factor=1.0
# --viewer.quit-on-train-completion True \

# ns-export gaussian-splat --load-config $DATA_PATH/splatfacto/config.yml --output-dir $DATA_PATH/splatfacto