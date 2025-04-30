PARENT_DIR=$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training

for ARKIT_PATH in $PARENT_DIR/*; do
    if [ -d "$ARKIT_PATH" ]; then
        echo "Processing directory: $ARKIT_PATH"

        if [ ! -d "$ARKIT_PATH/splatfacto" ]; then
          echo "Training Gaussian Splatting model..."
          ns-train splatfacto \
            --experiment-name '' \
            --timestamp '' \
            --output-dir $ARKIT_PATH \
            --viewer.quit-on-train-completion True \
            --max-num-iterations 30000 \
            arkit-data \
            --data $ARKIT_PATH \
            --auto-scale-poses False \
            --train-split-fraction 1.0 \
            --center-method='none'
        fi

        if [ -d "$ARKIT_PATH/splatfacto" ]; then
            if [ ! -f "$ARKIT_PATH/splatfacto/splat.ply" ]; then
                echo "Exporting Gaussian Splatting model for $ARKIT_PATH..."
                ns-export gaussian-splat --load-config "$ARKIT_PATH/splatfacto/config.yml" --output-dir "$ARKIT_PATH/splatfacto"
            else
                echo "Splat file already exists for $ARKIT_PATH, skipping export."
            fi
            # # Export the Gaussian Splatting model
            # ns-export gaussian-splat --load-config $ARKIT_PATH/splatfacto/config.yml --output-dir $ARKIT_PATH/splatfacto

            echo "Found splatfacto directory in $ARKIT_PATH, proceeding with mesh generation..."
            rm -r "$ARKIT_PATH/meshes" 2>/dev/null || true

            # Generate mesh slices using the python script
            python "$HOME/legged_gym/mesh_generation/generate_mesh_slices.py" \
              --config="$HOME/legged_gym/mesh_generation/configs/arkit.py" \
              --config.visualize=False \
              --config.load_dir="$ARKIT_PATH"
        else
            echo "Skipping mesh generation for $ARKIT_PATH: splatfacto directory not found."
        fi
    fi
done
