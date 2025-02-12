from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def update_image(new_image, fig, im):
    # To visualize environment RGB.
    if len(new_image.shape) == 4:
        rows = cols = int(np.floor(np.sqrt(new_image.shape[0])))
        new_image = new_image[:rows**2]
        def image_grid(imgs, rows, cols):
            assert len(imgs) == rows*cols
            imgs = (255 * imgs).astype(np.uint8)
            img = Image.fromarray(imgs[0])

            w, h = img.size
            grid = Image.new('RGB', size=(cols*w, rows*h))
            grid_w, grid_h = grid.size
            
            for i, img in enumerate(imgs):
                img = Image.fromarray(img)
                grid.paste(img, box=(i%cols*w, i//cols*h))
            return grid
        to_plot = image_grid(new_image, rows, cols)
    else:
        to_plot = new_image

    im.set_data(np.array(to_plot))
    fig.canvas.flush_events()
    fig.canvas.draw()
    plt.pause(0.001)
