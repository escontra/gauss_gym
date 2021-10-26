import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def update_occupancy_grid(env, fig, plots, env_id, occupancy_grids, titles, show=True):
  first_time = fig is None
  if first_time:
    assert plots is None, 'plots should be None if fig is None'
    if show:
      plt.ion()
    fig = plt.figure(figsize=(3, 3 * len(occupancy_grids)), tight_layout=True)
    plots = [None] * len(occupancy_grids)

  new_plots = []
  for i, (occupancy_grid, title, plot) in enumerate(
    zip(occupancy_grids, titles, plots)
  ):
    if isinstance(occupancy_grid, np.ndarray):
      occupancy_grid = torch.from_numpy(occupancy_grid)

    if env is None:
      x = torch.linspace(-1, 1, occupancy_grid.shape[-3])
      y = torch.linspace(-1, 1, occupancy_grid.shape[-2])
      grid_x, grid_y = torch.meshgrid(x, y)
      heights = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1)
      heights = heights[None].repeat(occupancy_grid.shape[0], 1, 1, 1)
    else:
      heights = env.sensors['raycast_grid'].ray_starts.clone()

    first_nonzero = torch.argmax(
      occupancy_grid.to(torch.int32), dim=-1
    )  # Find first non-zero in each row
    heights[..., -1] = first_nonzero
    heights = heights[env_id].reshape(-1, 3)

    heights_np = heights.cpu().numpy()
    x = heights_np[:, 0]
    y = heights_np[:, 1]
    z = heights_np[:, 2]

    if plot is None:
      ax = fig.add_subplot(len(occupancy_grids), 1, i + 1, projection='3d')
      new_plots.append(
        ax.scatter(
          x, y, z, c=z, cmap='viridis', s=50, vmin=0, vmax=occupancy_grid.shape[-1]
        )
      )
      ax.set_xlim([x.min(), x.max()])
      ax.set_ylim([y.min(), y.max()])
      ax.set_zlim([0, occupancy_grid.shape[-1]])
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_title(title)
      ax.set_box_aspect([1, 1, 1])
    else:
      plot._offsets3d = (x, y, z)
      plot.set_array(z)
      new_plots.append(plot)

  if first_time and show:
    plt.show(block=False)

  fig.tight_layout()
  fig.canvas.flush_events()
  fig.canvas.draw()
  if show:
    plt.pause(0.001)
  return fig, new_plots


def update_image(env, fig, im, env_id, image, show=True):
  new_image = image[env_id] if env_id >= 0 else image
  if isinstance(image, torch.Tensor):
    new_image = new_image.cpu().numpy()

  # Convert from channels first to channels last format
  if len(new_image.shape) == 4:
    new_image = new_image.transpose(0, 2, 3, 1)
    n, h, w, c = new_image.shape
  elif len(new_image.shape) == 3:
    new_image = new_image.transpose(1, 2, 0)
    n = 1
    h, w, c = new_image.shape
  else:
    raise ValueError(f'Invalid image shape: {new_image.shape}')

  first_time = fig is None
  if first_time:
    if show:
      plt.ion()
    fig_height = 2.5
    fig_width = (w / h) * fig_height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)
    im = ax.imshow(np.zeros((h * n, w * n, 3), dtype=np.uint8))

  # To visualize environment RGB.
  if len(new_image.shape) == 4:
    rows = cols = int(np.floor(np.sqrt(new_image.shape[0])))
    new_image = new_image[: rows**2]

    def image_grid(imgs, rows, cols):
      assert len(imgs) == rows * cols
      img = Image.fromarray(imgs[0])

      w, h = img.size
      grid = Image.new('RGB', size=(cols * w, rows * h))
      grid_w, grid_h = grid.size

      for i, img in enumerate(imgs):
        img = Image.fromarray(img)
        grid.paste(img, box=(i % cols * w, i // cols * h))
      return grid

    to_plot = image_grid(new_image, rows, cols)
  else:
    to_plot = new_image

  im.set_data(np.array(to_plot))

  if first_time and show:
    plt.show(block=False)

  fig.tight_layout()
  fig.canvas.flush_events()
  fig.canvas.draw()
  if show:
    plt.pause(0.001)

  return fig, im
