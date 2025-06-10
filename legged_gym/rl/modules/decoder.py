import numpy as np
import torch
import torch.nn as nn


class ConvDecoder3D(nn.Module):

  def __init__(
      self,
      in_channels: int,
      out_resolution: tuple,
      out_channels: int,
      depth: int = 64,
      mults: tuple = (2, 3, 4, 4),
      kernel: int = 4,
      act_final: bool = False,
    ):
    super().__init__()
    self.in_channels = in_channels
    self.out_resolution = out_resolution
    self.out_channels = out_channels
    self.depth = depth
    self.mults = mults
    self.kernel = kernel
    self.depths = tuple(self.depth * mult for mult in self.mults)

    factor = 2 ** len(self.depths)
    self.minres = [int(x // factor) for x in out_resolution]
    for i in range(len(self.minres)):
      assert 3 <= self.minres[i] <= 16, self.minres
    self.init_shape = (self.depths[-1], *self.minres)
    self.space_proj = nn.Sequential(
      nn.Linear(self.in_channels, np.prod(self.init_shape)),
      nn.GELU())

    conv_transpose_layers = []
    example_input = torch.randn(1, self.depths[-1], *self.minres)
    for i, depth in reversed(list(enumerate(self.depths[:-1]))):
      conv_transpose_layers.append(nn.ConvTranspose3d(
        in_channels=example_input.shape[1],
        out_channels=depth,
        kernel_size=self.kernel,
        stride=2,
        padding=0))
      example_input = conv_transpose_layers[-1](example_input)
      conv_transpose_layers.append(nn.GELU())
    self.conv_transpose_layers = nn.Sequential(*conv_transpose_layers)


    # Calculate padding for final conv transpose layer.
    padding = []
    curr_spatial_dims = example_input.shape[2:]
    for spatial_idx in range(len(curr_spatial_dims)):
      input_size = curr_spatial_dims[spatial_idx]
      output_size = self.out_resolution[spatial_idx]
      stride = 2
      kernel_size = self.kernel
      output_padding = 0
      padding.append(int(np.ceil(((input_size - 1) * stride + (kernel_size - 1) + output_padding - output_size) / 2)))

    final_conv_layers = [
      nn.ConvTranspose3d(
        in_channels=self.depths[0], out_channels=self.out_channels, kernel_size=self.kernel, stride=2,
        padding=padding
      )
    ]
    if act_final:
      final_conv_layers.append(nn.GELU())
    self.final_conv = nn.Sequential(*final_conv_layers)

  def __call__(self, x):
    bshape = x.shape[:-1]
    x = x.reshape((np.prod(bshape), -1))
    x = self.space_proj(x)
    x = x.reshape((x.shape[0], *self.init_shape))
    x = self.conv_transpose_layers(x)
    x = self.final_conv(x)
    x = x.reshape((*bshape, self.out_channels, *self.out_resolution))
    return x
