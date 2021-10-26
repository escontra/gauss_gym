import torch

from gauss_gym.rl.modules.decoder import ConvDecoder3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = torch.randn(2, 2, 512, device=device)

decoder = ConvDecoder3D(
  in_channels=512, out_resolution=(12, 12, 12), out_channels=512, mults=(2, 3)
)
decoder = decoder.to(device)
output_tensor = decoder(input_tensor)

print('Final output shape:')
print(output_tensor.shape)
