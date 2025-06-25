from urllib.request import urlopen
from PIL import Image
import timm
import torch
import numpy as np
import time
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torchaug

img = Image.open(urlopen(
    # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMXGTLRjhi5Z8LdEFgaFNyR82Pn8wKmdC4Xw&s'
    # 'https://content.homenetiol.com/2000157/2065512/0x0/19ce342addbd4a038bbcadb7dbb00977.jpg'
    'https://pictures.dealer.com/o/ourismanfairfaxtoyotascion/1234/98e607372ddc4e13abe24a8a01dab278.jpg?impolicy=downsize_bkpt&w=396'
))
img = img.resize((img.width // 4, img.height // 4))

print(img.size)

batch_size = 4

img = (torch.from_numpy(np.array(img)).permute(2, 0, 1).to(torch.float32) / 255.0)[None]
img = img.repeat(batch_size, 1, 1, 1)

transform = torchaug.transforms.SequentialTransform(
  [
    torchaug.transforms.RandomPhotometricDistort(
        brightness=(0.6, 1.4), 
        contrast=(0.6, 1.4),
        saturation=(0.6, 1.4),
        hue=(-0.05, 0.05),
        p_transform=0.5,
        p=0.5,
        batch_transform=True,
    ),
    torchaug.transforms.RandomAutocontrast(
      p=0.5,
      batch_transform=True,
    ),
    torchaug.transforms.RandomGaussianBlur(
      kernel_size=(3, 3),
      sigma=(0.2, 1.0),
      p=0.5,
      batch_transform=True,
    ),
  ],
  inplace=False,
  batch_inplace=False,
  batch_transform=True,
  permute_chunks=False,
)
# transform = v2.ColorJitter(
#     brightness=(0.6, 1.4), 
#     contrast=(0.6, 1.4),
#     saturation=(0.6, 1.4),
#     hue=(0, 0),
# )
print(transform)

for _ in range(10):
  img_blur = transform(img)

  # Convert to channels last for plotting
  img_batch_plot = img.permute(0, 2, 3, 1).cpu().numpy()
  img_blur_batch_plot = img_blur.permute(0, 2, 3, 1).cpu().numpy()

  # Stack images vertically for each column
  img_stacked = np.vstack([img_batch_plot[i] for i in range(batch_size)])
  img_blur_stacked = np.vstack([img_blur_batch_plot[i] for i in range(batch_size)])

  fig, axs = plt.subplots(1, 2, figsize=(12, 6))

  axs[0].imshow(img_stacked)
  axs[0].set_title('Original Images (Batch)')
  axs[0].axis('off')

  axs[1].imshow(img_blur_stacked)
  axs[1].set_title('Blurred Images (Batch)')
  axs[1].axis('off')

  plt.tight_layout()
  plt.show()
