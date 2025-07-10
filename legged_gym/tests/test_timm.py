from urllib.request import urlopen
from PIL import Image
import timm
import torch
import numpy as np
import time
from torchvision import transforms
import matplotlib.pyplot as plt

img = Image.open(urlopen(
    # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMXGTLRjhi5Z8LdEFgaFNyR82Pn8wKmdC4Xw&s'
    # 'https://content.homenetiol.com/2000157/2065512/0x0/19ce342addbd4a038bbcadb7dbb00977.jpg'
    'https://pictures.dealer.com/o/ourismanfairfaxtoyotascion/1234/98e607372ddc4e13abe24a8a01dab278.jpg?impolicy=downsize_bkpt&w=396'
))
# img_array = torch.tensor(np.array(img)).to('cuda')

# model = timm.create_model('test_convnext.r160_in1k', pretrained=True)
model = timm.create_model('resnet10t', pretrained=True)
model = model.eval()
print(model.default_cfg)
model = model.to('cuda')
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# model = timm.create_model('test_efficientnet', pretrained=True)
# model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
print(transforms)

# img_array = img_array.permute(2, 0, 1).to(torch.float32) / 255.0
# img_array = img_array.unsqueeze(0).repeat(1, 1, 1, 1)
# img_array = transforms(img_array)
# print(img_array.shape)

out = model(transforms(img).unsqueeze(0).to('cuda'))
print(out.shape)

top5_probabilities, top5_class_indices = torch.topk(out.softmax(dim=1) * 100, k=5)


# print(timm.data.IMAGENET_DEFAULT_LABELS)

import requests

url = "https://github.com/pytorch/hub/raw/master/imagenet_classes.txt"
class_names = requests.get(url).text.strip().split('\n')

for label in top5_class_indices[0]:
  print(class_names[label])

# img_array = img_array.unsqueeze(0).repeat(8, 1, 1, 1)

# img_array_orig = img_array.clone()

# img_array = img_array.unsqueeze(1)
# img_array_jitter = transforms.ColorJitter(
#     brightness=0.4,
#     contrast=0.4,
#     saturation=0.4,
#     hue=0.
# )(img_array)[:, 0]
# print(img_array_jitter.shape)

# def plot_transforms(img_array_orig, img_array_jitter):
#   fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#   # Reshape to channels last and stack along height
#   img_orig = img_array_orig.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
#   img_jitter = img_array_jitter.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
  
#   # Stack images along height axis
#   img_orig_stacked = np.vstack([img_orig[i] for i in range(img_orig.shape[0])])
#   img_jitter_stacked = np.vstack([img_jitter[i] for i in range(img_jitter.shape[0])])
  
#   # Plot
#   axs[0].imshow(img_orig_stacked)
#   axs[0].set_title('Original')
#   axs[0].axis('off')
  
#   axs[1].imshow(img_jitter_stacked)
#   axs[1].set_title('Jittered')
#   axs[1].axis('off')
  
#   plt.tight_layout()
#   plt.show()

# plot_transforms(img_array_orig, img_array_jitter)




# print(img_array.shape)
# torch.cuda.synchronize()
# start = time.time()
# num_repeats = 100
# for _ in range(num_repeats):
#   img_transformed = transforms(img_array)
# torch.cuda.synchronize()
# total_time = time.time() - start
# print("Transform time:", total_time / num_repeats) 
# print(img_transformed.shape)


# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)