---
tags:
- image-classification
- timm
- transformers
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
---
# Model card for test_convnext.r160_in1k

A very small test ConvNeXt image classification model for testing and sanity checks. Trained on ImageNet-1k by Ross Wightman.

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 0.3
  - GMACs: 0.0
  - Activations (M): 0.6
  - Image size: 160 x 160
- **Dataset:** ImageNet-1k
- **Papers:**
  - PyTorch Image Models: https://github.com/huggingface/pytorch-image-models
- **Original:** https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('test_convnext.r160_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'test_convnext.r160_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 24, 40, 40])
    #  torch.Size([1, 32, 20, 20])
    #  torch.Size([1, 48, 10, 10])
    #  torch.Size([1, 64, 5, 5])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'test_convnext.r160_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 64, 5, 5) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
### By Top-1

|model                           |img_size|top1  |top5  |param_count|
|--------------------------------|--------|------|------|-----------|
|test_convnext3.r160_in1k        |192     |54.558|79.356|0.47       |
|test_convnext2.r160_in1k        |192     |53.62 |78.636|0.48       |
|test_convnext2.r160_in1k        |160     |53.51 |78.526|0.48       |
|test_convnext3.r160_in1k        |160     |53.328|78.318|0.47       |
|test_convnext.r160_in1k         |192     |48.532|74.944|0.27       |
|test_nfnet.r160_in1k            |192     |48.298|73.446|0.38       |
|test_convnext.r160_in1k         |160     |47.764|74.152|0.27       |
|test_nfnet.r160_in1k            |160     |47.616|72.898|0.38       |
|test_efficientnet.r160_in1k     |192     |47.164|71.706|0.36       |
|test_efficientnet_evos.r160_in1k|192     |46.924|71.53 |0.36       |
|test_byobnet.r160_in1k          |192     |46.688|71.668|0.46       |
|test_efficientnet_evos.r160_in1k|160     |46.498|71.006|0.36       |
|test_efficientnet.r160_in1k     |160     |46.454|71.014|0.36       |
|test_byobnet.r160_in1k          |160     |45.852|70.996|0.46       |
|test_efficientnet_ln.r160_in1k  |192     |44.538|69.974|0.36       |
|test_efficientnet_gn.r160_in1k  |192     |44.448|69.75 |0.36       |
|test_efficientnet_ln.r160_in1k  |160     |43.916|69.404|0.36       |
|test_efficientnet_gn.r160_in1k  |160     |43.88 |69.162|0.36       |
|test_vit2.r160_in1k             |192     |43.454|69.798|0.46       |
|test_resnet.r160_in1k           |192     |42.376|68.744|0.47       |
|test_vit2.r160_in1k             |160     |42.232|68.982|0.46       |
|test_vit.r160_in1k              |192     |41.984|68.64 |0.37       |
|test_resnet.r160_in1k           |160     |41.578|67.956|0.47       |
|test_vit.r160_in1k              |160     |40.946|67.362|0.37       |

## Citation
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
