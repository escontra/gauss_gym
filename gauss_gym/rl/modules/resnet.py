import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(
      in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(
      out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != self.expansion * out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(
          in_channels,
          self.expansion * out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
        ),
        nn.BatchNorm2d(self.expansion * out_channels),
      )

  def forward(self, x):
    residual = self.shortcut(x)
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += residual
    out = F.relu(out)
    return out


class NatureCNN(nn.Module):
  def __init__(self, in_channels, num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(
      in_channels, 32, kernel_size=8, stride=4, padding=1, bias=False
    )
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.linear1 = nn.Linear(64, 128, bias=True)
    self.linear2 = nn.Linear(128, num_classes, bias=True)

  def forward(self, x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.conv3(x)
    x = nn.functional.relu(x)
    x = x.mean(dim=(-1, -2))
    x = self.linear1(x)
    x = nn.functional.relu(x)
    x = self.linear2(x)
    return x


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, init_channels=64, num_classes=10):
    super().__init__()
    self.in_channels = init_channels
    self.conv1 = nn.Conv2d(
      3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
    )
    self.bn1 = nn.BatchNorm2d(self.in_channels)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, init_channels, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, init_channels * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, init_channels * 4, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, init_channels * 8, num_blocks[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(init_channels * 8 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride))
      self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out


def ResNet18():
  return ResNet(BasicBlock, [2, 2, 2, 2])


if __name__ == '__main__':
  model = ResNet18()
  input_tensor = torch.randn(1, 3, 224, 224)
  output = model(input_tensor)
