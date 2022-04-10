from typing import Optional, Callable, Any

import torch
import torchvision
from torch import nn as nn, Tensor
from torchvision.models import GoogLeNet
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.resnet import Bottleneck, ResNet, _resnet
import torch.nn.functional as F

import common.dave


class MyBottleneck(Bottleneck):

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.relu3 = torch.nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', MyBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class MyBasicConv2d(BasicConv2d):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__(in_channels, out_channels, **kwargs)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class MyInceptionAux(InceptionAux):

    def __init__(self, in_channels: int, num_classes: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        if conv_block is None:
            conv_block = MyBasicConv2d
        super().__init__(in_channels, num_classes, conv_block)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.relu(self.fc1(x))
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class MyInception(Inception):

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int,
                 pool_proj: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        if conv_block is None:
            conv_block = MyBasicConv2d
        super().__init__(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block)


def googlenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "GoogLeNet":
    kwargs['blocks'] = [MyBasicConv2d, MyInception, MyInceptionAux]
    return torchvision.models.googlenet(pretrained, progress, **kwargs)


MODEL_CLASSES = (
    ('alexnet', torchvision.models.alexnet),
    ('squeezenet', torchvision.models.squeezenet1_1),
    # big memory requirement
    ('vgg19', torchvision.models.vgg19),
    ('mobilenet', torchvision.models.mobilenet_v3_small),
    ('googlenet', googlenet),
    # big memory requirement
    ('resnet50', resnet50),
    ('shufflenet', torchvision.models.shufflenet_v2_x0_5),
    ('e2e', common.dave.Dave2)
)

LOSS_CLASSES = (
    ('alexnet', nn.CrossEntropyLoss),
    ('squeezenet', nn.CrossEntropyLoss),
    # big memory requirement
    ('vgg19', nn.CrossEntropyLoss),
    ('mobilenet', nn.CrossEntropyLoss),
    ('googlenet', nn.CrossEntropyLoss),
    # big memory requirement
    ('resnet50', nn.CrossEntropyLoss),
    ('shufflenet', nn.CrossEntropyLoss),
    ('e2e', nn.MSELoss),
)