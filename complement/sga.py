import csv
import subprocess
from copy import copy
from typing import Optional, Callable

import numpy as np
import torch
import torchvision
from pytorchfi.core import fault_injection as pfi_core
from torch import Tensor, nn as nn
from torchvision.models.resnet import _resnet, Bottleneck

from datasets import get_data_loader
from pruning.injection import convert, RangerReLU, ClipperReLU
from pruning.parameters import DEFAULTS, BASELINE_CONFIG
from storage import load


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


model = _resnet('resnet50', MyBottleneck, [3, 4, 6, 3], True, True)
loss = torch.nn.CrossEntropyLoss()


class ClipperAttackRelu(ClipperReLU):
    gradients = []
    activations = []

    def __init__(self, inplace: bool = False, bounds=None):
        super().__init__(inplace, bounds)
        self.module_index = None

    def record_grad(self, grad):
        self.gradients.append((self.module_index, grad))

    def forward(self, input: Tensor) -> Tensor:
        result = super().forward(input)
        result.register_hook(self.record_grad)
        self.activations.append((self.module_index, result))
        return result


model, max_injection_index = convert(model, mapping={
    torch.nn.ReLU: ClipperAttackRelu
}, in_place=True)
model_baseline_config = copy(BASELINE_CONFIG)
model_baseline_config['model'] = 'resnet50'

bounds = []
with open('Resnet50_bounds_ImageNet_train20p_act.txt') as bounds_file:
    r = csv.reader(bounds_file)
    for row in r:
        bounds.append(tuple(map(float, row)))

relu_counter = 0
for j, m in enumerate(model.modules()):
    if isinstance(m, ClipperAttackRelu):
        m.bounds = bounds[relu_counter]
        m.module_index = relu_counter
        relu_counter += 1
print(relu_counter)

model.eval()
data_loader = get_data_loader()

model.zero_grad()

for i, (x, y) in enumerate(data_loader):
    model_output = model(x)
    l = loss(model_output, y)
    l.backward()
    print(i)


k = 5
print(torch.topk(model_output, k=k))
parameters = list(model.parameters())
grads = []
for i in range(len(parameters)):
    topk = torch.topk(parameters[i].grad.flatten(), k=64)
    for j, g in zip(topk.indices, topk.values):
        grads.append((g, i, j))

grads.sort(reverse=True)
skip = 63
for i in range(skip, skip + 1, 1):
    g, layer, index = grads[i]
    tensor_index = np.unravel_index(index, parameters[layer].shape)
    print(layer, tensor_index)
    with torch.no_grad():
        parameters[layer][tensor_index] *= 2 ** 16
    print(torch.topk(model(x), k=k))
