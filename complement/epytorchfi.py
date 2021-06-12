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


class RangerAttackReLU(RangerReLU):
    gradients = []
    activations = []
    very_index = None

    def __init__(self, inplace: bool = False, bounds=None):
        super().__init__(inplace, bounds)
        self.module_index = None

    def record_grad(self, grad):
        self.gradients.append((self.module_index, grad))

    def forward(self, input: Tensor) -> Tensor:
        if self.very_index and self.module_index == self.very_index[0]:
            input[self.very_index[1:]] = bounds[self.very_index[0]][1] * 10
        result = super().forward(input)
        result.register_hook(self.record_grad)
        self.activations.append((self.module_index, result))
        return result


model, max_injection_index = convert(model, mapping={
    torch.nn.ReLU: RangerAttackReLU
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
    if isinstance(m, RangerAttackReLU):
        m.bounds = bounds[relu_counter]
        m.module_index = relu_counter
        relu_counter += 1
print(relu_counter)

model.eval()
data_loader = get_data_loader()

x, y = next(iter(data_loader))
shape = x.shape
pfi_model = pfi_core(model, shape[2], shape[3], shape[0])

# ClipperAttackRelu.very_index = (3, 0, 237, 62, 34)

model_output = model(x)
l = loss(model_output, y + 100)
l.backward()
k = 5
print(torch.topk(model_output, k=k))

activations_map = dict(RangerAttackReLU.activations)
gradients_map = dict(RangerAttackReLU.gradients)

exploitability = {}

for index, gradient in RangerAttackReLU.gradients:
    lower, upper = bounds[index]
    activations = activations_map[index]
    z = torch.zeros(activations.shape, device=activations.device)
    down_opportunity = torch.maximum(activations - lower, z)
    up_opportunity = torch.maximum(upper - activations, z)
    exploitability[index] = torch.maximum(
        down_opportunity * gradient * (gradient >= 0),
        up_opportunity * gradient * (gradient < 0)
    )

layer = []
C = []
H = []
W = []
err_val = []
b = []

# maximum_grads = [max([(torch.max(e[1]), gradients_map[e[0]], e[0]) for e in exploitability.items()])]
maximum_grads = [max([(torch.max(e[1]), gradients_map[e[0]], e[0]) for e in exploitability.items()])]

for maximum_grad in maximum_grads:
    tensor_index = np.unravel_index(torch.argmax(torch.abs(maximum_grad[1])), maximum_grad[1].shape)
    print(tensor_index)
    _layer = maximum_grad[2]
    _b, _C, _H, _W = tensor_index
    if maximum_grad[1][tensor_index] <= 0:
        _err_val = bounds[_layer][1]
    else:
        _err_val = bounds[_layer][0]
    layer.append(_layer)
    C.append(_C)
    H.append(_H)
    W.append(_W)
    err_val.append(_err_val)
    b.append(_b)
print(layer)
inj = pfi_model.declare_neuron_fi(batch=b, conv_num=layer, c=C, h=H, w=W, value=err_val)
print(torch.topk(inj(x), k=k))
