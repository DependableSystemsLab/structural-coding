import csv
import os
import pickle
from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor, nn as nn
from torchvision.models.resnet import _resnet, Bottleneck

from complement.parameters import CONFIG, DEFAULTS
from complement.settings import BATCH_SIZE
from datasets import get_data_loader
from injection import convert, bitflip, ClipperReLU
from storage import store


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

if CONFIG['protection'] == 'clipper':

    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: ClipperReLU
    }, in_place=True)
    bounds = []
    with open('Resnet50_bounds_ImageNet_train20p_act.txt') as bounds_file:
        r = csv.reader(bounds_file)
        for row in r:
            bounds.append(tuple(map(float, row)))

    relu_counter = 0
    for j, m in enumerate(model.modules()):
        if isinstance(m, ClipperReLU):
            m.bounds = bounds[relu_counter]
            m.module_index = relu_counter
            relu_counter += 1
    print(relu_counter)

model.eval()
data_loader = get_data_loader()
parameters = list(model.parameters())

k = 5

one_time_stuff = 'nonrecurring.pkl'
if os.path.exists(one_time_stuff):
    with open(one_time_stuff, mode='rb') as grad_file:
        grads, baseline, rands = pickle.load(grad_file)
else:
    model.zero_grad()
    grads = []
    rands = []
    baseline = []

    for i, (x, y) in enumerate(data_loader):
        model_output = model(x)
        l = loss(model_output, y)
        l.backward()
        baseline.append({'top5': torch.topk(model_output, k=k).indices,
                         'label': y,
                         'batch': i,
                         'batch_size': BATCH_SIZE,
                         'amount': 0,
                         'injections': []})
        print("Done with batch {}".format(i))

    for i in range(len(parameters)):
        grad_flatten = parameters[i].grad.flatten()
        rand_flatten = torch.rand(grad_flatten.shape, device=grad_flatten.device)
        topk = torch.topk(grad_flatten, k=64)
        for j, g in zip(topk.indices, topk.values):
            grads.append((g, i, j))
        topk = torch.topk(rand_flatten, k=64)
        for j, g in zip(topk.indices, topk.values):
            rands.append((g, i, j))
    grads.sort(reverse=True)
    rands.sort(reverse=True)
    with open(one_time_stuff, mode='wb') as grad_file:
        pickle.dump((grads, baseline, rands), grad_file)

if CONFIG['ranking'] == 'gradient':
    g, layer, index = grads[CONFIG['rank']]
else:
    g, layer, index = rands[CONFIG['rank']]
tensor_index = np.unravel_index(index, parameters[layer].shape)
print(layer, tensor_index)
with torch.no_grad():
    parameters[layer][tensor_index] = bitflip(parameters[layer][tensor_index], CONFIG['bit_position'])

evaluation = []

for i, (x, y) in enumerate(data_loader):
    model_output = model(x)
    evaluation.append({'top5': torch.topk(model_output, k=k).indices,
                       'label': y,
                       'batch': i,
                       'batch_size': BATCH_SIZE,
                       'amount': 1,
                       'injections': [grads[CONFIG['rank']]]})
    print("Done with batch {} after injection".format(i))

store(CONFIG, evaluation, DEFAULTS)
