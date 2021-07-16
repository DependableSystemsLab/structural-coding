import csv
import os
import pickle
import random
from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor, nn as nn
from torchvision.models.resnet import _resnet, Bottleneck

from complement.models import FashionMNISTTutorial
from complement.parameters import CONFIG, DEFAULTS
from complement.settings import BATCH_SIZE
from datasets import get_data_loader, get_fashion_mnist, get_image_net
from injection import convert, bitflip, ClipperReLU, top_percent
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


if CONFIG['model'] == 'resnet50':
    model = _resnet('resnet50', MyBottleneck, [3, 4, 6, 3], True, True)
elif CONFIG['model'] == 'FashionMNISTTutorial':
    model = FashionMNISTTutorial(pretrained=True)
elif CONFIG['model'] == 'FashionMNISTTutorial_smooth':
    model = FashionMNISTTutorial(pretrained=True, weights='fashion_mnist_tutorial_smooth.pkl')
else:
    assert False


loss = torch.nn.CrossEntropyLoss()

if CONFIG['protection'] == 'clipper':

    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: ClipperReLU
    }, in_place=True)
    bounds = []
    if CONFIG['model'] == 'resnet50':
        bounds_filename = 'Resnet50_bounds_ImageNet_train20p_act.txt'
    elif CONFIG['model'] == 'FashionMNISTTutorial':
        bounds_filename = 'FashionMNISTTutorial_bounds.txt'
    elif CONFIG['model'] == 'FashionMNISTTutorial_smooth':
        bounds_filename = 'FashionMNISTTutorial_bounds_smooth.txt'
    else:
        assert False
    with open(bounds_filename) as bounds_file:
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
if CONFIG['model'] == 'resnet50':
    if CONFIG['sampler'] == 'none':
        data_loader = get_image_net()
    elif CONFIG['sampler'] == 'critical':
        data_loader = get_image_net(sampler=(4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                             143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                             419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                             536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                             784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                             948, 966, 998))
    else:
        assert False
    one_time_stuff = 'nonrecurring_resnet50.pkl'
elif CONFIG['model'] == 'FashionMNISTTutorial':
    data_loader = get_fashion_mnist()
    one_time_stuff = 'nonrecurring_FashionMNISTTutorial.pkl'
elif CONFIG['model'] == 'FashionMNISTTutorial_smooth':
    data_loader = get_fashion_mnist()
    one_time_stuff = 'nonrecurring_FashionMNISTTutorial_smooth.pkl'
else:
    assert False

parameters = list(model.parameters())

sizes = []
for i in range(len(parameters)):
    s = 1
    for d in parameters[i].shape:
        s *= d
    sizes.append(s)

k = 5

if os.path.exists(one_time_stuff):
    with open(one_time_stuff, mode='rb') as grad_file:
        grads, baseline, rands, protected_20_rands, _ = pickle.load(grad_file)
else:
    model.zero_grad()
    grads = []
    protected_20_rands = []
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
        max_flatten = min(64, len(grad_flatten))
        topk = torch.topk(grad_flatten, k=max_flatten)
        for j, g in zip(topk.indices, topk.values):
            grads.append((g, i, j))
        topk = torch.topk(-(1. * top_percent(grad_flatten, 0.20)) + rand_flatten, k=max_flatten)
        for j, g in zip(topk.indices, topk.values):
            protected_20_rands.append((g, i, j))
    abs_indices = set()
    while len(abs_indices) < 250000:
        abs_indices.add(random.randint(0, sum(sizes)))
    rands = list(abs_indices)
    random.shuffle(rands)
    grads.sort(reverse=True)
    protected_20_rands.sort(reverse=True)
    with open(one_time_stuff, mode='wb') as grad_file:
        pickle.dump((grads, baseline, rands, protected_20_rands, [p.grad for p in parameters]), grad_file)

if CONFIG['ranking'] == 'gradient':
    g, layer, index = grads[CONFIG['rank']]
elif CONFIG['ranking'] == 'random':
    absolute_index = rands[CONFIG['rank']]
    layer = None
    index = None
    g = None
    for i, s in enumerate(sizes):
        if absolute_index < s:
            layer = i
            index = absolute_index
            break
        else:
            absolute_index -= s
elif CONFIG['ranking'] == 'gradient_protected_20':
    g, layer, index = protected_20_rands[CONFIG['rank']]
else:
    assert False

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
