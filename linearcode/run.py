import csv
import os
import pickle
import random
import time

import numpy as np
import torch

from linearcode.models import get_model
from linearcode.parameters import CONFIG, DEFAULTS
from settings import BATCH_SIZE
from datasets import get_fashion_mnist, get_image_net
from injection import convert, bitflip, ClipperReLU, top_percent, RangerReLU, StructuralCodedConv2d, \
    ReorderingCodedConv2d, StructuralCodedLinear
from linearcode.parameters import DOMAIN
from storage import store

model = get_model()


loss = torch.nn.CrossEntropyLoss()

if CONFIG['protection'] in ('ranger', 'clipper'):

    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: ClipperReLU if CONFIG['protection'] == 'clipper' else RangerReLU
    }, in_place=True)
    bounds = []
    if CONFIG['model'] == 'resnet50':
        bounds_filename = 'bounds/resnet50.txt'
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
        if isinstance(m, torch.nn.ReLU):
            m.bounds = bounds[relu_counter]
            m.module_index = relu_counter
            relu_counter += 1
    print(relu_counter)
elif CONFIG['protection'] == 'sc':
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: StructuralCodedConv2d,
        torch.nn.Linear: StructuralCodedLinear,
    }, in_place=True, extra_kwargs={
        'k': CONFIG['flips'],
        'threshold': 1,
        'n': 256
    })
elif CONFIG['protection'] == 'roc':
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: ReorderingCodedConv2d
    }, in_place=True)
else:
    assert CONFIG['protection'] == 'none'

model.eval()
if CONFIG['model'] in ('resnet50', 'alexnet'):
    if CONFIG['sampler'] == 'none':
        data_loader = get_image_net()
    elif CONFIG['sampler'] == 'critical':
        data_loader = get_image_net(sampler=(4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                             143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                             419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                             536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                             784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                             948, 966, 998))
    elif CONFIG['sampler'] == 'tiny':
        data_loader = get_image_net(sampler=(4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                             143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                             419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                             536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                             784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                             948, 966, 998)[:BATCH_SIZE])
    else:
        assert False
    if CONFIG['model'] == 'resnet50':
        one_time_stuff = 'nonrecurring_resnet50.pkl'
    elif CONFIG['model'] == 'alexnet':
        one_time_stuff = 'nonrecurring_alexnet.pkl'

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
                         'batch_size': BATCH_SIZE,})
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

    rands = {}
    for flips in DOMAIN['flips']:
        rands[flips] = []
        for _ in DOMAIN['injection']:
            layer = next(iter(random.choices(range(len(sizes)), sizes, k=1)))
            abs_indices = set()
            while len(abs_indices) < flips:
                abs_indices.add(random.randrange(0, sizes[layer] * 32))
            rands[flips].append([layer, list(abs_indices)])

    grads.sort(reverse=True)
    protected_20_rands.sort(reverse=True)
    with open(one_time_stuff, mode='wb') as grad_file:
        pickle.dump((grads, baseline, rands, protected_20_rands, [p.grad for p in parameters]), grad_file)

layer, absolute_indices = rands[CONFIG['flips']][CONFIG['injection']]
target_modules = []

print('flipping', layer, absolute_indices)
for index in absolute_indices:
    parameter_index = index // 32
    bit_index = index % 32
    tensor_index = np.unravel_index(parameter_index, parameters[layer].shape)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if m.weight is parameters[layer] or m.bias is parameters[layer]:
                    m.injected = True
                    m.observe = True
                else:
                    m.injected = False
                    m.observe = False
                for p_index, p in enumerate(parameters):
                    if p is m.weight:
                        m.layer = p_index
                target_modules.append(m)
        corrupted = bitflip(parameters[layer][tensor_index], bit_index)
        print(parameters[layer][tensor_index], '->', corrupted, bit_index)
        parameters[layer][tensor_index] = corrupted
        print(layer, tensor_index, parameters[layer][tensor_index], parameters[layer].shape)

target_modules = list(sorted(set(target_modules), key=lambda m: m.layer))

evaluation = []

with torch.no_grad():
    for i, (x, y) in enumerate(data_loader):
        for t in target_modules:
            t.detected = False
        start_time = time.time()
        model_output = model(x)
        indices = torch.topk(model_output, k=k).indices
        end_time = time.time()
        evaluation.append({'top5': indices,
                           'label': y,
                           'batch': i,
                           'elapsed_time': end_time - start_time,
                           'detection': [(t.detected, t.injected) for t in target_modules],
                           'batch_size': BATCH_SIZE})
        print("Done with batch {} after injection".format(i))

    store(CONFIG, evaluation, DEFAULTS)
