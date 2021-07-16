import bisect
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from analysis import sdc, merge
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    baselines = {}
    _, baselines['FashionMNISTTutorial'] = load_pickle('nonrecurring_FashionMNISTTutorial')[:2]
    _, baselines['FashionMNISTTutorial_smooth'] = load_pickle('nonrecurring_FashionMNISTTutorial_smooth')[:2]
    _, baselines['resnet50'] = load_pickle('nonrecurring_resnet50')[:2]
    sdcs = defaultdict(list)
    key = lambda c: (c['model'], c['protection'], c['ranking'])
    missing = False
    percent = len(SLURM_ARRAY) // 100
    for i, config in enumerate(SLURM_ARRAY):
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            sdcs[key(config)].append(sdc(baselines[config['model']], faulty))
        if i % percent == 0:
            print(i // percent, '%')
    if missing:
        return
    for i in sdcs:
        print(i, sum(j for j, _ in sdcs[i]) / len(sdcs[i]))


def draw_image_dependence():

    _, baseline, _ = load_pickle('nonrecurring')
    baseline_top1, labels = merge(baseline)
    per_image_sdc = torch.zeros(baseline_top1.shape, device=baseline_top1.device)
    missing = False
    percent = len(SLURM_ARRAY) // 100
    for i, config in enumerate(SLURM_ARRAY):
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            faulty = load(config, defaults=DEFAULTS)
            faulty_top1, _ = merge(faulty)
            per_image_sdc = per_image_sdc + 1. * torch.logical_and(baseline_top1 == labels, faulty_top1 != labels)
        if i % percent == 0:
            print(i // percent, '%')
    if missing:
        return
    for i in range(100):
        print(per_image_sdc)


def draw_heuristics():
    _, baseline, rands, _, all_grads = load_pickle('nonrecurring_resnet50')
    sizes = [0]
    for p in all_grads:
        s = 1
        for d in p.shape:
            s *= d
        sizes.append(s)
    for i in range(1, len(sizes), 1):
        sizes[i] += sizes[i - 1]
    ranks = []
    missing = False
    percent = len(SLURM_ARRAY) // 100
    sdcs = defaultdict(list)
    for i, config in enumerate(SLURM_ARRAY):
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            faulty = load(config, defaults=DEFAULTS)
            s, _ = sdc(baseline, faulty)
            r = config['rank']
            sdcs[r].append(s)

        if i % percent == 0:
            print(i // percent, '%')
    if missing:
        return
    for r, s in sdcs.items():
        layer = bisect.bisect_left(sizes, rands[r]) - 1
        flatten_index = rands[r] - sizes[layer]
        grad = float(all_grads[layer].flatten()[flatten_index])
        critical = 0
        if sum(s) / len(s) >= 0.009:
            critical = 1
        ranks.append((r, grad, critical))
    y_rand = []
    y_grad = []
    y = 0
    for r, _, critical in ranks:
        y += critical
        y_rand.append(y)
    y = 0
    for r, _, critical in sorted(ranks, key=lambda record: record[1], reverse=True):
        y += critical
        y_grad.append(y)
    plt.plot(range(len(y_rand)), y_rand, label='rand')
    plt.plot(range(len(y_grad)), y_grad, label='grad')
    plt.legend()
    plt.show()


draw_heuristics()
