from copy import copy

import torch

from storage import load
from pruning.parameters import BASELINE_CONFIG, DOMAIN, DEFAULTS
from matplotlib import pyplot as plt


def sdc(baseline, target):
    baseline_top1 = None
    label = None
    target_top1 = None
    for e in baseline:
        top1 = e['top5'].T[0]
        if baseline_top1 is None:
            baseline_top1 = top1
        else:
            baseline_top1 = torch.cat((baseline_top1, top1), dim=0)
        if label is None:
            label = e['label']
        else:
            label = torch.cat((label, e['label']), dim=0)
    for e in target:
        top1 = e['top5'].T[0]
        if target_top1 is None:
            target_top1 = top1
        else:
            target_top1 = torch.cat((target_top1, top1), dim=0)
    print(len(baseline_top1), len(label), len(target_top1))


def draw_compression_correlation():
    for model in DOMAIN['model']:
        config = copy(BASELINE_CONFIG)
        config['model'] = model
        baseline_data = load(config, DEFAULTS)
        config['pruning_factor'] = 0.1
        pruned_data = load(config, DEFAULTS)
        config['inject'] = True
        faulty_data = load(config, DEFAULTS)
        config['protection'] = 'clipper'
        clipper_data = load(config, DEFAULTS)
        sdc(baseline_data, pruned_data)


if __name__ == '__main__':
    draw_compression_correlation()
