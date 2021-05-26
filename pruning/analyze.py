import math
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
    unit_baseline_top1 = baseline_top1
    unit_label = label
    while len(baseline_top1) < len(target_top1):
        baseline_top1 = torch.cat((baseline_top1, unit_baseline_top1), dim=0)
        label = torch.cat((label, unit_label), dim=0)
    n = len(target)
    correct = label == target_top1
    base_correct = label == baseline_top1
    corrupted = torch.logical_and(torch.logical_not(correct), base_correct)
    sdc = torch.sum(corrupted) / torch.sum(base_correct)
    z = 1.96  # 95% confidence interval
    error = z * math.sqrt(sdc * (1 - sdc) / n)
    return float(sdc), error


def draw_compression_correlation():
    for model in DOMAIN['model']:
        config = copy(BASELINE_CONFIG)
        config['model'] = model
        baseline_data = load(config, DEFAULTS)
        config['pruning_factor'] = 0.4
        pruned_data = load(config, DEFAULTS)
        config['inject'] = True
        config['pruning_factor'] = 0.0
        faulty_data = load(config, DEFAULTS)
        config['protection'] = 'clipper'
        clipper_data = load(config, DEFAULTS)
        pruned_sdc, pruned_error = sdc(baseline_data, pruned_data)
        clipped_sdc, clipped_error = sdc(baseline_data, clipper_data)
        faulty_sdc, faulty_error = sdc(baseline_data, faulty_data)
        print(model, clipped_sdc / faulty_sdc, pruned_sdc)


if __name__ == '__main__':
    draw_compression_correlation()
