import math

import torch


def sdc(baseline, target):
    baseline_top1, label = merge(baseline)
    target_top1 = None
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


def merge(baseline):
    baseline_top1 = None
    label = None
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
    return baseline_top1, label