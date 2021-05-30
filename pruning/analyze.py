import math
from collections import defaultdict
from copy import copy

import torch

from storage import load
from pruning.parameters import BASELINE_CONFIG, DOMAIN, DEFAULTS
from matplotlib import pyplot as plt


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


def draw_compression_correlation():
    x = []
    y = []
    yerr = []
    labels = []
    prunability = get_prunability()
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
        x.append(prunability[model])
        y.append(clipped_sdc)
        yerr.append(clipped_error)
        labels.append(model)
    plt.errorbar(x, y, yerr=yerr, fmt='o')
    for _x, _y, label in zip(x, y, labels):
        plt.annotate(label, (_x, _y))
    plt.xlabel('prunability')
    plt.ylabel('clipper SDC')
    plt.show()


def draw_prunability():
    for model in DOMAIN['model']:
        x = []
        y = []
        for factor in DOMAIN['pruning_factor']:
            config = copy(BASELINE_CONFIG)
            config['model'] = model
            config['pruning_factor'] = factor
            pruned_data = load(config, DEFAULTS)
            top1, label = merge(pruned_data)
            y.append(float(torch.sum(top1 == label)) / len(label))
            x.append(factor)
        plt.plot(x, y, label=model)
    plt.xlabel('portion of removed weights')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def get_prunability():
    prunability = defaultdict(float)
    base_accuracy = defaultdict(float)
    for model in DOMAIN['model']:
        for factor in DOMAIN['pruning_factor']:
            config = copy(BASELINE_CONFIG)
            config['model'] = model
            config['pruning_factor'] = factor
            pruned_data = load(config, DEFAULTS)
            top1, label = merge(pruned_data)
            accuracy = float(torch.sum(top1 == label)) / len(label)
            if model not in base_accuracy:
                base_accuracy[model] = accuracy
            if accuracy >= 0.95 * base_accuracy[model]:
                prunability[model] = max(prunability[model], factor)
    return prunability


def draw_compress_resilience():
    for model in ('vgg16', 'resnet50'):
        for protection in ('none', 'clipper'):
            x = []
            y = []
            for factor in DOMAIN['pruning_factor']:
                config = copy(BASELINE_CONFIG)
                config['model'] = model
                config['pruning_factor'] = factor
                config['protection'] = protection
                config['inject'] = True
                config['faults'] = 10
                pruned_data = load(config, DEFAULTS)
                top1, label = merge(pruned_data)
                y.append(float(torch.sum(top1 == label)) / len(label))
                x.append(factor)
            plt.plot(x, y, label=model)
    plt.xlabel('portion of removed weights')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # draw_prunability()
    # draw_compression_correlation()
    draw_compress_resilience()

