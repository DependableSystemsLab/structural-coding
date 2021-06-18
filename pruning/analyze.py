from collections import defaultdict
from copy import copy

import torch

from analysis import sdc, merge
from storage import load
from pruning.parameters import BASELINE_CONFIG, DOMAIN, DEFAULTS
from matplotlib import pyplot as plt


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
    for model in ('vgg16', ):
        for protection in ('none', 'clipper'):
            for inject, faults in ((True, 10), (False, 0)):
                x = []
                y = []
                for factor in (0., 0.05, 0.15, 0.2):
                    config = copy(BASELINE_CONFIG)
                    config['model'] = model
                    config['pruning_factor'] = factor
                    config['pruning_method'] = 'structured'
                    config['protection'] = protection
                    config['inject'] = inject
                    config['faults'] = faults
                    pruned_data = load(config, DEFAULTS)
                    if pruned_data is None:
                        continue
                    config['protection'] = 'none'
                    config['faults'] = 0
                    config['inject'] = False
                    baseline = load(config, DEFAULTS)
                    top1, label = merge(pruned_data)
                    # y.append(float(torch.sum(top1 == label)) / len(label))
                    y.append(sdc(baseline, pruned_data))
                    x.append(factor)
                if x:
                    plt.errorbar(x, [t[0] for t in y], [t[1] for t in y], label='{} {}'.format(protection, faults))
                    # plt.plot(x, y, label='{} {}'.format(protection, faults))
    plt.xlabel('portion of removed weights')
    plt.ylabel('accuracy')
    plt.title('VGG SDC')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # draw_prunability()
    # draw_compression_correlation()
    draw_compress_resilience()

