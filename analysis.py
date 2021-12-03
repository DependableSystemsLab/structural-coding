import math
import operator
from functools import reduce

import torch


def sdc(baseline, target, over_approximate=False):
    baseline_top1, label = merge(baseline)
    # baseline_top1 = baseline_top1[[4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
    #                                          143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
    #                                          419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
    #                                          536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
    #                                          784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
    #                                          948, 966, 998]]
    # label = label[[4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
    #                                          143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
    #                                          419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
    #                                          536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
    #                                          784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
    #                                          948, 966, 998]]
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
    sdc = (torch.sum(corrupted) + (1 if over_approximate else 0)) / torch.sum(base_correct)
    z = 1.96  # 95% confidence interval
    error = z * math.sqrt(sdc * (1 - sdc) / n)
    return float(sdc), error


def elapsed_time(baseline, target):
    return sum(e['elapsed_time'] for e in target), 0


def sc_detection_hit_rate(baseline, target):
    baseline_top1, label = merge(baseline)
    baseline_top1 = baseline_top1[[4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                             143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                             419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                             536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                             784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                             948, 966, 998]]
    label = label[[4, 10, 14, 16, 23, 27, 39, 51, 53, 64, 68, 109, 111, 120, 124, 131, 139,
                                             143, 162, 215, 236, 242, 276, 284, 303, 332, 374, 384, 397, 405, 408, 413,
                                             419, 420, 423, 424, 431, 432, 447, 448, 462, 466, 485, 502, 503, 511, 532,
                                             536, 538, 540, 563, 581, 621, 662, 673, 677, 690, 693, 701, 733, 767, 774,
                                             784, 789, 806, 808, 828, 851, 872, 877, 885, 907, 912, 915, 928, 929, 934,
                                             948, 966, 998]]
    sdc_batches = 0
    detected_sdc_batches = 0
    for e in target:
        start_index = e['batch'] * e['batch_size'] % len(baseline_top1)
        baseline_top1_chunk = baseline_top1[start_index: start_index + e['batch_size']]
        if torch.sum(torch.logical_and(baseline_top1_chunk == e['label'],
                                       e['top5'].T[0] != e['label'])) > 0:
            sdc_batches += 1
            if any(all(d) for d in e['detection']):
                detected_sdc_batches += 1
    return (detected_sdc_batches, sdc_batches), 0

def detection(baseline, target, term='sdc'):
    baseline_top1, label = merge(baseline)
    target_top1 = None
    detection = None
    for e in target:
        top1 = e['top5'].T[0]
        detected_channels = None
        for d in e['detection']:
            if d is not None:
                detected_channels = d
                break
        evaluation_detection = torch.zeros(top1.shape, device=top1.device)
        if detected_channels is not None:
            for channel in detected_channels:
                evaluation_detection[channel[0]] += 1
        if target_top1 is None:
            target_top1 = top1
            detection = evaluation_detection
        else:
            target_top1 = torch.cat((target_top1, top1), dim=0)
            detection = torch.cat((detection, evaluation_detection), dim=0)
    unit_baseline_top1 = baseline_top1
    unit_label = label
    while len(baseline_top1) < len(target_top1):
        baseline_top1 = torch.cat((baseline_top1, unit_baseline_top1), dim=0)
        label = torch.cat((label, unit_label), dim=0)

    n = len(target)
    correct = label == target_top1
    base_correct = label == baseline_top1
    corrupted = torch.logical_and(torch.logical_not(correct), base_correct)
    detected_uncorrected = torch.logical_and(detection > 0, corrupted)
    sdc = torch.sum(corrupted) / torch.sum(base_correct)
    due = torch.sum(detected_uncorrected) / torch.sum(base_correct)
    term = eval(term)
    z = 1.96  # 95% confidence interval
    error = z * math.sqrt(term * (1 - term) / n)
    return float(term), error


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