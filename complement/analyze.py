from collections import defaultdict

import torch

from analysis import sdc, merge
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    _, baseline, _ = load_pickle('nonrecurring')
    print(sdc(baseline, baseline))
    sdcs = defaultdict(list)
    for config in SLURM_ARRAY:
        faulty = load(config, defaults=DEFAULTS)
        sdcs[config['rank']].append(sdc(baseline, faulty))
    for i in range(64):
        print(sum(j for j, _ in sdcs[i]) / len(sdcs[i]))


def draw_image_dependence():

    _, baseline, _ = load_pickle('nonrecurring')
    baseline_top1, labels = merge(baseline)
    per_image_sdc = torch.zeros(baseline_top1.shape, device=baseline_top1.device)
    for config in SLURM_ARRAY:
        faulty = load(config, defaults=DEFAULTS)
        faulty_top1, _ = merge(faulty)
        per_image_sdc = per_image_sdc + 1. * torch.logical_and(baseline_top1 == labels, faulty_top1 != labels)
    for i in range(100):
        print(per_image_sdc)


draw_image_dependence()
