import bisect
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from analysis import sdc, merge
from complement.models import get_model
from complement.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc():
    baselines = {}
    _, baselines['FashionMNISTTutorial'] = load_pickle('nonrecurring_FashionMNISTTutorial')[:2]
    _, baselines['FashionMNISTTutorial_smooth'] = load_pickle('nonrecurring_FashionMNISTTutorial_smooth')[:2]
    _, baselines['resnet50'], rands = load_pickle('nonrecurring_resnet50')[:3]
    sdcs = defaultdict(list)

    model = get_model()
    parameters = list(model.parameters())
    sizes = [0]
    for p in parameters:
        s = 1
        for d in p.shape:
            s *= d
        sizes.append(s)
    for i in range(1, len(sizes), 1):
        sizes[i] += sizes[i - 1]

    key = lambda c: (c['protection'], )
    missing = False
    percent = len(SLURM_ARRAY) // 100
    for i, config in enumerate(SLURM_ARRAY):
        layer = bisect.bisect_left(sizes, rands[config['rank']]) - 1
        if layer == 159:
            continue
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
    model = get_model()
    parameters = list(model.parameters())
    sizes = [0]
    for p in all_grads:
        s = 1
        for d in p.shape:
            s *= d
        sizes.append(s)
    for i in range(1, len(sizes), 1):
        sizes[i] += sizes[i - 1]
    ranks = defaultdict(list)
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
        param = float(parameters[layer].flatten()[flatten_index])
        critical = 0
        if sum(s) / len(s) >= 0.01 and layer != 159:
        # if sum(s) / len(s) >= 0.01:
            critical = 1
        ranks['grad'].append((r, grad, critical))
        ranks['|grad|'].append((r, abs(grad), critical))
        ranks['-1/|grad|'].append((r, -1 / max(1e-20, abs(grad)), critical))
        ranks['value'].append((r, param, critical))
        ranks['value/|grad|'].append((r, 1 / max(1e-20, abs(grad)) * param, critical))
        ranks['|grad|/|value|'].append((r, 1 / max(1e-20, abs(param)) * abs(grad), critical))
        ranks['-layer'].append((r, -layer, critical))
        ranks['|value|'].append((r, abs(param), critical))
    y_rand = []
    y = 0
    for r, _, critical in ranks['grad']:
        y += critical
        y_rand.append(y)
    plt.plot(range(len(y_rand)), y_rand, label='random : found')
    for key in ranks:
        y = 0
        y_grad = []
        sorted_params = sorted(ranks[key], key=lambda record: record[1], reverse=True)
        for r, _, critical in sorted_params:
            y += critical
            y_grad.append(y)
        plt.plot(range(len(y_grad)), y_grad, label='{} : found'.format(key))

    # plt.plot(range(len(y_rand)), [y / (i + 1) for i, y in enumerate(y_rand)], label='random : precision')
    # plt.plot(range(len(y_rand)), [y / max(y_rand) for i, y in enumerate(y_rand)], label='random : recall')
    # plt.plot(range(len(y_grad)), [y / (i + 1) for i, y in enumerate(y_grad)], label='value / |gradient| : precision')
    # plt.plot(range(len(y_grad)), [y / max(y_grad) for i, y in enumerate(y_grad)], label='value / |gradient| : recall')
    # plt.title('Precision / Recall')
    plt.ylabel('Critical Parameters Found')
    plt.xlabel('Parameter Trials')
    plt.legend()
    plt.show()


def draw_layers():
    _, baseline, rands, _, all_grads = load_pickle('nonrecurring_resnet50')
    model = get_model()
    parameters = list(model.parameters())
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
            sdcs[config['protection'], r].append(s)

        if i % percent == 0:
            print(i // percent, '%')
    if missing:
        return
    layers = defaultdict(lambda: defaultdict(int))
    for (p, r), s in sdcs.items():
        layer = bisect.bisect_left(sizes, rands[r]) - 1
        if sum(s) / len(s) >= 0.01:
            layers[p][layer] += 1
    for p in ('none', 'clipper', 'ranger', 'radar'):
        plt.plot(*zip(*sorted(layers[p].items())), label=p)
    plt.legend()
    plt.show()


def draw_precision(criticality_threshold=0.01, excluded_params=(159, 160), title='', constraints=(), path=''):
    _, baseline, rands, _, all_grads = load_pickle('nonrecurring_resnet50')
    model = get_model()
    parameters = list(model.parameters())
    sizes = [0]
    for p in all_grads:
        s = 1
        for d in p.shape:
            s *= d
        sizes.append(s)
    for i in range(1, len(sizes), 1):
        sizes[i] += sizes[i - 1]
    ranks = defaultdict(list)
    missing = False
    sub_array = [c for c in SLURM_ARRAY if all(constraint(c) for constraint in constraints)]
    percent = len(sub_array) // 100
    sdcs = defaultdict(list)
    for i, config in enumerate(sub_array):
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
        param = float(parameters[layer].flatten()[flatten_index])
        critical = 0
        # if sum(s) / len(s) >= 0.01 and layer != 159:
        if sum(s) / len(s) >= criticality_threshold and layer not in excluded_params:
            critical = 1
        # ranks['grad'].append((r, grad, critical))
        ranks['|grad|'].append((r, abs(grad), critical))
        ranks['rand'].append((r, r, critical))
        # ranks['1/|grad|'].append((r, 1 / max(1e-20, abs(grad)), critical))
        # ranks['value'].append((r, param, critical))
        ranks['value/|grad|'].append((r, 1 / max(1e-20, abs(grad)) * param, critical))
        # ranks['-|grad|/|value|'].append((r, -1 / max(1e-20, abs(param)) * abs(grad), critical))
        # ranks['layer'].append((r, layer, critical))
        # ranks['|value|'].append((r, abs(param), critical))
    cut_off_percentage = 0.1
    cut_off = int(len(next(iter(ranks.values()))) * cut_off_percentage)
    plt.figure(figsize=(12, 8))
    for key in ranks:
        founds = []
        tp, fp, tn, fn = [], [], [], []
        sorted_params = sorted(ranks[key], key=lambda record: record[1], reverse=True)
        found = 0
        for i, (r, _, critical) in enumerate(sorted_params):
            trial = i + 1
            heuristic_selection_start = i + 1
            found += critical
            left_budget = cut_off - found
            heuristic_selection_end = i + 1 + left_budget
            heuristic_selection = sorted_params[heuristic_selection_start: heuristic_selection_end]
            heuristic_exclusion = sorted_params[heuristic_selection_end:]
            founds.append(found)
            tp.append(found + sum(c for _, _, c in heuristic_selection))
            fp.append(sum(1 - c for _, _, c in heuristic_selection))
            tn.append(trial - found + sum(1 - c for _, _, c in heuristic_exclusion))
            fn.append(sum(c for _, _, c in heuristic_exclusion))
            if found >= cut_off:
                break
        # plt.plot(range(1, len(founds) + 1), founds, label='{} found'.format(key))
        # plt.plot(range(1, len(tp) + 1), tp, label='{} tp'.format(key))
        # plt.plot(range(1, len(tn) + 1), tn, label='{} tn'.format(key))
        # plt.plot(range(1, len(fp) + 1), fp, label='{} fp'.format(key))
        # plt.plot(range(1, len(fn) + 1), fn, label='{} fn'.format(key))
        plt.plot(range(1, len(fn) + 1), [_tp / (_tp + _fp) for _tp, _fp, _tn, _fn in zip(tp, fp, tn, fn)], label='{} precision'.format(key))
        plt.plot(range(1, len(fn) + 1), [_tp / (_tp + _fn) for _tp, _fp, _tn, _fn in zip(tp, fp, tn, fn)], label='{} recall'.format(key))

    plt.ylabel('Measures')
    plt.xlabel('Parameter Trials')
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()


draw_precision(constraints=(
    lambda c: c['protection'] == 'none',
), title='All Layers w/o Protection (SDC > 5% Considered Critical)', excluded_params=(), path='../ubcthesis/images/heuristic_protection:none_layers:all.png', criticality_threshold=0.05)
draw_precision(constraints=(
    lambda c: c['protection'] == 'clipper',
), title='All Layers with Clipper (SDC > 5% Considered Critical)', excluded_params=(), path='../ubcthesis/images/heuristic_protection:clipper_layers:all.png', criticality_threshold=0.05)
draw_precision(constraints=(
    lambda c: c['protection'] == 'none',
), title='Excluding Last Layer w/o Protection (SDC > 1% Considered Critical)', path='../ubcthesis/images/heuristic_protection:none_layers:wolast.png')
draw_precision(constraints=(
    lambda c: c['protection'] == 'clipper',
), title='Excluding Last Layer with Clipper (SDC > 1% Considered Critical)', path='../ubcthesis/images/heuristic_protection:clipper_layers:wolast.png')
