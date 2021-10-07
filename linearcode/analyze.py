import bisect
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from analysis import sdc, merge, detection, elapsed_time, sc_detection_hit_rate
from linearcode.models import get_model
from linearcode.parameters import SLURM_ARRAY, DEFAULTS
from storage import load, load_pickle


def draw_sdc(partial=True):
    return draw_metric(partial, sdc)

def draw_metric(partial=True, metric_function=sdc):
    baselines = {}
    _, baselines['resnet50'], rands = load_pickle('nonrecurring_resnet50')[:3]
    # _, baselines['alexnet'], rands = load_pickle('nonrecurring_alexnet')[:3]
    sdcs = defaultdict(list)

    # model = get_model()
    # parameters = list(model.parameters())
    # sizes = [0]
    # for p in parameters:
    #     s = 1
    #     for d in p.shape:
    #         s *= d
    #     sizes.append(s)
    # for i in range(1, len(sizes), 1):
    #     sizes[i] += sizes[i - 1]

    key = lambda c: (c['protection'], c['flips'])
    missing = False
    percent = len(SLURM_ARRAY) // 100
    for i, config in enumerate(SLURM_ARRAY):
        layer, _ = rands[config['flips']][config['injection']]
        # if layer == 159:
        #     continue
        # ('none',)
        # 0.15676944520741992
        # ('clipper',)
        # 0.005379470284075879
        faulty = load(config, defaults=DEFAULTS)
        if faulty is None:
            print("{} in parameters array is missing.".format(i))
            missing = True
        else:
            sdcs[key(config)].extend(faulty)
        if i % percent == 0:
            print(i // percent, '%')
            # if partial:
            #     print(',none,clipper,sc')
                # for flips in (1, 2, 4, 8, 16, 32):
                #     print(flips, end=',')
                #     for i in sdcs:
                #         if i[1] != flips:
                #             continue
                #         print(sdc(baselines['resnet50'], sdcs[i])[0], end=',')
                #     print()
    if not partial and missing:
        return
    print(',none,clipper,sc')
    for flips in (1, 2, 4, 8, 16, 32):
        print(flips, end=',')
        for i in sdcs:
            if i[1] != flips:
                continue
            print(metric_function(baselines['resnet50'], sdcs[i])[0], end=',')
        print()


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


def draw_precision():
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
        # if sum(s) / len(s) >= 0.01 and layer != 159:
        if sum(s) / len(s) >= 0.01:
            critical = 1
        # ranks['grad'].append((r, grad, critical))
        # ranks['-|grad|'].append((r, -abs(grad), critical))
        # ranks['1/|grad|'].append((r, 1 / max(1e-20, abs(grad)), critical))
        # ranks['value'].append((r, param, critical))
        ranks['value/|grad|'].append((r, 1 / max(1e-20, abs(grad)) * param, critical))
        # ranks['-|grad|/|value|'].append((r, -1 / max(1e-20, abs(param)) * abs(grad), critical))
        ranks['layer'].append((r, layer, critical))
        # ranks['|value|'].append((r, abs(param), critical))
    y_rand = []
    y = 0
    for r, _, critical in ranks['grad']:
        y += critical
        y_rand.append(y)
    cut_off = 400
    y_rand = y_rand[:cut_off]
    plt.plot(range(len(y_rand)), [y / (cut_off - (i + 1 - y)) for i, y in enumerate(y_rand)],
             label='random : precision')
    plt.plot(range(len(y_rand)), [y / max(y_rand) for i, y in enumerate(y_rand)], label='random : recall')
    for key in ranks:
        y = 0
        y_grad = []
        sorted_params = sorted(ranks[key], key=lambda record: record[1], reverse=True)
        for r, _, critical in sorted_params:
            y += critical
            y_grad.append(y)
        y_grad = y_grad[:cut_off]
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


# draw_sdc(partial=True)
# draw_metric(partial=True, metric_function=elapsed_time)
draw_metric(partial=True, metric_function=sc_detection_hit_rate)


# data = [(
#     ('clipper', 2), (0.00363141018897295, 0.000833659585670892), (0.0002596153935883194, 0.0002232800445650557)),
#     (('clipper', 4), (0.009038461372256279, 0.0013116462552972459), (0.0004839743487536907, 0.00030482257613978004)),
#     (('clipper', 8), (0.011048076674342155, 0.001448678468120462), (0.0010801282478496432, 0.0004552438208953905)),
#     (('clipper', 16), (0.019903846085071564, 0.0019357261879632873), (0.002294871723279357, 0.000663164675666337)),
#     (('clipper', 32), (0.03829166665673256, 0.00265959128481501), (0.00557692302390933, 0.0010321053020137427),)]
#
# sdcs = []
# sdcerrs = []
# dues = []
# dueerrs = []
#
# for (_, flips), (sdc, sdcerr), (due, dueerr) in data:
#     sdcs.append(sdc)
#     sdcerrs.append(sdcerr)
#     dues.append(due)
#     dueerrs.append(dueerr)
#
# flips = (2, 4, 8, 16, 32)
# plt.errorbar(flips, sdcs, yerr=sdcerrs, label='sdc')
# plt.errorbar(flips, dues, yerr=dueerrs, label='due')
# plt.legend()
# plt.title('DUE vs SDC')
# plt.xlabel('# of bit flips')
# plt.show()
