import bisect
import math
import os
from collections import defaultdict
from copy import copy

import matplotlib.pyplot as plt
import numpy
import torch

from analysis import sdc, merge
from common.models import MODEL_CLASSES
from linearcode.fault import inject_memory_fault, get_target_modules
from linearcode.models import get_model
from linearcode.parameters import SLURM_ARRAY, DEFAULTS, query_configs, DOMAIN
from linearcode.protection import PROTECTIONS
from storage import load, load_pickle, get_storage_filename


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
# draw_metric(partial=True, metric_function=sc_detection_hit_rate)


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


def _sdc_protection_scales_with_ber():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: c['flips'] < 1,
        lambda c: not c['model'] in ('e2e', 'vgg19'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))
    for baseline_config in baseline_configs:
        data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
        baseline = data[0]
        print(baseline_config['model'])
        for flips in DOMAIN['flips']:
            print(flips, end=',')
            for protection in ('sc', 'clipper', 'none', 'tmr'):
                if flips == 0 or flips >= 1:
                    continue
                config = copy(baseline_config)
                config['flips'] = flips
                config['protection'] = protection
                data = load(config, {**DEFAULTS, 'injection': config['injection']})
                if data:
                    concat_data = []
                    for e in data:
                        concat_data.extend(e)
                    print(sdc(baseline, concat_data)[0], end=',')
            print()


def demonstrate_tmr_failure():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: c['flips'] < 1,
        lambda c: not c['model'] in ('e2e', 'vgg19'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
        lambda c: c['model'] == 'resnet50',
    ))
    baseline_config = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
        lambda c: c['model'] == 'resnet50',
    ))[0]
    data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
    baseline = data[0]
    data = load({**baseline_config,
                 'protection': 'tmr',
                 'flips': 0.00000552972}, {**DEFAULTS, 'injection': baseline_config['injection']})
    for datum in data:
        sdc_value = sdc(baseline, datum)[0]
        if sdc_value > 0:
            print(datum[0]['config'], sdc_value)
            model_class = dict(MODEL_CLASSES)['resnet50']
            model = model_class(pretrained=True)

            #  protect model
            model = PROTECTIONS['tmr'](model, datum[0]['config'])
            model.eval()

            # corrupt model
            # original_weights = [m.weight.clone() for m in get_target_modules(model)]
            bit_indices_to_flip, size = inject_memory_fault(model, datum[0]['config'])
            p = datum[0]['config']['flips']
            probability_of_two_in_same_index = (3 * (1 - p) * p ** 2)
            print('Model bit size', size)
            print('Probability of not seeing this', (1 - probability_of_two_in_same_index) ** (400 * size // 3))
            sizes = [m.weight.nelement() * m.weight.element_size() * 8 for m in get_target_modules(model)]
            while bit_indices_to_flip:
                first_param_indices = [i for i in bit_indices_to_flip if i < sizes[0]]
                bit_indices_to_flip = [i - sizes[0] for i in bit_indices_to_flip if i >= sizes[0]]
                injections = len(set(first_param_indices))
                distinct_injections = len(set(i % (sizes[0] // 3) for i in first_param_indices))
                if injections != distinct_injections:
                    print('yo', injections, distinct_injections)
                sizes = sizes[1:]


def demonstrate_quantization_accuracy():
    baseline_constraints = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: c['quantization'],
        lambda c: not c['model'] in ('e2e', 'vgg19'),
        lambda c: c['protection'] == 'none',
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    )
    baseline_configs = query_configs(baseline_constraints)
    for baseline_config in baseline_configs:
        baseline_config['protection'] = 'sc'
        baseline_config['model'] = 'resnet50'
        baseline_config['flips'] = 0
        data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
        unrolled = []
        for d in data:
            unrolled.extend(d)
        unrolled = merge(unrolled)
        print(baseline_config['model'], torch.sum(unrolled[0] == unrolled[1]) / unrolled[0].nelement())


def sdc_protection_scales_with_granularity():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: isinstance(c['flips'], str) or c['flips'] == 0,
        lambda c: not c['model'] in ('e2e', 'vgg19'),
        # lambda c: c['model'] in ("alexnet", 'mobilenet'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))

    for flips in DOMAIN['flips']:
        # if flips not in ("row", "row-4", "rowhammer"):
        #     continue
        if not isinstance(flips, str):
            continue
        # if flips in ('bank', 'chip'):
        #     continue
        # if not isinstance(flips, float):
        #     continue
        print(flips)
        for protection in (
            'sc',
            # 'clipper',
            'none',
            # 'tmr',
            # 'radar',
            # 'milr',
            # 'ranger',
        ):
            if not isinstance(protection, str):
                continue
            print(protection)
            filename = get_storage_filename({'fig': 'sdc_protection_scales_with_granularity',
                                             'flips': flips,
                                             'protection': protection},
                                            extension='.tex', storage='../thesis/data/')
            with open(filename, mode='w') as data_file:
                for baseline_config in baseline_configs:
                    data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
                    baseline = data[0]
                    config = copy(baseline_config)
                    config['flips'] = flips
                    config['protection'] = protection
                    data = load(config, {**DEFAULTS, 'injection': config['injection']})
                    if data:
                        concat_data = []
                        for e in data:
                            concat_data.extend(e)
                        print(baseline_config['model'], *sdc(baseline, concat_data, over_approximate=protection=='sc'), file=data_file)
                        print(baseline_config['model'], *sdc(baseline, concat_data, over_approximate=protection=='sc'))


def regression_recovery():
    base_query = (
        lambda c: c['dataset'] == 'driving_dataset_test',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: c['model'] == 'e2e',
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))

    for baseline_config in baseline_configs:
        data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
        baseline = data[0]
        baseline_losses = [float(e['loss']) for e in baseline]
        baseline_loss = sum(baseline_losses) / len(baseline_losses)
        for protection in (
            'sc',
            'clipper',
            'none',
            'tmr',
            'radar',
            'milr',
            'ranger',
        ):
            filename = get_storage_filename({'fig': 'regression_recovery',
                                             'model': baseline_config['model'],
                                             'protection': protection},
                                            extension='.tex', storage='../thesis/data/')
            with open(filename, mode='w') as data_file:
                for flips in DOMAIN['flips']:
                    if isinstance(flips, int):
                        continue
                    config = copy(baseline_config)
                    config['flips'] = flips
                    config['protection'] = protection
                    print(config)
                    data = load(config, {**DEFAULTS, 'injection': config['injection']})
                    if data:
                        concat_data = []
                        for e in data:
                            concat_data.extend(e)
                        losses = [float(e['loss']) for e in concat_data
                                  if not numpy.isnan(e['loss']) and not numpy.isinf(e['loss'])]
                        protection_loss = sum(losses) / len(losses)
                        if isinstance(flips, float):
                            flips_repr = '$10^{{{}}}$'.format(int(round(math.log(float(flips), 10))))
                        else:
                            flips_repr = flips
                        print(flips_repr, protection_loss / baseline_loss, file=data_file)


def rewhammer_recovery():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: isinstance(c['flips'], str) or c['flips'] == 0,
        lambda c: not c['model'] in ('e2e', 'vgg19'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))

    for protection in (
            'sc',
            'clipper',
            'none',
            'tmr',
            'radar',
            'milr',
            'ranger',
    ):

        filename = get_storage_filename({'fig': 'rowhammer_recovery',
                                         'protection': protection},
                                        extension='.tex', storage='../thesis/data/')
        with open(filename, mode='w') as data_file:
            for baseline_config in baseline_configs:
                data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
                baseline = data[0]
                config = copy(baseline_config)
                config['flips'] = 'rowhammer'
                config['protection'] = protection
                print(config)
                data = load(config, {**DEFAULTS, 'injection': config['injection']})
                if data:
                    concat_data = []
                    for e in data:
                        concat_data.extend(e)
                    print(config['model'], *sdc(baseline, concat_data), file=data_file)


def sdc_protection_scales_with_faults():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: isinstance(c['flips'], int),
        lambda c: not c['model'] in ('e2e', 'vgg19'),
        # lambda c: c['model'] in ("alexnet", 'mobilenet'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))

    for baseline_config in baseline_configs:
        data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
        baseline = data[0]
        for protection in (
            'sc',
            'clipper',
            'none',
            'tmr',
            'radar',
            'milr',
            'ranger',
        ):
            if not isinstance(protection, str):
                continue
            filename = get_storage_filename({'fig': 'sdc_protection_scales_with_faults',
                                             'model': baseline_config['model'],
                                             'protection': protection},
                                            extension='.tex', storage='../thesis/data/')
            with open(filename, mode='w') as data_file:
                for flips in DOMAIN['flips']:
                    if not isinstance(flips, int) or flips == 0:
                        continue
                    config = copy(baseline_config)
                    config['flips'] = flips
                    config['protection'] = protection
                    data = load(config, {**DEFAULTS, 'injection': config['injection']})
                    if data:
                        concat_data = []
                        for e in data:
                            concat_data.extend(e)
                        print(flips, *sdc(baseline, concat_data, over_approximate=protection=='sc'), file=data_file)


def sdc_protection_scales_with_ber():
    base_query = (
        lambda c: c['dataset'] == 'imagenet_ds_128',
        lambda c: c['sampler'] == 'none',
        lambda c: not c['quantization'],
        lambda c: not c['model'] in ('e2e', 'vgg19'),
        # lambda c: c['model'] in ("alexnet", 'mobilenet'),
    )
    baseline_configs = query_configs(base_query + (
        lambda c: all((c['flips'] == 0, c['injection'] == 0, c['protection'] == 'none')),
    ))

    for flips in DOMAIN['flips']:
        if not isinstance(flips, float):
            continue
        for protection in (
            'sc',
            'clipper',
            'none',
            'tmr',
            'radar',
            'milr',
            'ranger',
        ):
            filename = get_storage_filename({'fig': 'sdc_protection_scales_with_ber',
                                             'flips': '10e{}'.format(int(round(math.log(float(flips), 10)))),
                                             'protection': protection},
                                            extension='.tex', storage='../thesis/data/')
            with open(filename, mode='w') as data_file:
                for baseline_config in baseline_configs:
                    data = load(baseline_config, {**DEFAULTS, 'injection': baseline_config['injection']})
                    baseline = data[0]
                    config = copy(baseline_config)
                    config['flips'] = flips
                    config['protection'] = protection
                    data = load(config, {**DEFAULTS, 'injection': config['injection']})
                    if data:
                        concat_data = []
                        for e in data:
                            concat_data.extend(e)
                        print(baseline_config['model'], *sdc(baseline, concat_data, over_approximate=protection=='sc'), file=data_file)


def parameter_pages():

    for model_name, model_class in MODEL_CLASSES:
        model = model_class()
        numbers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                numbers.append(module.weight.nelement() / 1024)
        print(model_name, sum(numbers) / len(numbers), min(numbers), sorted(numbers))


parameter_pages()
