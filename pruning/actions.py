import random
from copy import copy

import torch
from torch.nn.utils import prune

from pruning.datasets import get_data_loader
from pruning.injection import InjectionMixin, convert, ObserverRelu, InjectionConv2D, InjectionLinear, RangerReLU
from pruning.models import get_model
from pruning.parameters import CONFIG, DEFAULTS, BASELINE_CONFIG
from settings import BATCH_SIZE
from storage import extend, load


def evaluate():
    model = get_model()
    if CONFIG['inject']:
        if CONFIG['protection'] == 'none':
            model, max_injection_index = convert(model, mapping={
                torch.nn.Conv2d: InjectionConv2D,
                torch.nn.Linear: InjectionLinear,
            })
        else:
            model, max_injection_index = convert(model, mapping={
                torch.nn.Conv2d: InjectionConv2D,
                torch.nn.Linear: InjectionLinear,
                torch.nn.ReLU: RangerReLU
            })
            model_baseline_config = copy(BASELINE_CONFIG)
            model_baseline_config['model'] = CONFIG['model']
            bounds = load(model_baseline_config, DEFAULTS, 'baseline')[-1]['bounds']
            for j, m in enumerate(model.modules()):
                if isinstance(m, RangerReLU):
                    m.bounds = bounds[j]
    else:
        model, max_injection_index = convert(model, mapping={
            torch.nn.ReLU: ObserverRelu
        })

    if CONFIG['pruning_factor']:

        parameters_to_prune = ()
        for name, param in model.named_parameters():
            if name.endswith('weight'):
                module_path = name.split('.')[:-1]
                module = model
                for edge in module_path:
                    module = getattr(module, edge)
                parameters_to_prune += ((module, 'weight'),)

        if CONFIG['pruning_method'] == 'global_unstructured':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=CONFIG['pruning_factor'],
            )
        elif CONFIG['pruning_method'] == 'structured':
            for module, name in parameters_to_prune:
                prune.ln_structured(module, name, CONFIG['pruning_factor'], 1, dim=0)
        else:
            assert False

    model.eval()
    dataset = get_data_loader()
    evaluation = []
    seed = 2021
    random.seed(seed)
    for i, (image, label) in enumerate(dataset):
        InjectionMixin.counter = 0
        if CONFIG['faults']:
            InjectionMixin.indices.clear()
            for _ in range(CONFIG['faults']):
                InjectionMixin.indices.append(random.randint(0, max_injection_index))
        model_out = model(image)
        top5 = torch.topk(model_out, 5).indices
        bounds = {}
        for j, m in enumerate(model.modules()):
            if isinstance(m, ObserverRelu):
                bounds[j] = (m.min, m.max)
        evaluation.append({'top5': top5,
                           'label': label,
                           'batch': i,
                           'batch_size': BATCH_SIZE,
                           'amount': InjectionMixin.counter,
                           'injections': copy(InjectionMixin.indices),
                           'seed': seed,
                           'bounds': bounds})
        print('Did batch {}'.format(i), flush=True)
    return evaluation


if __name__ == '__main__':

    if CONFIG['inject']:
        while True:
            extend(CONFIG, evaluate(), DEFAULTS)
    else:
        extend(CONFIG, evaluate(), DEFAULTS)
