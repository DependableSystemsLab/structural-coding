import torch
from torch.nn.utils import prune

from pruning.datasets import get_data_loader
from pruning.injection import InjectionMixin, convert, ObserverRelu, InjectionConv2D, InjectionLinear, ClipperRelu
from pruning.models import get_model
from pruning.parameters import CONFIG, DEFAULTS, BASELINE_CONFIG
from storage import extend, load


def evaluate():
    model = get_model()
    parameters_to_prune = ()
    for name, param in model.named_parameters():
        if name.endswith('weight'):
            module_path = name.split('.')[:-1]
            module = model
            for edge in module_path:
                module = getattr(module, edge)
            parameters_to_prune += ((module, 'weight'),)
    if CONFIG['pruning_factor']:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=CONFIG['pruning_factor'],
        )
    elif CONFIG['inject']:
        if CONFIG['protection'] == 'none':
            model = convert(model)
        else:
            model = convert(model, mapping={
                torch.nn.Conv2d: InjectionConv2D,
                torch.nn.Linear: InjectionLinear,
                torch.nn.ReLU: ClipperRelu
            })
            bounds = load(BASELINE_CONFIG, DEFAULTS, 'baseline')[-1]['bounds']
            for j, m in enumerate(model.modules()):
                if isinstance(m, ClipperRelu):
                    m.bounds = bounds[j]
    else:
        model = convert(model, {torch.nn.ReLU: ObserverRelu})
    model.eval()
    dataset = get_data_loader()
    evaluation = []
    for i, (image, label) in enumerate(dataset):
        InjectionMixin.counter = 0
        model_out = model(image)
        top5 = torch.topk(model_out, 5).indices
        bounds = {}
        for j, m in enumerate(model.modules()):
            if isinstance(m, ObserverRelu):
                bounds[j] = (m.min, m.max)
        evaluation.append({'top5': top5,
                           'label': label,
                           'batch': i,
                           'amount': InjectionMixin.counter,
                           'bounds': bounds})
        print('Did batch {}'.format(i))
    return evaluation


if __name__ == '__main__':

    if CONFIG['inject']:
        while True:
            extend(CONFIG, evaluate(), DEFAULTS)
    else:
        extend(CONFIG, evaluate(), DEFAULTS)
