import torch

from injection import convert, ClipperReLU, ClipperHardswish, StructuralCodedConv2d, StructuralCodedLinear, \
    NormalizedConv2d, NormalizedLinear, TMRLinear, TMRConv2d, RADARConv2d, RADARLinear, QStructuralCodedConv2d, \
    QStructuralCodedLinear, FRADARConv2d, FRADARLinear, MILRLinear, MILRConv2d, ClipperELU, RangerReLU, RangerHardswish, \
    RangerELU, NormalizedConv2dGroups


def apply_sc(model, config):
    if not isinstance(config['flips'], str) and config['flips'] // 1 == config['flips']:
        k = config['flips'] or 32
    else:
        k = 32
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: StructuralCodedConv2d,
        torch.nn.qat.Conv2d: QStructuralCodedConv2d,
        torch.nn.Linear: StructuralCodedLinear,
        torch.nn.qat.Linear: QStructuralCodedLinear,
    }, extra_kwargs={
        'k': k,
        'threshold': config.get('threshold', 0.00),
        'n': config.get('n', 256),
    })
    return model


def normalize_model(model, _):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: NormalizedConv2d,
        torch.nn.Linear: NormalizedLinear,
    })
    return model


def normalize_groups(model, _):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: NormalizedConv2dGroups,
    })
    return model


def apply_tmr(model, config):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: TMRConv2d,
        torch.nn.Linear: TMRLinear,
    })
    return model


def apply_radar(model, config):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: FRADARConv2d,
        torch.nn.Linear: FRADARLinear,
    })
    return model


def apply_milr(model, config):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: MILRConv2d,
        torch.nn.Linear: MILRLinear,
    })
    return model


def apply_clipper(model, config):
    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: ClipperReLU,
        torch.nn.Hardswish: ClipperHardswish,
        torch.nn.ELU: ClipperELU,
    }, in_place=True)
    bounds = []
    bounds_filename = 'bounds/{}.txt'.format(config['model'])
    with open(bounds_filename) as bounds_file:
        for row in bounds_file:
            bounds.append(eval(row.strip('\n')))

    relu_counter = 0
    for j, m in enumerate(model.modules()):
        if hasattr(m, 'bounds'):
            m.bounds = bounds[relu_counter]
            m.module_index = relu_counter
            relu_counter += 1
    return model


def apply_ranger(model, config):
    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: RangerReLU,
        torch.nn.Hardswish: RangerHardswish,
        torch.nn.ELU: RangerELU,
    }, in_place=True)
    bounds = []
    bounds_filename = 'bounds/{}.txt'.format(config['model'])
    with open(bounds_filename) as bounds_file:
        for row in bounds_file:
            bounds.append(eval(row.strip('\n')))

    relu_counter = 0
    for j, m in enumerate(model.modules()):
        if hasattr(m, 'bounds'):
            m.bounds = bounds[relu_counter]
            m.module_index = relu_counter
            relu_counter += 1
    return model


PROTECTIONS = {
    'before_quantization': {
        'radar': lambda model, config: model,
        'sc': normalize_model,
        # 'sc': lambda model, config: model,
        'clipper': apply_clipper,
        'ranger': apply_ranger,
        'tmr': apply_tmr,
        'milr': normalize_groups,
    },
    'after_quantization': {
        'radar': apply_radar,
        'sc': apply_sc,
        'clipper': lambda model, config: model,
        'ranger': lambda model, config: model,
        'tmr': lambda model, config: model,
        'milr': apply_milr,
    }
}


def apply_sc_automatically(model, n, k):
    model = normalize_model(model, None)
    model = apply_sc(model, {'flips': k, 'n': n})
    return model
