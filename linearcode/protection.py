import torch

from injection import convert, ClipperReLU, ClipperHardswish, StructuralCodedConv2d, StructuralCodedLinear, \
    NormalizedConv2d, NormalizedLinear


def apply_sc(model, config):
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: NormalizedConv2d,
        torch.nn.Linear: NormalizedLinear,
    })
    model, _ = convert(model, mapping={
        torch.nn.Conv2d: StructuralCodedConv2d,
        torch.nn.Linear: StructuralCodedLinear,
    }, in_place=True, extra_kwargs={
        'k': 32,
        'threshold': 1,
        'n': 256
    })
    return model


def apply_clipper(model, config):
    model, max_injection_index = convert(model, mapping={
        torch.nn.ReLU: ClipperReLU,
        torch.nn.Hardswish: ClipperHardswish,
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
    'none': lambda x, config: x,
    'sc': apply_sc,
    'clipper': apply_clipper,
}
