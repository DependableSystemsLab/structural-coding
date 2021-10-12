import torch

from injection import convert, ClipperReLU, ClipperHardswish


def apply_sc(model, config):
    assert False


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


PROTECTIONS = {
    'none': lambda x, config: x,
    'sc': apply_sc,
    'clipper': apply_clipper,
}
