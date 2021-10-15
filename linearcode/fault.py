import numpy.random
import torch
from torch.nn import Parameter

from injection import bitflip


def inject_memory_fault(model, config):
    rnd = numpy.random.RandomState(config['injection'])
    if config['flips'] == 0:
        return
    if config['flips'] // 1 == config['flips']:  # if is integer
        assert False
    ber = config['flips']
    parameters = get_flattened_weights(model)
    modules, parameters = zip(*parameters)
    bit_width = 32
    if config['quantization']:
        bit_width = 8
    size = sum(map(lambda p: p.shape[0] * bit_width, parameters))
    count = rnd.binomial(size, ber)
    print('Injecting', count, 'faults.')
    bit_indices_to_flip = set()
    while len(bit_indices_to_flip) < count:
        bit_indices_to_flip.add(rnd.randint(0, size - 1))
    pointer = iter(parameters)
    module_pointer = iter(modules)
    parameter = next(pointer)
    module = next(module_pointer)
    offset = 0
    with torch.no_grad():
        for bit_index in sorted(bit_indices_to_flip):
            parameter_index = bit_index // bit_width - offset
            while parameter_index >= len(parameter) and parameter is not None:
                offset += len(parameter)
                parameter_index -= len(parameter)
                parameter = next(pointer, None)
                module = next(module_pointer, None)
            if config['quantization']:
                scales = parameter.q_per_channel_scales()
                zero_points = parameter.q_per_channel_scales()
                axis = parameter.q_per_channel_axis()
                int_repr = parameter.int_repr()
                int_repr[parameter_index] = bitflip(int(int_repr[parameter_index]), bit_index % bit_width)
                corrupted = torch._make_per_channel_quantized_tensor(int_repr, scales, zero_points, axis)
                module.weight = corrupted
            else:
                parameter[parameter_index] = bitflip(float(parameter[parameter_index]), bit_index % bit_width)


def get_flattened_weights(model):
    parameters = []
    for m in model.modules():
        if hasattr(m, 'weight') and type(m) in (
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.quantized.Linear,
            torch.nn.quantized.Conv2d,
        ):
            weight = m.weight
            if callable(weight):
                weight = weight()
            parameters.append((m, weight.flatten()))
    return parameters
