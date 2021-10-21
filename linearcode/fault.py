import numpy.random
import torch
import torch.nn.qat

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
                repeat = parameter.shape[0] // module.weight_fake_quant.scale.flatten().shape[0]
                scale = torch.repeat_interleave(module.weight_fake_quant.scale, repeat)[parameter_index]
                zero_point = torch.repeat_interleave(module.weight_fake_quant.zero_point, repeat)[parameter_index]
                quantized = torch.clamp(
                    torch.round(parameter[parameter_index] / scale + zero_point),
                    module.weight_fake_quant.quant_min,
                    module.weight_fake_quant.quant_max)
                parameter[parameter_index] = (bitflip(int(quantized), bit_index % bit_width) - zero_point) * scale
            else:
                parameter[parameter_index] = bitflip(float(parameter[parameter_index]), bit_index % bit_width)
    return bit_indices_to_flip, size


def get_flattened_weights(model):
    parameters = []
    for m in get_target_modules(model):

            weight = m.weight
            if callable(weight):
                weight = weight()
            parameters.append((m, weight.flatten()))
    return parameters


def get_target_modules(model):
    modules = []
    for m in model.modules():
        if hasattr(m, 'weight') and any(map(lambda x: isinstance(m, x), (
                torch.nn.Linear,
                torch.nn.Conv2d,
                torch.nn.qat.Linear,
                torch.nn.qat.Conv2d,
        ))):
            modules.append(m)
    return modules
