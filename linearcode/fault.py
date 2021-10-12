import numpy.random
import torch

from injection import bitflip


def inject_memory_fault(model, config, quantized=False):
    assert not quantized
    rnd = numpy.random.RandomState(config['injection'])
    if config['flips'] == 0:
        return
    if config['flips'] // 1 == config['flips']:  # if is integer
        assert False
    ber = config['flips']
    parameters = get_flattened_weights(model)
    size = sum(map(lambda p: p.shape[0] * 32, parameters))
    count = rnd.binomial(size, ber)
    print('Injecting', count, 'faults.')
    bit_indices_to_flip = set()
    while len(bit_indices_to_flip) < count:
        bit_indices_to_flip.add(rnd.randint(0, size - 1))
    pointer = iter(parameters)
    parameter = next(pointer)
    offset = 0
    with torch.no_grad():
        for bit_index in sorted(bit_indices_to_flip):
            parameter_index = bit_index // 32 - offset
            while parameter_index >= len(parameter) and parameter is not None:
                offset += len(parameter)
                parameter_index -= len(parameter)
                parameter = next(pointer, None)
            parameter[parameter_index] = bitflip(float(parameter[parameter_index]), bit_index % 32)


def get_flattened_weights(model):
    parameters = []
    for m in model.modules():
        if hasattr(m, 'weight'):
            parameters.append(m.weight.flatten())
    return parameters
