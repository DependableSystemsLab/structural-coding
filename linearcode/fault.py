import ctypes

import numpy.random
import torch
import torch.nn.qat

from injection import bitflip
from settings import PROBABILITIES

_4KB = 2 ** 10 * 8
_2B = 2 * 8


def inject_memory_fault(model, config):

    rnd = numpy.random.RandomState(config['injection'])

    parameters = get_flattened_weights(model)
    modules, parameters = zip(*parameters)
    bit_width = 32
    if config['quantization']:
        bit_width = 8
    size = sum(map(lambda p: p.shape[0] * bit_width, parameters))
    bit_indices_to_flip = set()
    granularity = 1
    pages = size // _4KB
    words = size // _2B
    if config['flips'] == 0:
        return bit_indices_to_flip, size
    if isinstance(config['flips'], str):
        if config['flips'] == 'rowhammer':
            victim_rows = rnd.randint(4, 60)
            vulnerable_cells = 200
            ber = vulnerable_cells / victim_rows / 2 / _4KB
            for victim_row in range(2 * victim_rows):
                start = rnd.randint(0, pages - 1) * _4KB
                count = rnd.binomial(_4KB, ber)
                initial_count = len(bit_indices_to_flip)
                while len(bit_indices_to_flip) < count + initial_count:
                    bit_indices_to_flip.add(rnd.randint(start, start + _4KB - 1))
        elif config['flips'] == 'word':
            granularity = _2B
            start = rnd.randint(0, words - 1) * _2B
            bit_indices_to_flip.add(start)
        elif config['flips'] == 'column':
            number_of_corrupted_chunks = 3
            corrupted_chunk_starts = set()
            while len(corrupted_chunk_starts) < number_of_corrupted_chunks:
                corrupted_chunk_starts.add(rnd.randint(0, pages - 1) * _4KB)
            start = rnd.randint(0, _4KB / _2B - 1) * _2B
            for corrupted_chunk_start in corrupted_chunk_starts:
                bit_indices_to_flip.add(start + corrupted_chunk_start)
            granularity = _2B
        elif config['flips'] == 'bank':
            bank_index = rnd.randint(0, 63) * _2B
            for i in range(bank_index, size, 64 * _2B):
                bit_indices_to_flip.add(i)
            granularity = _2B
        elif config['flips'] == 'chip':
            bank_index = rnd.randint(0, 7) * _2B
            for i in range(bank_index, size, 8 * _2B):
                bit_indices_to_flip.add(i)
            granularity = _2B
        elif config['flips'] == 'row':
            victim_rows = 1
            ber = 0.3
            for victim_row in range(2 * victim_rows):
                start = rnd.randint(0, pages - 1) * _4KB
                count = rnd.binomial(_4KB, ber)
                initial_count = len(bit_indices_to_flip)
                while len(bit_indices_to_flip) < count + initial_count:
                    bit_indices_to_flip.add(rnd.randint(start, start + _4KB - 1))
        elif config['flips'] == 'flr':
            filter_index = config['injection']
            start = 0
            end = None
            for m in modules:
                p = m.weight
                if p.shape[0] <= filter_index:
                    filter_index -= p.shape[0]
                    start += p[0].nelement() * bit_width
                else:
                    filter_size = p[0].nelement() * bit_width
                    start += filter_index * filter_size
                    end = start + filter_size
                    break
            bit_indices_to_flip.add(rnd.randint(start, end))
        else:
            assert False
    else:

        if config['flips'] // 1 == config['flips']:  # if is integer
            parameter_index = rnd.choice(range(len(parameters)), 1, p=[p.nelement() * bit_width / size for p in parameters])[0]
            start = 0
            for i in range(parameter_index):
                start += parameters[i].nelement()
            while len(bit_indices_to_flip) < config['flips']:
                bit_indices_to_flip.add(rnd.randint(start, parameters[parameter_index].nelement() * bit_width - 1))
        else:
            # sensitivity model
            ber = config['flips']
            count = rnd.binomial(size, ber)
            while len(bit_indices_to_flip) < count:
                bit_indices_to_flip.add(rnd.randint(0, size - 1))

    with torch.no_grad():
        flip_bits(bit_indices_to_flip, bit_width, config, modules, parameters, granularity, rnd)

    return bit_indices_to_flip, size


def flip_bits(bit_indices_to_flip, bit_width, config, modules, parameters, granularity, rnd):
    print('Injecting', len(bit_indices_to_flip), 'faults at granularity {}'.format(granularity))
    print(config)
    if granularity == 1:
        pointer = iter(parameters)
        module_pointer = iter(modules)
        parameter = next(pointer)
        module = next(module_pointer)
        offset = 0
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
    else:
        all_params = torch.cat(parameters)
        assert not config['quantization'], "not yet implemented"
        if granularity == _4KB:
            sorted_bit_indices_to_flip = sorted(bit_indices_to_flip)
            bit_indices_to_flip = numpy.array(sorted_bit_indices_to_flip)
            rnd.shuffle(sorted_bit_indices_to_flip)
            for i, j in zip(sorted_bit_indices_to_flip, bit_indices_to_flip):
                start = int(i // bit_width)
                end = start + granularity // bit_width
                source_start = int(j // bit_width)
                source_end = source_start + granularity // bit_width
                all_params[start: end] = all_params[source_start: source_end]
        else:
            for i in sorted(bit_indices_to_flip):
                destination = int(i // bit_width)
                if i % bit_width == _2B:
                    short_index = 1
                elif i % bit_width == 0:
                    short_index = 0
                else:
                    assert False
                (2 * ctypes.c_uint16).from_address(all_params[destination].data_ptr())[
                    short_index
                ] = rnd.randint(0, 2 ** 16 - 1)
        o = 0
        for p in parameters:
            p[:] = all_params[o: o + p.nelement()]
            o += p.nelement()


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


if __name__ == '__main__':
    module = torch.nn.Linear(4096, 4096)
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 16})
    assert len(indices) == 16
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 0})
    assert len(indices) == 0
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': PROBABILITIES[0]})
    assert len(indices) > 0
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'rowhammer'})
    indices = sorted(indices)
    corrupted_chunk_indices = set(i // _4KB for i in indices)
    assert len(corrupted_chunk_indices) > 0
    for i in indices:
        offset = min(abs(i - chunk_index * _4KB) for chunk_index in corrupted_chunk_indices)
        assert offset < _4KB, offset
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'word'})
    assert max(indices) - min(indices) < _2B
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'row'})
    assert len(set(i // _4KB for i in indices)) == 2
    indices, _ = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'column'})
    assert len(indices) == 3
    assert all(i % _2B == 0 for i in indices)
    assert len(set((i // _2B) % (_4KB // _2B) for i in indices)) == 1
    indices, size = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'bank'})
    assert len(set(i % (64 * _2B) for i in indices)) == 1
    indices, size = inject_memory_fault(module, {'quantization': False, 'injection': 0, 'flips': 'chip'})
    assert len(set(i % (8 * _2B) for i in indices)) == 1
