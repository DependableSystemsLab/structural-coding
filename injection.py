from copy import deepcopy
from struct import pack, unpack

import torch.nn
from torch import Tensor


def convert(module, mapping=None, in_place=False, injection_index=None):
    if injection_index is None:
        injection_index = CounterReference()
    assert mapping is not None
    if not in_place:
        module = deepcopy(module)

    reassign = {}
    for name, mod in module.named_children():
        if list(mod.named_children()):
            convert(mod, mapping, True, injection_index)
            continue
        if mod.__class__ in mapping:
            reassign[name] = mapping[mod.__class__].from_original(mod)
            if hasattr(reassign[name], 'weight'):
                reassign[name].injection_index = injection_index.counter
                weight_shape = reassign[name].weight.shape
                weight_size = 1
                for d in weight_shape:
                    weight_size *= d
                injection_length = weight_size * 9
                injection_index.counter += injection_length
                reassign[name].injection_length = injection_length

    for key, value in reassign.items():
        module._modules[key] = value

    return module, injection_index.counter


def bitflip(f, pos):
    """ Single bit-flip in 32 bit floats """

    f_ = pack('f', f)
    b = list(unpack('BBBB', f_))
    [q, r] = divmod(pos, 8)
    b[q] ^= 1 << r
    f_ = pack('BBBB', *b)
    f = unpack('f', f_)
    return f[0]


class ClipperReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False, bounds=None):
        super().__init__(inplace)
        self.bounds = bounds

    def forward(self, input: Tensor) -> Tensor:
        forward = super().forward(input)
        forward = torch.nan_to_num(forward, 0, 0, 0)
        result = torch.clip(forward, *self.bounds)
        result *= result != self.bounds[1]
        return result

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


class CounterReference:

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0