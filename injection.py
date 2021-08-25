from copy import deepcopy
from struct import pack, unpack

import torch.nn
from torch import Tensor
from torch.nn.common_types import _size_2_t


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
        self.profile = False
        self.detection = None

    def forward(self, input: Tensor) -> Tensor:
        forward = super().forward(input)
        if self.profile:
            if self.bounds is None:
                self.bounds = (float(torch.min(forward)), float(torch.max(forward)))
            else:
                self.bounds = (
                    min(float(torch.min(forward)), self.bounds[0]),
                    max(float(torch.max(forward)), self.bounds[1])
                )
        forward = torch.nan_to_num(forward, self.bounds[1] + 1, self.bounds[1] + 1, self.bounds[0] - 1)

        self.detection = torch.any(torch.any(torch.logical_or(forward > self.bounds[1], forward < self.bounds[0]), -1), -1)
        if not self.detection.any():
            self.detection = None
        else:
            self.detection = torch.nonzero(self.detection)
        result = torch.clip(forward, *self.bounds)
        result *= result != self.bounds[1]
        return result

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


class RangerReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False, bounds=None):
        super().__init__(inplace)
        self.bounds = bounds
        self.profile = False

    def forward(self, input: Tensor) -> Tensor:
        forward = super().forward(input)
        forward = torch.nan_to_num(forward, self.bounds[1], self.bounds[1], self.bounds[0])
        if self.profile:
            if self.bounds is None:
                self.bounds = (float(torch.min(forward)), float(torch.max(forward)))
            else:
                self.bounds = (
                    min(float(torch.min(forward)), self.bounds[0]),
                    max(float(torch.max(forward)), self.bounds[1])
                )
        result = torch.clip(forward, *self.bounds)
        return result

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


class SmootherReLU(torch.nn.ReLU):

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.sigma == 0:
            return input
        noise = torch.normal(0., float(torch.std(input) * self.sigma), input.shape, device=input.device)
        return noise + input

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


class CounterReference:

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0


def top_percent(tensor, percent):
    size = 1
    for d in tensor.shape:
        size *= d
    desired_size = round(percent * size)
    minimum = torch.min(tensor)
    maximum = torch.max(tensor)
    for i in range(20):
        between = (maximum + minimum) / 2
        s = torch.sum(tensor >= between)
        if s > desired_size:
            minimum = between
        elif s < desired_size:
            maximum = between
        else:
            break
    return tensor > between


class StructuralCodedConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.key = 1
        self.step = 1
        self.r = 2
        self.injected = False

    @staticmethod
    def code(weight, r, key, step, dim):
        counter = key
        redundant_kernels = []
        channel_dimension = len(weight.shape) - len(dim) - 1
        for _ in range(r):
            redundant_kernel = torch.zeros([d for i, d in enumerate(weight.shape) if i != channel_dimension],
                                           device=weight.device)
            for c in range(weight.shape[channel_dimension]):
                kernel = weight.__getitem__([c if i == channel_dimension else slice(None, None, None) for i in range(len(weight.shape))])
                redundant_kernel += counter * kernel
                counter += step
            redundant_kernels.append(redundant_kernel)
        coded_weight = torch.cat((weight, torch.stack(redundant_kernels, channel_dimension)), channel_dimension)
        return coded_weight

    @staticmethod
    def checksum(code, r, key, step, dim, holdout):
        assert step == 1
        channel_dimension = len(code.shape) - len(dim) - 1
        channels_count = code.shape[channel_dimension]

        def channel_index(c):
            return [c if i == channel_dimension else slice(None, None, None) for i in range(len(code.shape))]

        original_channel_count = channels_count - r

        first = torch.FloatTensor([i for i in range(key, key + original_channel_count)], device=code.device)
        second = torch.FloatTensor([i for i in range(key + original_channel_count, key + 2 * original_channel_count)], device=code.device)
        scale_factor = first[holdout] / second[holdout]
        checksum_weights = first - second * scale_factor
        checksum_values = code.__getitem__(channel_index(original_channel_count)) - scale_factor * code.__getitem__(channel_index(original_channel_count + 1))
        original_channels = code.__getitem__([
            slice(None, original_channel_count) if i == channel_dimension else slice(None, None, None)
            for i in range(len(code.shape))])
        for c in range(original_channel_count):
            checksum_values -= checksum_weights[c] * original_channels.__getitem__(channel_index(c))

        return checksum_values

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        instance = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                       original.padding, original.dilation, original.groups, original.bias is not None,
                       original.padding_mode)
        coded_weights = cls.code(original.weight, instance.r, instance.key, instance.step, (1, 2, 3))
        instance.weight = torch.nn.Parameter(coded_weights)
        instance.bias = original.bias
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = super().forward(input)
        decoded_feature_maps = redundant_feature_maps[:, :-self.r]
        # hmm = self.code(decoded_feature_maps, self.r, self.key, self.step, (2, 3))

        if self.injected:
            print(torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), 411)))
            print(torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), 412)))
        return decoded_feature_maps
