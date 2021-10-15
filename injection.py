from copy import deepcopy
from struct import pack, unpack
from typing import overload
from torch.nn import functional as F

import torch.nn
import torch.nn.quantized
from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import _size_2_t

from sc import StructuralCode, ErasureCode
from utils import lcs, biggest_power_of_two, biggest_divisor_smaller_than


def convert(module, mapping=None, in_place=False, injection_index=None, extra_kwargs=None):
    if extra_kwargs is None:
        extra_kwargs = {}
    if injection_index is None:
        injection_index = CounterReference()
    assert mapping is not None
    if not in_place:
        module = deepcopy(module)

    reassign = {}
    for name, mod in module.named_children():
        if list(mod.named_children()):
            convert(mod, mapping, True, injection_index, extra_kwargs)
            continue
        if mod.__class__ in mapping:
            if extra_kwargs:
                reassign[name] = mapping[mod.__class__].from_original(mod, extra_kwargs)
            else:
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

    if isinstance(f, float):
        n_bytes = 4
        packing = 'f'
    elif isinstance(f, int):
        n_bytes = 1
        packing = 'b'
    else:
        assert False

    f_ = pack(packing, f)
    b = list(unpack('B' * n_bytes, f_))
    [q, r] = divmod(pos, 8)
    b[q] ^= 1 << r
    f_ = pack('B' * n_bytes, *b)
    f = unpack(packing, f_)
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
            return forward
        forward = torch.nan_to_num(forward, self.bounds[1] + 1, self.bounds[1] + 1, self.bounds[0] - 1)
        self.detection = torch.any(torch.any(torch.logical_or(forward > self.bounds[1], forward < self.bounds[0]), -1),
                                   -1)
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


def clipper(activation_class: [torch.nn.ReLU, torch.nn.Hardswish]):
    class ClipperActivation(activation_class):
        def __init__(self, *args, bounds=None, **kwargs):
            super().__init__(*args, **kwargs)
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
                return forward
            forward = torch.nan_to_num(forward, self.bounds[1] + 1, self.bounds[1] + 1, self.bounds[0] - 1)
            self.detection = torch.any(
                torch.any(torch.logical_or(forward > self.bounds[1], forward < self.bounds[0]), -1), -1)
            if not self.detection.any():
                self.detection = None
            else:
                self.detection = torch.nonzero(self.detection)
            result = torch.clip(forward, *self.bounds)
            result *= result != self.bounds[1]
            return result

        @classmethod
        def from_original(cls, original: activation_class):
            return cls()

    return ClipperActivation


ClipperReLU = clipper(torch.nn.ReLU)
ClipperHardswish = clipper(torch.nn.Hardswish)


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
                 padding_mode: str = 'zeros', k=1, threshold=0.1, n=256):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.k = k
        self.n = n
        self.threshold = threshold
        self.sc = StructuralCode(self.n, self.k, self.threshold)
        self.ec = ErasureCode(self.k)
        self.simple_checksum = None
        self.simple_checksum_tensors = None
        self.injected = False
        self.layer = None
        self.detected = False

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d, extra_kwarg=None):
        if extra_kwarg is None:
            extra_kwarg = {}
        instance = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                       original.padding, original.dilation, original.groups, original.bias is not None,
                       original.padding_mode, **extra_kwarg)
        coded_weights = instance.sc.code(original.weight)
        instance.weight = torch.nn.Parameter(coded_weights)
        checksum_tensors = (instance.weight,)
        if original.bias is not None:
            coded_bias = instance.sc.code(original.bias)
            instance.bias = torch.nn.Parameter(coded_bias)
            checksum_tensors += (instance.bias,)
        instance.simple_checksum_tensors = checksum_tensors
        instance.simple_checksum = instance.ec.checksum(instance.simple_checksum_tensors)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = super().forward(input)
        decoded = self.sc.decode(redundant_feature_maps, dim=1)
        if decoded is not None:
            return decoded
        self.detected = True
        erasure = self.ec.erasure(self.simple_checksum_tensors, self.simple_checksum)
        return self.sc.decode(redundant_feature_maps, 1, erasure)


class StructuralCodedLinear(torch.nn.Linear):
    maximum_channels = 128

    def __init__(self, in_features: int, out_features: int, bias: bool = True, k=1, threshold=0.1, n=256) -> None:
        super().__init__(in_features, out_features, bias)
        self.injected = False
        self.detected = False
        self.layer = None
        self.k = k
        self.n = n
        self.threshold = threshold
        self.sc = StructuralCode(self.n, self.k, self.threshold)
        self.ec = ErasureCode(self.k)
        self.simple_checksum = None
        self.simple_checksum_tensors = None

    @staticmethod
    def group(tensor: Tensor, size, dim=0):
        return tensor.reshape(tensor.shape[:dim] + (tensor.shape[dim] // size, size) + tensor.shape[dim + 1:])

    @staticmethod
    def ungroup(tensor: Tensor, dim=0):
        return tensor.reshape(
            tensor.shape[:dim] + (tensor.shape[dim] * tensor.shape[dim + 1],) + tensor.shape[dim + 2:])

    @classmethod
    def from_original(cls, original: torch.nn.Linear, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        instance = cls(original.in_features, original.out_features, original.bias is not None, **extra_kwargs)
        coded_weights = instance.sc.code(original.weight)
        instance.weight = torch.nn.Parameter(coded_weights)
        instance.simple_checksum_tensors = (instance.weight,)
        if original.bias is not None:
            coded_bias = instance.sc.code(original.bias)
            instance.bias = torch.nn.Parameter(coded_bias)
            instance.simple_checksum_tensors += (instance.bias,)
        instance.simple_checksum = instance.ec.checksum(instance.simple_checksum_tensors)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = super().forward(input)
        decoded = self.sc.decode(redundant_feature_maps, dim=1)
        if decoded is not None:
            return decoded
        self.detected = True
        erasure = self.ec.erasure(self.simple_checksum_tensors, self.simple_checksum)
        return self.sc.decode(redundant_feature_maps, 1, erasure)


class ReorderingCodedConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.observe = False
        self.standard_direction = None
        self.standard_order = None

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        instance = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                       original.padding, original.dilation, original.groups, original.bias is not None,
                       original.padding_mode)
        instance.weight = original.weight
        instance.bias = original.bias
        instance.standard_direction = torch.ones((1,) + instance.weight.shape[1:])
        instance.standard_order = instance.get_channel_order(instance.weight, instance.standard_direction)
        return instance

    def get_channel_order(self, tensor: Tensor, direction: Tensor, start_dim=1, threshold=1e-3):
        product = torch.sum(tensor * direction, dim=tuple(range(start_dim, len(tensor.shape))))
        # product = product * (abs(product) > threshold)
        return torch.sort(product).indices

    def forward(self, input: Tensor) -> Tensor:
        result = super().forward(input)
        if self.observe:
            # n = reduce(operator.mul, self.weight.shape[1:])
            # identity = torch.zeros((n, ) + self.weight.shape[1:])
            # for i in range(n):
            #     identity[i].view(n)[i] = 1
            # transformation = self._conv_forward(input, identity, self.bias).view(n, -1)
            # m = transformation.shape[1]
            # standard_direction = torch.ones((1, ) + self.weight.shape[1:]).view(n)
            # test_weight = self.weight[:1, :, :, :]
            # reference = self._conv_forward(input, test_weight, self.bias).view(m)
            # new = torch.matmul(test_weight.view(n), transformation)
            # inverse = torch.pinverse(transformation)
            # transformed_direction = torch.matmul(inverse, standard_direction)
            current_order = self.get_channel_order(self.weight, self.standard_direction)
            current = list(map(int, current_order))
            standard = list(map(int, self.standard_order))
            if torch.sum(current_order == self.standard_order) != len(self.standard_order):
                indices = lcs(standard, current)
                erasure = list(map(lambda ind: current[ind], set(standard) - set(ind[1] for ind in indices)))
                print(*erasure, 'violated')
                self.observe = erasure[0]
        return result


class BloomStructuralCodedConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.injected = False
        self.r = out_channels // 5

    @staticmethod
    def code(weight, r, key, step, dim):
        counter = key
        redundant_kernels = []
        channel_dimension = len(weight.shape) - len(dim) - 1
        for _ in range(r):
            redundant_kernel = torch.zeros([d for i, d in enumerate(weight.shape) if i != channel_dimension],
                                           device=weight.device)
            for c in range(weight.shape[channel_dimension]):
                ind = [c if i == channel_dimension else slice(None, None, None) for i in range(len(weight.shape))]
                if len(ind) == 1:
                    ind = ind[0]
                kernel = weight.__getitem__(ind)
                redundant_kernel += counter * kernel
                counter += step
            redundant_kernels.append(redundant_kernel)
        coded_weight = torch.cat((weight, torch.stack(redundant_kernels, channel_dimension)), channel_dimension)
        return coded_weight

    @staticmethod
    def checksum(code, r, key, step, dim, holdout):
        channel_dimension = len(code.shape) - len(dim) - 1
        channels_count = code.shape[channel_dimension]

        def channel_index(c):
            return [c if i == channel_dimension else slice(None, None, None) for i in range(len(code.shape))]

        original_channel_count = channels_count - r

        first = torch.FloatTensor([key + i * step for i in range(original_channel_count)], device=code.device)
        second = torch.FloatTensor([key + i * step for i in range(original_channel_count, 2 * original_channel_count)],
                                   device=code.device)
        scale_factor = first[holdout] / second[holdout]
        checksum_weights = first - second * scale_factor
        checksum_values = code.__getitem__(channel_index(original_channel_count)) - scale_factor * code.__getitem__(
            channel_index(original_channel_count + 1))
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
        if original.bias is not None:
            coded_bias = cls.code(original.bias, instance.r, instance.key, instance.step, ())
            instance.bias = torch.nn.Parameter(coded_bias)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = super().forward(input)
        decoded_feature_maps = redundant_feature_maps[:, :-self.r]
        hmm = self.code(decoded_feature_maps, self.r, self.key, self.step, (2, 3))
        checksums = torch.sum((hmm - redundant_feature_maps), (1,))

        print(torch.sum(checksums), self.injected)
        image_index = 2
        if self.injected:
            pass

            # detectable = numpy.unravel_index(torch.argmax(torch.sum(redundant_feature_maps[image_index], (0, ))), redundant_feature_maps[image_index].shape[1:])
            # print(detectable)
            # to_checksum = redundant_feature_maps[image_index].__getitem__((slice(None), ) + detectable)
            # to_checksum = redundant_feature_maps[image_index]
            # print(torch.sum(self.checksum(to_checksum, self.r, self.key, self.step, (2, 3), 369)))
            # print(torch.sum(self.checksum(to_checksum, self.r, self.key, self.step, (2, 3), 370)))
            # print(torch.sum(self.checksum(to_checksum, self.r, self.key, self.step, (2, 3), 371)))
            # print(torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), 411)))
            # print(torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), 412)))
        return decoded_feature_maps


class NormalizedConv2d(torch.nn.Module):

    def __init__(self, original):
        super().__init__()
        self.groups = original.groups
        group_in_channels = original.in_channels // original.groups
        self.group_in_channels = group_in_channels
        if group_in_channels > 64:
            divisions = group_in_channels // biggest_divisor_smaller_than(group_in_channels, 64)
        else:
            divisions = 1
        self.divisions = divisions
        division_in_channels = group_in_channels // divisions
        self.division_in_channels = division_in_channels
        for i in range(self.groups):
            for j in range(self.divisions):
                group_out_channels = original.out_channels // self.groups
                convolution = torch.nn.Conv2d(original.in_channels // divisions // self.groups,
                                              group_out_channels,
                                              original.kernel_size,
                                              original.stride,
                                              original.padding, original.dilation, 1, original.bias is not None,
                                              original.padding_mode)
                group_base_index = i * group_in_channels
                division_base_index = j * division_in_channels
                start = division_base_index
                end = start + division_in_channels
                weights = original.weight[group_base_index: group_base_index + group_out_channels, start: end]
                if original.bias is not None:
                    convolution.bias = original.bias
                convolution.weight = torch.nn.Parameter(weights)
                self.__setattr__('conv_{}_{}'.format(i, j), convolution)

    def forward(self, input: Tensor) -> Tensor:
        result = []
        for i in range(self.groups):
            division_result = None
            for j in range(self.divisions):
                group_base_index = i * self.group_in_channels
                division_base_index = j * self.division_in_channels
                start = group_base_index + division_base_index
                end = start + self.division_in_channels
                forward = getattr(self, 'conv_{}_{}'.format(i, j)).forward(input[:, start: end])
                if division_result is None:
                    division_result = forward
                else:
                    division_result += forward
            result.append(division_result)
        return torch.cat(result, dim=1)

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        return cls(original)


class NormalizedLinear(torch.nn.Module):

    def __init__(self, original: torch.nn.Linear):
        super().__init__()
        if original.in_features > 512:
            divisions = original.in_features // biggest_divisor_smaller_than(original.in_features, 512)
        else:
            divisions = 1
        self.divisions = divisions
        division_in_features = original.in_features // divisions
        self.division_in_features = division_in_features
        for j in range(self.divisions):
            convolution = torch.nn.Linear(division_in_features, original.out_features, original.bias is not None)
            division_base_index = j * division_in_features
            start = division_base_index
            end = start + division_in_features
            weights = original.weight[:, start: end]
            if original.bias is not None:
                convolution.bias = original.bias
            convolution.weight = torch.nn.Parameter(weights)
            self.__setattr__('linear_{}'.format(j), convolution)

    def forward(self, input: Tensor) -> Tensor:
        division_result = None
        for j in range(self.divisions):
            division_base_index = j * self.division_in_features
            start = division_base_index
            end = start + self.division_in_features
            forward = getattr(self, 'linear_{}'.format(j)).forward(input[:, start: end])
            if division_result is None:
                division_result = forward
            else:
                division_result += forward
        return division_result

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        return cls(original)


class TMRLinear(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        result = cls(original.in_features, original.out_features, original.bias is not None)
        result.weight = torch.nn.Parameter(torch.cat((original.weight, ) * 3))
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        original_size = self.weight.shape[0] // 3
        first, second, third = (
            self.weight[:original_size],
            self.weight[original_size: 2 * original_size],
            self.weight[original_size * 2:]
        )
        recovered = (first != second) * third + first * (first == second)
        return F.linear(input, recovered, self.bias)


class TMRConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        result = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode)
        result.weight = torch.nn.Parameter(torch.cat((original.weight,) * 3))
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        original_size = self.weight.shape[0] // 3
        first, second, third = (
            self.weight[:original_size],
            self.weight[original_size: 2 * original_size],
            self.weight[original_size * 2:]
        )
        recovered = (first != second) * third + first * (first == second)
        return self._conv_forward(input, recovered, self.bias)


class RADARLinear(torch.nn.quantized.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_original(cls, original: torch.nn.quantized.Linear):
        result = cls(original.in_features, original.out_features, original.bias is not None)
        result.weight = original.weight
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)


class RADARConv2d(torch.nn.quantized.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    @classmethod
    def from_original(cls, original: torch.nn.quantized.Conv2d):
        result = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode)
        result.weight = original.weight
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)



