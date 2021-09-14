from copy import deepcopy
from struct import pack, unpack

import torch.nn
from torch import Tensor
from torch.nn.common_types import _size_2_t

from utils import lcs


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
        second = torch.FloatTensor([key + i * step for i in range(original_channel_count, 2 * original_channel_count)], device=code.device)
        scale_factor = first[holdout] / second[holdout]
        checksum_weights = first - second * scale_factor
        checksum_values = code.__getitem__(channel_index(original_channel_count)) - scale_factor * code.__getitem__(channel_index(original_channel_count + 1))
        original_channels = code.__getitem__([
            slice(None, original_channel_count) if i == channel_dimension else slice(None, None, None)
            for i in range(len(code.shape))])
        checksum_values -= torch.sum(checksum_weights * original_channels, (channel_dimension, ))

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
        calculated_redundant_fmaps = hmm[:, self.out_channels:, :, :]
        weight_coded_redundant_fmaps = redundant_feature_maps[:, self.out_channels:, :, :]
        checksum = torch.sum(
            calculated_redundant_fmaps - weight_coded_redundant_fmaps
        )
        if abs(checksum) > 1000:
            holdout_checksums = []
            reduced = torch.sum(redundant_feature_maps, (2, 3))
            for holdout in range(decoded_feature_maps.shape[1]):
                holdout_checksum = torch.sum(self.checksum(reduced, self.r, self.key, self.step, (), holdout))
                # holdout_checksum = torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), holdout))
                holdout_checksums.append(holdout_checksum)
            for _r in range(self.r):
                holdout_checksums.append(torch.sum(calculated_redundant_fmaps[:, _r, :, :] - weight_coded_redundant_fmaps[:, _r, :, :]))
            corrupted = min(enumerate(holdout_checksums), key=lambda c: abs(c[1]))[0]
            if corrupted < self.out_channels:
                first = torch.FloatTensor([self.key + i * self.step for i in range(self.out_channels)],
                                          device=decoded_feature_maps.device)
                for c in range(self.out_channels):
                    if c == corrupted:
                        continue
                    weight_coded_redundant_fmaps[:, 0, :, :] -= first[c] * decoded_feature_maps[:, c, :, :]
                decoded_feature_maps[:, corrupted, :, :] = weight_coded_redundant_fmaps[:, 0, :, :] / first[corrupted]
            # print('recon', corrupted)
        return decoded_feature_maps


class StructuralCodedLinear(torch.nn.Linear):

    maximum_channels = 128

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.key = 1
        self.step = 1
        self.r = 2
        self.group_size = out_features // self.maximum_channels
        while out_features % self.group_size != 0:
            self.group_size += 1
        self.out_channels = out_features // self.group_size
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
        second = torch.FloatTensor([key + i * step for i in range(original_channel_count, 2 * original_channel_count)], device=code.device)
        scale_factor = first[holdout] / second[holdout]
        checksum_weights = first - second * scale_factor
        checksum_values = code.__getitem__(channel_index(original_channel_count)) - scale_factor * code.__getitem__(channel_index(original_channel_count + 1))
        original_channels = code.__getitem__([
            slice(None, original_channel_count) if i == channel_dimension else slice(None, None, None)
            for i in range(len(code.shape))])
        checksum_values -= torch.sum(checksum_weights * original_channels, (channel_dimension, ))

        return checksum_values

    @staticmethod
    def group(tensor: Tensor, size, dim=0):
        return tensor.reshape(tensor.shape[:dim] + (tensor.shape[dim] // size, size) + tensor.shape[dim + 1:])

    @staticmethod
    def ungroup(tensor: Tensor, dim=0):
        return tensor.reshape(tensor.shape[:dim] + (tensor.shape[dim] * tensor.shape[dim + 1],) + tensor.shape[dim + 2:])

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        instance = cls(original.in_features, original.out_features, original.bias is not None)
        coded_weights = cls.code(cls.group(original.weight, instance.group_size), instance.r, instance.key, instance.step, (1, 2, ))
        instance.weight = torch.nn.Parameter(cls.ungroup(coded_weights))
        if original.bias is not None:
            coded_bias = cls.code(cls.group(original.bias, instance.group_size), instance.r, instance.key, instance.step, (1, ))
            instance.bias = torch.nn.Parameter(cls.ungroup(coded_bias))
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = self.group(super().forward(input), self.group_size, dim=1)
        decoded_feature_maps = redundant_feature_maps[:, :-self.r]
        hmm = self.code(decoded_feature_maps, self.r, self.key, self.step, (2, ))
        calculated_redundant_fmaps = hmm[:, self.out_channels:]
        weight_coded_redundant_fmaps = redundant_feature_maps[:, self.out_channels:]
        checksum = torch.sum(
            calculated_redundant_fmaps - weight_coded_redundant_fmaps
        )
        if abs(checksum) > 1000:
            holdout_checksums = []
            reduced = torch.sum(redundant_feature_maps, (2, ))
            for holdout in range(decoded_feature_maps.shape[1]):
                print(holdout)
                holdout_checksum = torch.sum(self.checksum(reduced, self.r, self.key, self.step, (2,), holdout))
                # holdout_checksum = torch.sum(self.checksum(redundant_feature_maps, self.r, self.key, self.step, (2, 3), holdout))
                holdout_checksums.append(holdout_checksum)
            for _r in range(self.r):
                holdout_checksums.append(torch.sum(calculated_redundant_fmaps[:, _r] - weight_coded_redundant_fmaps[:, _r]))
            corrupted = min(enumerate(holdout_checksums), key=lambda c: abs(c[1]))[0]
            if corrupted < self.out_channels:
                first = torch.FloatTensor([self.key + i * self.step for i in range(self.out_channels)],
                                          device=decoded_feature_maps.device)
                for c in range(self.out_channels):
                    if c == corrupted:
                        continue
                    weight_coded_redundant_fmaps[:, 0] -= first[c] * decoded_feature_maps[:, c]
                decoded_feature_maps[:, corrupted] = weight_coded_redundant_fmaps[:, 0] / first[corrupted]
                print('recon')
        return self.ungroup(decoded_feature_maps, dim=1)


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
        second = torch.FloatTensor([key + i * step for i in range(original_channel_count, 2 * original_channel_count)], device=code.device)
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
        if original.bias is not None:
            coded_bias = cls.code(original.bias, instance.r, instance.key, instance.step, ())
            instance.bias = torch.nn.Parameter(coded_bias)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        redundant_feature_maps = super().forward(input)
        decoded_feature_maps = redundant_feature_maps[:, :-self.r]
        hmm = self.code(decoded_feature_maps, self.r, self.key, self.step, (2, 3))
        checksums = torch.sum((hmm - redundant_feature_maps), (1, ))

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

