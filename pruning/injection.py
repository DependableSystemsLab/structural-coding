from abc import ABCMeta, abstractmethod
from typing import Optional

import torch.quantization
from torch import Tensor
from torch.nn import functional as F

from pruning.parameters import CONFIG


class InjectionMixinBase(metaclass=ABCMeta):
    counter = 0
    indices = []

    @abstractmethod
    def perturb_weight_sign(self, sign):
        pass

    @abstractmethod
    def perturb_weight_exponent(self, exponent):
        pass

    def perturb_weight(self, weight):
        original_weight = weight
        sign = 1 * (weight < 0)
        perturbed_sign = self.perturb_weight_sign(sign)
        denormalized_mask = torch.logical_and(0 <= weight, weight < 2 ** (-126))
        exponent = torch.masked_fill(torch.floor(torch.log2(torch.abs(weight))) + 127, denormalized_mask, 0)
        perturbed_exponent = self.perturb_weight_exponent(exponent)
        # add normalization bias
        weight = weight + (2 ** (-126)) * (torch.logical_and(perturbed_exponent != 0, exponent == 0))
        # apply change in sign and exponent
        weight = ((-1) ** perturbed_sign) * torch.abs(weight) * (2 ** (perturbed_exponent - exponent))
        if hasattr(self, 'weight_mask'):
            weight = torch.masked_fill(weight, torch.logical_not(self.weight_mask), 0)
        InjectionMixin.counter += int(torch.sum(original_weight != weight))
        return weight


class BerInjectionMixin(InjectionMixinBase):
    ber = 1e-9

    def perturb_weight_sign(self, sign):
        flipped_sign = 1 - sign
        flipped_mask = torch.rand(sign.shape, device=sign.device) < self.ber
        return flipped_mask * flipped_sign + torch.logical_not(flipped_mask) * sign

    def perturb_weight_exponent(self, exponent):
        additive = torch.zeros(exponent.shape, device=exponent.device)
        for bit_magnitude in (1, 2, 4, 8, 16, 32, 64, 127):
            mask = torch.logical_not(torch.rand(exponent.shape, device=exponent.device) < self.ber)
            flip_sign = torch.masked_fill(- (torch.floor(exponent / bit_magnitude) % 2 - 0.5) * 2, mask, 0)
            additive += flip_sign * bit_magnitude
        return exponent + additive


class PositionInjectionMixin(InjectionMixinBase):

    def perturb_weight(self, weight):
        if [
            i for i in self.indices
            if 0 <= i - self.injection_index < self.injection_length
        ]:
            return super().perturb_weight(weight)
        else:
            return weight

    def perturb_weight_sign(self, sign):
        indices_to_inject = [(i - self.injection_index) // 9 for i in self.indices
                             if (
                                     self.injection_index <= i < self.injection_index + self.injection_length and
                                     (i - self.injection_index) % 9 == 0
                             )]
        shape = sign.shape
        if indices_to_inject:
            sign = torch.clone(sign.flatten())
            for i in indices_to_inject:
                sign[i] = 1 - sign[i]
            return sign.reshape(shape)
        else:
            return sign

    def perturb_weight_exponent(self, exponent):
        indices_to_inject = [i - self.injection_index for i in self.indices
                             if (
                                     self.injection_index <= i < self.injection_index + self.injection_length and
                                     i % 9 != 0
                             )]
        shape = exponent.shape
        if indices_to_inject:
            exponent = torch.clone(exponent.flatten())
            for index in indices_to_inject:
                weight_index = index // 9
                bit_index = index % 9 - 1
                bit_magnitude = 2 ** bit_index
                if (exponent[weight_index] // bit_magnitude) % 2:
                    exponent[weight_index] -= bit_magnitude
                else:
                    exponent[weight_index] += bit_magnitude
            return exponent.reshape(shape)
        else:
            return exponent


if not CONFIG['faults']:
    InjectionMixin = BerInjectionMixin
else:
    InjectionMixin = PositionInjectionMixin


class InjectionConv2D(torch.nn.Conv2d, InjectionMixin):

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        module = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode)
        module.load_state_dict(original.state_dict())
        return module

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        perturbed_weight = self.perturb_weight(weight)
        result = super()._conv_forward(input, perturbed_weight, bias)
        return result


class InjectionLinear(torch.nn.Linear, InjectionMixin):

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        module = cls(original.in_features, original.out_features, bias=original.bias is not None)
        module.load_state_dict(original.state_dict())
        return module

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.perturb_weight(self.weight), self.bias)


class ObserverRelu(torch.nn.ReLU):

    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.min = self.max = 0

    def forward(self, input: Tensor) -> Tensor:
        result = super().forward(input)
        self.min = min(self.min, float(torch.min(result)))
        self.max = max(self.max, float(torch.max(result)))
        return result

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


class RangerReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False, bounds=None):
        super().__init__(inplace)
        self.bounds = bounds

    def forward(self, input: Tensor) -> Tensor:
        return torch.clip(super().forward(input), *self.bounds)

    @classmethod
    def from_original(cls, original: torch.nn.ReLU):
        return cls()


