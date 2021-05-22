from copy import deepcopy
from typing import Optional

import torch.quantization
from torch import Tensor
from torch.nn import functional as F


class InjectionMixin:

    ber = 1e-9
    counter = 0

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

    def perturb_weight(self, weight):
        original_weight = weight
        sign = 1 * (weight < 0)
        perturbed_sign = self.perturb_weight_sign(sign)
        denormalized_mask = torch.logical_and(0 <= weight, weight < 2 ** (-126))
        normalized_mask = torch.logical_not(denormalized_mask)
        exponent = normalized_mask * (torch.floor(torch.log2(torch.abs(weight))) + 127)
        perturbed_exponent = self.perturb_weight_exponent(exponent)
        # add normalization bias
        weight = weight + (2 ** (-126)) * (torch.logical_and(perturbed_exponent != 0, exponent == 0))
        # apply change in sign and exponent
        weight = ((-1) ** perturbed_sign) * torch.abs(weight) * (2 ** (perturbed_exponent - exponent))
        InjectionMixin.counter += int(torch.sum(original_weight != weight))
        return weight


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


def convert(module, mapping=None, in_place=False):
    if mapping is None:
        mapping = {
            torch.nn.Conv2d: InjectionConv2D,
            torch.nn.Linear: InjectionLinear
        }
    if not in_place:
        module = deepcopy(module)

    reassign = {}
    for name, mod in module.named_children():
        if list(mod.named_children()):
            convert(mod, mapping, True)
            continue
        if mod.__class__ in mapping:
            reassign[name] = mapping[mod.__class__].from_original(mod)

    for key, value in reassign.items():
        module._modules[key] = value

    return module
