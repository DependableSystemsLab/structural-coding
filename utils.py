import ctypes
import math

import torch
from torch import Tensor


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = []
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + [(i -1, j - 1)]
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1], key=lambda l: len(l))

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def biggest_power_of_two(n):
    result = 1
    while n % 2 == 0:
        n = n // 2
        result *= 2
    return result


def biggest_divisor_smaller_than(n, k):
    for i in range(k, -1, -1):
        if n % i == 0:
            return i
    return 1


def quantize_tensor(weight, fake_quant, signed=True):
    repeat = weight[0].flatten().shape[0] // fake_quant.scale[0].flatten().shape[0]
    shape_of_you = (fake_quant.scale.shape[0],) + weight.shape[1:]
    scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(shape_of_you)
    zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(shape_of_you)
    quantized = torch.clamp(
        torch.round(
            weight[:scale.shape[0]] / scale + zero_point),
        fake_quant.quant_min,
        fake_quant.quant_max)
    quantized = torch.cat((quantized, weight[quantized.shape[0]:]), 0)
    if not signed:
        quantized = quantized - fake_quant.quant_min
    return quantized


def dequantize_tensor(quantized, fake_quant, signed=True):
    if not signed:
        quantized = quantized + fake_quant.quant_min

    repeat = quantized[0].flatten().shape[0] // fake_quant.scale[0].flatten().shape[0]
    shape_of_you = (fake_quant.scale.shape[0],) + quantized.shape[1:]
    scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(shape_of_you)
    zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(shape_of_you)
    return torch.cat(((quantized[:scale.shape[0]] - zero_point) * scale,
                      quantized[scale.shape[0]:]), 0)


def radar_checksum(quantized):
    unsigned = 256 * (quantized < 0) + quantized
    checksum = unsigned // 256 * 2 + unsigned // 128 % 2
    return checksum


def fradar_checksum(weight: Tensor):
    assert weight.element_size() == 4
    allocate_memory = weight.flatten()[:math.ceil(weight.nelement() / 4)].clone()
    checksum_array = (ctypes.c_ubyte * (allocate_memory.nelement() * allocate_memory.element_size())).from_address(allocate_memory.data_ptr())
    original_array = (ctypes.c_ubyte * (weight.nelement() * weight.element_size())).from_address(weight.data_ptr())
    for i in range(0, weight.nelement(), 4):
        checksum_array[i // 4] = original_array[i]
    return allocate_memory


def recover_with_fradar(weight: Tensor, weight_redundancy: Tensor):
    assert weight.element_size() == 4
    checksum = weight_redundancy
    weight = weight.clone()
    calculated_checksum = fradar_checksum(weight)
    checksum_array = (ctypes.c_ubyte * (checksum.nelement() * checksum.element_size())).from_address(checksum.data_ptr())
    calculated_checksum_array = (ctypes.c_ubyte * (calculated_checksum.nelement() * calculated_checksum.element_size())).from_address(calculated_checksum.data_ptr())
    result_array = (ctypes.c_ubyte * (weight.nelement() * weight.element_size())).from_address(weight.data_ptr())
    for i in range(0, weight.nelement(), 4):
        if checksum_array[i // 4] != calculated_checksum_array[i // 4]:
            result_array[i] = 0
    return weight


def bit_tmr(param, param1, param2):
    return ((~((param ^ param1) | 256)) & param1) | ((param ^ param1) & param2)


def recover_with_tmr(weight: Tensor):
    original_size = weight.shape[0] // 3
    first, second, third = (
        weight[:original_size],
        weight[original_size: 2 * original_size],
        weight[original_size * 2:]
    )
    number_of_bytes = first.nelement() * first.element_size()
    result = first.clone()
    first_bytes, second_bytes, third_bytes, result_bytes = (
        (ctypes.c_ubyte * number_of_bytes).from_address(first.data_ptr()),
        (ctypes.c_ubyte * number_of_bytes).from_address(second.data_ptr()),
        (ctypes.c_ubyte * number_of_bytes).from_address(third.data_ptr()),
        (ctypes.c_ubyte * number_of_bytes).from_address(result.data_ptr()),
    )
    for i in range(number_of_bytes):
        result_bytes[i] = bit_tmr(first_bytes[i], second_bytes[i], third_bytes[i])
    return result

