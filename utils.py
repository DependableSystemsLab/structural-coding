import torch


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


def quantize_tensor(weight, fake_quant):
    repeat = weight.flatten().shape[0] // fake_quant.scale.flatten().shape[0]
    scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(weight.shape)
    zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(weight.shape)
    quantized = torch.clamp(
        torch.round(
            weight / scale + zero_point),
        fake_quant.quant_min,
        fake_quant.quant_max)
    return quantized


def dequantize_tensor(quantized, fake_quant):
    repeat = quantized.flatten().shape[0] // fake_quant.scale.flatten().shape[0]
    scale = torch.repeat_interleave(fake_quant.scale, repeat).reshape(quantized.shape)
    zero_point = torch.repeat_interleave(fake_quant.zero_point, repeat).reshape(quantized.shape)
    return (quantized - zero_point) * scale


def radar_checksum(quantized):
    unsigned = 256 * (quantized < 0) + quantized
    checksum = unsigned // 256 * 2 + unsigned // 128 % 2
    return checksum
