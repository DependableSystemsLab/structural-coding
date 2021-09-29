import random
from itertools import combinations
from typing import Optional, Tuple

import torch
from torch import Tensor, LongTensor


class ErasureCode:
    def __init__(self, k) -> None:
        self.k = k

    @staticmethod
    def checksum(tensor: [Tensor, Tuple[Tensor]], dim: int = 0) -> Tensor:
        if isinstance(tensor, tuple) and len(tensor) > 1:
            return ErasureCode.checksum(tensor[0]) + ErasureCode.checksum(tensor[1:])
        else:
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            return torch.sum(tensor, dim=tuple(d for d in range(len(tensor.shape)) if d != dim))

    def erasure(self, tensor: [Tensor, Tuple[Tensor]], checksum: Tensor, dim: int = 0) -> Tensor:
        return torch.sort(torch.topk(torch.abs(self.checksum(tensor, dim) - checksum), self.k).indices).values


class StructuralCode:

    def __init__(self, n, k, threshold=0) -> None:
        self.n = n
        self.k = k
        self._weights = None
        self.threshold = threshold

    def _code(self, tensor: Tensor, dim: int = 0, weight_stop=None, weights=None) -> Tensor:
        if weights is None:
            weights = self._generate_redundant_weights()
        if weight_stop is None:
            weight_stop = self.k
        return self.out_transform(torch.matmul(self.in_transpose(tensor, dim), weights[:tensor.shape[dim], :weight_stop]), dim)

    def code(self, tensor: Tensor, dim: int = 0, weight_stop=None, weights=None) -> Tensor:
        if tensor.shape[dim] > self.n:
            tensor = self.in_transpose(tensor, dim, 0)
            first = tensor[:self.n]
            rest = tensor[self.n:]
            result = torch.cat((
                first, self._code(first, 0, weight_stop, weights),
                self.code(rest, 0, weight_stop, weights)
            ), 0)
            return self.out_transform(result, 0, dim)
        return torch.cat((tensor, self._code(tensor, dim, weight_stop, weights)), dim)

    def decode(self, tensor: Tensor, dim: int = 0, erasure: Tensor = None) -> Optional[Tensor]:
        tensor = self.in_transpose(tensor, dim, 0)
        if tensor.shape[0] > self.n + self.k:
            first = tensor[:self.n + self.k]
            rest = tensor[self.n + self.k:]
            first_kwargs = {
                'erasure': None
            }
            rest_kwargs = {
                'erasure': None
            }
            if erasure is not None:
                first_kwargs['erasure'] = LongTensor([i for i in erasure if i < self.n + self.k])
                rest_kwargs['erasure'] = LongTensor([i - self.n - self.k for i in erasure if i >= self.n + self.k])
            decoded_first = self.decode(first, 0, **first_kwargs)
            if decoded_first is None:
                return None
            decoded_rest = self.decode(rest, 0, **rest_kwargs)
            if decoded_rest is None:
                return None
            return self.out_transform(torch.cat((decoded_first, decoded_rest), 0), dim, 0)
        if tensor.shape[0] < self.n + self.k:
            n = tensor.shape[0] - self.k
        else:
            n = self.n
        systematic = tensor[: n]
        checksum = torch.sum(tensor[n] - self.checksum(systematic))
        if abs(checksum) <= self.threshold:
            return self.out_transform(systematic, dim, 0)
        # print(checksum)
        if erasure is None:
            return None
        erasure = torch.sort(erasure).values
        healthy_indices = LongTensor(list(set(range(tensor.shape[0])) - set(map(int, erasure))))
        systematic_healthy_indices = LongTensor([i for i in set(range(tensor.shape[0])) - set(map(int, erasure)) if i < n])
        weights = torch.cat((torch.eye(n), self._generate_redundant_weights()[:n]), 1)

        # keep healthy redundant vectors
        weights = weights[:, healthy_indices]
        redundant_part = tensor[healthy_indices]

        patch = torch.pinverse(weights)

        reconstructed_erasure = self.in_transpose(torch.matmul(self.out_transform(redundant_part, 0), patch))

        reconstructed_erasure[systematic_healthy_indices] = tensor[systematic_healthy_indices]

        return self.out_transform(reconstructed_erasure, 0, dim)


    def _generate_redundant_weights(self) -> Tensor:
        if self._weights is None:
            seed = torch.seed()
            torch.manual_seed(0)
            self._weights = torch.rand((self.n, self.k))
            torch.manual_seed(seed)
            self._weights[:, 0] = 1
        return torch.clone(self._weights)

    def _generate_bloom_keys(self):
        return list(combinations(range(self.k), self.k // 2))

    def in_transpose(self, tensor: Tensor, dim=0, destination=None) -> Tensor:
        if destination is None:
            destination = len(tensor.shape) - 1
        return torch.transpose(tensor, dim, destination)

    def out_transform(self, tensor: Tensor, dim=0, destination=None) -> Tensor:
        if destination is None:
            destination = len(tensor.shape) - 1
        return torch.transpose(tensor, destination, dim)

    def checksum(self, systematic: Tensor, dim=0) -> Tensor:
        return self._code(systematic, dim, 1)[0]

    def extract_systematic(self, tensor: Tensor, dim: int = 0) -> Tensor:
        tensor = self.in_transpose(tensor, dim, 0)
        systematic = tensor[: self.n]
        return self.out_transform(systematic, dim, 0)


if __name__ == '__main__':
    k = 8
    coding_dim = 1
    block_size = 64
    plain = torch.rand((2, block_size + 63, 2))
    coding = StructuralCode(block_size, k)
    erasure_coding = ErasureCode(k)
    codeword = coding.code(plain, coding_dim)
    erasure_checksum = erasure_coding.checksum(plain, coding_dim)
    corrupted_index = []
    for _ in range(k):
        corruption_index = random.randrange(0, block_size + k)
        codeword[:, corruption_index, :] = random.randrange(0, block_size)
        corrupted_index.append(corruption_index)
    # assert coding.decode(codeword, 1) is None
    # erasure = erasure_coding.erasure(coding.extract_systematic(codeword, coding_dim), erasure_checksum, coding_dim)
    corrupted_index = sorted(corrupted_index)
    erasure = LongTensor(corrupted_index)
    decoded_plain = coding.decode(codeword, 1, erasure)
    print(corrupted_index)
    print(erasure)
    corrupted_index = [c for c in corrupted_index if c < block_size]
    print(plain - decoded_plain)
