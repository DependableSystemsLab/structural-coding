import random
from itertools import combinations
from typing import Optional

import torch
from torch import Tensor, LongTensor


class ErasureCode:
    def __init__(self, k) -> None:
        self.k = k

    @staticmethod
    def checksum(tensor: Tensor, dim: int = 0) -> Tensor:
        return torch.sum(tensor, dim=tuple(d for d in range(len(tensor.shape)) if d != dim))

    def erasure(self, tensor: Tensor, checksum: Tensor, dim: int = 0) -> Tensor:
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
        return self.out_transform(torch.matmul(self.in_transpose(tensor, dim), weights[:, :weight_stop]), dim)

    def code(self, tensor: Tensor, dim: int = 0, weight_stop=None, weights=None) -> Tensor:
        return torch.cat((tensor, self._code(tensor, dim, weight_stop, weights)), dim)

    def decode(self, tensor: Tensor, dim: int = 0, erasure: Tensor = None) -> Optional[Tensor]:
        tensor = self.in_transpose(tensor, dim, 0)
        systematic = tensor[: self.n]
        checksum = torch.sum(tensor[self.n] - self.checksum(systematic))
        if abs(checksum) <= self.threshold:
            return self.out_transform(systematic, dim, 0)
        if erasure is None:
            return None
        erasure = torch.sort(erasure).values
        healthy_indices = LongTensor(list(set(range(tensor.shape[0])) - set(map(int, erasure))))
        systematic_healthy_indices = LongTensor([i for i in set(range(tensor.shape[0])) - set(map(int, erasure)) if i < self.n])
        weights = torch.cat((torch.eye(self.n), self._generate_redundant_weights()), 1)

        # keep healthy redundant vectors
        weights = weights[:, healthy_indices]
        redundant_part = tensor[healthy_indices]

        print(weights.shape)
        patch = torch.pinverse(weights)

        reconstructed_erasure = self.in_transpose(torch.matmul(self.out_transform(redundant_part, 0), patch))

        reconstructed_erasure[systematic_healthy_indices] = tensor[systematic_healthy_indices]

        return self.out_transform(reconstructed_erasure, 0, dim)


    def _generate_redundant_weights(self) -> Tensor:
        if self._weights is None:
            self._weights = torch.rand((self.n, self.k))
        return torch.clone(self._weights)
        result = torch.ones((self.n, self.k))
        counter = 0
        for j in range(self.n):
            for i in range(self.k):
                result[j][i] += counter
                counter += 1
        return result

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


k = 64
coding_dim = 1
block_size = 64
plain = torch.rand((2, block_size, 2))
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
