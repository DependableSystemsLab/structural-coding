import random
from itertools import combinations

import torch
from torch import Tensor, FloatTensor


class StructuralCode:

    def __init__(self, n, k, p, threshold=0) -> None:
        self.n = n
        self.k = k
        self.p = p
        self.threshold = threshold

    def code(self, tensor: Tensor, dim: int = 0, weight_stop=None, weights=None) -> Tensor:
        if weights is None:
            weights = self._generate_redundant_weights()
        if weight_stop is None:
            weight_stop = self.p + self.k
        return torch.cat((tensor, self.out_transform(torch.matmul(self.in_transpose(tensor, dim), weights[:, :weight_stop]), dim)), dim)

    def decode(self, tensor: Tensor, dim: int = 0) -> Tensor:
        tensor = self.in_transpose(tensor, dim, 0)
        systematic = tensor[: self.n]
        bloom = tensor[self.n: self.n + self.k]
        checksum = torch.sum(bloom[:2] - self.checksum(systematic)[self.n:])
        if abs(checksum) <= self.threshold:
            return self.out_transform(systematic, dim, 0)
        linear = tensor[self.n:]
        bloom = self.brute_decode(linear)
        faulty_bloom = self.code(systematic)[self.n: self.n + self.k]
        bloom_pattern = torch.abs(torch.sum(bloom - faulty_bloom, tuple(range(1, len(bloom.shape))))) > self.threshold
        print(bloom_pattern)

    def patch(self, tensor: Tensor, dim: int = 0) -> Tensor:
        return None

    def _generate_redundant_weights(self) -> Tensor:
        result = torch.zeros((self.n, self.k))
        bloom_keys = self._generate_bloom_keys()
        for i in range(self.n):
            for j in bloom_keys[i]:
                result[i][j] += 1
        brute_weights = self._generate_brute_weights()
        result = torch.cat((result, torch.matmul(result, brute_weights)), 1)
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

    def checksum(self, systematic: Tensor, dim=0):
        return self.code(systematic, dim, 2)

    def brute_decode(self, linear: Tensor) -> Tensor:
        brute_weights = self._generate_brute_weights()
        checksum = torch.sum(torch.matmul(self.in_transpose(linear[:self.k]), brute_weights[:, 0]) - linear[self.k])
        if checksum <= self.threshold:
            return linear[:self.k]
        assert False
        weights = torch.cat((torch.eye(self.k), brute_weights))
        for in_hold in combinations(range(self.k + self.p), self.k + 1):
            hold = linear[in_hold]
            hold_weights = weights[in_hold]
            target = hold[self.k]
            target_weights = hold_weights[self.k]


    def _generate_brute_weights(self) -> Tensor:
        result = torch.ones((self.k, self.p))
        counter = 0
        for i in range(1, self.p):
            for j in range(self.k):
                result[j][i] += counter
                counter += 1
        return result


coding = StructuralCode(64, 8, 4)

plain = torch.ones((2, 64, 2))
codeword = coding.code(plain, 1)
for _ in range(3):
    corruption_index = random.randrange(0, 64)
    codeword[:, corruption_index, :] = random.randrange(0, 64)
print(plain - coding.decode(codeword, 1))
