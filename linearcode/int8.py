import random

import numpy.random
import torch
from galois import GF
import numpy as np
from torch import Tensor, LongTensor

from sc import StructuralCode, Field, ErasureCode


class TensorIndex:

    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = (1, ) + shape
        self.index = [0 for _ in self.shape]
        self.index[-1] = -1

    def __next__(self):
        for i in range(len(self.shape) - 1, -1, -1):
            self.index[i] += 1
            if self.index[i] < self.shape[i]:
                break
            self.index[i] = 0
            if i == 0:
                raise StopIteration
        return tuple(self.index)[1:]

    def __iter__(self):
        return self


class IntField(Field):

    def matmul(self, a, b):
        a_shape = a.shape[:-2]
        b_shape = b.shape[:-2]
        result = self.galois_field(numpy.zeros(a_shape + b_shape + (a.shape[-2], b.shape[-1],), dtype='int'))
        for i in TensorIndex(result.shape[:-2]):
            result[i] = a[i[:len(a_shape)]] @ b[i[:len(b_shape)]]
        return self.to_torch(result)

    def __init__(self, galois_field: GF = None) -> None:
        super().__init__()
        if galois_field is None:
            galois_field = GF(2 ** 8)
        self.galois_field = galois_field

    def invert(self, matrix: Tensor):
        return np.linalg.inv(self.to_field(matrix))

    def random(self, rnd: numpy.random.RandomState, n: int, k: int) -> Tensor:
        return self.to_torch(rnd.randint(0, self.galois_field.order, (n, k)))

    def to_torch(self, tensor) -> Tensor:
        return torch.FloatTensor(tensor)

    def to_field(self, tensor: Tensor):
        return self.galois_field(tensor.type(torch.ShortTensor).numpy())


if __name__ == '__main__':
    k = 8
    coding_dim = 3
    block_size = 64
    gf = GF(2 ** 8)
    plain = torch.randint(0, gf.order, (5, 3, 2, block_size + 63, 3))
    # plain = torch.rand((3, 2, block_size + 63, 2))
    coding = StructuralCode(block_size, k, field=IntField(gf))
    # coding = StructuralCode(block_size, k)
    erasure_coding = ErasureCode(block_size, k)
    codeword = coding.code(plain, coding_dim)
    erasure_checksum = erasure_coding.checksum(codeword, coding_dim)
    corrupted_index = []
    for _ in range(k):
        corruption_index = random.randrange(0, block_size + k)
        codeword[tuple(slice(None) if d != coding_dim else corruption_index for d, _ in enumerate(codeword.shape))] = random.randrange(0, gf.order)
        corrupted_index.append(corruption_index)
    # assert coding.decode(codeword, 1) is None
    # erasure = erasure_coding.erasure(coding.extract_systematic(codeword, coding_dim), erasure_checksum, coding_dim)
    corrupted_index = sorted(corrupted_index)
    erasure = erasure_coding.erasure(codeword, erasure_checksum, coding_dim)
    decoded_plain = coding.decode(codeword, coding_dim, erasure)
    print(corrupted_index)
    print(erasure)
    corrupted_index = [c for c in corrupted_index if c < block_size]
    print(plain - decoded_plain)
    print(torch.sum(plain - decoded_plain))
