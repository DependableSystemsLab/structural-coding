import copy
import operator
from copy import deepcopy
from functools import reduce
from struct import pack, unpack
from typing import overload, Optional

import numpy.random
from torch.nn import functional as F

import torch.nn
import torch.nn.quantized
import torch.nn.qat
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from linearcode.int8 import IntField
from sc import StructuralCode, ErasureCode
from utils import lcs, biggest_divisor_smaller_than, quantize_tensor, radar_checksum, \
    recover_with_tmr, recover_with_fradar, fradar_checksum, dequantize_tensor

BER_OPTIMAL_CONFIGS = {(1, 1, 3, 3, 'one-size-fits-all'): (288, 32),
 (1, 1, 3, 3, 2.160046875e-08): (2, 1),
 (1, 1, 3, 3, 3.456075e-07): (2, 1),
 (1, 1, 3, 3, 5.52972e-06): (2, 1),
 (1, 1, 5, 5, 'one-size-fits-all'): (288, 32),
 (1, 1, 5, 5, 2.160046875e-08): (2, 1),
 (1, 1, 5, 5, 3.456075e-07): (2, 1),
 (1, 1, 5, 5, 5.52972e-06): (2, 1),
 (1, 10, 2.160046875e-08): (2, 1),
 (1, 10, 3.456075e-07): (2, 1),
 (1, 10, 5.52972e-06): (2, 1),
 (1, 10, 'one-size-fits-all'): (288, 32),
 (8, 16, 1, 1, 2.160046875e-08): (9, 1),
 (8, 16, 1, 1, 'one-size-fits-all'): (288, 32),
 (8, 16, 1, 1, 3.456075e-07): (9, 1),
 (8, 16, 1, 1, 5.52972e-06): (9, 1),
 (10, 50, 'one-size-fits-all'): (288, 32),
 (10, 50, 2.160046875e-08): (11, 1),
 (10, 50, 3.456075e-07): (11, 1),
 (10, 50, 5.52972e-06): (11, 1),
 (16, 3, 3, 3, 'one-size-fits-all'): (288, 32),
 (16, 3, 3, 3, 2.160046875e-08): (17, 1),
 (16, 3, 3, 3, 3.456075e-07): (17, 1),
 (16, 3, 3, 3, 5.52972e-06): (17, 1),
 (16, 8, 1, 1, 2.160046875e-08): (17, 1),
 (16, 8, 1, 1, 'one-size-fits-all'): (288, 32),
 (16, 8, 1, 1, 3.456075e-07): (17, 1),
 (16, 8, 1, 1, 5.52972e-06): (17, 1),
 (16, 16, 1, 1, 'one-size-fits-all'): (288, 32),
 (16, 16, 1, 1, 2.160046875e-08): (17, 1),
 (16, 16, 1, 1, 3.456075e-07): (17, 1),
 (16, 16, 1, 1, 5.52972e-06): (17, 1),
 (16, 60, 1, 1, 3.456075e-07): (17, 1),
 (16, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (16, 60, 1, 1, 2.160046875e-08): (17, 1),
 (16, 60, 1, 1, 5.52972e-06): (18, 2),
 (16, 64, 1, 1, 2.160046875e-08): (17, 1),
 (16, 64, 1, 1, 3.456075e-07): (17, 1),
 (16, 64, 1, 1, 5.52972e-06): (18, 2),
 (16, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 3, 3, 3, 'one-size-fits-all'): (288, 32),
 (24, 3, 3, 3, 2.160046875e-08): (25, 1),
 (24, 3, 3, 3, 3.456075e-07): (25, 1),
 (24, 3, 3, 3, 5.52972e-06): (25, 1),
 (24, 3, 5, 5, 'one-size-fits-all'): (288, 32),
 (24, 3, 5, 5, 2.160046875e-08): (25, 1),
 (24, 3, 5, 5, 3.456075e-07): (25, 1),
 (24, 3, 5, 5, 5.52972e-06): (26, 2),
 (24, 24, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 24, 1, 1, 2.160046875e-08): (25, 1),
 (24, 24, 1, 1, 3.456075e-07): (25, 1),
 (24, 24, 1, 1, 5.52972e-06): (25, 1),
 (24, 36, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 36, 1, 1, 2.160046875e-08): (25, 1),
 (24, 36, 1, 1, 3.456075e-07): (25, 1),
 (24, 36, 1, 1, 5.52972e-06): (26, 2),
 (24, 44, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 44, 1, 1, 2.160046875e-08): (25, 1),
 (24, 44, 1, 1, 3.456075e-07): (25, 1),
 (24, 44, 1, 1, 5.52972e-06): (26, 2),
 (24, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 48, 1, 1, 2.160046875e-08): (25, 1),
 (24, 48, 1, 1, 3.456075e-07): (25, 1),
 (24, 48, 1, 1, 5.52972e-06): (26, 2),
 (24, 64, 1, 1, 2.160046875e-08): (25, 1),
 (24, 64, 1, 1, 3.456075e-07): (25, 1),
 (24, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (24, 64, 1, 1, 5.52972e-06): (26, 2),
 (32, 16, 3, 3, 3.456075e-07): (33, 1),
 (32, 16, 3, 3, 'one-size-fits-all'): (288, 32),
 (32, 16, 3, 3, 2.160046875e-08): (33, 1),
 (32, 16, 3, 3, 5.52972e-06): (36, 4),
 (32, 48, 1, 1, 2.160046875e-08): (33, 1),
 (32, 48, 1, 1, 3.456075e-07): (33, 1),
 (32, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (32, 48, 1, 1, 5.52972e-06): (34, 2),
 (32, 60, 1, 1, 2.160046875e-08): (33, 1),
 (32, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (32, 60, 1, 1, 3.456075e-07): (33, 1),
 (32, 60, 1, 1, 5.52972e-06): (34, 2),
 (32, 64, 1, 1, 2.160046875e-08): (33, 1),
 (32, 64, 1, 1, 3.456075e-07): (33, 1),
 (32, 64, 1, 1, 5.52972e-06): (34, 2),
 (32, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (36, 24, 5, 5, 'one-size-fits-all'): (288, 32),
 (36, 24, 5, 5, 2.160046875e-08): (37, 1),
 (36, 24, 5, 5, 3.456075e-07): (38, 2),
 (36, 24, 5, 5, 5.52972e-06): (46, 10),
 (40, 48, 1, 1, 2.160046875e-08): (41, 1),
 (40, 48, 1, 1, 3.456075e-07): (41, 1),
 (40, 48, 1, 1, 5.52972e-06): (42, 2),
 (40, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (40, 60, 1, 1, 2.160046875e-08): (41, 1),
 (40, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (40, 60, 1, 1, 3.456075e-07): (41, 1),
 (40, 60, 1, 1, 5.52972e-06): (42, 2),
 (48, 16, 3, 3, 3.456075e-07): (49, 1),
 (48, 16, 3, 3, 'one-size-fits-all'): (288, 32),
 (48, 16, 3, 3, 2.160046875e-08): (49, 1),
 (48, 16, 3, 3, 5.52972e-06): (52, 4),
 (48, 36, 5, 5, 'one-size-fits-all'): (288, 32),
 (48, 36, 5, 5, 2.160046875e-08): (49, 1),
 (48, 36, 5, 5, 3.456075e-07): (51, 3),
 (48, 36, 5, 5, 5.52972e-06): (64, 16),
 (48, 48, 1, 1, 2.160046875e-08): (49, 1),
 (48, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (48, 48, 1, 1, 3.456075e-07): (49, 1),
 (48, 48, 1, 1, 5.52972e-06): (50, 2),
 (48, 60, 1, 1, 2.160046875e-08): (49, 1),
 (48, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (48, 60, 1, 1, 3.456075e-07): (49, 1),
 (48, 60, 1, 1, 5.52972e-06): (51, 3),
 (48, 64, 1, 1, 2.160046875e-08): (49, 1),
 (48, 64, 1, 1, 3.456075e-07): (49, 1),
 (48, 64, 1, 1, 5.52972e-06): (51, 3),
 (48, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (50, 100, 'one-size-fits-all'): (288, 32),
 (50, 100, 2.160046875e-08): (51, 1),
 (50, 100, 3.456075e-07): (51, 1),
 (50, 100, 5.52972e-06): (54, 4),
 (64, 3, 3, 3, 2.160046875e-08): (65, 1),
 (64, 3, 3, 3, 3.456075e-07): (65, 1),
 (64, 3, 3, 3, 5.52972e-06): (66, 2),
 (64, 3, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 3, 7, 7, 'one-size-fits-all'): (288, 32),
 (64, 3, 7, 7, 2.160046875e-08): (65, 1),
 (64, 3, 7, 7, 3.456075e-07): (65, 1),
 (64, 3, 7, 7, 5.52972e-06): (69, 5),
 (64, 3, 11, 11, 2.160046875e-08): (65, 1),
 (64, 3, 11, 11, 3.456075e-07): (66, 2),
 (64, 3, 11, 11, 5.52972e-06): (74, 10),
 (64, 3, 11, 11, 'one-size-fits-all'): (288, 32),
 (64, 16, 1, 1, 2.160046875e-08): (65, 1),
 (64, 16, 1, 1, 3.456075e-07): (65, 1),
 (64, 16, 1, 1, 5.52972e-06): (66, 2),
 (64, 16, 1, 1, 'one-size-fits-all'): (288, 32),
 (64, 16, 3, 3, 2.160046875e-08): (65, 1),
 (64, 16, 3, 3, 3.456075e-07): (65, 1),
 (64, 16, 3, 3, 5.52972e-06): (69, 5),
 (64, 16, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 24, 3, 3, 2.160046875e-08): (65, 1),
 (64, 24, 3, 3, 3.456075e-07): (66, 2),
 (64, 24, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 24, 3, 3, 5.52972e-06): (71, 7),
 (64, 32, 3, 3, 2.160046875e-08): (65, 1),
 (64, 32, 3, 3, 3.456075e-07): (66, 2),
 (64, 32, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 32, 3, 3, 5.52972e-06): (72, 8),
 (64, 48, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 48, 3, 3, 2.160046875e-08): (65, 1),
 (64, 48, 3, 3, 3.456075e-07): (66, 2),
 (64, 48, 3, 3, 5.52972e-06): (75, 11),
 (64, 60, 1, 1, 2.160046875e-08): (65, 1),
 (64, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (64, 60, 1, 1, 3.456075e-07): (65, 1),
 (64, 60, 1, 1, 5.52972e-06): (67, 3),
 (64, 64, 1, 1, 2.160046875e-08): (65, 1),
 (64, 64, 1, 1, 3.456075e-07): (65, 1),
 (64, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (64, 64, 1, 1, 5.52972e-06): (67, 3),
 (64, 64, 3, 3, 3.456075e-07): (66, 2),
 (64, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (64, 64, 3, 3, 2.160046875e-08): (65, 1),
 (64, 64, 3, 3, 5.52972e-06): (78, 14),
 (72, 16, 1, 1, 'one-size-fits-all'): (288, 32),
 (72, 16, 1, 1, 2.160046875e-08): (73, 1),
 (72, 16, 1, 1, 3.456075e-07): (73, 1),
 (72, 16, 1, 1, 5.52972e-06): (74, 2),
 (72, 48, 1, 1, 2.160046875e-08): (73, 1),
 (72, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (72, 48, 1, 1, 3.456075e-07): (73, 1),
 (72, 48, 1, 1, 5.52972e-06): (75, 3),
 (88, 24, 1, 1, 'one-size-fits-all'): (288, 32),
 (88, 24, 1, 1, 2.160046875e-08): (89, 1),
 (88, 24, 1, 1, 3.456075e-07): (89, 1),
 (88, 24, 1, 1, 5.52972e-06): (90, 2),
 (96, 24, 1, 1, 'one-size-fits-all'): (288, 32),
 (96, 24, 1, 1, 2.160046875e-08): (97, 1),
 (96, 24, 1, 1, 3.456075e-07): (97, 1),
 (96, 24, 1, 1, 5.52972e-06): (98, 2),
 (96, 32, 3, 3, 2.160046875e-08): (97, 1),
 (96, 32, 3, 3, 3.456075e-07): (98, 2),
 (96, 32, 3, 3, 5.52972e-06): (107, 11),
 (96, 32, 3, 3, 'one-size-fits-all'): (288, 32),
 (96, 48, 1, 1, 2.160046875e-08): (97, 1),
 (96, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (96, 48, 1, 1, 3.456075e-07): (97, 1),
 (96, 48, 1, 1, 5.52972e-06): (100, 4),
 (96, 60, 1, 1, 3.456075e-07): (97, 1),
 (96, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (96, 60, 1, 1, 2.160046875e-08): (97, 1),
 (96, 60, 1, 1, 5.52972e-06): (100, 4),
 (96, 64, 1, 1, 2.160046875e-08): (97, 1),
 (96, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (96, 64, 1, 1, 3.456075e-07): (97, 1),
 (96, 64, 1, 1, 5.52972e-06): (100, 4),
 (100, 384, 'one-size-fits-all'): (288, 32),
 (100, 384, 2.160046875e-08): (101, 1),
 (100, 384, 3.456075e-07): (102, 2),
 (100, 384, 5.52972e-06): (114, 14),
 (112, 64, 1, 1, 3.456075e-07): (113, 1),
 (112, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (112, 64, 1, 1, 2.160046875e-08): (113, 1),
 (112, 64, 1, 1, 5.52972e-06): (117, 5),
 (120, 32, 1, 1, 2.160046875e-08): (121, 1),
 (120, 32, 1, 1, 'one-size-fits-all'): (288, 32),
 (120, 32, 1, 1, 3.456075e-07): (121, 1),
 (120, 32, 1, 1, 5.52972e-06): (123, 3),
 (120, 40, 1, 1, 2.160046875e-08): (121, 1),
 (120, 40, 1, 1, 'one-size-fits-all'): (288, 32),
 (120, 40, 1, 1, 3.456075e-07): (121, 1),
 (120, 40, 1, 1, 5.52972e-06): (124, 4),
 (128, 32, 1, 1, 2.160046875e-08): (129, 1),
 (128, 32, 1, 1, 3.456075e-07): (129, 1),
 (128, 32, 1, 1, 5.52972e-06): (131, 3),
 (128, 32, 1, 1, 'one-size-fits-all'): (288, 32),
 (128, 32, 3, 3, 2.160046875e-08): (129, 1),
 (128, 32, 3, 3, 3.456075e-07): (130, 2),
 (128, 32, 3, 3, 5.52972e-06): (142, 14),
 (128, 32, 3, 3, 'one-size-fits-all'): (288, 32),
 (128, 48, 1, 1, 2.160046875e-08): (129, 1),
 (128, 48, 1, 1, 3.456075e-07): (129, 1),
 (128, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (128, 48, 1, 1, 5.52972e-06): (132, 4),
 (128, 48, 3, 3, 3.456075e-07): (131, 3),
 (128, 48, 3, 3, 'one-size-fits-all'): (288, 32),
 (128, 48, 3, 3, 2.160046875e-08): (129, 1),
 (128, 48, 3, 3, 5.52972e-06): (147, 19),
 (128, 64, 1, 1, 2.160046875e-08): (129, 1),
 (128, 64, 1, 1, 3.456075e-07): (129, 1),
 (128, 64, 1, 1, 5.52972e-06): (133, 5),
 (128, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (128, 64, 3, 3, 3.456075e-07): (132, 4),
 (128, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (128, 64, 3, 3, 2.160046875e-08): (129, 1),
 (128, 64, 3, 3, 5.52972e-06): (152, 24),
 (144, 40, 1, 1, 2.160046875e-08): (145, 1),
 (144, 40, 1, 1, 'one-size-fits-all'): (288, 32),
 (144, 40, 1, 1, 3.456075e-07): (145, 1),
 (144, 40, 1, 1, 5.52972e-06): (148, 4),
 (144, 48, 1, 1, 2.160046875e-08): (145, 1),
 (144, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (144, 48, 1, 1, 3.456075e-07): (145, 1),
 (144, 48, 1, 1, 5.52972e-06): (148, 4),
 (144, 64, 1, 1, 2.160046875e-08): (145, 1),
 (144, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (144, 64, 1, 1, 3.456075e-07): (145, 1),
 (144, 64, 1, 1, 5.52972e-06): (149, 5),
 (160, 48, 1, 1, 2.160046875e-08): (161, 1),
 (160, 48, 1, 1, 3.456075e-07): (161, 1),
 (160, 48, 1, 1, 5.52972e-06): (165, 5),
 (160, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (160, 64, 1, 1, 3.456075e-07): (161, 1),
 (160, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (160, 64, 1, 1, 2.160046875e-08): (161, 1),
 (160, 64, 1, 1, 5.52972e-06): (166, 6),
 (192, 48, 1, 1, 2.160046875e-08): (193, 1),
 (192, 48, 1, 1, 3.456075e-07): (193, 1),
 (192, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (192, 48, 1, 1, 5.52972e-06): (197, 5),
 (192, 48, 3, 3, 2.160046875e-08): (193, 1),
 (192, 48, 3, 3, 3.456075e-07): (196, 4),
 (192, 48, 3, 3, 'one-size-fits-all'): (288, 32),
 (192, 48, 3, 3, 5.52972e-06): (217, 25),
 (192, 60, 1, 1, 2.160046875e-08): (193, 1),
 (192, 60, 1, 1, 3.456075e-07): (193, 1),
 (192, 60, 1, 1, 5.52972e-06): (198, 6),
 (192, 60, 1, 1, 'one-size-fits-all'): (288, 32),
 (192, 64, 1, 1, 2.160046875e-08): (193, 1),
 (192, 64, 1, 1, 3.456075e-07): (193, 1),
 (192, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (192, 64, 1, 1, 5.52972e-06): (198, 6),
 (192, 64, 3, 3, 3.456075e-07): (196, 4),
 (192, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (192, 64, 3, 3, 2.160046875e-08): (193, 1),
 (192, 64, 3, 3, 5.52972e-06): (225, 33),
 (192, 64, 5, 5, 2.160046875e-08): (194, 2),
 (192, 64, 5, 5, 3.456075e-07): (200, 8),
 (192, 64, 5, 5, 5.52972e-06): (277, 85),
 (192, 64, 5, 5, 'one-size-fits-all'): (288, 32),
 (208, 48, 3, 3, 3.456075e-07): (212, 4),
 (208, 48, 3, 3, 'one-size-fits-all'): (288, 32),
 (208, 48, 3, 3, 2.160046875e-08): (209, 1),
 (208, 48, 3, 3, 5.52972e-06): (235, 27),
 (224, 56, 3, 3, 3.456075e-07): (228, 4),
 (224, 56, 3, 3, 'one-size-fits-all'): (288, 32),
 (224, 56, 3, 3, 2.160046875e-08): (225, 1),
 (224, 56, 3, 3, 5.52972e-06): (257, 33),
 (240, 40, 1, 1, 2.160046875e-08): (241, 1),
 (240, 40, 1, 1, 3.456075e-07): (241, 1),
 (240, 40, 1, 1, 5.52972e-06): (245, 5),
 (240, 40, 1, 1, 'one-size-fits-all'): (288, 32),
 (240, 64, 1, 1, 2.160046875e-08): (241, 1),
 (240, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (240, 64, 1, 1, 3.456075e-07): (242, 2),
 (240, 64, 1, 1, 5.52972e-06): (247, 7),
 (256, 48, 1, 1, 2.160046875e-08): (257, 1),
 (256, 48, 1, 1, 3.456075e-07): (257, 1),
 (256, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (256, 48, 1, 1, 5.52972e-06): (262, 6),
 (256, 64, 1, 1, 2.160046875e-08): (257, 1),
 (256, 64, 1, 1, 3.456075e-07): (258, 2),
 (256, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (256, 64, 1, 1, 5.52972e-06): (264, 8),
 (256, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (256, 64, 3, 3, 2.160046875e-08): (257, 1),
 (256, 64, 3, 3, 3.456075e-07): (261, 5),
 (256, 64, 3, 3, 5.52972e-06): (297, 41),
 (288, 36, 1, 1, 2.160046875e-08): (257, 1),
 (288, 36, 1, 1, 'one-size-fits-all'): (288, 32),
 (288, 36, 1, 1, 3.456075e-07): (257, 1),
 (288, 36, 1, 1, 5.52972e-06): (261, 5),
 (288, 48, 1, 1, 2.160046875e-08): (257, 1),
 (288, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (288, 48, 1, 1, 3.456075e-07): (257, 1),
 (288, 48, 1, 1, 5.52972e-06): (262, 6),
 (288, 48, 3, 3, 2.160046875e-08): (257, 1),
 (288, 48, 3, 3, 3.456075e-07): (260, 4),
 (288, 48, 3, 3, 'one-size-fits-all'): (288, 32),
 (288, 48, 3, 3, 5.52972e-06): (288, 32),
 (320, 40, 3, 3, 2.160046875e-08): (257, 1),
 (320, 40, 3, 3, 3.456075e-07): (260, 4),
 (320, 40, 3, 3, 'one-size-fits-all'): (288, 32),
 (320, 40, 3, 3, 5.52972e-06): (283, 27),
 (384, 64, 1, 1, 2.160046875e-08): (257, 1),
 (384, 64, 1, 1, 3.456075e-07): (258, 2),
 (384, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (384, 64, 1, 1, 5.52972e-06): (264, 8),
 (384, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (384, 64, 3, 3, 2.160046875e-08): (257, 1),
 (384, 64, 3, 3, 3.456075e-07): (261, 5),
 (384, 64, 3, 3, 5.52972e-06): (297, 41),
 (512, 64, 1, 1, 3.456075e-07): (258, 2),
 (512, 64, 1, 1, 5.52972e-06): (264, 8),
 (512, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (512, 64, 1, 1, 2.160046875e-08): (257, 1),
 (512, 64, 3, 3, 3.456075e-07): (261, 5),
 (512, 64, 3, 3, 'one-size-fits-all'): (288, 32),
 (512, 64, 3, 3, 2.160046875e-08): (257, 1),
 (512, 64, 3, 3, 5.52972e-06): (297, 41),
 (576, 48, 1, 1, 2.160046875e-08): (257, 1),
 (576, 48, 1, 1, 'one-size-fits-all'): (288, 32),
 (576, 48, 1, 1, 3.456075e-07): (257, 1),
 (576, 48, 1, 1, 5.52972e-06): (262, 6),
 (1000, 64, 1, 1, 2.160046875e-08): (257, 1),
 (1000, 64, 1, 1, 3.456075e-07): (258, 2),
 (1000, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (1000, 64, 1, 1, 5.52972e-06): (264, 8),
 (1000, 512, 2.160046875e-08): (257, 1),
 (1000, 512, 3.456075e-07): (261, 5),
 (1000, 512, 5.52972e-06): (293, 37),
 (1000, 512, 'one-size-fits-all'): (288, 32),
 (1024, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (1024, 64, 1, 1, 2.160046875e-08): (257, 1),
 (1024, 64, 1, 1, 3.456075e-07): (258, 2),
 (1024, 64, 1, 1, 5.52972e-06): (264, 8),
 (1024, 288, 2.160046875e-08): (257, 1),
 (1024, 288, 3.456075e-07): (259, 3),
 (1024, 288, 5.52972e-06): (279, 23),
 (1024, 288, 'one-size-fits-all'): (288, 32),
 (1024, 512, 5.52972e-06): (293, 37),
 (1024, 512, 'one-size-fits-all'): (288, 32),
 (1024, 512, 2.160046875e-08): (257, 1),
 (1024, 512, 3.456075e-07): (261, 5),
 (2048, 64, 1, 1, 'one-size-fits-all'): (288, 32),
 (2048, 64, 1, 1, 2.160046875e-08): (257, 1),
 (2048, 64, 1, 1, 3.456075e-07): (258, 2),
 (2048, 64, 1, 1, 5.52972e-06): (264, 8),
 (4096, 512, 2.160046875e-08): (257, 1),
 (4096, 512, 'one-size-fits-all'): (288, 32),
 (4096, 512, 3.456075e-07): (261, 5),
 (4096, 512, 5.52972e-06): (293, 37)}



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
        if list(mod.named_children()) and mod.__class__ not in mapping:
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
    if pos == 'all':
        flip_pattern = (1 << 8) - 1
        for i in range(len(b)):
            b[i] ^= flip_pattern
    else:
        [q, r] = divmod(pos, 8)
        flip_pattern = 1 << r
        b[q] ^= flip_pattern
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


def range_restriction(activation_class: [torch.nn.ReLU, torch.nn.Hardswish], clip=True):
    class RangeRestrictionActivation(activation_class):
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
            if clip:
                result *= result != self.bounds[1]
            return result

        @classmethod
        def from_original(cls, original: activation_class):
            return cls()

    return RangeRestrictionActivation


ClipperReLU = range_restriction(torch.nn.ReLU)
ClipperHardswish = range_restriction(torch.nn.Hardswish)
ClipperELU = range_restriction(torch.nn.ELU)

RangerReLU = range_restriction(torch.nn.ReLU, False)
RangerHardswish = range_restriction(torch.nn.Hardswish, False)
RangerELU = range_restriction(torch.nn.ELU, False)


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
        self.ec = ErasureCode(self.n, self.k)
        self.simple_checksum = None
        self.simple_checksum_tensors = None
        self.injected = False
        self.layer = None
        self.detected = False

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d, extra_kwarg=None):
        if extra_kwarg is None:
            extra_kwarg = {}
        else:
            extra_kwarg = copy.copy(extra_kwarg)
            if 'ber' in extra_kwarg:
                _n, _k = BER_OPTIMAL_CONFIGS[tuple(original.weight.shape) + (extra_kwarg.pop('ber'),)]
                extra_kwarg['k'] = _k
                extra_kwarg['n'] = _n - _k

        instance = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                       original.padding, original.dilation, original.groups, original.bias is not None,
                       original.padding_mode, **extra_kwarg)
        coded_weights = instance.sc.code(original.weight)
        instance.weight = torch.nn.Parameter(coded_weights)
        checksum_tensors = instance.weight
        instance.simple_checksum_tensors = checksum_tensors
        instance.simple_checksum = instance.ec.checksum(instance.simple_checksum_tensors)
        return instance

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        decoded = self.sc.decode(self.weight, dim=0)
        if decoded is None:
            self.detected = True
            erasure = self.ec.erasure(self.simple_checksum_tensors, self.simple_checksum)
            decoded = self.sc.decode(self.weight, 0, erasure)
            self.weight[:] = self.sc.code(decoded, 0)
            self.simple_checksum = self.ec.checksum(self.simple_checksum_tensors)
        return super()._conv_forward(input, decoded, bias)


class QStructuralCodedConv2d(torch.nn.qat.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', qconfig=None, k=1, threshold=0.1, n=256):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         qconfig)
        self.k = k
        self.n = n
        self.threshold = threshold
        self.sc = StructuralCode(self.n, self.k, self.threshold, field=IntField())
        self.ec = ErasureCode(self.n, self.k)
        self.simple_checksum = None
        self.simple_checksum_tensors = None
        self.injected = False
        self.layer = None
        self.detected = False

    @classmethod
    def from_original(cls, original: torch.nn.qat.Conv2d, extra_kwarg):
        if extra_kwarg is None:
            extra_kwarg = {}
        instance = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode, original.qconfig, **extra_kwarg)
        instance.weight_fake_quant = original.weight_fake_quant
        instance.activation_post_process = original.activation_post_process
        instance.load_state_dict(original.state_dict())

        coded_weights = dequantize_tensor(instance.sc.code(quantize_tensor(original.weight, instance.weight_fake_quant, False)), instance.weight_fake_quant, False)
        instance.weight = torch.nn.Parameter(coded_weights)
        checksum_tensors = instance.weight
        instance.simple_checksum_tensors = checksum_tensors
        instance.simple_checksum = instance.ec.checksum(quantize_tensor(instance.simple_checksum_tensors, instance.weight_fake_quant, False))

        return instance

    def forward(self, input: Tensor) -> Tensor:
        quantized_weights = quantize_tensor(self.weight, self.weight_fake_quant, False)
        decoded = self.sc.decode(quantized_weights, dim=0)
        if decoded is None:
            self.detected = True
            erasure = self.ec.erasure(quantize_tensor(self.simple_checksum_tensors, self.weight_fake_quant, False), self.simple_checksum)
            decoded = self.sc.decode(quantized_weights, 0, erasure)
            self.weight[:] = dequantize_tensor(self.sc.code(decoded, 0), self.weight_fake_quant, False)
            self.simple_checksum = self.ec.checksum(quantize_tensor(self.simple_checksum_tensors, self.weight_fake_quant, False))
        return self._conv_forward(input, self.weight_fake_quant(dequantize_tensor(decoded, self.weight_fake_quant, False)), self.bias)


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
        self.ec = ErasureCode(self.n, self.k)
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
        else:
            if 'ber' in extra_kwargs:
                extra_kwargs = copy.copy(extra_kwargs)
                _n, _k = BER_OPTIMAL_CONFIGS[tuple(original.weight.shape) + (extra_kwargs.pop('ber'), )]
                extra_kwargs['k'] = _k
                extra_kwargs['n'] = _n - _k
        instance = cls(original.in_features, original.out_features, original.bias is not None, **extra_kwargs)
        coded_weights = instance.sc.code(original.weight)
        instance.weight = torch.nn.Parameter(coded_weights)
        instance.simple_checksum_tensors = instance.weight
        instance.simple_checksum = instance.ec.checksum(instance.simple_checksum_tensors)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        decoded = self.sc.decode(self.weight, dim=0)
        if decoded is None:
            self.detected = True
            erasure = self.ec.erasure(self.simple_checksum_tensors, self.simple_checksum)
            decoded = self.sc.decode(self.weight, 0, erasure)
            self.weight[:] = self.sc.code(decoded, 0)
            self.simple_checksum = self.ec.checksum(self.simple_checksum_tensors)
        return F.linear(input, decoded, self.bias)


class QStructuralCodedLinear(torch.nn.qat.Linear):

    def __init__(self, in_features, out_features, bias=True, qconfig=None, k=1, threshold=0.1, n=256):
        super().__init__(in_features, out_features, bias, qconfig)
        self.injected = False
        self.detected = False
        self.layer = None
        self.k = k
        self.n = n
        self.threshold = threshold
        self.sc = StructuralCode(self.n, self.k, self.threshold, field=IntField())
        self.ec = ErasureCode(self.n, self.k)
        self.simple_checksum = None
        self.simple_checksum_tensors = None

    @classmethod
    def from_original(cls, original: torch.nn.qat.Linear, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        instance = cls(original.in_features, original.out_features, original.bias is not None, original.qconfig,
                     **extra_kwargs)
        instance.weight_fake_quant = original.weight_fake_quant
        instance.activation_post_process = original.activation_post_process
        instance.load_state_dict(original.state_dict())

        coded_weights = dequantize_tensor(instance.sc.code(quantize_tensor(original.weight, instance.weight_fake_quant, False)), instance.weight_fake_quant, False)
        instance.weight = torch.nn.Parameter(coded_weights)
        instance.simple_checksum_tensors = instance.weight

        instance.simple_checksum = instance.ec.checksum(quantize_tensor(instance.simple_checksum_tensors, instance.weight_fake_quant, False))

        return instance

    def forward(self, input: Tensor) -> Tensor:
        quantized_weights = quantize_tensor(self.weight, self.weight_fake_quant, False)
        decoded = self.sc.decode(quantized_weights, dim=0)
        if decoded is None:
            self.detected = True
            erasure = self.ec.erasure(self.simple_checksum_tensors, self.simple_checksum)
            decoded = self.sc.decode(self.weight, 0, erasure)
            self.weight[:] = dequantize_tensor(self.sc.code(decoded, 0), self.weight_fake_quant, False)
            self.simple_checksum = self.ec.checksum(quantize_tensor(self.simple_checksum_tensors, self.weight_fake_quant, False))
        return F.linear(input, self.weight_fake_quant(dequantize_tensor(decoded, self.weight_fake_quant, False)), self.bias)


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
                    convolution.bias = torch.nn.Parameter(original.bias[group_base_index: group_base_index + group_out_channels] / divisions)
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
        # if original.groups > 1:
        #     result = torch.nn.Conv2d(original.in_channels, original.out_channels, original.kernel_size, original.stride,
        #                              original.padding, original.dilation, original.groups, original.bias is not None,
        #                              original.padding_mode)
        #     result.weight = original.weight
        #     result.bias = original.bias
        #     return result
        return cls(original)


class NormalizedConv2dGroups(torch.nn.Module):

    def __init__(self, original):
        super().__init__()
        self.groups = original.groups
        group_in_channels = original.in_channels // original.groups
        self.group_in_channels = group_in_channels
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
                    convolution.bias = torch.nn.Parameter(original.bias[group_base_index: group_base_index + group_out_channels] / divisions)
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
            linear = torch.nn.Linear(division_in_features, original.out_features, original.bias is not None)
            division_base_index = j * division_in_features
            start = division_base_index
            end = start + division_in_features
            weights = original.weight[:, start: end]
            if original.bias is not None:
                linear.bias = torch.nn.Parameter(original.bias / divisions)
            linear.weight = torch.nn.Parameter(weights)
            self.__setattr__('linear_{}'.format(j), linear)

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
        assert torch.all(recover_with_tmr(result.weight) == original.weight)
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        recovered = recover_with_tmr(self.weight)
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
        assert torch.all(recover_with_tmr(result.weight) == original.weight)
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        recovered = recover_with_tmr(self.weight)
        return self._conv_forward(input, recovered, self.bias)


class RADARLinear(torch.nn.qat.Linear):

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)

    @classmethod
    def from_original(cls, original: torch.nn.qat.Linear):
        result = cls(original.in_features, original.out_features, original.bias is not None, original.qconfig)
        result.weight_fake_quant = original.weight_fake_quant
        result.activation_post_process = original.activation_post_process
        result.load_state_dict(original.state_dict())
        result.backup = original.weight.clone()
        return result

    def forward(self, input: Tensor) -> Tensor:
        current_checksum = radar_checksum(quantize_tensor(self.weight, self.weight_fake_quant))
        original_checksum = radar_checksum(quantize_tensor(self.backup, self.weight_fake_quant))
        self.weight *= (original_checksum == current_checksum)
        return super().forward(input)


class RADARConv2d(torch.nn.qat.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         qconfig)

    @classmethod
    def from_original(cls, original: torch.nn.qat.Conv2d):
        result = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode, original.qconfig)
        result.weight_fake_quant = original.weight_fake_quant
        result.activation_post_process = original.activation_post_process
        result.load_state_dict(original.state_dict())
        result.backup = original.weight.clone()
        return result

    def forward(self, input: Tensor) -> Tensor:
        current_checksum = radar_checksum(quantize_tensor(self.weight, self.weight_fake_quant))
        original_checksum = radar_checksum(quantize_tensor(self.backup, self.weight_fake_quant))
        self.weight *= (original_checksum == current_checksum)
        return super().forward(input)


class FRADARLinear(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        result = cls(original.in_features, original.out_features, original.bias is not None)
        result.weight = original.weight
        result.weight_redundancy = torch.nn.Parameter(fradar_checksum(original.weight))
        return result

    def forward(self, input: Tensor) -> Tensor:
        recovered = recover_with_fradar(self.weight, self.weight_redundancy)
        return F.linear(input, recovered, self.bias)


class FRADARConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        result = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode)
        result.weight = original.weight
        result.weight_redundancy = torch.nn.Parameter(fradar_checksum(original.weight))
        result.bias = original.bias
        return result

    def forward(self, input: Tensor) -> Tensor:
        recovered = recover_with_fradar(self.weight, self.weight_redundancy)
        return self._conv_forward(input, recovered, self.bias)


MILR_BATCH_SIZE = 4


class MILRLinear(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_original(cls, original: torch.nn.Linear):
        result = cls(original.in_features, original.out_features, original.bias is not None)
        result.weight = original.weight
        result.bias = original.bias
        random_input = cls.get_random_input(result)
        result.checkpoint = original.forward(random_input) - original.bias
        return result

    def get_random_input(self):
        rnd = numpy.random.RandomState(2021)
        random_input = rnd.random((MILR_BATCH_SIZE, self.in_features))
        return torch.FloatTensor(random_input) * 1000

    def forward(self, input: Tensor) -> Tensor:
        detection = torch.max(super().forward(self.get_random_input()) - self.bias != self.checkpoint, 0).values
        if torch.any(detection):
            for i in range(self.weight.shape[0]):
                if detection[i]:
                    self.weight[i] = torch.nn.Parameter(torch.matmul(
                        torch.pinverse(self.get_random_input()),
                        self.checkpoint[:, i],
                    ))
            self.checkpoint = super().forward(self.get_random_input()) - self.bias
        return super().forward(input)


class MILRConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def get_random_input(self):
        rnd = numpy.random.RandomState(2021)
        random_input = rnd.random((MILR_BATCH_SIZE, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        return torch.FloatTensor(random_input)

    @classmethod
    def from_original(cls, original: torch.nn.Conv2d):
        result = cls(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                     original.padding, original.dilation, original.groups, original.bias is not None,
                     original.padding_mode)
        result.weight = original.weight
        result.bias = original.bias
        result.checkpoint = original._conv_forward(result.get_random_input(), original.weight, None)
        return result

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        detection = torch.amax(self.checkpoint != super()._conv_forward(self.get_random_input(), weight, None), (0, 2, 3))
        if torch.any(detection):
            weight = weight.clone()
            n = reduce(operator.mul, weight.shape[1:])
            identity = torch.zeros((n,) + weight.shape[1:])
            for i in range(n):
                identity[i].view(n)[i] = 1
            for i in range(weight.shape[0]):
                if detection[i]:
                    transformation = super()._conv_forward(self.get_random_input(), identity, None).view(n, -1)
                    checkpoint_channel = self.checkpoint[:, i].reshape(1, -1)
                    inverse_transformation = torch.pinverse(transformation)
                    original_weight = torch.matmul(checkpoint_channel, inverse_transformation)
                    self.weight[i] = original_weight.view(*weight.shape[1:])
            weight = self.weight
            self.checkpoint = super()._conv_forward(self.get_random_input(), weight, None)
        return super()._conv_forward(input, weight, bias)



