from unittest import TestCase, skip

from common.tests import TestSDC
from settings import PROBABILITIES


class TestBERCorrupts(TestSDC, TestCase):
    __test__ = False
    baseline_config = {'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny',
                       'dataset': 'imagenet', 'flips': 0, 'protection': 'none'}
    config = {'flips': PROBABILITIES[0]}
    message = 'BER did not cause SDC.'

    def criterion(self, s, e):
        return s > 0


class TestSCProtects(TestSDC, TestCase):
    __test__ = False
    baseline_config = {'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny',
                       'dataset': 'imagenet', 'flips': 0, 'protection': 'none'}
    config = {'flips': PROBABILITIES[0], 'protection': 'sc'}
    message = 'BER did not cause SDC.'

    def criterion(self, s, e):
        return s == 0


class TestQSCProtects(TestSDC, TestCase):
    baseline_config = {'injection': 0, 'model': 'resnet50', 'quantization': False, 'sampler': 'tiny',
                       'dataset': 'imagenet', 'flips': 0, 'protection': 'none'}
    config = {'flips': 'row', 'protection': 'sc'}
    message = 'BER did not cause SDC.'

    def criterion(self, s, e):
        return s == 0


class TestQBERCorrupts(TestSDC, TestCase):
    baseline_config = {'injection': 0, 'model': 'resnet50', 'quantization': True, 'sampler': 'tiny',
                       'dataset': 'imagenet', 'flips': 0, 'protection': 'none'}
    config = {'flips': 'row', 'protection': 'none'}
    message = 'BER did not cause SDC.'

    def criterion(self, s, e):
        return s > 0
