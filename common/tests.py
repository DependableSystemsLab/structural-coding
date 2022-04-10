import abc
from unittest import TestCase

from analysis import sdc
from linearcode.map import evaluate_config


class TestSDC:
    baseline_config = None
    config = None
    message = None

    @abc.abstractmethod
    def criterion(self, s, e):
        pass

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.baseline = evaluate_config(cls.baseline_config)

    def setUp(self) -> None:
        super().setUp()

    def get_config(self):
        return {
            **self.baseline_config,
            **self.config,
        }

    def test_sdc(self):
        evaluation = evaluate_config(self.get_config())
        s, e = sdc(self.baseline, evaluation)
        self.assertTrue(self.criterion(s, e), msg=self.message)