from __future__ import division, print_function, generators, unicode_literals
import numpy as np
from typing import *


class AVGMeter(object):

    def __init__(self, precition: int = 6, log_freq=None):
        self.precision = precition
        self.log_freq = log_freq
        self.sum = 0
        self.count = 0
        self.values = []
        self.reset()

    def append(self, x: float):
        self.values.append(x)
        self.sum += x
        self.count += 1

    def reset(self):
        self.sum = 0.0  # type: float
        self.count = 0  # type: int
        self.values = []  # type: List[float]

    def avg_of_last(self, last_size=2) -> float:
        x = np.mean(self.values[-last_size:])  # type: float
        return float(round(x, self.precision))

    @property
    def avg(self) -> float:
        return round(self.sum / self.count, self.precision)

    @property
    def avgl(self) -> float:
        assert self.log_freq is not None
        return self.avg_of_last(self.log_freq)

    @property
    def last(self) -> Optional[float]:
        if len(self.values) > 0:
            return self.values[-1]

    @property
    def clen(self) -> Optional[int]:
        return len(self.values)

    def __str__(self) -> str:
        return '{:.{p}f}'.format(self.avg, p=self.precision)
