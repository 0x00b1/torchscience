"""Constants for filter design module."""

import math

# Butterworth filter Q factor (1/sqrt(2))
# This gives maximally flat passband response
Q_BUTTERWORTH: float = 1.0 / math.sqrt(2.0)  # 0.7071067811865476

# Common Q factor values for audio filters
Q_WIDE: float = 0.5
Q_MEDIUM: float = 1.0
Q_NARROW: float = 2.0
