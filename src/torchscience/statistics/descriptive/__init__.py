"""Descriptive statistics functions.

This module provides functions for computing descriptive statistics
such as kurtosis, skewness, and other moments of distributions.
"""

from ._histogram import histogram
from ._kurtosis import kurtosis

__all__ = [
    "histogram",
    "kurtosis",
]
