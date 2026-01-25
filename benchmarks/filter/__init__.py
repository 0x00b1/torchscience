"""Benchmarks for filter design functions.

This module provides benchmarking utilities and benchmark classes for
comparing torchscience filter design functions against scipy baselines.
"""

from .bench_adaptive_filters import BenchAdaptiveFilters
from .bench_filter_application import BenchFilterApplication
from .bench_filter_design import BenchFilterDesign

__all__ = [
    "BenchFilterDesign",
    "BenchFilterApplication",
    "BenchAdaptiveFilters",
]
