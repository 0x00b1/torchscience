"""Integral transforms for signal processing.

This module provides implementations of various integral transforms
commonly used in signal processing and mathematical physics.

Functions
---------
hilbert_transform
    Compute the Hilbert transform of a signal with optional padding and windowing.
inverse_hilbert_transform
    Compute the inverse Hilbert transform of a signal.
"""

from ._hilbert_transform import hilbert_transform
from ._inverse_hilbert_transform import inverse_hilbert_transform

__all__ = [
    "hilbert_transform",
    "inverse_hilbert_transform",
]
