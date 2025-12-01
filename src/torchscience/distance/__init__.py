"""Statistical distance functions for probability distributions.

This module provides distance metrics between probability distributions
with full autograd support.

Functions
---------
hellinger_distance
    Hellinger distance (symmetric, bounded [0, 1]).
total_variation_distance
    Total variation distance (symmetric, bounded [0, 1]).
bhattacharyya_distance
    Bhattacharyya distance (symmetric, non-negative).
"""

from ._bhattacharyya_distance import bhattacharyya_distance
from ._hellinger_distance import hellinger_distance
from ._minkowski_distance import minkowski_distance
from ._total_variation_distance import total_variation_distance

__all__ = [
    "bhattacharyya_distance",
    "hellinger_distance",
    "minkowski_distance",
    "total_variation_distance",
]
