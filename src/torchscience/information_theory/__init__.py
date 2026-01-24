"""Compatibility alias for the legacy `torchscience.information_theory` package.

This module re-exports information theory operators from
`torchscience.information` to maintain backward compatibility with tests and
user imports that still reference `torchscience.information_theory`.
"""

from torchscience.information import (
    chi_squared_divergence,
    conditional_entropy,
    cross_entropy,
    f_divergence,
    jensen_shannon_divergence,
    joint_entropy,
    kullback_leibler_divergence,
    mutual_information,
    pointwise_mutual_information,
    renyi_divergence,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)

__all__ = [
    "chi_squared_divergence",
    "conditional_entropy",
    "cross_entropy",
    "f_divergence",
    "jensen_shannon_divergence",
    "joint_entropy",
    "kullback_leibler_divergence",
    "mutual_information",
    "pointwise_mutual_information",
    "renyi_divergence",
    "renyi_entropy",
    "shannon_entropy",
    "tsallis_entropy",
]
