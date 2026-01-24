"""Shared types for transform module."""

from typing import Literal

# Re-export PaddingMode from pad module for convenience
from torchscience.pad._pad import PaddingMode

# Normalization mode for FFT-based transforms
NormMode = Literal["forward", "backward", "ortho"]

__all__ = ["PaddingMode", "NormMode"]
