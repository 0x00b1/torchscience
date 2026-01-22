"""Mathematical morphology operations.

This module provides N-dimensional morphological operations with full PyTorch
integration including autograd, torch.compile, autocast, and vmap support.

Operations
----------
erosion : Morphological erosion (minimum over structuring element).
dilation : Morphological dilation (maximum over structuring element).
opening : Erosion followed by dilation.
closing : Dilation followed by erosion.
"""

from torchscience.morphology._closing import closing
from torchscience.morphology._dilation import dilation
from torchscience.morphology._erosion import erosion
from torchscience.morphology._opening import opening

__all__ = [
    "closing",
    "dilation",
    "erosion",
    "opening",
]
