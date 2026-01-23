"""Tensor product spline for 2D interpolation."""

from ._tensor_product_derivative import tensor_product_derivative
from ._tensor_product_evaluate import tensor_product_evaluate
from ._tensor_product_fit import tensor_product_fit
from ._tensor_product_spline import TensorProductSpline, tensor_product_spline

__all__ = [
    "TensorProductSpline",
    "tensor_product_derivative",
    "tensor_product_evaluate",
    "tensor_product_fit",
    "tensor_product_spline",
]
