"""Hypothesis strategies for PyTorch operator testing."""

from ._all_complex_dtypes import all_complex_dtypes
from ._all_dtypes import all_dtypes
from ._all_floating_dtypes import all_floating_dtypes
from ._available_devices import available_devices
from ._avoiding_poles import avoiding_poles
from ._avoiding_values import avoiding_values
from ._broadcastable_shapes import broadcastable_shapes
from ._broadcastable_tensor_pairs import broadcastable_tensor_pairs
from ._chebyshev_domain import chebyshev_domain
from ._complex_avoiding_real_axis import complex_avoiding_real_axis
from ._complex_dtypes import complex_dtypes
from ._complex_numbers import complex_numbers
from ._hyperbolic_domain import hyperbolic_domain
from ._integers_as_floats import integers_as_floats
from ._low_precision_dtypes import low_precision_dtypes
from ._negative_real_numbers import negative_real_numbers
from ._non_integer_real_numbers import non_integer_real_numbers
from ._positive_real_numbers import positive_real_numbers
from ._real_number_dtypes import real_number_dtypes
from ._real_numbers import real_numbers
from ._recurrence_test_values import recurrence_test_values
from ._reflection_test_values import reflection_test_values
from ._shapes import shapes
from ._tensors import tensors

__all__ = [
    # Numeric strategies
    "positive_real_numbers",
    "negative_real_numbers",
    "real_numbers",
    "non_integer_real_numbers",
    "integers_as_floats",
    "chebyshev_domain",
    "hyperbolic_domain",
    "avoiding_values",
    "avoiding_poles",
    # Complex strategies
    "complex_numbers",
    "complex_avoiding_real_axis",
    # Tensor strategies
    "shapes",
    "tensors",
    "broadcastable_shapes",
    "broadcastable_tensor_pairs",
    "recurrence_test_values",
    "reflection_test_values",
    # Dtype strategies
    "real_number_dtypes",
    "complex_dtypes",
    "low_precision_dtypes",
    "all_floating_dtypes",
    "all_complex_dtypes",
    "all_dtypes",
    # Device strategies
    "available_devices",
]
