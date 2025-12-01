from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor

from torchscience.testing.sympy_utils import SymPyReference


class SymPyReferenceMixin:
    """Mixin providing SymPy reference verification tests."""

    descriptor: "OperatorDescriptor"

    def test_sympy_reference_real(self):
        """Verify real values match SymPy reference."""
        if "test_sympy_reference_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.sympy_func is None:
            pytest.skip("No SymPy function provided")

        inputs = self._make_standard_inputs(dtype=torch.float64, shape=(5,))
        result = self.descriptor.func(*inputs)

        reference = SymPyReference(self.descriptor.sympy_func)
        expected = reference.to_torch(*inputs, dtype=result.dtype)

        rtol = self.descriptor.tolerances.sympy_rtol
        atol = self.descriptor.tolerances.sympy_atol
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_sympy_reference_complex(self):
        """Verify complex values match SymPy reference."""
        if "test_sympy_reference_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.sympy_func is None:
            pytest.skip("No SymPy function provided")

        inputs = self._make_standard_inputs(dtype=torch.complex128, shape=(5,))
        result = self.descriptor.func(*inputs)

        reference = SymPyReference(self.descriptor.sympy_func)
        expected = reference.to_torch(*inputs, dtype=result.dtype)

        rtol = (
            self.descriptor.tolerances.sympy_rtol * 10
        )  # Relaxed for complex
        atol = self.descriptor.tolerances.sympy_atol * 10
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
