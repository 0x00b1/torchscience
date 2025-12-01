from typing import TYPE_CHECKING, Tuple

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class AutogradMixin:
    """Mixin providing autograd tests for operators."""

    descriptor: "OperatorDescriptor"

    def _make_gradcheck_inputs(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...] = (3,),
    ) -> Tuple[torch.Tensor, ...]:
        """Generate inputs suitable for gradient checking."""
        inputs = []
        for spec in self.descriptor.input_specs:
            tensor = self._make_input_for_spec(spec, dtype, "cpu", shape)
            if (
                spec.supports_grad
                and dtype.is_floating_point
                or dtype.is_complex
            ):
                tensor = tensor.requires_grad_(True)
            inputs.append(tensor)
        return tuple(inputs)

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gradcheck_real(self, dtype: torch.dtype):
        """Test gradient correctness for real dtypes using torch.autograd.gradcheck."""
        if "test_gradcheck_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradcheck(
            func,
            inputs,
            eps=tol.gradcheck_eps,
            atol=tol.gradcheck_atol,
            rtol=tol.gradcheck_rtol,
        )

    @pytest.mark.parametrize("dtype", [torch.complex128])
    def test_gradcheck_complex(self, dtype: torch.dtype):
        """Test gradient correctness for complex dtypes."""
        if "test_gradcheck_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradcheck(
            func,
            inputs,
            eps=tol.gradcheck_eps,
            atol=tol.gradcheck_atol * 10,  # Relaxed for complex
            rtol=tol.gradcheck_rtol * 10,
        )

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gradgradcheck_real(self, dtype: torch.dtype):
        """Test second-order gradient correctness for real dtypes."""
        if "test_gradgradcheck_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradgradcheck(
            func,
            inputs,
            eps=tol.gradgradcheck_eps,
            atol=tol.gradgradcheck_atol,
            rtol=tol.gradgradcheck_rtol,
        )

    @pytest.mark.parametrize("dtype", [torch.complex128])
    def test_gradgradcheck_complex(self, dtype: torch.dtype):
        """Test second-order gradient correctness for complex dtypes."""
        if "test_gradgradcheck_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradgradcheck(
            func,
            inputs,
            eps=tol.gradgradcheck_eps,
            atol=tol.gradgradcheck_atol * 10,  # Relaxed for complex
            rtol=tol.gradgradcheck_rtol * 10,
        )
