import pytest
import torch
import torch.testing

import torchscience.special_functions

# Optional: mpmath for high-precision reference
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def reference_u(a, z):
    """Reference implementation using mpmath."""
    if not HAS_MPMATH:
        return None
    mpmath.mp.dps = 50
    result = mpmath.pcfu(float(a), float(z))
    return complex(result)


class TestParabolicCylinderU:
    """Tests for the parabolic cylinder function U(a, z)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not installed")
    def test_mpmath_agreement_basic(self):
        """Test agreement with mpmath for basic values."""
        test_cases = [
            (0.0, 1.0),
            (0.5, 2.0),
            (1.0, 0.5),
            (-0.5, 1.5),
            (2.0, 3.0),
        ]
        for a_val, z_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = torchscience.special_functions.parabolic_cylinder_u(a, z)
            expected = reference_u(a_val, z_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected.real], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-8,
                msg=f"Failed for a={a_val}, z={z_val}",
            )

    def test_special_value_at_origin(self):
        """Test U(0, 0) = Gamma(1/4) / (2^(3/4) * sqrt(pi))."""
        import math

        a = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        # U(0,0) ≈ 0.7978845608
        expected = math.gamma(0.25) / (2**0.75 * math.sqrt(math.pi))
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-6,
        )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck_z(self):
        """Test first-order gradient correctness for z."""
        a = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z_):
            return torchscience.special_functions.parabolic_cylinder_u(a, z_)

        assert torch.autograd.gradcheck(
            func, z, eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_both_inputs(self):
        """Test first-order gradient correctness for both a and z."""
        a = torch.tensor([0.5, 1.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(a_, z_):
            return torchscience.special_functions.parabolic_cylinder_u(a_, z_)

        assert torch.autograd.gradcheck(
            func, (a, z), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradgradcheck_z(self):
        """Test second-order gradient correctness for z."""
        a = torch.tensor([0.0, 0.5], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(z_):
            return torchscience.special_functions.parabolic_cylinder_u(a, z_)

        assert torch.autograd.gradgradcheck(
            func, z, eps=1e-6, atol=1e-3, rtol=1e-3
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting between a and z."""
        a = torch.tensor([[0.0], [1.0]], dtype=torch.float64)  # (2, 1)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        assert result.shape == (2, 3)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        a = torch.tensor([0.0, 1.0], dtype=dtype)
        z = torch.tensor([1.0, 2.0], dtype=dtype)
        result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        a = torch.tensor([0.0, 1.0], dtype=dtype)
        z = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=dtype)
        result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        assert result.dtype == dtype

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        a = torch.randn(10, device="meta")
        z = torch.randn(10, device="meta")
        result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        assert result.shape == (10,)
        assert result.device.type == "meta"

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        a = torch.tensor([0.0, 1.0], dtype=torch.float32)
        z = torch.tensor([1.0, 2.0], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.parabolic_cylinder_u(a, z)
        assert result.dtype == torch.float32

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        a = torch.tensor([0.0, 1.0], dtype=torch.float64)
        z = torch.randn(5, 2, dtype=torch.float64)

        def fn(z_row):
            return torchscience.special_functions.parabolic_cylinder_u(
                a, z_row
            )

        result = torch.vmap(fn)(z)
        assert result.shape == (5, 2)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.special_functions.parabolic_cylinder_u
        )
        a = torch.tensor([0.0, 1.0], dtype=torch.float64)
        z = torch.randn(2, dtype=torch.float64)
        result = compiled_fn(a, z)
        expected = torchscience.special_functions.parabolic_cylinder_u(a, z)
        torch.testing.assert_close(result, expected)
