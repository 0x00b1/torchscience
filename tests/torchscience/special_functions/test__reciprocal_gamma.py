import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestReciprocalGamma:
    """Tests for the reciprocal gamma function."""

    def test_positive_integers(self):
        """Test 1/Gamma(n) = 1/(n-1)! for positive integers."""
        z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        result = torchscience.special_functions.reciprocal_gamma(z)
        # 1/Gamma(1)=1, 1/Gamma(2)=1, 1/Gamma(3)=1/2, 1/Gamma(4)=1/6, etc.
        expected = torch.tensor(
            [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_poles_return_zero(self):
        """Test that 1/Gamma at poles (non-positive integers) returns zero."""
        poles = torch.tensor(
            [0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float64
        )
        result = torchscience.special_functions.reciprocal_gamma(poles)
        expected = torch.zeros(6, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_half_integer_values(self):
        """Test 1/Gamma at half-integer values."""
        sqrt_pi = math.sqrt(math.pi)
        z = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        result = torchscience.special_functions.reciprocal_gamma(z)
        # 1/Gamma(0.5) = 1/sqrt(pi), 1/Gamma(1.5) = 2/sqrt(pi), 1/Gamma(2.5) = 4/(3*sqrt(pi))
        expected = torch.tensor(
            [1.0 / sqrt_pi, 2.0 / sqrt_pi, 4.0 / (3.0 * sqrt_pi)],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_non_integers(self):
        """Test 1/Gamma at negative non-integer values."""
        sqrt_pi = math.sqrt(math.pi)
        z = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float64)
        result = torchscience.special_functions.reciprocal_gamma(z)
        # 1/Gamma(-0.5) = -1/(2*sqrt(pi))
        # 1/Gamma(-1.5) = 3/(4*sqrt(pi))
        # 1/Gamma(-2.5) = -15/(8*sqrt(pi))
        expected = torch.tensor(
            [
                -1.0 / (2.0 * sqrt_pi),
                3.0 / (4.0 * sqrt_pi),
                -15.0 / (8.0 * sqrt_pi),
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_comparison_with_inverse_gamma(self):
        """Test that reciprocal_gamma(z) == 1/gamma(z) for non-poles."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.reciprocal_gamma(z)
        expected = 1.0 / torchscience.special_functions.gamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_values(self):
        """Test reciprocal gamma with complex values."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result = torchscience.special_functions.reciprocal_gamma(z)
        expected = 1.0 / torchscience.special_functions.gamma(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_at_poles(self):
        """Test complex reciprocal gamma at poles returns zero."""
        z = torch.tensor(
            [0.0 + 0.0j, -1.0 + 0.0j, -2.0 + 0.0j], dtype=torch.complex128
        )
        result = torchscience.special_functions.reciprocal_gamma(z)
        expected = torch.zeros(3, dtype=torch.complex128)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_gradcheck(self):
        """Test gradient correctness via torch.autograd.gradcheck."""
        z = torch.tensor(
            [0.5, 1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True
        )

        def func(x):
            return torchscience.special_functions.reciprocal_gamma(x)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-5, rtol=1e-5
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness via torch.autograd.gradgradcheck."""
        z = torch.tensor(
            [1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True
        )

        def func(x):
            return torchscience.special_functions.reciprocal_gamma(x)

        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradient_formula_manual(self):
        """Manually verify backward formula: d/dz [1/Gamma(z)] = -psi(z)/Gamma(z)."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.reciprocal_gamma(z)
        y.backward()

        # Expected: -digamma(2) * reciprocal_gamma(2)
        psi_2 = torchscience.special_functions.digamma(
            torch.tensor([2.0], dtype=torch.float64)
        )
        rg_2 = torchscience.special_functions.reciprocal_gamma(
            torch.tensor([2.0], dtype=torch.float64)
        )
        expected_grad = -psi_2 * rg_2

        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_meta_tensor_support(self):
        """Test that meta tensors are supported for shape inference."""
        z = torch.empty(3, 4, dtype=torch.float64, device="meta")
        result = torchscience.special_functions.reciprocal_gamma(z)
        assert result.shape == (3, 4)
        assert result.device == torch.device("meta")
        assert result.dtype == torch.float64

    def test_complex_conjugate_symmetry(self):
        """Test 1/Gamma(conj(z)) = conj(1/Gamma(z))."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result_z = torchscience.special_functions.reciprocal_gamma(z)
        result_conj_z = torchscience.special_functions.reciprocal_gamma(
            z.conj()
        )
        torch.testing.assert_close(
            result_conj_z, result_z.conj(), rtol=1e-10, atol=1e-10
        )

    def test_dtypes(self):
        """Test that various floating-point dtypes are supported."""
        for dtype in [torch.float32, torch.float64]:
            z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            result = torchscience.special_functions.reciprocal_gamma(z)
            assert result.dtype == dtype
            assert torch.isfinite(result).all()

    def test_complex_dtypes(self):
        """Test that complex dtypes are supported."""
        for dtype in [torch.complex64, torch.complex128]:
            z = torch.tensor([1.0 + 0.5j, 2.0 - 0.3j], dtype=dtype)
            result = torchscience.special_functions.reciprocal_gamma(z)
            assert result.dtype == dtype
            assert torch.isfinite(result.real).all()
            assert torch.isfinite(result.imag).all()

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        z1 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)  # Shape (2, 1)
        z2 = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float64
        )  # Shape (1, 3)

        # Just test that the shapes broadcast correctly
        result1 = torchscience.special_functions.reciprocal_gamma(z1)
        result2 = torchscience.special_functions.reciprocal_gamma(z2)
        assert result1.shape == (2, 1)
        assert result2.shape == (1, 3)

    def test_large_positive_values(self):
        """Test with large positive values (should approach zero)."""
        z = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        result = torchscience.special_functions.reciprocal_gamma(z)
        # 1/Gamma(n) gets very small for large n
        assert torch.isfinite(result).all()
        assert (result > 0).all()
        # Should be monotonically decreasing
        assert result[0] > result[1] > result[2]

    def test_relation_to_gamma(self):
        """Test that reciprocal_gamma(z) * gamma(z) = 1 for non-poles."""
        z = torch.tensor(
            [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64
        )
        product = torchscience.special_functions.reciprocal_gamma(
            z
        ) * torchscience.special_functions.gamma(z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(product, expected, rtol=1e-10, atol=1e-10)

    def test_entire_function_property(self):
        """Test that 1/Gamma is entire (finite everywhere)."""
        # Test a grid including poles of gamma
        z_values = torch.tensor(
            [0.0, -1.0, -2.0, -3.0, 0.5, -0.5, 1.0, 2.0, 100.0],
            dtype=torch.float64,
        )
        result = torchscience.special_functions.reciprocal_gamma(z_values)
        # All results should be finite (including zeros at poles)
        assert torch.isfinite(result).all()

    def test_gradient_near_poles(self):
        """Test that gradients near (but not at) poles are finite."""
        z = torch.tensor(
            [0.001, -0.999, -1.999], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.reciprocal_gamma(z)
        y.sum().backward()
        assert torch.isfinite(z.grad).all()

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_dtype_requires_conversion(self, int_dtype):
        """Test that integer inputs require explicit conversion to float."""
        z_int = torch.tensor([1, 2, 3], dtype=int_dtype)
        with pytest.raises(NotImplementedError):
            torchscience.special_functions.reciprocal_gamma(z_int)
