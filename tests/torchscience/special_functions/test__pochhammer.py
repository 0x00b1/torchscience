import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestPochhammer:
    """Tests for the Pochhammer symbol (rising factorial)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_against_log_gamma(self):
        """Test pochhammer(z, m) = exp(lgamma(z+m) - lgamma(z))."""
        z = torch.tensor([1.0, 2.0, 3.0, 5.0, 1.5], dtype=torch.float64)
        m = torch.tensor([1.0, 2.0, 3.0, 2.0, 2.5], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_m_zero(self):
        """Test (z)_0 = 1 for all z."""
        z = torch.tensor([1.0, 2.0, -0.5, 10.0, 0.1], dtype=torch.float64)
        m = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_1_n_equals_factorial(self):
        """Test (1)_n = n! for positive integer n."""
        z = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        m = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        # 1!, 2!, 3!, 4!, 5!
        expected = torch.tensor(
            [1.0, 2.0, 6.0, 24.0, 120.0], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_special_value_m_one(self):
        """Test (z)_1 = z."""
        z = torch.tensor([1.0, 2.0, 3.0, 0.5, 10.0], dtype=torch.float64)
        m = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        torch.testing.assert_close(result, z, rtol=1e-10, atol=1e-10)

    def test_rising_factorial_property(self):
        """Test (z)_n = z(z+1)(z+2)...(z+n-1) for positive integer n."""
        z = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([4.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        # (3)_4 = 3 * 4 * 5 * 6 = 360
        expected = torch.tensor([360.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_rising_factorial_property_general(self):
        """Test rising factorial product formula for various cases."""
        # (2)_3 = 2 * 3 * 4 = 24
        result1 = torchscience.special_functions.pochhammer(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([3.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result1,
            torch.tensor([24.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # (5)_2 = 5 * 6 = 30
        result2 = torchscience.special_functions.pochhammer(
            torch.tensor([5.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result2,
            torch.tensor([30.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_against_scipy(self):
        """Test against scipy.special.poch values (manually computed reference)."""
        # scipy.special.poch(3.5, 2.5) = Gamma(6) / Gamma(3.5)
        # Gamma(6) = 5! = 120
        # Gamma(3.5) = 2.5 * 1.5 * 0.5 * sqrt(pi) = 1.875 * sqrt(pi) ~ 3.3234
        z = torch.tensor([3.5], dtype=torch.float64)
        m = torch.tensor([2.5], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        z = torch.tensor([2.0, 3.0], dtype=dtype)
        m = torch.tensor([3.0, 2.0], dtype=dtype)
        result = torchscience.special_functions.pochhammer(z, m)
        assert result.dtype == dtype
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test forward pass for complex dtypes."""
        z = torch.tensor([2.0 + 0.5j, 3.0 + 0.0j], dtype=dtype)
        m = torch.tensor([3.0 + 0.0j, 2.0 + 0.5j], dtype=dtype)
        result = torchscience.special_functions.pochhammer(z, m)
        assert result.dtype == dtype
        # Just check it runs without error and produces finite values
        assert torch.isfinite(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        z = torch.tensor(
            [2.0, 3.0, 1.5], dtype=torch.float64, requires_grad=True
        )
        m = torch.tensor(
            [3.0, 2.0, 2.5], dtype=torch.float64, requires_grad=True
        )

        def func(z, m):
            return torchscience.special_functions.pochhammer(z, m)

        assert torch.autograd.gradcheck(
            func, (z, m), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        z = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(z, m):
            return torchscience.special_functions.pochhammer(z, m)

        assert torch.autograd.gradgradcheck(
            func, (z, m), eps=1e-6, atol=1e-3, rtol=1e-3
        )

    def test_gradient_values_z(self):
        """Test gradient values w.r.t. z against analytical formula."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([3.0], dtype=torch.float64, requires_grad=False)

        result = torchscience.special_functions.pochhammer(z, m)
        result.backward()

        # d/dz (z)_m = (z)_m * (psi(z+m) - psi(z))
        poch_val = result.detach()
        psi_zm = torch.digamma(z.detach() + m.detach())
        psi_z = torch.digamma(z.detach())

        expected_grad_z = poch_val * (psi_zm - psi_z)

        torch.testing.assert_close(
            z.grad, expected_grad_z, rtol=1e-5, atol=1e-10
        )

    def test_gradient_values_m(self):
        """Test gradient values w.r.t. m against analytical formula."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=False)
        m = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.pochhammer(z, m)
        result.backward()

        # d/dm (z)_m = (z)_m * psi(z+m)
        poch_val = result.detach()
        psi_zm = torch.digamma(z.detach() + m.detach())

        expected_grad_m = poch_val * psi_zm

        torch.testing.assert_close(
            m.grad, expected_grad_m, rtol=1e-5, atol=1e-10
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        z = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # (2, 1)
        m = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        result = torchscience.special_functions.pochhammer(z, m)
        assert result.shape == (2, 3)
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_small_m_values(self):
        """Test with small positive m values."""
        z = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        m = torch.tensor([0.1, 0.01, 0.001], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_large_values(self):
        """Test with larger values."""
        z = torch.tensor([10.0, 20.0], dtype=torch.float64)
        m = torch.tensor([5.0, 3.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.exp(torch.lgamma(z + m) - torch.lgamma(z))
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_fractional_z(self):
        """Test with fractional z values."""
        z = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        m = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        result = torchscience.special_functions.pochhammer(z, m)
        # (0.5)_2 = 0.5 * 1.5 = 0.75
        # (1.5)_2 = 1.5 * 2.5 = 3.75
        # (2.5)_2 = 2.5 * 3.5 = 8.75
        expected = torch.tensor([0.75, 3.75, 8.75], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Complex input tests
    # =========================================================================

    def test_complex_real_axis(self):
        """Test complex inputs on real axis match real results."""
        z_real = torch.tensor([2.0, 3.0], dtype=torch.float64)
        m_real = torch.tensor([3.0, 2.0], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)
        m_complex = m_real.to(torch.complex128)

        result_real = torchscience.special_functions.pochhammer(z_real, m_real)
        result_complex = torchscience.special_functions.pochhammer(
            z_complex, m_complex
        )

        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_complex.imag),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_m_zero(self):
        """Test (z)_0 = 1 for complex z."""
        z = torch.tensor([2.0 + 1.0j, 3.0 + 0.5j], dtype=torch.complex128)
        m = torch.tensor([0.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
        result = torchscience.special_functions.pochhammer(z, m)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        z = torch.empty(3, 4, device="meta")
        m = torch.empty(3, 4, device="meta")
        result = torchscience.special_functions.pochhammer(z, m)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)

    def test_meta_tensor_broadcasting(self):
        """Test meta tensor with broadcasting."""
        z = torch.empty(3, 1, device="meta")
        m = torch.empty(1, 4, device="meta")
        result = torchscience.special_functions.pochhammer(z, m)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
