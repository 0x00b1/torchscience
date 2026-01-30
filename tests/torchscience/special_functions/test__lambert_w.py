import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestLambertW:
    """Tests for the Lambert W function."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_w_0_equals_0(self):
        """Test W(0) = 0 for principal branch."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_w_e_equals_1(self):
        """Test W_0(e) = 1."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([math.e], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_branch_point(self):
        """Test W(-1/e) = -1 at the branch point."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([-1.0 / math.e], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        expected = torch.tensor([-1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_inverse_property(self):
        """Test W(z) * exp(W(z)) = z (defining property)."""
        k = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        w = torchscience.special_functions.lambert_w(k, z)
        reconstructed = w * torch.exp(w)
        torch.testing.assert_close(reconstructed, z, rtol=1e-10, atol=1e-10)

    def test_principal_branch_positive(self):
        """Test principal branch for positive z values."""
        k = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-10, atol=1e-10)

    def test_principal_branch_negative(self):
        """Test principal branch for negative z in valid range."""
        k = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        # z must be >= -1/e for principal branch
        z = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-10, atol=1e-10)

    def test_secondary_branch(self):
        """Test secondary branch (k=-1) for valid range."""
        k = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        # z must be in [-1/e, 0) for secondary branch
        z = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-8, atol=1e-8)
        # Secondary branch should give more negative values
        assert (result < -1).all()

    def test_branches_differ(self):
        """Test that k=0 and k=-1 give different results for same z."""
        z_val = -0.2
        k0 = torch.tensor([0.0], dtype=torch.float64)
        km1 = torch.tensor([-1.0], dtype=torch.float64)
        z = torch.tensor([z_val], dtype=torch.float64)

        w0 = torchscience.special_functions.lambert_w(k0, z)
        wm1 = torchscience.special_functions.lambert_w(km1, z)

        # Both should satisfy the defining property
        torch.testing.assert_close(
            w0 * torch.exp(w0), z, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            wm1 * torch.exp(wm1), z, rtol=1e-8, atol=1e-8
        )

        # But they should be different
        assert not torch.allclose(w0, wm1)
        # Principal branch: W_0 in [-1, 0) for z in (-1/e, 0)
        assert w0.item() >= -1 and w0.item() < 0
        # Secondary branch: W_{-1} < -1 for z in (-1/e, 0)
        assert wm1.item() < -1

    def test_known_value_w_1(self):
        """Test W_0(1) ~ 0.5671432904097838."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Known value from Wolfram Alpha / scipy
        expected = torch.tensor([0.5671432904097838], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Test forward pass for float dtypes."""
        k = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = torchscience.special_functions.lambert_w(k, z)
        assert result.dtype == dtype
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test forward pass for complex dtypes."""
        k = torch.tensor([0.0 + 0.0j, 0.0 + 0.0j], dtype=dtype)
        z = torch.tensor([1.0 + 0.0j, 1.0 + 1.0j], dtype=dtype)
        result = torchscience.special_functions.lambert_w(k, z)
        assert result.dtype == dtype
        # Just check it runs without error and produces finite values
        assert torch.isfinite(result).all()

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        # Note: k is a discrete parameter with zero gradient
        # We only check gradient w.r.t. z
        k = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        z = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def func(z):
            return torchscience.special_functions.lambert_w(k, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        k = torch.tensor([0.0, 0.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(z):
            return torchscience.special_functions.lambert_w(k, z)

        assert torch.autograd.gradgradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradient_values(self):
        """Test gradient values against analytical formula."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.lambert_w(k, z)
        result.backward()

        # d/dz W(z) = W(z) / (z * (1 + W(z)))
        w = result.detach()
        expected_grad = w / (z.detach() * (1 + w))

        torch.testing.assert_close(
            z.grad, expected_grad, rtol=1e-6, atol=1e-10
        )

    def test_gradient_at_z_1(self):
        """Test gradient at z=1."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        result = torchscience.special_functions.lambert_w(k, z)
        result.backward()

        # W(1) ~ 0.5671432904097838
        # W'(1) = W(1) / (1 * (1 + W(1))) ~ 0.3618
        w = 0.5671432904097838
        expected_grad = torch.tensor([w / (1 + w)], dtype=torch.float64)

        torch.testing.assert_close(
            z.grad, expected_grad, rtol=1e-4, atol=1e-10
        )

    # =========================================================================
    # Broadcasting tests
    # =========================================================================

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        k = torch.tensor([[0.0], [0.0]], dtype=torch.float64)  # (2, 1)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        result = torchscience.special_functions.lambert_w(k, z)
        assert result.shape == (2, 3)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(
            reconstructed, z.expand(2, 3), rtol=1e-10, atol=1e-10
        )

    def test_scalar_k(self):
        """Test with scalar k and vector z."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_large_values(self):
        """Test with large z values."""
        k = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        z = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-8, atol=1e-8)

    def test_small_positive_values(self):
        """Test with small positive z values."""
        k = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        z = torch.tensor([0.01, 0.001, 0.0001], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        # For small z, W(z) ~ z - z^2
        # Verify via inverse property
        reconstructed = result * torch.exp(result)
        torch.testing.assert_close(reconstructed, z, rtol=1e-10, atol=1e-10)

    def test_invalid_principal_branch(self):
        """Test that principal branch returns NaN for z < -1/e."""
        k = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([-0.5], dtype=torch.float64)  # Below -1/e ~ -0.368
        result = torchscience.special_functions.lambert_w(k, z)
        assert torch.isnan(result).all()

    def test_invalid_secondary_branch_positive(self):
        """Test that secondary branch returns NaN for z >= 0."""
        k = torch.tensor([-1.0], dtype=torch.float64)
        z = torch.tensor([0.1], dtype=torch.float64)
        result = torchscience.special_functions.lambert_w(k, z)
        assert torch.isnan(result).all()

    # =========================================================================
    # Complex input tests
    # =========================================================================

    def test_complex_inverse_property(self):
        """Test W(z) * exp(W(z)) = z for complex inputs."""
        k = torch.tensor([0.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
        z = torch.tensor([1.0 + 1.0j, 2.0 - 0.5j], dtype=torch.complex128)
        w = torchscience.special_functions.lambert_w(k, z)
        reconstructed = w * torch.exp(w)
        torch.testing.assert_close(reconstructed, z, rtol=1e-8, atol=1e-8)

    def test_complex_real_axis(self):
        """Test complex inputs on real axis match real results."""
        k_real = torch.tensor([0.0], dtype=torch.float64)
        z_real = torch.tensor([2.0], dtype=torch.float64)
        k_complex = k_real.to(torch.complex128)
        z_complex = z_real.to(torch.complex128)

        result_real = torchscience.special_functions.lambert_w(k_real, z_real)
        result_complex = torchscience.special_functions.lambert_w(
            k_complex, z_complex
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

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test with meta tensors (shape inference only)."""
        k = torch.empty(3, 4, device="meta")
        z = torch.empty(3, 4, device="meta")
        result = torchscience.special_functions.lambert_w(k, z)
        assert result.device.type == "meta"
        assert result.shape == (3, 4)
