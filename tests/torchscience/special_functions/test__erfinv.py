import pytest
import scipy.special
import torch
import torch.testing

import torchscience.special_functions


class TestErfinv:
    """Tests for the inverse error function erfinv(x)."""

    # =========================================================================
    # Forward correctness tests
    # =========================================================================

    def test_special_values(self):
        """Test special values: erfinv(0) = 0, erfinv(nan) = nan."""
        # erfinv(0) = 0
        result = torchscience.special_functions.erfinv(
            torch.tensor(0.0, dtype=torch.float64)
        )
        assert result.item() == pytest.approx(0.0, abs=1e-15)

        # erfinv(nan) = nan
        assert torchscience.special_functions.erfinv(
            torch.tensor(float("nan"))
        ).isnan()

    def test_boundary_values(self):
        """Test boundary values: erfinv(+/-1) = +/-inf."""
        # erfinv(1) = +inf
        result = torchscience.special_functions.erfinv(
            torch.tensor(1.0, dtype=torch.float64)
        )
        assert result.item() == float("inf")

        # erfinv(-1) = -inf
        result = torchscience.special_functions.erfinv(
            torch.tensor(-1.0, dtype=torch.float64)
        )
        assert result.item() == float("-inf")

    def test_out_of_domain(self):
        """Test out-of-domain values return NaN."""
        x = torch.tensor([1.5, -1.5, 2.0, -2.0], dtype=torch.float64)
        result = torchscience.special_functions.erfinv(x)
        assert result.isnan().all()

    def test_scipy_agreement(self):
        """Test agreement with scipy.special.erfinv."""
        x = torch.tensor(
            [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9], dtype=torch.float64
        )
        result = torchscience.special_functions.erfinv(x)
        expected = torch.tensor(
            [scipy.special.erfinv(xi.item()) for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    def test_scipy_agreement_near_boundary(self):
        """Test agreement with scipy near the boundaries."""
        x = torch.tensor(
            [0.99, 0.999, 0.9999, -0.99, -0.999, -0.9999], dtype=torch.float64
        )
        result = torchscience.special_functions.erfinv(x)
        expected = torch.tensor(
            [scipy.special.erfinv(xi.item()) for xi in x], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_roundtrip_erf_erfinv(self):
        """Test roundtrip: erf(erfinv(x)) = x."""
        x = torch.tensor(
            [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9], dtype=torch.float64
        )
        result = torch.erf(torchscience.special_functions.erfinv(x))
        torch.testing.assert_close(result, x, rtol=1e-10, atol=1e-12)

    def test_roundtrip_erfinv_erf(self):
        """Test roundtrip: erfinv(erf(y)) = y for reasonable y values."""
        y = torch.tensor(
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64
        )
        result = torchscience.special_functions.erfinv(torch.erf(y))
        torch.testing.assert_close(result, y, rtol=1e-10, atol=1e-12)

    def test_symmetry(self):
        """Test odd symmetry: erfinv(-x) = -erfinv(x)."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        result_pos = torchscience.special_functions.erfinv(x)
        result_neg = torchscience.special_functions.erfinv(-x)
        torch.testing.assert_close(
            result_neg, -result_pos, rtol=1e-14, atol=1e-14
        )

    def test_known_value(self):
        """Test known value: erfinv(0.5) ~ 0.4769362762044699."""
        result = torchscience.special_functions.erfinv(
            torch.tensor(0.5, dtype=torch.float64)
        )
        expected = 0.4769362762044699
        assert result.item() == pytest.approx(expected, rel=1e-10)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradcheck(self):
        """Test first-order gradient correctness."""
        x = torch.tensor(
            [-0.5, -0.1, 0.1, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(
            torchscience.special_functions.erfinv, x, eps=1e-6
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.tensor(
            [-0.5, -0.1, 0.1, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradgradcheck(
            torchscience.special_functions.erfinv, x, eps=1e-6
        )

    def test_gradient_formula(self):
        """Verify gradient formula: d/dx erfinv(x) = sqrt(pi)/2 * exp(erfinv(x)^2)."""
        import math

        x = torch.tensor(
            [-0.5, 0.0, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.special_functions.erfinv(x)
        grad = torch.autograd.grad(y.sum(), x)[0]

        # Expected: sqrt(pi)/2 * exp(y^2)
        sqrt_pi_over_2 = math.sqrt(math.pi) / 2
        expected = sqrt_pi_over_2 * torch.exp(y.detach() ** 2)
        torch.testing.assert_close(grad, expected, rtol=1e-8, atol=1e-10)

    # =========================================================================
    # Backend tests
    # =========================================================================

    def test_meta_tensor(self):
        """Test meta tensor shape inference."""
        x = torch.randn(10, device="meta")
        result = torchscience.special_functions.erfinv(x)
        assert result.shape == x.shape
        assert result.device == x.device

    def test_autocast(self):
        """Test autocast (mixed precision) support."""
        x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = torchscience.special_functions.erfinv(x)
        # Special functions use float32 for accuracy under autocast
        assert result.dtype == torch.float32
        # Verify results match non-autocast version
        expected = torchscience.special_functions.erfinv(x)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # PyTorch integration tests
    # =========================================================================

    def test_vmap(self):
        """Verify vmap compatibility."""
        x = (
            torch.rand(5, 10, dtype=torch.float64) * 1.8 - 0.9
        )  # Range [-0.9, 0.9]
        result = torch.vmap(torchscience.special_functions.erfinv)(x)
        expected = torchscience.special_functions.erfinv(x)
        torch.testing.assert_close(result, expected)

    def test_compile(self):
        """Verify torch.compile compatibility."""
        compiled_fn = torch.compile(torchscience.special_functions.erfinv)
        x = torch.rand(100, dtype=torch.float64) * 1.8 - 0.9
        result = compiled_fn(x)
        expected = torchscience.special_functions.erfinv(x)
        torch.testing.assert_close(result, expected)

    def test_compile_with_autograd(self):
        """Verify torch.compile works with gradients."""
        compiled_fn = torch.compile(torchscience.special_functions.erfinv)
        x = (torch.rand(100, dtype=torch.float64) * 1.8 - 0.9).requires_grad_(
            True
        )
        result = compiled_fn(x)
        result.sum().backward()
        assert x.grad is not None
        # Verify gradient matches uncompiled version
        x2 = x.detach().clone().requires_grad_(True)
        expected = torchscience.special_functions.erfinv(x2)
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x2.grad)

    def test_broadcasting(self):
        """Verify broadcasting works correctly."""
        x1 = torch.rand(3, 1, dtype=torch.float64) * 0.8
        x2 = torch.rand(1, 4, dtype=torch.float64) * 0.8
        # x1 + x2 will have values in [0, 1.6] which may be outside domain
        # So use values that stay in domain
        combined = x1 * 0.5 + x2 * 0.5  # Stays in [0, 0.8]
        result = torchscience.special_functions.erfinv(combined)
        assert result.shape == (3, 4)

    # =========================================================================
    # dtype tests
    # =========================================================================

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_float_dtypes(self, dtype):
        """Test various floating point dtypes."""
        x = torch.tensor([0.1, 0.5, 0.9], dtype=dtype)
        result = torchscience.special_functions.erfinv(x)
        assert result.dtype == dtype
