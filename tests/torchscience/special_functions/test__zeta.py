import math

import pytest
import torch
import torch.testing
from scipy import special as scipy_special

import torchscience.special_functions


class TestZeta:
    """Tests for the Riemann zeta function."""

    def test_known_values(self):
        """Test zeta at known values."""
        s = torch.tensor([2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)

        # Known values
        expected = torch.tensor(
            [
                math.pi**2 / 6,  # zeta(2) = Basel problem
                1.2020569031595943,  # zeta(3) = Apery's constant
                math.pi**4 / 90,  # zeta(4)
                math.pi**6 / 945,  # zeta(6)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_large_s_approaches_one(self):
        """Test that zeta(s) -> 1 as s -> infinity."""
        s = torch.tensor([10.0, 20.0, 30.0, 50.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        # zeta(10) ~ 1.00099, zeta(20) ~ 1.000001, etc.
        # The function converges to 1, but slowly
        assert (result > 1).all(), "zeta(s) should be > 1 for all s > 1"
        assert (result < 1.001).all(), (
            "zeta(s) should be close to 1 for large s"
        )
        # Check decreasing trend towards 1
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1], (
                f"zeta should decrease towards 1 as s increases"
            )

    def test_pole_at_one(self):
        """Test that zeta(1) returns infinity."""
        s = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        assert torch.isinf(result).all()

    def test_invalid_domain_returns_nan(self):
        """Test that s <= 1 (except s=1) returns NaN."""
        s = torch.tensor([0.5, 0.0, -1.0, -2.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        assert torch.isnan(result).all()

    def test_comparison_with_scipy(self):
        """Test agreement with scipy.special.zeta for s > 1."""
        s_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
        s = torch.tensor(s_values, dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)

        # scipy.special.zeta takes (x, q) where default q=1 gives Riemann zeta
        expected = torch.tensor(
            [scipy_special.zeta(sv, 1) for sv in s_values],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_monotonically_decreasing(self):
        """Test that zeta(s) decreases as s increases for s > 1."""
        s = torch.tensor([1.5, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        # Should be strictly decreasing
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1], (
                f"zeta({s[i].item()}) = {result[i].item()} should be > "
                f"zeta({s[i + 1].item()}) = {result[i + 1].item()}"
            )

    def test_all_positive_for_s_greater_than_one(self):
        """Test that zeta(s) > 0 for s > 1."""
        s = torch.tensor([1.1, 1.5, 2.0, 3.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        assert (result > 0).all()

    def test_gradient(self):
        """Test that gradients exist and are correct sign."""
        s = torch.tensor(
            [2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.zeta(s)
        y.sum().backward()

        # Derivative of zeta is negative for s > 1
        assert (s.grad < 0).all(), "d/ds zeta(s) should be negative for s > 1"

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        s = torch.tensor(
            [1.5, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.zeta(s)
        y.sum().backward()
        assert torch.isfinite(s.grad).all()

    def test_second_order_gradient(self):
        """Test second-order gradients exist and are finite."""
        s = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.zeta(s)

        (grad1,) = torch.autograd.grad(y, s, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, s)

        assert torch.isfinite(grad2).all()
        # Second derivative should be positive (zeta is convex for s > 1)
        assert grad2.item() > 0

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        s = torch.tensor(
            [2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.zeta,
            (s,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        s = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.zeta,
            (s,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_float32_precision(self):
        """Test float32 inputs."""
        s = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        result = torchscience.special_functions.zeta(s)
        expected = torch.tensor(
            [math.pi**2 / 6, 1.2020569, math.pi**4 / 90],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_broadcasting(self):
        """Test that broadcasting works."""
        s = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        assert result.shape == s.shape
        assert torch.isfinite(result).all()

    def test_empty_tensor(self):
        """Test empty tensor input."""
        s = torch.tensor([], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        assert result.shape == torch.Size([0])

    def test_meta_tensor(self):
        """Test meta tensor support."""
        s = torch.tensor([2.0, 3.0], dtype=torch.float64, device="meta")
        result = torchscience.special_functions.zeta(s)
        assert result.device.type == "meta"
        assert result.shape == s.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Test CUDA support."""
        s = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, device="cuda")
        result = torchscience.special_functions.zeta(s)
        assert result.device.type == "cuda"
        expected_cpu = torchscience.special_functions.zeta(s.cpu())
        torch.testing.assert_close(
            result.cpu(), expected_cpu, rtol=1e-7, atol=1e-7
        )

    def test_near_pole_stability(self):
        """Test numerical stability near the pole at s=1."""
        s = torch.tensor([1.01, 1.001, 1.0001], dtype=torch.float64)
        result = torchscience.special_functions.zeta(s)
        # Values should be large and finite
        assert torch.isfinite(result).all()
        # zeta(1+eps) ~ 1/eps + gamma as eps -> 0
        # So zeta(1.01) ~ 100, zeta(1.001) ~ 1000, etc.
        assert (result > 10).all()

    def test_derivative_near_pole(self):
        """Test derivative stability near the pole."""
        s = torch.tensor([1.1, 1.5], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.zeta(s)
        y.sum().backward()
        assert torch.isfinite(s.grad).all()
