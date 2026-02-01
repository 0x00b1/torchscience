import math

import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestLogMultivariateGamma:
    """Tests for the log_multivariate_gamma function."""

    def test_d_equals_1_matches_log_gamma(self):
        """Test that d=1 equals log_gamma(a)."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0, 10.0], dtype=torch.float64)
        result = torchscience.special_functions.log_multivariate_gamma(a, 1)
        expected = torchscience.special_functions.log_gamma(a)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_known_values_d2(self):
        """Test against known values for d=2."""
        # log(Gamma_2(a)) = (d*(d-1)/4)*log(pi) + log_gamma(a) + log_gamma(a - 0.5)
        # For d=2: (2*1/4)*log(pi) = 0.5*log(pi)
        a = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.log_multivariate_gamma(a, 2)

        # Compute expected value manually
        d = 2
        expected = (
            (d * (d - 1) / 4) * math.log(math.pi)
            + torchscience.special_functions.log_gamma(a)
            + torchscience.special_functions.log_gamma(a - 0.5)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_known_values_d3(self):
        """Test against known values for d=3."""
        # log(Gamma_3(a)) = (d*(d-1)/4)*log(pi) + log_gamma(a) + log_gamma(a - 0.5) + log_gamma(a - 1)
        # For d=3: (3*2/4)*log(pi) = 1.5*log(pi)
        a = torch.tensor([2.5, 3.0, 5.0], dtype=torch.float64)
        result = torchscience.special_functions.log_multivariate_gamma(a, 3)

        # Compute expected value manually
        d = 3
        expected = (
            (d * (d - 1) / 4) * math.log(math.pi)
            + torchscience.special_functions.log_gamma(a)
            + torchscience.special_functions.log_gamma(a - 0.5)
            + torchscience.special_functions.log_gamma(a - 1.0)
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_comparison_with_scipy(self):
        """Test agreement with scipy.special.multigammaln."""
        scipy_special = pytest.importorskip("scipy.special")

        a_values = [2.0, 3.0, 5.0, 10.0]
        d_values = [1, 2, 3, 5]

        for d in d_values:
            # Ensure a > (d-1)/2
            valid_a = [av for av in a_values if av > (d - 1) / 2]
            if not valid_a:
                continue

            a = torch.tensor(valid_a, dtype=torch.float64)
            result = torchscience.special_functions.log_multivariate_gamma(
                a, d
            )

            # Compare with scipy
            for i, av in enumerate(valid_a):
                expected = scipy_special.multigammaln(av, d)
                torch.testing.assert_close(
                    result[i],
                    torch.tensor(expected, dtype=torch.float64),
                    rtol=1e-8,
                    atol=1e-8,
                )

    def test_gradcheck(self):
        """Test first-order gradients with gradcheck."""
        # a must be > (d-1)/2, so for d=3, a > 1.0
        a = torch.tensor(
            [2.5, 3.5, 5.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a):
            return torchscience.special_functions.log_multivariate_gamma(a, 3)

        torch.autograd.gradcheck(fn, (a,), eps=1e-5, atol=1e-4, rtol=1e-4)

    def test_gradgradcheck(self):
        """Test second-order gradients with gradgradcheck."""
        a = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)

        def fn(a):
            return torchscience.special_functions.log_multivariate_gamma(a, 2)

        torch.autograd.gradgradcheck(fn, (a,), eps=1e-5, atol=1e-4, rtol=1e-4)

    def test_gradient_equals_sum_of_digammas(self):
        """Test d/da log(Gamma_d(a)) = sum of digamma terms."""
        a = torch.tensor([3.0, 5.0], dtype=torch.float64, requires_grad=True)
        d = 3

        y = torchscience.special_functions.log_multivariate_gamma(a, d)
        y.sum().backward()

        # Expected: sum_{j=1}^{d} digamma(a + (1-j)/2)
        expected_grad = torch.zeros_like(a.detach())
        for j in range(1, d + 1):
            expected_grad += torch.special.digamma(a.detach() + (1 - j) / 2)

        torch.testing.assert_close(a.grad, expected_grad, rtol=1e-8, atol=1e-8)

    def test_meta_tensor_support(self):
        """Test that meta tensor shapes are inferred correctly."""
        a = torch.empty(3, 4, device="meta")
        result = torchscience.special_functions.log_multivariate_gamma(a, 2)
        assert result.shape == a.shape
        assert result.device.type == "meta"

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        a = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
        d = 2
        result = torchscience.special_functions.log_multivariate_gamma(a, d)
        assert result.shape == a.shape

        # Verify each element
        for i in range(2):
            for k in range(2):
                expected = (
                    (d * (d - 1) / 4) * math.log(math.pi)
                    + torchscience.special_functions.log_gamma(
                        torch.tensor([a[i, k].item()], dtype=torch.float64)
                    )
                    + torchscience.special_functions.log_gamma(
                        torch.tensor(
                            [a[i, k].item() - 0.5], dtype=torch.float64
                        )
                    )
                )
                torch.testing.assert_close(
                    result[i, k],
                    expected.squeeze(),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_float32(self):
        """Test with float32 dtype."""
        a = torch.tensor([3.0, 5.0], dtype=torch.float32)
        result = torchscience.special_functions.log_multivariate_gamma(a, 2)
        assert result.dtype == torch.float32

        # Compare with float64 result
        a64 = a.to(torch.float64)
        result64 = torchscience.special_functions.log_multivariate_gamma(
            a64, 2
        )
        torch.testing.assert_close(
            result.to(torch.float64), result64, rtol=1e-5, atol=1e-5
        )

    def test_different_d_values(self):
        """Test with various dimension values."""
        a = torch.tensor([10.0], dtype=torch.float64)

        for d in [1, 2, 3, 4, 5, 10]:
            result = torchscience.special_functions.log_multivariate_gamma(
                a, d
            )
            assert torch.isfinite(result).all(), f"Failed for d={d}"

            # Manually compute expected
            expected = (d * (d - 1) / 4) * math.log(math.pi)
            for j in range(1, d + 1):
                expected += torchscience.special_functions.log_gamma(
                    torch.tensor([a.item() + (1 - j) / 2], dtype=torch.float64)
                ).item()

            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_large_d(self):
        """Test with larger dimension values."""
        a = torch.tensor([50.0], dtype=torch.float64)
        d = 20

        result = torchscience.special_functions.log_multivariate_gamma(a, d)
        assert torch.isfinite(result).all()

    def test_second_order_gradient_finite(self):
        """Test that second-order gradients are finite."""
        a = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.log_multivariate_gamma(a, 2)

        (grad1,) = torch.autograd.grad(y, a, create_graph=True)
        (grad2,) = torch.autograd.grad(grad1, a)

        assert torch.isfinite(grad2).all()
        # Second derivative should be sum of trigammas, which are positive
        assert grad2.item() > 0
