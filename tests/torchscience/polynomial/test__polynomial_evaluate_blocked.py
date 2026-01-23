# tests/torchscience/polynomial/test__polynomial_evaluate_blocked.py
import torch

from torchscience.polynomial import polynomial, polynomial_evaluate


class TestPolynomialEvaluateBlocked:
    """Tests for blocked Clenshaw evaluation."""

    def test_evaluate_large_points(self):
        """Evaluation with many points is correct."""
        p = polynomial(torch.randn(10, dtype=torch.float64))
        x = torch.linspace(-1, 1, 10000, dtype=torch.float64)

        result = polynomial_evaluate(p, x)

        # Verify a sample of points
        for i in [0, 1000, 5000, 9999]:
            expected = sum(p[j] * x[i] ** j for j in range(len(p)))
            torch.testing.assert_close(
                result[i], expected, atol=1e-10, rtol=1e-10
            )

    def test_evaluate_high_degree_large_points(self):
        """High-degree polynomial with many points."""
        p = polynomial(torch.randn(100, dtype=torch.float64))
        x = torch.linspace(-0.5, 0.5, 100000, dtype=torch.float64)

        result = polynomial_evaluate(p, x)
        assert result.shape == (100000,)

    def test_evaluate_batched_large_points(self):
        """Batched evaluation with many points."""
        p = polynomial(
            torch.randn(5, 10, dtype=torch.float64)
        )  # 5 polynomials
        x = torch.linspace(-1, 1, 10000, dtype=torch.float64)

        result = polynomial_evaluate(p, x)
        assert result.shape == (5, 10000)
