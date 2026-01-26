import pytest
import torch
import torch.testing

from torchscience.optimization.minimization import minimize


class TestMinimizeDispatch:
    @pytest.mark.parametrize(
        "method",
        [
            "l-bfgs",
            "conjugate-gradient",
            "newton-cg",
            "trust-region",
            "nelder-mead",
        ],
    )
    def test_quadratic_all_methods(self, method):
        """All methods should solve a simple quadratic."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([3.0, 4.0]), method=method)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""

        def f(x):
            return (x**2).sum()

        with pytest.raises(ValueError, match="Unknown method"):
            minimize(f, torch.tensor([1.0]), method="unknown")
