# tests/torchscience/root_finding/test__convergence.py
import torch

from torchscience.root_finding._convergence import (
    check_convergence,
    default_tolerances,
)


class TestDefaultTolerances:
    """Tests for dtype-aware default tolerances."""

    def test_float64_tolerances(self):
        """float64 has tightest tolerances."""
        tols = default_tolerances(torch.float64)
        assert tols["xtol"] == 1e-12
        assert tols["rtol"] == 1e-9
        assert tols["ftol"] == 1e-12

    def test_float32_tolerances(self):
        """float32 has medium tolerances."""
        tols = default_tolerances(torch.float32)
        assert tols["xtol"] == 1e-6
        assert tols["rtol"] == 1e-5
        assert tols["ftol"] == 1e-6

    def test_float16_tolerances(self):
        """float16 has loose tolerances."""
        tols = default_tolerances(torch.float16)
        assert tols["xtol"] == 1e-3
        assert tols["rtol"] == 1e-2
        assert tols["ftol"] == 1e-3

    def test_bfloat16_tolerances(self):
        """bfloat16 has loose tolerances."""
        tols = default_tolerances(torch.bfloat16)
        assert tols["xtol"] == 1e-3
        assert tols["rtol"] == 1e-2
        assert tols["ftol"] == 1e-3


class TestCheckConvergence:
    """Tests for convergence checking."""

    def test_converged_by_xtol(self):
        """Converged when x change is small."""
        x_old = torch.tensor([1.0, 2.0, 3.0])
        x_new = torch.tensor([1.0 + 1e-8, 2.0, 3.0 + 1e-8])
        f_new = torch.tensor([1.0, 1.0, 1.0])  # Large f, but x converged

        converged = check_convergence(
            x_old, x_new, f_new, xtol=1e-6, rtol=0.0, ftol=0.0
        )

        assert converged.tolist() == [True, True, True]

    def test_converged_by_ftol(self):
        """Converged when f is small."""
        x_old = torch.tensor([1.0, 2.0])
        x_new = torch.tensor([2.0, 3.0])  # Large x change
        f_new = torch.tensor([1e-10, 1e-10])  # Small f

        converged = check_convergence(
            x_old, x_new, f_new, xtol=0.0, rtol=0.0, ftol=1e-6
        )

        assert converged.tolist() == [True, True]

    def test_converged_by_rtol(self):
        """Converged when relative x change is small."""
        x_old = torch.tensor([100.0, 1000.0])
        x_new = torch.tensor([100.0001, 1000.001])  # Small relative change
        f_new = torch.tensor([1.0, 1.0])  # Large f

        converged = check_convergence(
            x_old, x_new, f_new, xtol=0.0, rtol=1e-5, ftol=0.0
        )

        assert converged.tolist() == [True, True]

    def test_not_converged(self):
        """Not converged when neither condition met."""
        x_old = torch.tensor([1.0])
        x_new = torch.tensor([2.0])  # Large change
        f_new = torch.tensor([1.0])  # Large f

        converged = check_convergence(
            x_old, x_new, f_new, xtol=1e-6, rtol=1e-6, ftol=1e-6
        )

        assert converged.tolist() == [False]

    def test_mixed_convergence(self):
        """Some elements converged, others not."""
        x_old = torch.tensor([1.0, 1.0, 1.0])
        x_new = torch.tensor([1.0 + 1e-8, 2.0, 1.5])  # Third: large x change
        f_new = torch.tensor([1.0, 1e-10, 1.0])

        converged = check_convergence(
            x_old, x_new, f_new, xtol=1e-6, rtol=0.0, ftol=1e-6
        )

        # First: x converged, Second: f converged, Third: neither
        assert converged.tolist() == [True, True, False]
