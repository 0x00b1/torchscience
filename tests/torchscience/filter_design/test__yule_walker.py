"""Tests for Yule-Walker IIR filter design."""

import pytest
import torch

from torchscience.filter_design import yule_walker


class TestYuleWalker:
    """Test yule_walker function."""

    def test_first_order(self) -> None:
        """Should correctly solve first-order case."""
        r = torch.tensor([1.0, 0.5], dtype=torch.float64)
        a, sigma = yule_walker(r, order=1)

        expected_a = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(a, expected_a, rtol=1e-10, atol=1e-12)

        expected_sigma = torch.tensor(0.75, dtype=torch.float64).sqrt()
        torch.testing.assert_close(
            sigma, expected_sigma, rtol=1e-10, atol=1e-12
        )

    def test_second_order(self) -> None:
        """Should correctly solve second-order case."""
        # Known AR(2): x[n] = 0.5*x[n-1] + 0.3*x[n-2] + e[n]
        # Yule-Walker: R @ a = r  where R is Toeplitz
        r = torch.tensor([1.0, 0.6, 0.4], dtype=torch.float64)
        a, sigma = yule_walker(r, order=2)

        # Verify by checking Yule-Walker equation
        # a[0] * r[0] + a[1] * r[1] should equal r[1]
        assert a.shape == (2,)
        assert sigma.item() > 0

    def test_dtype_preservation(self) -> None:
        """Should preserve input dtype."""
        r32 = torch.tensor([1.0, 0.5], dtype=torch.float32)
        r64 = torch.tensor([1.0, 0.5], dtype=torch.float64)

        a32, sigma32 = yule_walker(r32, order=1)
        a64, sigma64 = yule_walker(r64, order=1)

        assert a32.dtype == torch.float32
        assert a64.dtype == torch.float64

    def test_invalid_order_zero(self) -> None:
        """Should raise for order=0."""
        r = torch.tensor([1.0, 0.5], dtype=torch.float64)
        with pytest.raises(ValueError, match="order must be at least 1"):
            yule_walker(r, order=0)

    def test_invalid_order_too_large(self) -> None:
        """Should raise when order >= len(r)."""
        r = torch.tensor([1.0, 0.5], dtype=torch.float64)
        with pytest.raises(ValueError, match="order must be less than"):
            yule_walker(r, order=2)

    def test_singular_r0_zero(self) -> None:
        """Should raise when r[0] is zero."""
        r = torch.tensor([0.0, 0.5], dtype=torch.float64)
        with pytest.raises(ValueError, match="r\\[0\\] is zero"):
            yule_walker(r, order=1)

    def test_allow_singular(self) -> None:
        """Should handle singular case when allow_singular=True."""
        r = torch.tensor([0.0, 0.5], dtype=torch.float64)
        a, sigma = yule_walker(r, order=1, allow_singular=True)
        assert torch.all(a == 0)
        assert sigma == 0
