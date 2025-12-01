"""Tests for Richardson extrapolation."""

import math

import torch

from torchscience.differentiation import richardson_extrapolation


class TestRichardsonExtrapolation:
    """Tests for Richardson extrapolation."""

    def test_improves_first_derivative(self):
        """Richardson extrapolation improves first derivative accuracy."""
        x = 1.0
        true_deriv = math.cos(x)

        def central_diff(h):
            return (math.sin(x + h) - math.sin(x - h)) / (2 * h)

        h = 0.1
        no_extrap = central_diff(h)
        no_extrap_error = abs(no_extrap - true_deriv)

        with_extrap = richardson_extrapolation(
            f=lambda h: torch.tensor(central_diff(h)),
            h=h,
            order=2,
            levels=2,
        )
        with_extrap_error = abs(with_extrap.item() - true_deriv)

        assert with_extrap_error < no_extrap_error / 10

    def test_improves_second_derivative(self):
        """Richardson extrapolation improves second derivative accuracy."""
        x = 1.0
        true_deriv = -math.sin(x)

        def central_second_diff(h):
            return (math.sin(x + h) - 2 * math.sin(x) + math.sin(x - h)) / (
                h**2
            )

        h = 0.1
        no_extrap = central_second_diff(h)
        no_extrap_error = abs(no_extrap - true_deriv)

        with_extrap = richardson_extrapolation(
            f=lambda h: torch.tensor(central_second_diff(h)),
            h=h,
            order=2,
            levels=2,
        )
        with_extrap_error = abs(with_extrap.item() - true_deriv)

        assert with_extrap_error < no_extrap_error / 10

    def test_multiple_levels(self):
        """More levels give better accuracy."""
        x = 1.0
        true_deriv = math.cos(x)

        def central_diff(h):
            return (math.sin(x + h) - math.sin(x - h)) / (2 * h)

        h = 0.1

        result_2 = richardson_extrapolation(
            f=lambda h: torch.tensor(central_diff(h)), h=h, order=2, levels=2
        )
        result_3 = richardson_extrapolation(
            f=lambda h: torch.tensor(central_diff(h)), h=h, order=2, levels=3
        )

        error_2 = abs(result_2.item() - true_deriv)
        error_3 = abs(result_3.item() - true_deriv)

        assert error_3 < error_2

    def test_works_with_tensors(self):
        """Works with tensor-valued functions."""
        x = torch.tensor([0.5, 1.0, 1.5])

        def diff(h):
            return (torch.sin(x + h) - torch.sin(x - h)) / (2 * h)

        result = richardson_extrapolation(f=diff, h=0.1, order=2, levels=2)

        assert result.shape == x.shape
        torch.testing.assert_close(result, torch.cos(x), rtol=1e-5, atol=1e-7)

    def test_custom_ratio(self):
        """Custom step size ratio works."""
        x = 1.0
        true_deriv = math.cos(x)

        def central_diff(h):
            return (math.sin(x + h) - math.sin(x - h)) / (2 * h)

        h = 0.1

        result = richardson_extrapolation(
            f=lambda h: torch.tensor(central_diff(h)),
            h=h,
            order=2,
            ratio=3.0,
            levels=2,
        )

        error = abs(result.item() - true_deriv)
        assert error < 1e-6
