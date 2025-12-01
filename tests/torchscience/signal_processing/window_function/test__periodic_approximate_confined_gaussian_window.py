import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestPeriodicApproximateConfinedGaussianWindow:
    """Tests for periodic_approximate_confined_gaussian_window."""

    def test_reference(self):
        """Compare against reference implementation."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            expected = self._reference_periodic_approximate_confined_gaussian(
                n, sigma.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        sigma = torch.tensor(0.4)
        for n in [0, 1, 5, 100]:
            result = wf.periodic_approximate_confined_gaussian_window(n, sigma)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        sigma = torch.tensor(0.4, dtype=dtype)
        result = wf.periodic_approximate_confined_gaussian_window(
            64, sigma, dtype=dtype
        )
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        sigma = torch.tensor(0.4)
        result = wf.periodic_approximate_confined_gaussian_window(0, sigma)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        result = wf.periodic_approximate_confined_gaussian_window(
            1, sigma, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_monotonic_increase_to_center(self):
        """Test that window values increase monotonically toward the center.

        The window should have a bell shape with values increasing from
        the boundary (zero) toward the center (maximum).
        """
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            center = n // 2
            # Values should increase from start toward center
            for i in range(center - 1):
                assert result[i] <= result[i + 1], (
                    f"Expected monotonic increase, but result[{i}]={result[i]} "
                    f"> result[{i + 1}]={result[i + 1]}"
                )
            # Values should decrease from center toward end
            for i in range(center, n - 1):
                assert result[i] >= result[i + 1], (
                    f"Expected monotonic decrease, but result[{i}]={result[i]} "
                    f"< result[{i + 1}]={result[i + 1]}"
                )

    def test_boundary_values_near_zero(self):
        """Test that boundary values are near zero (approximate confinement)."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            # First and last values should be exactly 0 due to the subtraction
            # of G(-0.5) and clamping
            assert result[0].item() == 0.0
            # Last value should also be very small or zero
            assert result[-1].item() < 0.1

    def test_non_negative_values(self):
        """Test that all window values are non-negative."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 32, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            assert (result >= 0).all()

    def test_center_is_maximum(self):
        """Test that the center has the maximum value."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            max_idx = result.argmax().item()
            center = n // 2
            # Maximum should be at or near center
            assert abs(max_idx - center) <= 1

    def test_sigma_affects_shape(self):
        """Test that sigma affects the window shape.

        For the approximate confined Gaussian, smaller sigma values produce
        wider effective windows because the boundary correction G(-0.5)
        is smaller, preserving more of the original Gaussian.
        """
        n = 64
        sigma_small = torch.tensor(0.2, dtype=torch.float64)
        sigma_large = torch.tensor(0.5, dtype=torch.float64)
        result_small = wf.periodic_approximate_confined_gaussian_window(
            n, sigma_small, dtype=torch.float64
        )
        result_large = wf.periodic_approximate_confined_gaussian_window(
            n, sigma_large, dtype=torch.float64
        )
        # Smaller sigma produces larger sum (wider effective window)
        # because boundary correction is smaller
        assert result_small.sum() > result_large.sum()
        # Smaller sigma produces higher max value
        assert result_small.max() > result_large.max()

    def test_gradient_flow(self):
        """Test that gradients flow through sigma parameter."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_approximate_confined_gaussian_window(
            32, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert sigma.grad is not None
        assert not torch.isnan(sigma.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_approximate_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_approximate_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradgradcheck(func, (sigma,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        sigma = torch.tensor(0.4)
        with pytest.raises(ValueError):
            wf.periodic_approximate_confined_gaussian_window(-1, sigma)

    def test_float_sigma_input(self):
        """Test that sigma can be passed as float."""
        result = wf.periodic_approximate_confined_gaussian_window(
            64, 0.4, dtype=torch.float64
        )
        assert result.shape == (64,)

    def test_device_support(self):
        """Test that device parameter is respected."""
        sigma = torch.tensor(0.4)
        result = wf.periodic_approximate_confined_gaussian_window(
            64, sigma, device="cpu"
        )
        assert result.device.type == "cpu"

    def test_various_sigma_values(self):
        """Test various sigma values produce valid windows."""
        n = 64
        for sigma_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sigma = torch.tensor(sigma_val, dtype=torch.float64)
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            # All values should be non-negative
            assert (result >= 0).all()
            # Maximum should be positive
            assert result.max() > 0
            # Shape should be correct
            assert result.shape == (n,)

    @staticmethod
    def _reference_periodic_approximate_confined_gaussian(
        n: int, sigma: float
    ) -> torch.Tensor:
        """Reference implementation of periodic approximate confined Gaussian window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        # For periodic window, denominator is n
        denom = float(n)
        center = denom / 2.0

        # Compute normalized positions t in [-0.5, 0.5)
        k = torch.arange(n, dtype=torch.float64)
        t = (k - center) / denom

        # Gaussian function: G(t) = exp(-0.5 * (t / sigma)^2)
        def gaussian(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(-0.5 * (x / sigma) ** 2)

        # w[k] = G(t_k) - G(-0.5)
        g_t = gaussian(t)
        g_boundary = gaussian(torch.tensor(-0.5, dtype=torch.float64))

        window = g_t - g_boundary

        # Ensure non-negative values
        window = torch.clamp(window, min=0.0)

        return window
