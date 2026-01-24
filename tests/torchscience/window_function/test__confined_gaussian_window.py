import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestConfinedGaussianWindow:
    """Tests for confined_gaussian_window and periodic_confined_gaussian_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [2, 5, 64, 128]:
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            expected = self._reference_confined_gaussian(
                n, sigma.item(), periodic=False
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [2, 5, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            expected = self._reference_confined_gaussian(
                n, std.item(), periodic=True
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_symmetric_boundary_zero(self):
        """Symmetric window should be exactly zero at both boundaries."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 32, 64, 128]:
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )
            torch.testing.assert_close(
                result[-1],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_periodic_boundary_zero(self):
        """Periodic window should be exactly zero at first boundary."""
        std = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 32, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        sigma = torch.tensor(0.4)
        for n in [0, 1, 5, 100]:
            result = wf.confined_gaussian_window(n, sigma)
            assert result.shape == (n,)
            result_periodic = wf.periodic_confined_gaussian_window(n, sigma)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        sigma = torch.tensor(0.4, dtype=dtype)
        result = wf.confined_gaussian_window(64, sigma, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_confined_gaussian_window(
            64, sigma, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        sigma = torch.tensor(0.4)
        result = wf.confined_gaussian_window(0, sigma)
        assert result.shape == (0,)
        result_periodic = wf.periodic_confined_gaussian_window(0, sigma)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        result = wf.confined_gaussian_window(1, sigma, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_confined_gaussian_window(
            1, sigma, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric confined Gaussian window is symmetric."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_positive_values_interior(self):
        """Test that window has positive values in interior."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            # Interior values (excluding boundaries) should be positive
            assert (result[1:-1] > 0).all()
            result_periodic = wf.periodic_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            assert (result_periodic[1:] > 0).all()

    def test_max_at_center(self):
        """Test that maximum occurs near center."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            max_idx = result.argmax().item()
            center = (n - 1) / 2
            # Max should be within 1 sample of center
            assert abs(max_idx - center) <= 1

    def test_gradient_flow(self):
        """Test that gradients flow through sigma parameter."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.confined_gaussian_window(32, sigma, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert sigma.grad is not None
        assert not torch.isnan(sigma.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_confined_gaussian_window(
            32, std, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert std.grad is not None
        assert not torch.isnan(std.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        std = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (std,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        sigma = torch.tensor(0.4)
        with pytest.raises(ValueError):
            wf.confined_gaussian_window(-1, sigma)
        with pytest.raises(ValueError):
            wf.periodic_confined_gaussian_window(-1, sigma)

    def test_float_sigma_input(self):
        """Test that sigma can be passed as float."""
        result = wf.confined_gaussian_window(64, 0.4, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_confined_gaussian_window(
            64, 0.4, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_sigma_affects_width(self):
        """Test that larger sigma produces wider window."""
        n = 64
        sigma_narrow = torch.tensor(0.1, dtype=torch.float64)
        sigma_wide = torch.tensor(0.4, dtype=torch.float64)
        result_narrow = wf.confined_gaussian_window(
            n, sigma_narrow, dtype=torch.float64
        )
        result_wide = wf.confined_gaussian_window(
            n, sigma_wide, dtype=torch.float64
        )
        # Both should still have zero boundaries
        torch.testing.assert_close(
            result_narrow[0],
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )
        torch.testing.assert_close(
            result_wide[0],
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )
        # Wider sigma should have larger sum (more area under curve)
        assert result_wide.sum() > result_narrow.sum()

    def test_various_sigma_values(self):
        """Test window is valid for various sigma values."""
        n = 64
        for sigma_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sigma = torch.tensor(sigma_val, dtype=torch.float64)
            result = wf.confined_gaussian_window(n, sigma, dtype=torch.float64)
            # Check boundaries are zero
            assert abs(result[0].item()) < 1e-10
            assert abs(result[-1].item()) < 1e-10
            # Check no NaN or Inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
            # Check positive interior
            assert (result[1:-1] > 0).all()

    @staticmethod
    def _reference_confined_gaussian(
        n: int, sigma: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation of confined Gaussian window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        if periodic:
            # Periodic version: uses n as denominator
            L = float(n)
            half_L = L / 2.0
            actual_sigma = sigma * L

            k = torch.arange(n, dtype=torch.float64)
            x = k - half_L

            def g(t):
                return torch.exp(-t * t / (2.0 * actual_sigma * actual_sigma))

            g_x = g(x)
            g_half_L = g(torch.tensor(half_L, dtype=torch.float64))
            g_neg_3half_L = g(torch.tensor(-1.5 * L, dtype=torch.float64))

            numerator = g(x - L) + g(x + L)
            denominator = g_neg_3half_L + g_half_L

            correction = g_half_L * numerator / denominator
            return g_x - correction
        else:
            # Symmetric version: normalized positions in [-0.5, 0.5]
            k = torch.arange(n, dtype=torch.float64)
            center = (n - 1) / 2.0
            t = (k - center) / (n - 1)

            def gaussian(x):
                return torch.exp(-0.5 * (x / sigma) ** 2)

            g_t = gaussian(t)
            g_t_minus_1 = gaussian(t - 1)
            g_t_plus_1 = gaussian(t + 1)
            g_half = gaussian(torch.tensor(0.5, dtype=torch.float64))
            g_three_half = gaussian(torch.tensor(1.5, dtype=torch.float64))

            correction = g_half / (g_half + g_three_half)
            return g_t - correction * (g_t_minus_1 + g_t_plus_1)
