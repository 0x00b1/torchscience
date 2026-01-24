import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestApproximateConfinedGaussianWindow:
    """Tests for approximate_confined_gaussian_window and periodic version."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            expected = self._reference_approximate_confined_gaussian(
                n, sigma.item(), periodic=False
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            expected = self._reference_approximate_confined_gaussian(
                n, sigma.item(), periodic=True
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        sigma = torch.tensor(0.4)
        for n in [0, 1, 5, 100]:
            result = wf.approximate_confined_gaussian_window(n, sigma)
            assert result.shape == (n,)
            result_periodic = wf.periodic_approximate_confined_gaussian_window(
                n, sigma
            )
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        sigma = torch.tensor(0.4, dtype=dtype)
        result = wf.approximate_confined_gaussian_window(
            64, sigma, dtype=dtype
        )
        assert result.dtype == dtype
        result_periodic = wf.periodic_approximate_confined_gaussian_window(
            64, sigma, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        sigma = torch.tensor(0.4)
        result = wf.approximate_confined_gaussian_window(0, sigma)
        assert result.shape == (0,)
        result_periodic = wf.periodic_approximate_confined_gaussian_window(
            0, sigma
        )
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        result = wf.approximate_confined_gaussian_window(
            1, sigma, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_approximate_confined_gaussian_window(
            1, sigma, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric window is symmetric."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value_symmetric(self):
        """Test that center of odd-length symmetric window is 1.0."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_endpoints_zero_symmetric(self):
        """Test that symmetric window endpoints are zero (confined property)."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 64, 128]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            # Endpoints should be essentially zero
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-7,
                rtol=0,
            )
            torch.testing.assert_close(
                result[-1],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-7,
                rtol=0,
            )

    def test_endpoints_near_zero_periodic(self):
        """Test that periodic window has first endpoint at zero."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 64, 128]:
            result = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            # First endpoint should be zero or very close
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-7,
                rtol=0,
            )

    def test_gradient_flow(self):
        """Test that gradients flow through sigma parameter."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = wf.approximate_confined_gaussian_window(
            32, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert sigma.grad is not None
        assert not torch.isnan(sigma.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
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
            return wf.approximate_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_approximate_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        sigma = torch.tensor(0.4)
        with pytest.raises(ValueError):
            wf.approximate_confined_gaussian_window(-1, sigma)
        with pytest.raises(ValueError):
            wf.periodic_approximate_confined_gaussian_window(-1, sigma)

    def test_float_sigma_input(self):
        """Test that sigma can be passed as float."""
        result = wf.approximate_confined_gaussian_window(
            64, 0.4, dtype=torch.float64
        )
        assert result.shape == (64,)
        result_periodic = wf.periodic_approximate_confined_gaussian_window(
            64, 0.4, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_sigma_affects_width(self):
        """Test that larger sigma produces wider window."""
        n = 64
        sigma_narrow = torch.tensor(0.2, dtype=torch.float64)
        sigma_wide = torch.tensor(0.5, dtype=torch.float64)
        result_narrow = wf.approximate_confined_gaussian_window(
            n, sigma_narrow, dtype=torch.float64
        )
        result_wide = wf.approximate_confined_gaussian_window(
            n, sigma_wide, dtype=torch.float64
        )
        # Wider sigma should have larger values away from center
        # Check at quarter points
        quarter = n // 4
        assert result_wide[quarter] > result_narrow[quarter]
        assert result_wide[3 * quarter] > result_narrow[3 * quarter]
        # Sum should be larger for wider window
        assert result_wide.sum() > result_narrow.sum()

    def test_values_non_negative(self):
        """Test that all values are non-negative."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            assert (result >= -1e-10).all()  # Allow tiny numerical errors
            result_periodic = wf.periodic_approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            assert (result_periodic >= -1e-10).all()

    def test_max_value_at_center(self):
        """Test that maximum value is at or near the center."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # For even n, max could be at center-1 or center
            assert abs(max_idx - expected_center) <= 1

    def test_max_value_is_one_odd_n(self):
        """Test that max value is 1.0 for odd n (symmetric)."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            torch.testing.assert_close(
                result.max(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_different_sigma_values(self):
        """Test various sigma values produce valid windows."""
        n = 32
        for sigma_val in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
            sigma = torch.tensor(sigma_val, dtype=torch.float64)
            result = wf.approximate_confined_gaussian_window(
                n, sigma, dtype=torch.float64
            )
            # Should be symmetric
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)
            # Should have zero endpoints
            assert abs(result[0].item()) < 1e-6
            assert abs(result[-1].item()) < 1e-6

    def test_comparison_with_gaussian(self):
        """Test that approximate confined Gaussian differs from standard Gaussian."""
        n = 64
        sigma = torch.tensor(0.4, dtype=torch.float64)
        confined = wf.approximate_confined_gaussian_window(
            n, sigma, dtype=torch.float64
        )
        standard = wf.gaussian_window(n, sigma, dtype=torch.float64)
        # Confined should have smaller endpoint values
        assert confined[0] < standard[0]
        assert confined[-1] < standard[-1]
        # But they should have similar shape in the middle
        center = n // 2
        # Both should be close to 1 at center for odd n
        assert abs(confined[center].item() - 1.0) < 0.1
        assert abs(standard[center].item() - 1.0) < 0.01

    def test_n_equals_two(self):
        """Test edge case n=2."""
        sigma = torch.tensor(0.4, dtype=torch.float64)
        result = wf.approximate_confined_gaussian_window(
            2, sigma, dtype=torch.float64
        )
        assert result.shape == (2,)
        # Both values should be essentially zero or equal
        torch.testing.assert_close(result[0], result[1], atol=1e-10, rtol=0)

    @staticmethod
    def _reference_approximate_confined_gaussian(
        n: int, sigma: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation of approximate confined Gaussian window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        if periodic:
            # Periodic version uses different normalization
            center = n / 2.0
            k = torch.arange(n, dtype=torch.float64)
            t = (k - center) / n

            def gaussian(x):
                return torch.exp(-0.5 * (x / sigma) ** 2)

            g_t = gaussian(t)
            g_boundary = gaussian(torch.tensor(-0.5, dtype=torch.float64))
            window = g_t - g_boundary
            return torch.clamp(window, min=0.0)
        else:
            # Symmetric version
            center = (n - 1) / 2.0
            k = torch.arange(n, dtype=torch.float64)
            normalized_x = (k - center) / (sigma * center)
            G = torch.exp(-0.5 * normalized_x * normalized_x)
            G_endpoint = torch.exp(
                torch.tensor(-0.5 / (sigma * sigma), dtype=torch.float64)
            )
            return (G - G_endpoint) / (1.0 - G_endpoint)
