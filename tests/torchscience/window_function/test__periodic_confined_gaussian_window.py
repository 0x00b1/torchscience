import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestPeriodicConfinedGaussianWindow:
    """Tests for periodic_confined_gaussian_window."""

    def test_reference(self):
        """Compare against reference implementation."""
        std = torch.tensor(0.3, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            expected = self._reference_periodic_confined_gaussian(
                n, std.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_boundary_zero(self):
        """Test that window reaches zero at k=0 (key property of confined Gaussian)."""
        std = torch.tensor(0.3, dtype=torch.float64)
        for n in [16, 32, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            # First value should be exactly zero
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-12,
                rtol=0,
            )

    def test_center_near_one(self):
        """Test that center value is close to 1."""
        std = torch.tensor(0.3, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            center_idx = n // 2
            # Center should be close to 1 (but not exactly due to confinement)
            assert result[center_idx] > 0.9
            assert result[center_idx] <= 1.0

    def test_output_shape(self):
        """Test output shape is (n,)."""
        std = torch.tensor(0.3)
        for n in [0, 1, 5, 100]:
            result = wf.periodic_confined_gaussian_window(n, std)
            assert result.shape == (n,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        std = torch.tensor(0.3)
        result = wf.periodic_confined_gaussian_window(0, std)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        std = torch.tensor(0.3, dtype=torch.float64)
        result = wf.periodic_confined_gaussian_window(
            1, std, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        std = torch.tensor(0.3)
        with pytest.raises(ValueError):
            wf.periodic_confined_gaussian_window(-1, std)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        std = torch.tensor(0.3, dtype=dtype)
        result = wf.periodic_confined_gaussian_window(64, std, dtype=dtype)
        assert result.dtype == dtype

    def test_gradient_flow(self):
        """Test that gradients flow through std parameter."""
        std = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_confined_gaussian_window(
            32, std, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert std.grad is not None
        assert not torch.isnan(std.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        std = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_confined_gaussian_window(
                32, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (std,), raise_exception=True)

    def test_float_std_input(self):
        """Test that std can be passed as float."""
        result = wf.periodic_confined_gaussian_window(
            64, 0.3, dtype=torch.float64
        )
        assert result.shape == (64,)

    def test_std_affects_width(self):
        """Test that larger std produces wider window."""
        n = 64
        std_narrow = torch.tensor(0.15, dtype=torch.float64)
        std_wide = torch.tensor(0.4, dtype=torch.float64)
        result_narrow = wf.periodic_confined_gaussian_window(
            n, std_narrow, dtype=torch.float64
        )
        result_wide = wf.periodic_confined_gaussian_window(
            n, std_wide, dtype=torch.float64
        )
        # Wider std should have larger values near edges (but not at edge 0)
        # Check at 1/4 position
        quarter_idx = n // 4
        assert result_wide[quarter_idx] > result_narrow[quarter_idx]
        # Sum should be larger for wider window
        assert result_wide.sum() > result_narrow.sum()

    def test_non_negative_values(self):
        """Test that window values are non-negative."""
        std = torch.tensor(0.3, dtype=torch.float64)
        for n in [16, 32, 64]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            assert (result >= 0).all()

    def test_max_value_near_center(self):
        """Test that maximum value is near the center."""
        std = torch.tensor(0.3, dtype=torch.float64)
        for n in [32, 64, 128]:
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            max_idx = result.argmax().item()
            center = n // 2
            # Max should be within 1 of center
            assert abs(max_idx - center) <= 1

    def test_different_std_values(self):
        """Test various std values produce valid windows."""
        n = 64
        for std_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            std = torch.tensor(std_val, dtype=torch.float64)
            result = wf.periodic_confined_gaussian_window(
                n, std, dtype=torch.float64
            )
            # Should have valid shape
            assert result.shape == (n,)
            # Should be non-negative
            assert (result >= 0).all()
            # First value should be zero
            assert abs(result[0].item()) < 1e-10

    def test_device_cpu(self):
        """Test CPU device."""
        std = torch.tensor(0.3, device="cpu")
        result = wf.periodic_confined_gaussian_window(64, std, device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_device_cuda(self):
        """Test CUDA device."""
        std = torch.tensor(0.3, device="cuda")
        result = wf.periodic_confined_gaussian_window(64, std, device="cuda")
        assert result.device.type == "cuda"

    def test_bell_shape(self):
        """Test that window has bell shape (monotonic from center to edges)."""
        std = torch.tensor(0.3, dtype=torch.float64)
        n = 64
        result = wf.periodic_confined_gaussian_window(
            n, std, dtype=torch.float64
        )
        center = n // 2

        # From center to end should be monotonically decreasing
        right_half = result[center:]
        for i in range(len(right_half) - 1):
            assert right_half[i] >= right_half[i + 1]

        # From start to center should be monotonically increasing
        left_half = result[: center + 1]
        for i in range(len(left_half) - 1):
            assert left_half[i] <= left_half[i + 1]

    @staticmethod
    def _reference_periodic_confined_gaussian(
        n: int, std: float
    ) -> torch.Tensor:
        """Reference implementation of periodic confined Gaussian window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        L = float(n)
        half_L = L / 2.0
        sigma = std * L

        k = torch.arange(n, dtype=torch.float64)
        x = k - half_L

        def g(t):
            return torch.exp(-t * t / (2.0 * sigma * sigma))

        g_x = g(x)
        g_half_L = g(torch.tensor(half_L, dtype=torch.float64))
        g_neg_3half_L = g(torch.tensor(-1.5 * L, dtype=torch.float64))

        numerator = g(x - L) + g(x + L)
        denominator = g_neg_3half_L + g_half_L

        correction = g_half_L * numerator / denominator

        return g_x - correction
