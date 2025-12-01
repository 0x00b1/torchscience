import math

import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestLanczosWindow:
    """Tests for lanczos_window and periodic_lanczos_window.

    The Lanczos window (sinc window) is defined as:
        w[k] = sinc(2k / denom - 1)
    where sinc(x) = sin(pi * x) / (pi * x) with sinc(0) = 1.
    """

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.lanczos_window(n, dtype=torch.float64)
            expected = self._reference_lanczos(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_lanczos_window(n, dtype=torch.float64)
            expected = self._reference_lanczos(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.lanczos_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_lanczos_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.lanczos_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_lanczos_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.lanczos_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_lanczos_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.lanczos_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_lanczos_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.lanczos_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_lanczos_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric lanczos window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.lanczos_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        for n in [5, 11, 65]:
            result = wf.lanczos_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_sinc_shape(self):
        """Test that lanczos window follows sinc curve."""
        n = 65
        result = wf.lanczos_window(n, dtype=torch.float64)
        # The window is sinc(2k/(n-1) - 1) = sinc(x) where x goes from -1 to 1
        # At x = -0.5 (k = n/4), sinc(-0.5) = sin(-pi/2) / (-pi/2) = -1 / (-pi/2) = 2/pi
        quarter_idx = n // 4
        expected_val = torch.tensor(2.0 / math.pi, dtype=torch.float64)
        torch.testing.assert_close(
            result[quarter_idx], expected_val, rtol=1e-2, atol=1e-2
        )

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.lanczos_window(64, dtype=torch.float64, requires_grad=True)
        assert result.requires_grad

    def test_values_in_valid_range(self):
        """Test that window values are mostly in reasonable range."""
        for n in [5, 64, 128]:
            result = wf.lanczos_window(n, dtype=torch.float64)
            # sinc values can be negative (first sidelobe), but bounded
            assert result.max() <= 1.0 + 1e-10
            # sinc(-1) and sinc(1) are 0, max sidelobe is about -0.217
            assert result.min() >= -0.25

    def test_n_equals_two(self):
        """Test n=2 case."""
        result = wf.lanczos_window(2, dtype=torch.float64)
        # For n=2, denom=1, so x = 2k/1 - 1 = 2k - 1
        # k=0: x=-1, sinc(-1) = sin(-pi)/(-pi) = 0
        # k=1: x=1, sinc(1) = sin(pi)/(pi) = 0
        expected = torch.tensor([0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @staticmethod
    def _reference_lanczos(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = float(n) if periodic else float(n - 1)
        if denom == 0:
            return torch.ones(n, dtype=torch.float64)
        result = torch.empty(n, dtype=torch.float64)
        for k in range(n):
            x = 2.0 * k / denom - 1.0
            if abs(x) < 1e-10:
                result[k] = 1.0
            else:
                pi_x = math.pi * x
                result[k] = math.sin(pi_x) / pi_x
        return result
