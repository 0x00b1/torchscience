import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestBartlettHannWindow:
    """Tests for bartlett_hann_window and periodic_bartlett_hann_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            expected = self._reference_bartlett_hann(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_bartlett_hann_window(n, dtype=torch.float64)
            expected = self._reference_bartlett_hann(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison(self):
        """Compare against scipy.signal.windows.barthann if available."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import barthann

        for n in [5, 10, 64, 128]:
            # scipy barthann is symmetric by default
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            expected = torch.tensor(barthann(n), dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_periodic_comparison(self):
        """Compare against scipy with sym=False if available."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import barthann

        for n in [5, 10, 64, 128]:
            result = wf.periodic_bartlett_hann_window(n, dtype=torch.float64)
            expected = torch.tensor(
                barthann(n, sym=False), dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.bartlett_hann_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_bartlett_hann_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.bartlett_hann_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_bartlett_hann_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.bartlett_hann_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_bartlett_hann_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.bartlett_hann_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_bartlett_hann_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.bartlett_hann_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_bartlett_hann_window(
            1, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Bartlett-Hann window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that symmetric Bartlett-Hann window has zero endpoints.

        For the Bartlett-Hann window with coefficients a0=0.62, a1=0.48, a2=0.38:
        w[0] = 0.62 - 0.48*|0 - 0.5| - 0.38*cos(0) = 0.62 - 0.24 - 0.38 = 0
        """
        for n in [5, 10, 64]:
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            # Bartlett-Hann has exactly zero endpoints
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

    def test_center_value(self):
        """Test that center of odd-length symmetric window has expected value.

        For the Bartlett-Hann window at the center (k = (n-1)/2):
        w[center] = 0.62 - 0.48*0 - 0.38*cos(pi) = 0.62 - 0 + 0.38 = 1.0
        """
        for n in [5, 11, 65]:
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_coefficients_verify(self):
        """Verify Bartlett-Hann coefficients produce expected endpoint/center values."""
        # Standard Bartlett-Hann coefficients
        a0, a1, a2 = 0.62, 0.48, 0.38
        # At endpoints (frac=0), cos(0)=1: w = a0 - a1*0.5 - a2*1 = 0.62 - 0.24 - 0.38 = 0
        endpoint_value = a0 - a1 * 0.5 - a2 * 1.0
        assert abs(endpoint_value - 0.0) < 1e-10
        # At center (frac=0.5), cos(pi)=-1: w = a0 - a1*0 - a2*(-1) = 0.62 + 0.38 = 1
        center_value = a0 - a1 * 0.0 - a2 * math.cos(math.pi)
        assert abs(center_value - 1.0) < 1e-10

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.bartlett_hann_window(
            64, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

    def test_values_in_range(self):
        """Test that all window values are in [0, 1] for n > 1."""
        for n in [5, 10, 64, 128]:
            result = wf.bartlett_hann_window(n, dtype=torch.float64)
            assert (result >= 0.0).all()
            assert (result <= 1.0).all()
            result_periodic = wf.periodic_bartlett_hann_window(
                n, dtype=torch.float64
            )
            assert (result_periodic >= 0.0).all()
            assert (result_periodic <= 1.0).all()

    @staticmethod
    def _reference_bartlett_hann(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation using the same coefficients as scipy."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        # Bartlett-Hann coefficients (same as scipy.signal.windows.barthann)
        a0, a1, a2 = 0.62, 0.48, 0.38
        frac = k / denom
        return (
            a0
            - a1 * torch.abs(frac - 0.5)
            - a2 * torch.cos(2 * math.pi * frac)
        )
