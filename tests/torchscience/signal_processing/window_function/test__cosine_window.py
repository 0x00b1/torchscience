import math

import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestCosineWindow:
    """Tests for cosine_window and periodic_cosine_window.

    Note: This is the sine window (also known as cosine window), which uses
    sin(pi * k / (n-1)) for symmetric, different from cosine-sum windows.
    """

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.cosine_window(n, dtype=torch.float64)
            expected = self._reference_cosine(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_cosine_window(n, dtype=torch.float64)
            expected = self._reference_cosine(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.cosine_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_cosine_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.cosine_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_cosine_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.cosine_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_cosine_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.cosine_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_cosine_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.cosine_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_cosine_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric cosine window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.cosine_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that symmetric cosine window has zero endpoints."""
        for n in [5, 10, 64]:
            result = wf.cosine_window(n, dtype=torch.float64)
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
        """Test that center of odd-length symmetric window is 1.0."""
        for n in [5, 11, 65]:
            result = wf.cosine_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_half_sine_shape(self):
        """Test that cosine window follows half sine curve."""
        n = 65
        result = wf.cosine_window(n, dtype=torch.float64)
        # At quarter points (n/4 and 3n/4), value should be sin(pi/4) = sqrt(2)/2
        quarter_idx = n // 4
        expected_val = torch.tensor(math.sin(math.pi / 4), dtype=torch.float64)
        torch.testing.assert_close(
            result[quarter_idx], expected_val, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            result[n - 1 - quarter_idx], expected_val, rtol=1e-2, atol=1e-2
        )

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.cosine_window(64, dtype=torch.float64, requires_grad=True)
        assert result.requires_grad

    @staticmethod
    def _reference_cosine(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        return torch.sin(math.pi * k / denom)
