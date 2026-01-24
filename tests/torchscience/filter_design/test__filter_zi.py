"""Tests for lfilter_zi and sosfilt_zi functions."""

import numpy as np
import pytest
import scipy.signal
import torch

from torchscience.filter_design import lfilter_zi, sosfilt_zi


class TestLfilterZi:
    """Tests for lfilter_zi function."""

    def test_lfilter_zi_matches_scipy_butterworth(self):
        """Test that lfilter_zi matches scipy for a Butterworth filter."""
        # Create a Butterworth lowpass filter
        b_np, a_np = scipy.signal.butter(5, 0.25)
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_matches_scipy_chebyshev(self):
        """Test that lfilter_zi matches scipy for a Chebyshev filter."""
        # Create a Chebyshev type I filter
        b_np, a_np = scipy.signal.cheby1(4, 0.5, 0.3)
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_matches_scipy_elliptic(self):
        """Test that lfilter_zi matches scipy for an elliptic filter."""
        # Create an elliptic filter
        b_np, a_np = scipy.signal.ellip(3, 0.5, 40, 0.2)
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_first_order(self):
        """Test lfilter_zi for a simple first-order filter."""
        # Simple first-order lowpass: y[n] = 0.5*x[n] + 0.5*y[n-1]
        # b = [0.5], a = [1, -0.5]
        b_np = np.array([0.5])
        a_np = np.array([1.0, -0.5])
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_second_order(self):
        """Test lfilter_zi for a second-order filter."""
        b_np = np.array([0.0675, 0.135, 0.0675])
        a_np = np.array([1.0, -1.143, 0.413])
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_unnormalized_a(self):
        """Test lfilter_zi with a[0] != 1."""
        # Multiply by 2 to get unnormalized coefficients
        b_np = np.array([0.1, 0.2, 0.1])
        a_np = np.array([2.0, -1.6, 0.6])
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_different_lengths(self):
        """Test lfilter_zi when b and a have different lengths."""
        # b longer than a
        b_np = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        a_np = np.array([1.0, -0.5, 0.2])
        zi_scipy = scipy.signal.lfilter_zi(b_np, a_np)

        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_lfilter_zi_output_shape(self):
        """Test that lfilter_zi output has correct shape."""
        b = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1], dtype=torch.float64)
        a = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64)
        zi = lfilter_zi(b, a)

        # Output should have length max(len(a), len(b)) - 1 = 4
        assert zi.shape == (4,)

    def test_lfilter_zi_dtype_preservation(self):
        """Test that lfilter_zi preserves dtype."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float32)
        a = torch.tensor([1.0, -0.3], dtype=torch.float32)
        zi = lfilter_zi(b, a)

        assert zi.dtype == torch.float32

    def test_lfilter_zi_step_response_no_transient(self):
        """Test that filtering with zi produces no transient on step input."""
        # Create a filter
        b_np, a_np = scipy.signal.butter(4, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        # Get initial conditions
        zi = lfilter_zi(b, a)

        # Filter a step input (all ones)
        x = torch.ones(20, dtype=torch.float64)

        # Use scipy.signal.lfilter with the computed zi
        # The output should be all ones (steady-state for step input)
        y, _ = scipy.signal.lfilter(
            b.numpy(), a.numpy(), x.numpy(), zi=zi.numpy()
        )

        # All outputs should be 1.0 (no transient)
        np.testing.assert_allclose(y, np.ones_like(y), rtol=1e-10, atol=1e-10)

    def test_lfilter_zi_invalid_1d_requirement(self):
        """Test that lfilter_zi raises error for non-1D input."""
        b = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)

        with pytest.raises(ValueError, match="1-D"):
            lfilter_zi(b, a)

    def test_lfilter_zi_empty_a_raises(self):
        """Test that lfilter_zi raises error for empty a."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([], dtype=torch.float64)

        with pytest.raises(ValueError):
            lfilter_zi(b, a)


class TestSosfiltZi:
    """Tests for sosfilt_zi function."""

    def test_sosfilt_zi_matches_scipy_butterworth(self):
        """Test that sosfilt_zi matches scipy for a Butterworth filter."""
        # Create a Butterworth filter in SOS form
        sos_np = scipy.signal.butter(5, 0.25, output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_sosfilt_zi_matches_scipy_chebyshev(self):
        """Test that sosfilt_zi matches scipy for a Chebyshev filter."""
        sos_np = scipy.signal.cheby1(6, 0.5, 0.3, output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_sosfilt_zi_matches_scipy_elliptic(self):
        """Test that sosfilt_zi matches scipy for an elliptic filter."""
        sos_np = scipy.signal.ellip(4, 0.5, 40, 0.2, output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_sosfilt_zi_single_section(self):
        """Test sosfilt_zi for a single biquad section."""
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_sosfilt_zi_output_shape(self):
        """Test that sosfilt_zi output has correct shape."""
        sos_np = scipy.signal.butter(8, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        # Output should have shape (n_sections, 2)
        n_sections = sos.shape[0]
        assert zi.shape == (n_sections, 2)

    def test_sosfilt_zi_dtype_preservation(self):
        """Test that sosfilt_zi preserves dtype."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float32)
        zi = sosfilt_zi(sos)

        assert zi.dtype == torch.float32

    def test_sosfilt_zi_step_response_no_transient(self):
        """Test that filtering with zi produces no transient on step input."""
        # Create a filter
        sos_np = scipy.signal.butter(6, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        # Get initial conditions
        zi = sosfilt_zi(sos)

        # Filter a step input (all ones)
        x = np.ones(30)

        # Use scipy.signal.sosfilt with the computed zi
        y, _ = scipy.signal.sosfilt(sos_np, x, zi=zi.numpy())

        # All outputs should be 1.0 (no transient)
        np.testing.assert_allclose(y, np.ones_like(y), rtol=1e-10, atol=1e-10)

    def test_sosfilt_zi_invalid_shape_raises(self):
        """Test that sosfilt_zi raises error for invalid SOS shape."""
        sos = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float64
        )  # Wrong shape

        with pytest.raises(ValueError, match="shape"):
            sosfilt_zi(sos)

    def test_sosfilt_zi_invalid_ndim_raises(self):
        """Test that sosfilt_zi raises error for wrong number of dimensions."""
        sos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="shape"):
            sosfilt_zi(sos)

    def test_sosfilt_zi_highpass(self):
        """Test sosfilt_zi for a highpass filter."""
        sos_np = scipy.signal.butter(5, 0.4, btype="high", output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )

    def test_sosfilt_zi_bandpass(self):
        """Test sosfilt_zi for a bandpass filter."""
        sos_np = scipy.signal.butter(4, [0.2, 0.4], btype="band", output="sos")
        zi_scipy = scipy.signal.sosfilt_zi(sos_np)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)

        np.testing.assert_allclose(
            zi.numpy(), zi_scipy, rtol=1e-10, atol=1e-12
        )
