"""Tests for sosfilt second-order sections filter implementation."""

import pytest
import scipy.signal
import torch

from torchscience.filter_design import sosfilt, sosfilt_zi


class TestSosfiltMatchesScipy:
    """Test sosfilt matches scipy.signal.sosfilt."""

    def test_sosfilt_butterworth(self) -> None:
        """Test sosfilt with a Butterworth filter."""
        sos_np = scipy.signal.butter(5, 0.25, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_chebyshev(self) -> None:
        """Test sosfilt with a Chebyshev type I filter."""
        sos_np = scipy.signal.cheby1(6, 0.5, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_elliptic(self) -> None:
        """Test sosfilt with an elliptic filter."""
        sos_np = scipy.signal.ellip(4, 0.5, 40, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_single_section(self) -> None:
        """Test sosfilt with a single biquad section."""
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_highpass(self) -> None:
        """Test sosfilt with a highpass filter."""
        sos_np = scipy.signal.butter(5, 0.4, btype="high", output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_bandpass(self) -> None:
        """Test sosfilt with a bandpass filter."""
        sos_np = scipy.signal.butter(4, [0.2, 0.4], btype="band", output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_bandstop(self) -> None:
        """Test sosfilt with a bandstop filter."""
        sos_np = scipy.signal.butter(
            3, [0.3, 0.5], btype="bandstop", output="sos"
        )
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_long_signal(self) -> None:
        """Test sosfilt with a long signal."""
        sos_np = scipy.signal.butter(6, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(10000, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )


class TestSosfiltWithInitialConditions:
    """Test sosfilt with initial conditions."""

    def test_sosfilt_with_zi(self) -> None:
        """Test sosfilt returns (y, zf) when zi is provided."""
        sos_np = scipy.signal.butter(5, 0.25, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        zi = sosfilt_zi(sos)
        x = torch.randn(100, dtype=torch.float64)

        result = sosfilt(sos, x, zi=zi * x[0])
        assert isinstance(result, tuple)
        assert len(result) == 2

        y, zf = result

        # Compare with scipy
        y_scipy, zf_scipy = scipy.signal.sosfilt(
            sos_np, x.numpy(), zi=zi.numpy() * x[0].item()
        )

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            zf, torch.from_numpy(zf_scipy), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_step_response_no_transient(self) -> None:
        """Test that zi removes transient from step input."""
        sos_np = scipy.signal.butter(6, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        zi = sosfilt_zi(sos)
        x = torch.ones(50, dtype=torch.float64)

        y, _ = sosfilt(sos, x, zi=zi)

        # Output should be all ones (steady-state)
        torch.testing.assert_close(
            y, torch.ones_like(y), rtol=1e-10, atol=1e-10
        )

    def test_sosfilt_zi_continuation(self) -> None:
        """Test that zf from one call can be used as zi for next call."""
        sos_np = scipy.signal.butter(4, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        n_sections = sos.shape[0]

        # Generate full signal and split
        x_full = torch.randn(200, dtype=torch.float64)
        x1 = x_full[:100]
        x2 = x_full[100:]

        # Filter full signal in one go
        y_full = sosfilt(sos, x_full)

        # Filter in two parts using zi/zf
        zi = torch.zeros(n_sections, 2, dtype=torch.float64)
        y1, zf1 = sosfilt(sos, x1, zi=zi)
        y2, _ = sosfilt(sos, x2, zi=zf1)

        # Concatenate should match full filtering
        y_concat = torch.cat([y1, y2])
        torch.testing.assert_close(y_concat, y_full, rtol=1e-10, atol=1e-10)


class TestSosfiltAxis:
    """Test sosfilt axis parameter."""

    def test_sosfilt_axis_minus_1(self) -> None:
        """Test filtering along last axis (default)."""
        sos_np = scipy.signal.butter(4, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y = sosfilt(sos, x, axis=-1)

        # Check each row
        for i in range(10):
            y_scipy = scipy.signal.sosfilt(sos_np, x[i].numpy())
            torch.testing.assert_close(
                y[i], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
            )

    def test_sosfilt_axis_0(self) -> None:
        """Test filtering along axis 0."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, 10, dtype=torch.float64)

        y = sosfilt(sos, x, axis=0)

        # Check each column
        for j in range(10):
            y_scipy = scipy.signal.sosfilt(sos_np, x[:, j].numpy())
            torch.testing.assert_close(
                y[:, j], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
            )

    def test_sosfilt_axis_1(self) -> None:
        """Test filtering along axis 1 of 3D tensor."""
        sos_np = scipy.signal.butter(2, 0.4, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(5, 100, 8, dtype=torch.float64)

        y = sosfilt(sos, x, axis=1)

        # Check a few slices
        for i in range(5):
            for k in range(8):
                y_scipy = scipy.signal.sosfilt(sos_np, x[i, :, k].numpy())
                torch.testing.assert_close(
                    y[i, :, k],
                    torch.from_numpy(y_scipy),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_sosfilt_axis_with_zi(self) -> None:
        """Test axis parameter works with initial conditions."""
        sos_np = scipy.signal.butter(4, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        n_sections = sos.shape[0]
        x = torch.randn(10, 100, dtype=torch.float64)

        # zi shape should match: (batch..., n_sections, 2)
        zi = torch.zeros(10, n_sections, 2, dtype=torch.float64)

        y, zf = sosfilt(sos, x, axis=-1, zi=zi)

        # Check shape
        assert y.shape == x.shape
        assert zf.shape == zi.shape

        # Compare first row with scipy
        y_scipy, zf_scipy = scipy.signal.sosfilt(
            sos_np, x[0].numpy(), zi=zi[0].numpy()
        )
        torch.testing.assert_close(
            y[0], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )


class TestSosfiltGradients:
    """Test sosfilt gradient computation."""

    def test_gradcheck_signal(self) -> None:
        """Gradient check for input signal."""
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(30, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda x_: sosfilt(sos, x_),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_sos_coeffs(self) -> None:
        """Gradient check for SOS coefficients (excluding a0).

        Note: a0 is always 1.0 in normalized SOS form and gradients through
        it aren't meaningful since lfilter normalizes by a0. We only check
        gradients for b0, b1, b2, a1, a2 columns.
        """
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        # Only check gradients for columns that make sense (not a0 at column 3)
        sos = torch.tensor(sos_np, dtype=torch.float64, requires_grad=True)
        x = torch.randn(30, dtype=torch.float64)

        # Mask out a0 column (index 3) from gradient checking
        def sosfilt_masked(sos_partial):
            # sos_partial is columns [0,1,2,4,5] flattened
            n_sections = sos_np.shape[0]
            # Reconstruct full sos
            sos_full = torch.zeros(n_sections, 6, dtype=torch.float64)
            sos_full[:, 0] = sos_partial[:n_sections]  # b0
            sos_full[:, 1] = sos_partial[n_sections : 2 * n_sections]  # b1
            sos_full[:, 2] = sos_partial[2 * n_sections : 3 * n_sections]  # b2
            sos_full[:, 3] = 1.0  # a0 fixed
            sos_full[:, 4] = sos_partial[3 * n_sections : 4 * n_sections]  # a1
            sos_full[:, 5] = sos_partial[4 * n_sections :]  # a2
            return sosfilt(sos_full, x)

        # Create partial parameter (excluding a0)
        n_sections = sos_np.shape[0]
        sos_partial = (
            torch.cat(
                [
                    sos[:, 0],  # b0
                    sos[:, 1],  # b1
                    sos[:, 2],  # b2
                    sos[:, 4],  # a1
                    sos[:, 5],  # a2
                ]
            )
            .clone()
            .detach()
            .requires_grad_(True)
        )

        torch.autograd.gradcheck(
            sosfilt_masked,
            (sos_partial,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_both(self) -> None:
        """Gradient check for both SOS coefficients (excluding a0) and signal.

        Note: a0 is always 1.0 in normalized SOS form and gradients through
        it aren't meaningful since lfilter normalizes by a0.
        """
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        n_sections = sos_np.shape[0]

        def sosfilt_masked(sos_partial, x_):
            # Reconstruct full sos from partial (excluding a0)
            sos_full = torch.zeros(n_sections, 6, dtype=torch.float64)
            sos_full[:, 0] = sos_partial[:n_sections]  # b0
            sos_full[:, 1] = sos_partial[n_sections : 2 * n_sections]  # b1
            sos_full[:, 2] = sos_partial[2 * n_sections : 3 * n_sections]  # b2
            sos_full[:, 3] = 1.0  # a0 fixed
            sos_full[:, 4] = sos_partial[3 * n_sections : 4 * n_sections]  # a1
            sos_full[:, 5] = sos_partial[4 * n_sections :]  # a2
            return sosfilt(sos_full, x_)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        sos_partial = (
            torch.cat(
                [
                    sos[:, 0],  # b0
                    sos[:, 1],  # b1
                    sos[:, 2],  # b2
                    sos[:, 4],  # a1
                    sos[:, 5],  # a2
                ]
            )
            .clone()
            .detach()
            .requires_grad_(True)
        )
        x = torch.randn(30, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            sosfilt_masked,
            (sos_partial, x),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestSosfiltDtypeAndDevice:
    """Test sosfilt dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float32)
        x = torch.randn(100, dtype=torch.float32)

        y = sosfilt(sos, x)

        assert y.dtype == torch.float32
        assert y.shape == x.shape

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfilt(sos, x)

        assert y.dtype == torch.float64
        assert y.shape == x.shape

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64, device="cpu")
        x = torch.randn(100, dtype=torch.float64, device="cpu")

        y = sosfilt(sos, x)

        assert y.device == x.device


class TestSosfiltEdgeCases:
    """Test sosfilt edge cases."""

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape."""
        sos_np = scipy.signal.butter(4, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        for shape in [(100,), (10, 100), (5, 10, 100)]:
            x = torch.randn(*shape, dtype=torch.float64)
            y = sosfilt(sos, x)
            assert y.shape == x.shape

    def test_single_sample(self) -> None:
        """Test with single sample input."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_many_sections(self) -> None:
        """Test with filter having many sections."""
        sos_np = scipy.signal.butter(12, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-9, atol=1e-9
        )


class TestSosfiltComplex:
    """Test sosfilt with complex inputs."""

    def test_complex_signal(self) -> None:
        """Test with complex input signal."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.complex128)

        y = sosfilt(sos, x)
        y_scipy = scipy.signal.sosfilt(sos_np, x.numpy())

        assert y.is_complex()
        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )


class TestSosfiltValidation:
    """Test sosfilt input validation."""

    def test_invalid_sos_shape_raises(self) -> None:
        """Test that invalid SOS shape raises error."""
        sos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        with pytest.raises(ValueError, match="shape"):
            sosfilt(sos, x)

    def test_invalid_sos_ndim_raises(self) -> None:
        """Test that wrong SOS dimensions raises error."""
        sos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        with pytest.raises(ValueError, match="shape"):
            sosfilt(sos, x)
