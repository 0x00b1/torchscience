"""Tests for lfilter IIR filter implementation."""

import scipy.signal
import torch

from torchscience.signal_processing.filter import lfilter, lfilter_zi


class TestLfilterMatchesScipy:
    """Test lfilter matches scipy.signal.lfilter."""

    def test_lfilter_simple_iir(self) -> None:
        """Test lfilter with a simple IIR filter."""
        # Simple first-order IIR filter
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_butterworth(self) -> None:
        """Test lfilter with a Butterworth filter."""
        b_np, a_np = scipy.signal.butter(5, 0.25)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_chebyshev(self) -> None:
        """Test lfilter with a Chebyshev type I filter."""
        b_np, a_np = scipy.signal.cheby1(4, 0.5, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_elliptic(self) -> None:
        """Test lfilter with an elliptic filter."""
        b_np, a_np = scipy.signal.ellip(3, 0.5, 40, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_fir_only(self) -> None:
        """Test lfilter with FIR filter (a = [1])."""
        b = torch.tensor([0.2, 0.3, 0.3, 0.2], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_long_signal(self) -> None:
        """Test lfilter with a long signal."""
        b_np, a_np = scipy.signal.butter(4, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(10000, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )


class TestLfilterWithInitialConditions:
    """Test lfilter with initial conditions."""

    def test_lfilter_with_zi(self) -> None:
        """Test lfilter returns (y, zf) when zi is provided."""
        b_np, a_np = scipy.signal.butter(4, 0.25)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        zi = lfilter_zi(b, a)
        x = torch.randn(100, dtype=torch.float64)

        result = lfilter(b, a, x, zi=zi * x[0])
        assert isinstance(result, tuple)
        assert len(result) == 2

        y, zf = result

        # Compare with scipy
        y_scipy, zf_scipy = scipy.signal.lfilter(
            b_np, a_np, x.numpy(), zi=zi.numpy() * x[0].item()
        )

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            zf, torch.from_numpy(zf_scipy), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_step_response_no_transient(self) -> None:
        """Test that zi removes transient from step input."""
        b_np, a_np = scipy.signal.butter(5, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        zi = lfilter_zi(b, a)
        x = torch.ones(50, dtype=torch.float64)

        y, _ = lfilter(b, a, x, zi=zi)

        # Output should be all ones (steady-state)
        torch.testing.assert_close(
            y, torch.ones_like(y), rtol=1e-10, atol=1e-10
        )

    def test_lfilter_zi_continuation(self) -> None:
        """Test that zf from one call can be used as zi for next call."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        # Generate full signal and split
        x_full = torch.randn(200, dtype=torch.float64)
        x1 = x_full[:100]
        x2 = x_full[100:]

        # Filter full signal in one go
        y_full = lfilter(b, a, x_full)

        # Filter in two parts using zi/zf
        zi = torch.zeros(len(a) - 1, dtype=torch.float64)
        y1, zf1 = lfilter(b, a, x1, zi=zi)
        y2, _ = lfilter(b, a, x2, zi=zf1)

        # Concatenate should match full filtering
        y_concat = torch.cat([y1, y2])
        torch.testing.assert_close(y_concat, y_full, rtol=1e-10, atol=1e-10)


class TestLfilterAxis:
    """Test lfilter axis parameter."""

    def test_lfilter_axis_minus_1(self) -> None:
        """Test filtering along last axis (default)."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y = lfilter(b, a, x, axis=-1)

        # Check each row
        for i in range(10):
            y_scipy = scipy.signal.lfilter(b_np, a_np, x[i].numpy())
            torch.testing.assert_close(
                y[i], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
            )

    def test_lfilter_axis_0(self) -> None:
        """Test filtering along axis 0."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, 10, dtype=torch.float64)

        y = lfilter(b, a, x, axis=0)

        # Check each column
        for j in range(10):
            y_scipy = scipy.signal.lfilter(b_np, a_np, x[:, j].numpy())
            torch.testing.assert_close(
                y[:, j], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
            )

    def test_lfilter_axis_1(self) -> None:
        """Test filtering along axis 1 of 3D tensor."""
        b_np, a_np = scipy.signal.butter(2, 0.4)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(5, 100, 8, dtype=torch.float64)

        y = lfilter(b, a, x, axis=1)

        # Check a few slices
        for i in range(5):
            for k in range(8):
                y_scipy = scipy.signal.lfilter(b_np, a_np, x[i, :, k].numpy())
                torch.testing.assert_close(
                    y[i, :, k],
                    torch.from_numpy(y_scipy),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_lfilter_axis_with_zi(self) -> None:
        """Test axis parameter works with initial conditions."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        # zi shape should match: (batch..., n_order)
        n_order = max(len(b), len(a)) - 1
        zi = torch.zeros(10, n_order, dtype=torch.float64)

        y, zf = lfilter(b, a, x, axis=-1, zi=zi)

        # Check shape
        assert y.shape == x.shape
        assert zf.shape == zi.shape

        # Compare first row with scipy
        y_scipy, zf_scipy = scipy.signal.lfilter(
            b_np, a_np, x[0].numpy(), zi=zi[0].numpy()
        )
        torch.testing.assert_close(
            y[0], torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )


class TestLfilterGradients:
    """Test lfilter gradient computation."""

    def test_gradcheck_signal(self) -> None:
        """Gradient check for input signal."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(50, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda x_: lfilter(b, a, x_),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_filter_coeffs(self) -> None:
        """Gradient check for filter coefficients."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(50, dtype=torch.float64)

        torch.autograd.gradcheck(
            lambda b_: lfilter(b_, a, x),
            (b,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_both(self) -> None:
        """Gradient check for both filter coefficients and signal."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        x = torch.randn(50, dtype=torch.float64, requires_grad=True)
        # Note: a has fixed leading coefficient of 1.0
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)

        def fn(b_, x_):
            return lfilter(b_, a, x_)

        torch.autograd.gradcheck(
            fn,
            (b, x),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestLfilterDtypeAndDevice:
    """Test lfilter dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float32)
        a = torch.tensor([1.0, -0.3], dtype=torch.float32)
        x = torch.randn(100, dtype=torch.float32)

        y = lfilter(b, a, x)

        assert y.dtype == torch.float32
        assert y.shape == x.shape

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)

        assert y.dtype == torch.float64
        assert y.shape == x.shape

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64, device="cpu")
        a = torch.tensor([1.0, -0.3], dtype=torch.float64, device="cpu")
        x = torch.randn(100, dtype=torch.float64, device="cpu")

        y = lfilter(b, a, x)

        assert y.device == x.device


class TestLfilterEdgeCases:
    """Test lfilter edge cases."""

    def test_unnormalized_a(self) -> None:
        """Test with a[0] != 1."""
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        a = torch.tensor([2.0, -0.6], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_b_longer_than_a(self) -> None:
        """Test when b has more coefficients than a."""
        b = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1], dtype=torch.float64)
        a = torch.tensor([1.0, -0.5], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_a_longer_than_b(self) -> None:
        """Test when a has more coefficients than b."""
        b = torch.tensor([0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.5, 0.2, -0.1], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_single_sample(self) -> None:
        """Test with single sample input."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        for shape in [(100,), (10, 100), (5, 10, 100)]:
            x = torch.randn(*shape, dtype=torch.float64)
            y = lfilter(b, a, x)
            assert y.shape == x.shape


class TestLfilterComplex:
    """Test lfilter with complex inputs."""

    def test_complex_signal(self) -> None:
        """Test with complex input signal."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.complex128)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        assert y.is_complex()
        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )

    def test_complex_coefficients(self) -> None:
        """Test with complex filter coefficients."""
        b = torch.tensor([0.5 + 0.1j, 0.5 - 0.1j], dtype=torch.complex128)
        a = torch.tensor([1.0, -0.3 + 0.1j], dtype=torch.complex128)
        x = torch.randn(100, dtype=torch.float64)

        y = lfilter(b, a, x)
        y_scipy = scipy.signal.lfilter(b.numpy(), a.numpy(), x.numpy())

        assert y.is_complex()
        torch.testing.assert_close(
            y, torch.from_numpy(y_scipy), rtol=1e-10, atol=1e-10
        )
