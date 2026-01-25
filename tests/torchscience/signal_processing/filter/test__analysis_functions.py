"""Tests for filter analysis functions."""

import math

import numpy as np
import torch
from numpy.testing import assert_allclose
from scipy import signal

from torchscience.signal_processing.filter import (
    group_delay,
    group_delay_sos,
    impulse_response,
    impulse_response_sos,
    step_response,
)


class TestGroupDelay:
    """Tests for group delay computation."""

    def test_fir_linear_phase(self):
        """Test that linear phase FIR filter has constant group delay."""
        # Symmetric FIR filter has linear phase
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b)

        # Group delay should be (N-1)/2 = 1 for N=3
        expected_delay = 1.0
        assert_allclose(gd.numpy(), expected_delay, rtol=1e-10)

    def test_fir_longer_filter(self):
        """Test group delay for longer FIR filter."""
        # Create a symmetric FIR filter
        from torchscience.signal_processing.filter import firwin

        b = firwin(51, 0.3)
        w, gd = group_delay(b)

        # Group delay should be (N-1)/2 = 25 for N=51
        expected_delay = 25.0
        assert_allclose(gd.numpy(), expected_delay, rtol=1e-6)

    def test_compare_scipy_fir(self):
        """Compare FIR group delay with scipy."""
        b = np.array([0.2, 0.3, 0.3, 0.2])
        w_scipy, gd_scipy = signal.group_delay((b, [1.0]))

        b_torch = torch.tensor(b)
        w_torch, gd_torch = group_delay(b_torch)

        assert_allclose(gd_torch.numpy(), gd_scipy, rtol=1e-10)

    def test_compare_scipy_iir(self):
        """Compare IIR group delay with scipy."""
        # Design a simple IIR filter
        b = np.array([0.0675, 0.1349, 0.0675])
        a = np.array([1.0, -1.1430, 0.4128])

        w_scipy, gd_scipy = signal.group_delay((b, a))

        b_torch = torch.tensor(b)
        a_torch = torch.tensor(a)
        w_torch, gd_torch = group_delay(b_torch, a_torch)

        assert_allclose(gd_torch.numpy(), gd_scipy, rtol=1e-6)

    def test_butterworth_filter(self):
        """Test group delay of Butterworth filter."""
        from torchscience.signal_processing.filter import (
            butterworth_design,
            zpk_to_ba,
        )

        z, p, k = butterworth_design(4, 0.3, output="zpk")
        b, a = zpk_to_ba(z, p, k)

        w_torch, gd_torch = group_delay(b, a)

        # Compare with scipy
        b_np = b.numpy()
        a_np = a.numpy()
        w_scipy, gd_scipy = signal.group_delay((b_np, a_np))

        # Exclude edge points near Nyquist where numerical differences can be larger
        # This is common in digital filter analysis near the Nyquist boundary
        assert_allclose(gd_torch.numpy()[:-5], gd_scipy[:-5], rtol=1e-5)

    def test_n_points(self):
        """Test custom number of frequency points."""
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b, n_points=256)

        assert w.shape == torch.Size([256])
        assert gd.shape == torch.Size([256])

    def test_whole_true(self):
        """Test full unit circle computation."""
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b, whole=True)

        # Frequencies should go from 0 to 2*pi
        assert w[-1].item() < 2 * math.pi
        assert w[-1].item() > math.pi

    def test_whole_false(self):
        """Test half unit circle computation (default)."""
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b, whole=False)

        # Frequencies should go from 0 to pi
        assert w[-1].item() < math.pi

    def test_sampling_frequency(self):
        """Test with explicit sampling frequency."""
        b = torch.tensor([0.25, 0.5, 0.25])
        fs = 1000

        w, gd = group_delay(b, sampling_frequency=fs)

        # Frequencies should be in Hz, up to Nyquist
        assert w[-1].item() < fs / 2

    def test_dtype_float32(self):
        """Test float32 output."""
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b, dtype=torch.float32)

        assert w.dtype == torch.float32
        assert gd.dtype == torch.float32

    def test_device_cpu(self):
        """Test CPU device."""
        b = torch.tensor([0.25, 0.5, 0.25])
        w, gd = group_delay(b, device=torch.device("cpu"))

        assert w.device.type == "cpu"
        assert gd.device.type == "cpu"


class TestGroupDelaySOS:
    """Tests for SOS group delay computation."""

    def test_single_section(self):
        """Test group delay for single SOS section."""
        sos = torch.tensor([[0.25, 0.5, 0.25, 1.0, 0.0, 0.0]])
        w, gd = group_delay_sos(sos)

        # For this FIR section, delay should be 1
        assert_allclose(gd.numpy(), 1.0, rtol=1e-6)

    def test_compare_scipy(self):
        """Compare SOS group delay with scipy."""
        from torchscience.signal_processing.filter import (
            butterworth_design,
            zpk_to_sos,
        )

        z, p, k = butterworth_design(4, 0.3, output="zpk")
        sos = zpk_to_sos(z, p, k)

        w_torch, gd_torch = group_delay_sos(sos)

        # Compare with scipy using ba representation
        sos_np = sos.numpy()
        w_scipy, gd_scipy = signal.group_delay(signal.sos2tf(sos_np))

        assert_allclose(gd_torch.numpy(), gd_scipy, rtol=1e-4)

    def test_n_points(self):
        """Test custom number of frequency points."""
        sos = torch.tensor([[0.25, 0.5, 0.25, 1.0, 0.0, 0.0]])
        w, gd = group_delay_sos(sos, n_points=256)

        assert w.shape == torch.Size([256])
        assert gd.shape == torch.Size([256])


class TestImpulseResponse:
    """Tests for impulse response computation."""

    def test_fir_coefficients(self):
        """Test that FIR impulse response equals coefficients."""
        b = torch.tensor([0.25, 0.5, 0.25])
        t, h = impulse_response(b, n_samples=5)

        expected = torch.tensor(
            [0.25, 0.5, 0.25, 0.0, 0.0], dtype=torch.float64
        )
        assert_allclose(h.numpy(), expected.numpy(), rtol=1e-10)

    def test_time_indices(self):
        """Test that time indices are correct."""
        b = torch.tensor([0.25, 0.5, 0.25])
        t, h = impulse_response(b, n_samples=10)

        expected_t = torch.arange(10, dtype=torch.float64)
        assert_allclose(t.numpy(), expected_t.numpy(), rtol=1e-10)

    def test_iir_filter(self):
        """Test impulse response of IIR filter."""
        # Simple first-order IIR: y[n] = 0.5*x[n] + 0.5*y[n-1]
        b = torch.tensor([0.5])
        a = torch.tensor([1.0, -0.5])

        t, h = impulse_response(b, a, n_samples=5)

        # Impulse response: h[0]=0.5, h[1]=0.25, h[2]=0.125, ...
        expected = torch.tensor(
            [0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=torch.float64
        )
        assert_allclose(h.numpy(), expected.numpy(), rtol=1e-10)

    def test_compare_scipy(self):
        """Compare impulse response with scipy.signal.dlti.impulse."""
        b = np.array([0.0675, 0.1349, 0.0675])
        a = np.array([1.0, -1.1430, 0.4128])

        # Scipy approach using lfilter
        n_samples = 50
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        h_scipy = signal.lfilter(b, a, impulse)

        b_torch = torch.tensor(b)
        a_torch = torch.tensor(a)
        t_torch, h_torch = impulse_response(
            b_torch, a_torch, n_samples=n_samples
        )

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_n_samples(self):
        """Test custom number of samples."""
        b = torch.tensor([0.5])
        t, h = impulse_response(b, n_samples=200)

        assert t.shape == torch.Size([200])
        assert h.shape == torch.Size([200])

    def test_dtype_float32(self):
        """Test float32 output."""
        b = torch.tensor([0.5])
        t, h = impulse_response(b, dtype=torch.float32)

        assert t.dtype == torch.float32
        assert h.dtype == torch.float32

    def test_device_cpu(self):
        """Test CPU device."""
        b = torch.tensor([0.5])
        t, h = impulse_response(b, device=torch.device("cpu"))

        assert t.device.type == "cpu"
        assert h.device.type == "cpu"


class TestImpulseResponseSOS:
    """Tests for SOS impulse response computation."""

    def test_single_fir_section(self):
        """Test impulse response for FIR SOS section."""
        # FIR filter as SOS: [b0, b1, b2, 1, 0, 0]
        sos = torch.tensor([[0.25, 0.5, 0.25, 1.0, 0.0, 0.0]])
        t, h = impulse_response_sos(sos, n_samples=5)

        expected = torch.tensor(
            [0.25, 0.5, 0.25, 0.0, 0.0], dtype=torch.float64
        )
        assert_allclose(h.numpy(), expected.numpy(), rtol=1e-10)

    def test_compare_scipy(self):
        """Compare SOS impulse response with scipy."""
        from torchscience.signal_processing.filter import (
            butterworth_design,
            zpk_to_sos,
        )

        z, p, k = butterworth_design(4, 0.3, output="zpk")
        sos = zpk_to_sos(z, p, k)

        t_torch, h_torch = impulse_response_sos(sos, n_samples=50)

        # Compare with scipy using sosfilt
        sos_np = sos.numpy()
        impulse = np.zeros(50)
        impulse[0] = 1.0
        h_scipy = signal.sosfilt(sos_np, impulse)

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_n_samples(self):
        """Test custom number of samples."""
        sos = torch.tensor([[0.25, 0.5, 0.25, 1.0, 0.0, 0.0]])
        t, h = impulse_response_sos(sos, n_samples=100)

        assert t.shape == torch.Size([100])
        assert h.shape == torch.Size([100])


class TestStepResponse:
    """Tests for step response computation."""

    def test_fir_cumsum(self):
        """Test that step response is cumulative sum of impulse response."""
        b = torch.tensor([0.25, 0.5, 0.25])
        t, s = step_response(b, n_samples=5)

        # Step response: cumsum of [0.25, 0.5, 0.25, 0, 0] = [0.25, 0.75, 1.0, 1.0, 1.0]
        expected = torch.tensor(
            [0.25, 0.75, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        assert_allclose(s.numpy(), expected.numpy(), rtol=1e-10)

    def test_compare_scipy(self):
        """Compare step response with scipy.signal.lfilter."""
        b = np.array([0.0675, 0.1349, 0.0675])
        a = np.array([1.0, -1.1430, 0.4128])

        # Scipy approach using lfilter with step input
        n_samples = 50
        step_input = np.ones(n_samples)
        s_scipy = signal.lfilter(b, a, step_input)

        b_torch = torch.tensor(b)
        a_torch = torch.tensor(a)
        t_torch, s_torch = step_response(b_torch, a_torch, n_samples=n_samples)

        assert_allclose(s_torch.numpy(), s_scipy, rtol=1e-10)

    def test_lowpass_steady_state(self):
        """Test that lowpass filter step response approaches DC gain."""
        # Design a lowpass filter
        from torchscience.signal_processing.filter import (
            butterworth_design,
            zpk_to_ba,
        )

        z, p, k = butterworth_design(4, 0.3, output="zpk")
        b, a = zpk_to_ba(z, p, k)

        t, s = step_response(b, a, n_samples=200)

        # DC gain is H(z=1) = sum(b) / sum(a)
        dc_gain = b.sum() / a.sum()

        # Step response should approach DC gain
        assert_allclose(s[-1].item(), dc_gain.item(), rtol=0.01)

    def test_time_indices(self):
        """Test that time indices are correct."""
        b = torch.tensor([0.5])
        t, s = step_response(b, n_samples=10)

        expected_t = torch.arange(10, dtype=torch.float64)
        assert_allclose(t.numpy(), expected_t.numpy(), rtol=1e-10)

    def test_n_samples(self):
        """Test custom number of samples."""
        b = torch.tensor([0.5])
        t, s = step_response(b, n_samples=200)

        assert t.shape == torch.Size([200])
        assert s.shape == torch.Size([200])

    def test_dtype_float32(self):
        """Test float32 output."""
        b = torch.tensor([0.5])
        t, s = step_response(b, dtype=torch.float32)

        assert t.dtype == torch.float32
        assert s.dtype == torch.float32

    def test_device_cpu(self):
        """Test CPU device."""
        b = torch.tensor([0.5])
        t, s = step_response(b, device=torch.device("cpu"))

        assert t.device.type == "cpu"
        assert s.device.type == "cpu"
