"""Tests for analog frequency response functions (freqs_zpk, freqs_ba)."""

import numpy as np
import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_design import (
    butterworth_prototype,
    freqs_ba,
    freqs_zpk,
    zpk_to_ba,
)


class TestFreqsZpkForward:
    """Test freqs_zpk forward correctness."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_freqs_zpk_matches_scipy(self, order: int) -> None:
        """freqs_zpk should match scipy.signal.freqs_zpk."""
        # Get Butterworth prototype
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        # Frequency points
        worN = torch.logspace(-2, 2, 100, dtype=torch.float64)

        # Compute with torchscience
        w_ts, h_ts = freqs_zpk(zeros, poles, gain, worN)

        # Compute with scipy
        z_np = zeros.numpy()
        p_np = poles.numpy()
        k_np = gain.item()
        w_scipy, h_scipy = scipy_signal.freqs_zpk(
            z_np, p_np, k_np, worN.numpy()
        )

        # Compare
        np.testing.assert_allclose(w_ts.numpy(), w_scipy, rtol=1e-10)
        np.testing.assert_allclose(h_ts.numpy(), h_scipy, rtol=1e-10)

    def test_freqs_zpk_butterworth_prototype_3db_point(self) -> None:
        """Butterworth prototype should have -3dB at cutoff (w=1)."""
        for order in range(1, 7):
            zeros, poles, gain = butterworth_prototype(
                order, dtype=torch.float64
            )

            # Evaluate at cutoff frequency w=1
            worN = torch.tensor([1.0], dtype=torch.float64)
            w, h = freqs_zpk(zeros, poles, gain, worN)

            # Magnitude should be 1/sqrt(2) at cutoff (-3dB point)
            magnitude = torch.abs(h[0]).item()
            expected = 1.0 / np.sqrt(2)
            assert abs(magnitude - expected) < 1e-10, (
                f"Order {order}: got {magnitude}, expected {expected}"
            )

    def test_freqs_zpk_dc_gain(self) -> None:
        """DC gain (w=0) should equal k for all-pole filter."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        worN = torch.tensor([0.0], dtype=torch.float64)
        w, h = freqs_zpk(zeros, poles, gain, worN)

        # For all-pole Butterworth, DC gain should be 1.0
        dc_gain = torch.abs(h[0]).item()
        assert abs(dc_gain - 1.0) < 1e-10

    def test_freqs_zpk_high_frequency_rolloff(self) -> None:
        """High frequency response should roll off for lowpass."""
        zeros, poles, gain = butterworth_prototype(4, dtype=torch.float64)

        # Evaluate at high frequency
        worN = torch.tensor([100.0], dtype=torch.float64)
        w, h = freqs_zpk(zeros, poles, gain, worN)

        # Magnitude should be very small
        magnitude = torch.abs(h[0]).item()
        assert magnitude < 1e-6

    def test_freqs_zpk_single_pole(self) -> None:
        """Test single pole system H(s) = 1/(s+1)."""
        zeros = torch.empty(0, dtype=torch.complex128)
        poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        gain = torch.tensor(1.0, dtype=torch.float64)

        worN = torch.tensor([0.0, 1.0, 10.0], dtype=torch.float64)
        w, h = freqs_zpk(zeros, poles, gain, worN)

        # At w=0: H(0) = 1/(0+1) = 1
        assert abs(h[0].item() - 1.0) < 1e-10

        # At w=1: H(j) = 1/(j+1), |H| = 1/sqrt(2)
        assert abs(torch.abs(h[1]).item() - 1.0 / np.sqrt(2)) < 1e-10

        # At w=10: H(10j) = 1/(10j+1), |H| = 1/sqrt(101)
        assert abs(torch.abs(h[2]).item() - 1.0 / np.sqrt(101)) < 1e-10

    def test_freqs_zpk_with_zeros(self) -> None:
        """Test system with both zeros and poles."""
        # H(s) = (s+2)/(s+1) has zero at -2, pole at -1
        zeros = torch.tensor([-2.0 + 0j], dtype=torch.complex128)
        poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        gain = torch.tensor(1.0, dtype=torch.float64)

        worN = torch.tensor([0.0], dtype=torch.float64)
        w, h = freqs_zpk(zeros, poles, gain, worN)

        # At w=0: H(0) = (0+2)/(0+1) = 2
        assert abs(h[0].item() - 2.0) < 1e-10


class TestFreqsZpkDtypes:
    """Test freqs_zpk dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype: torch.dtype) -> None:
        """Output h should be complex with matching precision."""
        if dtype == torch.float32:
            complex_dtype = torch.complex64
        else:
            complex_dtype = torch.complex128

        zeros = torch.empty(0, dtype=complex_dtype)
        poles = torch.tensor([-1.0 + 0j], dtype=complex_dtype)
        gain = torch.tensor(1.0, dtype=dtype)
        worN = torch.tensor([1.0], dtype=dtype)

        w, h = freqs_zpk(zeros, poles, gain, worN)

        assert w.dtype == dtype
        assert h.dtype == complex_dtype


class TestFreqsZpkDevice:
    """Test freqs_zpk device handling."""

    def test_cpu_device(self) -> None:
        """Should work on CPU."""
        zeros = torch.empty(0, dtype=torch.complex128, device="cpu")
        poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128, device="cpu")
        gain = torch.tensor(1.0, dtype=torch.float64, device="cpu")
        worN = torch.tensor([1.0], dtype=torch.float64, device="cpu")

        w, h = freqs_zpk(zeros, poles, gain, worN)

        assert w.device.type == "cpu"
        assert h.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self) -> None:
        """Should work on CUDA."""
        zeros = torch.empty(0, dtype=torch.complex128, device="cuda")
        poles = torch.tensor(
            [-1.0 + 0j], dtype=torch.complex128, device="cuda"
        )
        gain = torch.tensor(1.0, dtype=torch.float64, device="cuda")
        worN = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        w, h = freqs_zpk(zeros, poles, gain, worN)

        assert w.device.type == "cuda"
        assert h.device.type == "cuda"


class TestFreqsBaForward:
    """Test freqs_ba forward correctness."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_freqs_ba_matches_scipy(self, order: int) -> None:
        """freqs_ba should match scipy.signal.freqs."""
        # Get Butterworth prototype and convert to BA
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)
        b, a = zpk_to_ba(zeros, poles, gain)

        # Frequency points
        worN = torch.logspace(-2, 2, 100, dtype=torch.float64)

        # Compute with torchscience
        w_ts, h_ts = freqs_ba(b, a, worN)

        # Compute with scipy
        w_scipy, h_scipy = scipy_signal.freqs(
            b.numpy(), a.numpy(), worN.numpy()
        )

        # Compare
        np.testing.assert_allclose(w_ts.numpy(), w_scipy, rtol=1e-10)
        np.testing.assert_allclose(h_ts.numpy(), h_scipy, rtol=1e-8)

    def test_freqs_ba_simple_first_order(self) -> None:
        """Test simple first-order lowpass H(s) = 1/(s+1)."""
        # b = [1], a = [1, 1] represents H(s) = 1/(s+1)
        b = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([1.0, 1.0], dtype=torch.float64)

        worN = torch.tensor([0.0, 1.0], dtype=torch.float64)
        w, h = freqs_ba(b, a, worN)

        # At w=0: H(0) = 1
        assert abs(h[0].item() - 1.0) < 1e-10

        # At w=1: |H(j)| = 1/sqrt(2)
        assert abs(torch.abs(h[1]).item() - 1.0 / np.sqrt(2)) < 1e-10

    def test_freqs_ba_second_order(self) -> None:
        """Test second-order system H(s) = 1/(s^2 + s + 1)."""
        b = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        worN = torch.tensor([0.0, 1.0], dtype=torch.float64)
        w, h = freqs_ba(b, a, worN)

        # At w=0: H(0) = 1/(0+0+1) = 1
        assert abs(h[0].item() - 1.0) < 1e-10

        # At w=1: H(j) = 1/(j^2 + j + 1) = 1/(-1 + j + 1) = 1/j = -j
        expected = -1j
        assert abs(h[1].item() - expected) < 1e-10

    def test_freqs_ba_with_numerator(self) -> None:
        """Test system with non-trivial numerator H(s) = (s+1)/(s+2)."""
        # H(s) = (s+1)/(s+2) => b = [1, 1], a = [1, 2]
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)

        worN = torch.tensor([0.0], dtype=torch.float64)
        w, h = freqs_ba(b, a, worN)

        # At w=0: H(0) = (0+1)/(0+2) = 0.5
        assert abs(h[0].item() - 0.5) < 1e-10


class TestFreqsBaDtypes:
    """Test freqs_ba dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype: torch.dtype) -> None:
        """Output h should be complex with matching precision."""
        if dtype == torch.float32:
            complex_dtype = torch.complex64
        else:
            complex_dtype = torch.complex128

        b = torch.tensor([1.0], dtype=dtype)
        a = torch.tensor([1.0, 1.0], dtype=dtype)
        worN = torch.tensor([1.0], dtype=dtype)

        w, h = freqs_ba(b, a, worN)

        assert w.dtype == dtype
        assert h.dtype == complex_dtype


class TestFreqsBaDevice:
    """Test freqs_ba device handling."""

    def test_cpu_device(self) -> None:
        """Should work on CPU."""
        b = torch.tensor([1.0], dtype=torch.float64, device="cpu")
        a = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cpu")
        worN = torch.tensor([1.0], dtype=torch.float64, device="cpu")

        w, h = freqs_ba(b, a, worN)

        assert w.device.type == "cpu"
        assert h.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self) -> None:
        """Should work on CUDA."""
        b = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        a = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")
        worN = torch.tensor([1.0], dtype=torch.float64, device="cuda")

        w, h = freqs_ba(b, a, worN)

        assert w.device.type == "cuda"
        assert h.device.type == "cuda"


class TestFreqsConsistency:
    """Test consistency between freqs_zpk and freqs_ba."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_zpk_ba_consistency(self, order: int) -> None:
        """freqs_zpk and freqs_ba should give identical results."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)
        b, a = zpk_to_ba(zeros, poles, gain)

        worN = torch.logspace(-2, 2, 100, dtype=torch.float64)

        w_zpk, h_zpk = freqs_zpk(zeros, poles, gain, worN)
        w_ba, h_ba = freqs_ba(b, a, worN)

        np.testing.assert_allclose(h_zpk.numpy(), h_ba.numpy(), rtol=1e-8)


class TestFreqsEdgeCases:
    """Test edge cases for freqs functions."""

    def test_freqs_zpk_empty_frequencies(self) -> None:
        """Empty frequency array should return empty response."""
        zeros = torch.empty(0, dtype=torch.complex128)
        poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        gain = torch.tensor(1.0, dtype=torch.float64)
        worN = torch.empty(0, dtype=torch.float64)

        w, h = freqs_zpk(zeros, poles, gain, worN)

        assert w.numel() == 0
        assert h.numel() == 0

    def test_freqs_ba_empty_frequencies(self) -> None:
        """Empty frequency array should return empty response."""
        b = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([1.0, 1.0], dtype=torch.float64)
        worN = torch.empty(0, dtype=torch.float64)

        w, h = freqs_ba(b, a, worN)

        assert w.numel() == 0
        assert h.numel() == 0

    def test_freqs_zpk_no_zeros_no_poles(self) -> None:
        """System with no zeros and no poles should return gain."""
        zeros = torch.empty(0, dtype=torch.complex128)
        poles = torch.empty(0, dtype=torch.complex128)
        gain = torch.tensor(2.5, dtype=torch.float64)
        worN = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        w, h = freqs_zpk(zeros, poles, gain, worN)

        # H(s) = k = 2.5 for all frequencies
        expected = torch.full((3,), 2.5, dtype=torch.complex128)
        np.testing.assert_allclose(h.numpy(), expected.numpy(), rtol=1e-10)

    def test_freqs_ba_constant_gain(self) -> None:
        """System H(s) = k should return constant gain."""
        b = torch.tensor([2.5], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)
        worN = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        w, h = freqs_ba(b, a, worN)

        # H(s) = 2.5 for all frequencies
        expected = torch.full((3,), 2.5, dtype=torch.complex128)
        np.testing.assert_allclose(h.numpy(), expected.numpy(), rtol=1e-10)

    def test_freqs_zpk_scalar_gain(self) -> None:
        """Should accept float gain as well as tensor."""
        zeros = torch.empty(0, dtype=torch.complex128)
        poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        worN = torch.tensor([1.0], dtype=torch.float64)

        # Test with float gain
        w, h = freqs_zpk(zeros, poles, 1.0, worN)

        assert abs(torch.abs(h[0]).item() - 1.0 / np.sqrt(2)) < 1e-10
