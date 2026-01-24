"""Tests for Butterworth analog lowpass prototype."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.filter_design import butterworth_prototype


class TestButterworthPrototypeForward:
    """Test butterworth_prototype forward correctness."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_poles_match_scipy(self, order: int) -> None:
        """Poles should match scipy.signal.buttap."""
        zeros, poles, gain = butterworth_prototype(order, dtype=torch.float64)

        # Get scipy reference
        z_scipy, p_scipy, k_scipy = scipy_signal.buttap(order)

        # Check zeros (should be empty for Butterworth)
        assert zeros.numel() == 0
        assert len(z_scipy) == 0

        # Check poles (sort by angle for comparison)
        p_np = poles.numpy()
        p_sorted = sorted(p_np, key=lambda x: (x.real, x.imag))
        p_scipy_sorted = sorted(p_scipy, key=lambda x: (x.real, x.imag))

        for p_ts, p_sp in zip(p_sorted, p_scipy_sorted):
            assert abs(p_ts - p_sp) < 1e-10, f"Pole mismatch: {p_ts} vs {p_sp}"

        # Check gain
        assert abs(gain.item() - k_scipy) < 1e-10

    def test_order_1_single_real_pole(self) -> None:
        """Order 1 should have single real pole at -1."""
        zeros, poles, gain = butterworth_prototype(1, dtype=torch.float64)

        assert zeros.numel() == 0
        assert poles.numel() == 1
        assert abs(poles[0].real + 1.0) < 1e-10
        assert abs(poles[0].imag) < 1e-10
        assert abs(gain.item() - 1.0) < 1e-10

    def test_poles_on_unit_circle(self) -> None:
        """All poles should lie on the unit circle."""
        for order in range(1, 9):
            zeros, poles, gain = butterworth_prototype(
                order, dtype=torch.float64
            )
            for pole in poles:
                magnitude = abs(pole)
                assert abs(magnitude - 1.0) < 1e-10, (
                    f"Pole {pole} not on unit circle"
                )

    def test_poles_in_left_half_plane(self) -> None:
        """All poles should be in the left half-plane (stable)."""
        for order in range(1, 9):
            zeros, poles, gain = butterworth_prototype(
                order, dtype=torch.float64
            )
            for pole in poles:
                assert pole.real < 1e-10, f"Pole {pole} not in left half-plane"


class TestButterworthPrototypeDtypes:
    """Test butterworth_prototype dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype: torch.dtype) -> None:
        """Output should match requested dtype."""
        zeros, poles, gain = butterworth_prototype(4, dtype=dtype)

        # Poles are complex, so check the underlying real dtype
        if dtype == torch.float32:
            assert poles.dtype == torch.complex64
        else:
            assert poles.dtype == torch.complex128
        assert gain.dtype == dtype


class TestButterworthPrototypeDevice:
    """Test butterworth_prototype device handling."""

    def test_cpu_device(self) -> None:
        """Should work on CPU."""
        zeros, poles, gain = butterworth_prototype(
            4, device=torch.device("cpu")
        )
        assert poles.device.type == "cpu"
        assert gain.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self) -> None:
        """Should work on CUDA."""
        zeros, poles, gain = butterworth_prototype(
            4, device=torch.device("cuda")
        )
        assert poles.device.type == "cuda"
        assert gain.device.type == "cuda"


class TestButterworthPrototypeEdgeCases:
    """Test butterworth_prototype edge cases."""

    def test_invalid_order_zero(self) -> None:
        """Order 0 should raise error."""
        with pytest.raises((ValueError, RuntimeError)):
            butterworth_prototype(0)

    def test_invalid_order_negative(self) -> None:
        """Negative order should raise error."""
        with pytest.raises((ValueError, RuntimeError)):
            butterworth_prototype(-1)
