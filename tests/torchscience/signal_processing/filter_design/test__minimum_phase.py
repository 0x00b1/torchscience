"""Tests for minimum phase filter conversion."""

import pytest
import torch
from scipy.signal import minimum_phase as scipy_minimum_phase

from torchscience.signal_processing.filter_design import firwin, minimum_phase


class TestMinimumPhase:
    """Test minimum_phase function."""

    def test_matches_scipy_hilbert(self) -> None:
        """Output should match scipy for hilbert method."""
        h = firwin(65, 0.3, filter_type="lowpass", dtype=torch.float64)
        h_min = minimum_phase(h, method="hilbert")
        h_scipy = scipy_minimum_phase(h.numpy(), method="hilbert")

        min_len = min(len(h_min), len(h_scipy))
        torch.testing.assert_close(
            h_min[:min_len],
            torch.from_numpy(h_scipy[:min_len]),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_matches_scipy_homomorphic(self) -> None:
        """Output should match scipy for homomorphic method."""
        h = firwin(65, 0.3, filter_type="lowpass", dtype=torch.float64)
        h_min = minimum_phase(h, method="homomorphic")
        h_scipy = scipy_minimum_phase(h.numpy(), method="homomorphic")

        min_len = min(len(h_min), len(h_scipy))
        torch.testing.assert_close(
            h_min[:min_len],
            torch.from_numpy(h_scipy[:min_len]),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_magnitude_response_square_root(self) -> None:
        """Minimum phase filter should have sqrt of original magnitude response.

        Both the hilbert and homomorphic methods produce a filter with half
        the length whose magnitude response approximates the square root of
        the original filter's magnitude response.
        """
        h = firwin(65, 0.3, filter_type="lowpass", dtype=torch.float64)
        h_min = minimum_phase(h, method="homomorphic")

        n_fft = 1024
        H = torch.fft.fft(h, n=n_fft)
        H_min = torch.fft.fft(h_min, n=n_fft)

        # The minimum phase filter should have magnitude ~ sqrt(original)
        # Compare in the passband where the response is significant
        passband = slice(0, int(0.3 * n_fft // 2))
        torch.testing.assert_close(
            torch.sqrt(H.abs()[passband]),
            H_min.abs()[passband],
            rtol=5e-2,
            atol=1e-3,
        )

    def test_output_length(self) -> None:
        """Output length should be (n+1)//2."""
        h = firwin(65, 0.3, dtype=torch.float64)
        h_min = minimum_phase(h)
        assert len(h_min) == (65 + 1) // 2  # 33

    def test_output_length_even(self) -> None:
        """Output length for even-length input should be n//2."""
        h = firwin(64, 0.3, dtype=torch.float64)
        h_min = minimum_phase(h)
        assert len(h_min) == 64 // 2  # 32

    def test_dtype_preservation(self) -> None:
        """Should preserve dtype."""
        h32 = torch.randn(21, dtype=torch.float32)
        h64 = torch.randn(21, dtype=torch.float64)

        h_min32 = minimum_phase(h32)
        h_min64 = minimum_phase(h64)

        assert h_min32.dtype == torch.float32
        assert h_min64.dtype == torch.float64

    def test_invalid_method(self) -> None:
        """Should raise for invalid method."""
        h = torch.randn(21, dtype=torch.float64)
        with pytest.raises(ValueError, match="Unknown method"):
            minimum_phase(h, method="invalid")

    def test_custom_n_fft(self) -> None:
        """Should accept custom n_fft parameter."""
        h = firwin(65, 0.3, dtype=torch.float64)
        h_min = minimum_phase(h, n_fft=4096)
        assert len(h_min) == (65 + 1) // 2

    def test_device_preservation(self) -> None:
        """Should preserve device (CPU test)."""
        h = torch.randn(21, dtype=torch.float64, device="cpu")
        h_min = minimum_phase(h)
        assert h_min.device == h.device
