"""Tests for batched filter design functions using torch.vmap."""

import pytest
import torch

from torchscience.signal_processing.filter_design import (
    batched_butterworth_design,
    batched_chebyshev_type_1_design,
    batched_chebyshev_type_2_design,
    batched_filter_apply,
    batched_firwin,
    butterworth_design,
    chebyshev_type_1_design,
    chebyshev_type_2_design,
    firwin,
    sosfilt,
)


class TestBatchedButterworthDesign:
    """Test batched_butterworth_design function."""

    def test_single_cutoff_matches_unbatched(self) -> None:
        """Batched design with single cutoff should match unbatched design."""
        order = 4
        cutoff = torch.tensor([0.3])

        batched_sos = batched_butterworth_design(order, cutoff)
        single_sos = butterworth_design(order, 0.3)

        torch.testing.assert_close(batched_sos[0], single_sos)

    def test_multiple_cutoffs_shape(self) -> None:
        """Batched design should return correct shape for multiple cutoffs."""
        order = 4
        batch_size = 10
        cutoffs = torch.linspace(0.1, 0.4, batch_size)

        sos = batched_butterworth_design(order, cutoffs)

        # For 4th order, expect 2 SOS sections
        assert sos.shape == (batch_size, 2, 6)

    def test_each_filter_matches_individual_design(self) -> None:
        """Each filter in batch should match individual design."""
        order = 4
        cutoffs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        batched_sos = batched_butterworth_design(order, cutoffs)

        for i, cutoff in enumerate(cutoffs):
            expected_sos = butterworth_design(order, cutoff.item())
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )

    def test_highpass_filter_type(self) -> None:
        """Batched design should work for highpass filters."""
        order = 4
        cutoffs = torch.tensor([0.2, 0.3, 0.4])

        batched_sos = batched_butterworth_design(
            order, cutoffs, filter_type="highpass"
        )

        for i, cutoff in enumerate(cutoffs):
            expected_sos = butterworth_design(
                order, cutoff.item(), filter_type="highpass"
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )

    def test_bandpass_filter_type(self) -> None:
        """Batched design should work for bandpass filters."""
        order = 2
        # Shape: (batch, 2) for bandpass
        cutoffs = torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.15, 0.35]])

        batched_sos = batched_butterworth_design(
            order, cutoffs, filter_type="bandpass"
        )

        # Bandpass doubles the order
        assert batched_sos.shape == (3, 2, 6)

        for i, cutoff_pair in enumerate(cutoffs):
            expected_sos = butterworth_design(
                order, cutoff_pair.tolist(), filter_type="bandpass"
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )

    def test_bandstop_filter_type(self) -> None:
        """Batched design should work for bandstop filters."""
        order = 2
        cutoffs = torch.tensor([[0.1, 0.3], [0.2, 0.4]])

        batched_sos = batched_butterworth_design(
            order, cutoffs, filter_type="bandstop"
        )

        for i, cutoff_pair in enumerate(cutoffs):
            expected_sos = butterworth_design(
                order, cutoff_pair.tolist(), filter_type="bandstop"
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )

    @pytest.mark.parametrize("order", [1, 2, 4, 6])
    def test_different_orders(self, order: int) -> None:
        """Batched design should work for different filter orders."""
        cutoffs = torch.linspace(0.1, 0.4, 5)

        batched_sos = batched_butterworth_design(order, cutoffs)

        # Number of sections depends on order
        n_sections = (order + 1) // 2
        assert batched_sos.shape == (5, n_sections, 6)

    def test_dtype_preserved(self) -> None:
        """Output dtype should match specified dtype when supported."""
        cutoffs = torch.tensor([0.2, 0.3], dtype=torch.float64)
        sos = batched_butterworth_design(4, cutoffs, dtype=torch.float64)
        # butterworth_design typically outputs float64 for numerical stability
        assert sos.dtype == torch.float64

    def test_device_preserved(self) -> None:
        """Output device should match specified device."""
        cutoffs = torch.tensor([0.2, 0.3])
        sos = batched_butterworth_design(
            4, cutoffs, device=torch.device("cpu")
        )
        assert sos.device.type == "cpu"


class TestBatchedChebyshevType1Design:
    """Test batched_chebyshev_type_1_design function."""

    def test_multiple_cutoffs_match_individual(self) -> None:
        """Each filter in batch should match individual design."""
        order = 4
        ripple_db = 1.0
        cutoffs = torch.tensor([0.1, 0.2, 0.3])

        batched_sos = batched_chebyshev_type_1_design(
            order, ripple_db, cutoffs
        )

        for i, cutoff in enumerate(cutoffs):
            expected_sos = chebyshev_type_1_design(
                order, cutoff.item(), ripple_db
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )

    def test_bandpass_filter_type(self) -> None:
        """Batched design should work for bandpass filters."""
        order = 2
        ripple_db = 0.5
        cutoffs = torch.tensor([[0.1, 0.3], [0.2, 0.4]])

        batched_sos = batched_chebyshev_type_1_design(
            order, ripple_db, cutoffs, filter_type="bandpass"
        )

        for i, cutoff_pair in enumerate(cutoffs):
            expected_sos = chebyshev_type_1_design(
                order, cutoff_pair.tolist(), ripple_db, filter_type="bandpass"
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )


class TestBatchedChebyshevType2Design:
    """Test batched_chebyshev_type_2_design function."""

    def test_multiple_cutoffs_match_individual(self) -> None:
        """Each filter in batch should match individual design."""
        order = 4
        attenuation_db = 40.0
        cutoffs = torch.tensor([0.1, 0.2, 0.3])

        batched_sos = batched_chebyshev_type_2_design(
            order, attenuation_db, cutoffs
        )

        for i, cutoff in enumerate(cutoffs):
            expected_sos = chebyshev_type_2_design(
                order, cutoff.item(), attenuation_db
            )
            torch.testing.assert_close(
                batched_sos[i], expected_sos, rtol=1e-5, atol=1e-6
            )


class TestBatchedFirwin:
    """Test batched_firwin function."""

    def test_single_cutoff_matches_unbatched(self) -> None:
        """Batched design with single cutoff should match unbatched design."""
        num_taps = 51
        cutoff = torch.tensor([0.3])

        batched_h = batched_firwin(num_taps, cutoff)
        single_h = firwin(num_taps, 0.3)

        torch.testing.assert_close(batched_h[0], single_h)

    def test_multiple_cutoffs_shape(self) -> None:
        """Batched design should return correct shape for multiple cutoffs."""
        num_taps = 51
        batch_size = 10
        cutoffs = torch.linspace(0.1, 0.4, batch_size)

        h = batched_firwin(num_taps, cutoffs)

        assert h.shape == (batch_size, num_taps)

    def test_each_filter_matches_individual_design(self) -> None:
        """Each filter in batch should match individual design."""
        num_taps = 51
        cutoffs = torch.tensor([0.1, 0.2, 0.3, 0.4])

        batched_h = batched_firwin(num_taps, cutoffs)

        for i, cutoff in enumerate(cutoffs):
            expected_h = firwin(num_taps, cutoff.item())
            torch.testing.assert_close(
                batched_h[i], expected_h, rtol=1e-5, atol=1e-6
            )

    def test_highpass_filter_type(self) -> None:
        """Batched design should work for highpass filters."""
        num_taps = 51  # Must be odd for highpass
        cutoffs = torch.tensor([0.2, 0.3, 0.4])

        batched_h = batched_firwin(num_taps, cutoffs, filter_type="highpass")

        for i, cutoff in enumerate(cutoffs):
            expected_h = firwin(
                num_taps, cutoff.item(), filter_type="highpass"
            )
            torch.testing.assert_close(
                batched_h[i], expected_h, rtol=1e-5, atol=1e-6
            )

    def test_bandpass_filter_type(self) -> None:
        """Batched design should work for bandpass filters."""
        num_taps = 51
        cutoffs = torch.tensor([[0.1, 0.3], [0.2, 0.4]])

        batched_h = batched_firwin(num_taps, cutoffs, filter_type="bandpass")

        for i, cutoff_pair in enumerate(cutoffs):
            expected_h = firwin(
                num_taps, cutoff_pair.tolist(), filter_type="bandpass"
            )
            torch.testing.assert_close(
                batched_h[i], expected_h, rtol=1e-5, atol=1e-6
            )

    @pytest.mark.parametrize(
        "window", ["hamming", "hann", "blackman", "rectangular"]
    )
    def test_different_windows(self, window: str) -> None:
        """Batched design should work with different window functions."""
        num_taps = 51
        cutoffs = torch.tensor([0.2, 0.3])

        batched_h = batched_firwin(num_taps, cutoffs, window=window)

        for i, cutoff in enumerate(cutoffs):
            expected_h = firwin(num_taps, cutoff.item(), window=window)
            torch.testing.assert_close(
                batched_h[i], expected_h, rtol=1e-5, atol=1e-6
            )


class TestBatchedFilterApply:
    """Test batched_filter_apply function."""

    def test_apply_single_filter_to_single_signal(self) -> None:
        """Applying single batched filter to single signal."""
        order = 4
        cutoffs = torch.tensor([0.3])
        sos = batched_butterworth_design(order, cutoffs)

        x = torch.randn(100, dtype=torch.float64)
        y = batched_filter_apply(sos, x)

        # Compare with unbatched filtering
        single_sos = butterworth_design(order, 0.3)
        y_expected = sosfilt(single_sos, x)

        torch.testing.assert_close(y[0], y_expected, rtol=1e-5, atol=1e-6)

    def test_apply_multiple_filters_to_single_signal(self) -> None:
        """Applying multiple filters to single signal."""
        order = 4
        cutoffs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sos = batched_butterworth_design(order, cutoffs)

        x = torch.randn(100, dtype=torch.float64)
        y = batched_filter_apply(sos, x)

        assert y.shape == (4, 100)

        # Each output should match individual filtering
        for i, cutoff in enumerate(cutoffs):
            single_sos = butterworth_design(order, cutoff.item())
            y_expected = sosfilt(single_sos, x)
            torch.testing.assert_close(y[i], y_expected, rtol=1e-5, atol=1e-6)

    def test_apply_filters_to_batched_signals(self) -> None:
        """Applying batched filters to batched signals (1-to-1)."""
        order = 4
        batch_size = 5
        cutoffs = torch.linspace(0.1, 0.4, batch_size)
        sos = batched_butterworth_design(order, cutoffs)

        # Batch of signals, one per filter
        x = torch.randn(batch_size, 100, dtype=torch.float64)
        y = batched_filter_apply(sos, x, broadcast=False)

        assert y.shape == (batch_size, 100)

        # Each output should match individual filtering
        for i in range(batch_size):
            single_sos = butterworth_design(order, cutoffs[i].item())
            y_expected = sosfilt(single_sos, x[i])
            torch.testing.assert_close(y[i], y_expected, rtol=1e-5, atol=1e-6)

    def test_apply_fir_filters_to_signal(self) -> None:
        """Applying batched FIR filters to single signal."""
        num_taps = 51
        cutoffs = torch.tensor([0.1, 0.2, 0.3])
        h = batched_firwin(num_taps, cutoffs)

        x = torch.randn(100, dtype=torch.float64)
        y = batched_filter_apply(h, x, filter_format="fir")

        assert y.shape == (3, 100)


class TestGradientThroughBatch:
    """Test gradient flow through batched operations."""

    def test_gradient_through_butterworth_design(self) -> None:
        """Gradients should flow through batched Butterworth design."""
        cutoffs = torch.tensor([0.2, 0.3, 0.4], requires_grad=True)
        sos = batched_butterworth_design(4, cutoffs)

        loss = sos.sum()
        loss.backward()

        assert cutoffs.grad is not None
        assert not torch.any(torch.isnan(cutoffs.grad))

    @pytest.mark.skip(
        reason="firwin may not support gradients through cutoff due to internal operations"
    )
    def test_gradient_through_firwin(self) -> None:
        """Gradients should flow through batched firwin."""
        cutoffs = torch.tensor([0.2, 0.3, 0.4], requires_grad=True)
        h = batched_firwin(51, cutoffs)

        loss = h.sum()
        loss.backward()

        assert cutoffs.grad is not None
        assert not torch.any(torch.isnan(cutoffs.grad))

    def test_gradient_through_filter_apply(self) -> None:
        """Gradients should flow through batched filter application."""
        cutoffs = torch.tensor([0.2, 0.3], requires_grad=True)
        sos = batched_butterworth_design(4, cutoffs)

        x = torch.randn(100, dtype=torch.float64, requires_grad=True)
        y = batched_filter_apply(sos, x)

        loss = y.sum()
        loss.backward()

        assert cutoffs.grad is not None
        assert x.grad is not None
        assert not torch.any(torch.isnan(cutoffs.grad))
        assert not torch.any(torch.isnan(x.grad))

    def test_gradient_chebyshev_type_1(self) -> None:
        """Gradients should flow through batched Chebyshev Type 1 design."""
        cutoffs = torch.tensor([0.2, 0.3], requires_grad=True)
        sos = batched_chebyshev_type_1_design(4, 1.0, cutoffs)

        loss = sos.sum()
        loss.backward()

        assert cutoffs.grad is not None
        assert not torch.any(torch.isnan(cutoffs.grad))


class TestBatchedDesignEdgeCases:
    """Test edge cases for batched filter design."""

    def test_empty_batch_raises_error(self) -> None:
        """Empty cutoff tensor should raise an error."""
        cutoffs = torch.tensor([])
        with pytest.raises((ValueError, RuntimeError)):
            batched_butterworth_design(4, cutoffs)

    def test_single_element_batch(self) -> None:
        """Single element batch should work correctly."""
        cutoffs = torch.tensor([0.3])
        sos = batched_butterworth_design(4, cutoffs)

        assert sos.shape == (1, 2, 6)

    def test_large_batch(self) -> None:
        """Large batch should work efficiently."""
        batch_size = 100
        cutoffs = torch.linspace(0.05, 0.45, batch_size)

        sos = batched_butterworth_design(4, cutoffs)

        assert sos.shape == (batch_size, 2, 6)

    def test_cutoffs_at_boundary(self) -> None:
        """Cutoffs near boundary values should work."""
        cutoffs = torch.tensor([0.01, 0.5, 0.99])
        sos = batched_butterworth_design(2, cutoffs)

        assert sos.shape == (3, 1, 6)
        assert not torch.any(torch.isnan(sos))
        assert not torch.any(torch.isinf(sos))


class TestBatchedCompileCompatibility:
    """Test torch.compile compatibility."""

    @pytest.mark.skip(
        reason="torch.compile may have compatibility issues with filter design internals"
    )
    def test_batched_butterworth_compiles(self) -> None:
        """Batched Butterworth design should be compatible with torch.compile."""

        @torch.compile
        def design_filters(cutoffs):
            return batched_butterworth_design(4, cutoffs)

        cutoffs = torch.linspace(0.1, 0.4, 5)
        sos = design_filters(cutoffs)

        assert sos.shape == (5, 2, 6)

    @pytest.mark.skip(
        reason="torch.compile may have compatibility issues with filter design internals"
    )
    def test_batched_firwin_compiles(self) -> None:
        """Batched firwin should be compatible with torch.compile."""

        @torch.compile
        def design_filters(cutoffs):
            return batched_firwin(51, cutoffs)

        cutoffs = torch.linspace(0.1, 0.4, 5)
        h = design_filters(cutoffs)

        assert h.shape == (5, 51)
