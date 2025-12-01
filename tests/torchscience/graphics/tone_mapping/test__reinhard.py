"""Tests for Reinhard tone mapping operator."""

import torch
from torch.autograd import gradcheck


class TestReinhardBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_matches_input(self):
        """Output shape matches input shape."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, 3) * 10.0  # HDR values

        result = reinhard(input_hdr)

        assert result.shape == input_hdr.shape

    def test_output_range_basic(self):
        """Basic reinhard maps to [0, 1) for non-negative input."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(100, 3) * 100.0  # Large HDR values

        result = reinhard(input_hdr)

        assert (result >= 0).all()
        assert (result < 1).all()

    def test_output_range_extended(self):
        """Extended reinhard with white_point maps to [0, 1]."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(100, 3) * 10.0
        white_point = torch.tensor(10.0)

        result = reinhard(input_hdr, white_point=white_point)

        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-6).all()


class TestReinhardCorrectness:
    """Tests for numerical correctness."""

    def test_basic_formula(self):
        """Basic reinhard: L_out = L_in / (1 + L_in)."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float64)

        result = reinhard(input_hdr)

        expected = input_hdr / (1 + input_hdr)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_extended_formula(self):
        """Extended reinhard: L_out = L_in * (1 + L_in/L_w^2) / (1 + L_in)."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
        white_point = torch.tensor(4.0, dtype=torch.float64)

        result = reinhard(input_hdr, white_point=white_point)

        L_w_sq = white_point**2
        expected = input_hdr * (1 + input_hdr / L_w_sq) / (1 + input_hdr)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_white_point_maps_to_one(self):
        """Input at white_point maps to 1.0."""
        from torchscience.graphics.tone_mapping import reinhard

        white_point = torch.tensor(8.0, dtype=torch.float64)
        input_hdr = white_point.clone()

        result = reinhard(input_hdr.unsqueeze(0), white_point=white_point)

        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_zero_input(self):
        """Zero input maps to zero."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.zeros(5, dtype=torch.float64)

        result = reinhard(input_hdr)

        torch.testing.assert_close(result, torch.zeros(5, dtype=torch.float64))


class TestReinhardGradients:
    """Tests for gradient computation."""

    def test_gradcheck_basic(self):
        """Passes gradcheck for basic reinhard."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = (
            torch.rand(5, dtype=torch.float64, requires_grad=True) * 5.0 + 0.1
        )

        def func(x):
            return reinhard(x)

        assert gradcheck(func, (input_hdr,), raise_exception=True)

    def test_gradcheck_extended(self):
        """Passes gradcheck for extended reinhard with white_point."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = (
            torch.rand(5, dtype=torch.float64, requires_grad=True) * 5.0 + 0.1
        )
        white_point = torch.tensor(
            10.0, dtype=torch.float64, requires_grad=True
        )

        def func(x, wp):
            return reinhard(x, white_point=wp)

        assert gradcheck(func, (input_hdr, white_point), raise_exception=True)


class TestReinhardDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, dtype=torch.float32) * 10.0

        result = reinhard(input_hdr)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.tone_mapping import reinhard

        input_hdr = torch.rand(10, dtype=torch.float64) * 10.0

        result = reinhard(input_hdr)

        assert result.dtype == torch.float64
