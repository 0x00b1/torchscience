# tests/torchscience/distance/test__minkowski_distance.py
import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.distance import minkowski_distance


class TestMinkowskiDistanceBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_two_inputs(self):
        """Output shape is (m, n) for two input tensors."""
        x = torch.randn(5, 3)
        y = torch.randn(7, 3)

        dist = minkowski_distance(x, y, p=2.0)

        assert dist.shape == (5, 7)

    def test_output_shape_self_distance(self):
        """Output shape is (m, m) for self-pairwise distance."""
        x = torch.randn(4, 3)

        dist = minkowski_distance(x, p=2.0)

        assert dist.shape == (4, 4)

    def test_self_distance_diagonal_zero(self):
        """Diagonal of self-pairwise distance matrix is zero."""
        x = torch.randn(5, 3)

        dist = minkowski_distance(x, p=2.0)

        diagonal = torch.diag(dist)
        torch.testing.assert_close(
            diagonal, torch.zeros(5), rtol=1e-6, atol=1e-6
        )

    def test_distances_non_negative(self):
        """All distances are non-negative."""
        x = torch.randn(10, 4)
        y = torch.randn(8, 4)

        dist = minkowski_distance(x, y, p=2.0)

        assert (dist >= 0).all()

    def test_symmetric_for_self_distance(self):
        """Self-pairwise distance matrix is symmetric."""
        x = torch.randn(5, 3, dtype=torch.float64)

        dist = minkowski_distance(x, p=2.0)

        torch.testing.assert_close(dist, dist.T, rtol=1e-6, atol=1e-6)


class TestMinkowskiDistanceCorrectness:
    """Tests for numerical correctness against known values."""

    def test_euclidean_distance(self):
        """p=2 matches Euclidean distance."""
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        y = torch.tensor([[0.0, 0.0]])

        dist = minkowski_distance(x, y, p=2.0)

        expected = torch.tensor([[0.0], [5.0]])
        torch.testing.assert_close(dist, expected, rtol=1e-6, atol=1e-6)

    def test_manhattan_distance(self):
        """p=1 matches Manhattan distance."""
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        y = torch.tensor([[0.0, 0.0]])

        dist = minkowski_distance(x, y, p=1.0)

        expected = torch.tensor([[0.0], [7.0]])
        torch.testing.assert_close(dist, expected, rtol=1e-6, atol=1e-6)

    def test_matches_torch_cdist_p1(self):
        """Matches torch.cdist for p=1."""
        x = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(7, 5, dtype=torch.float64)

        ours = minkowski_distance(x, y, p=1.0)
        theirs = torch.cdist(x, y, p=1.0)

        torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-10)

    def test_matches_torch_cdist_p2(self):
        """Matches torch.cdist for p=2."""
        x = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(7, 5, dtype=torch.float64)

        ours = minkowski_distance(x, y, p=2.0)
        theirs = torch.cdist(x, y, p=2.0)

        torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-10)

    def test_matches_torch_cdist_p3(self):
        """Matches torch.cdist for p=3."""
        x = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(7, 5, dtype=torch.float64)

        ours = minkowski_distance(x, y, p=3.0)
        theirs = torch.cdist(x, y, p=3.0)

        torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-10)

    def test_weighted_distance(self):
        """Weighted distance computation is correct."""
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[1.0, 1.0]])
        w = torch.tensor([1.0, 4.0])

        dist = minkowski_distance(x, y, p=2.0, weight=w)

        # sqrt(1.0 * 1.0^2 + 4.0 * 1.0^2) = sqrt(5.0)
        expected = torch.tensor([[math.sqrt(5.0)]])
        torch.testing.assert_close(dist, expected, rtol=1e-6, atol=1e-6)

    def test_weighted_distance_p1(self):
        """Weighted Manhattan distance is correct."""
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[2.0, 3.0]])
        w = torch.tensor([1.0, 2.0])

        dist = minkowski_distance(x, y, p=1.0, weight=w)

        # 1.0 * |2.0| + 2.0 * |3.0| = 2.0 + 6.0 = 8.0
        expected = torch.tensor([[8.0]])
        torch.testing.assert_close(dist, expected, rtol=1e-6, atol=1e-6)

    def test_fractional_p(self):
        """Works correctly with fractional p values (0 < p < 1)."""
        x = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

        dist = minkowski_distance(x, y, p=0.5)

        # (|1|^0.5 + |1|^0.5)^(1/0.5) = (1 + 1)^2 = 4
        expected = torch.tensor([[4.0]], dtype=torch.float64)
        torch.testing.assert_close(dist, expected, rtol=1e-10, atol=1e-10)

    def test_large_p(self):
        """Works correctly with large p values."""
        x = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64)

        dist = minkowski_distance(x, y, p=10.0)

        # (1^10 + 2^10 + 0.5^10)^(1/10) ≈ 2.0035
        expected = (1.0**10 + 2.0**10 + 0.5**10) ** (1.0 / 10.0)
        torch.testing.assert_close(
            dist,
            torch.tensor([[expected]], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        )


class TestMinkowskiDistanceValidation:
    """Tests for input validation and error handling."""

    def test_x_must_be_2d(self):
        """Raise ValueError when x is not 2D."""
        x = torch.randn(5)
        y = torch.randn(3, 2)

        with pytest.raises(ValueError, match="x must be 2D"):
            minkowski_distance(x, y, p=2.0)

    def test_y_must_be_2d(self):
        """Raise ValueError when y is not 2D."""
        x = torch.randn(5, 2)
        y = torch.randn(3)

        with pytest.raises(ValueError, match="y must be 2D"):
            minkowski_distance(x, y, p=2.0)

    def test_dimension_mismatch(self):
        """Raise ValueError when feature dimensions don't match."""
        x = torch.randn(5, 3)
        y = torch.randn(4, 2)

        with pytest.raises(ValueError, match="Feature dimensions must match"):
            minkowski_distance(x, y, p=2.0)

    def test_p_must_be_positive(self):
        """Raise ValueError when p <= 0."""
        x = torch.randn(5, 3)

        with pytest.raises(ValueError, match="p must be > 0"):
            minkowski_distance(x, p=0.0)

        with pytest.raises(ValueError, match="p must be > 0"):
            minkowski_distance(x, p=-1.0)

    def test_weight_must_be_1d(self):
        """Raise ValueError when weight is not 1D."""
        x = torch.randn(5, 3)
        w = torch.randn(3, 3)

        with pytest.raises(ValueError, match="weight must be 1D"):
            minkowski_distance(x, p=2.0, weight=w)

    def test_weight_size_mismatch(self):
        """Raise ValueError when weight size doesn't match feature dim."""
        x = torch.randn(5, 3)
        w = torch.randn(4)

        with pytest.raises(
            ValueError, match="weight size .* must match feature dim"
        ):
            minkowski_distance(x, p=2.0, weight=w)

    def test_weight_must_be_non_negative(self):
        """Raise ValueError when weight contains negative values."""
        x = torch.randn(5, 3)
        w = torch.tensor([1.0, -1.0, 1.0])

        with pytest.raises(ValueError, match="weight must be non-negative"):
            minkowski_distance(x, p=2.0, weight=w)


class TestMinkowskiDistanceGradient:
    """Tests for gradient computation and autograd support."""

    def test_gradient_exists(self):
        """Gradient exists for differentiable inputs."""
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

        dist = minkowski_distance(x, y, p=2.0)
        loss = dist.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None

    def test_gradient_is_finite(self):
        """Gradient values are finite (not NaN or Inf)."""
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

        dist = minkowski_distance(x, y, p=2.0)
        loss = dist.sum()
        loss.backward()

        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()

    def test_gradcheck_p1(self):
        """Gradient is correct for p=1 (Manhattan distance)."""
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

        def func(x_val, y_val):
            return minkowski_distance(x_val, y_val, p=1.0)

        assert gradcheck(func, (x, y), eps=1e-6, atol=1e-4)

    def test_gradcheck_p2(self):
        """Gradient is correct for p=2 (Euclidean distance)."""
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

        def func(x_val, y_val):
            return minkowski_distance(x_val, y_val, p=2.0)

        assert gradcheck(func, (x, y), eps=1e-6, atol=1e-4)

    def test_gradcheck_p3(self):
        """Gradient is correct for p=3."""
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

        def func(x_val, y_val):
            return minkowski_distance(x_val, y_val, p=3.0)

        assert gradcheck(func, (x, y), eps=1e-6, atol=1e-4)

    def test_gradcheck_with_weights(self):
        """Gradient is correct with weighted distance."""
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        y = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        w = torch.rand(2, dtype=torch.float64)

        def func(x_val, y_val):
            return minkowski_distance(x_val, y_val, p=2.0, weight=w)

        assert gradcheck(func, (x, y), eps=1e-6, atol=1e-4)

    def test_gradient_weight_parameter(self):
        """Gradient computation works when weight has requires_grad."""
        x = torch.randn(3, 2, dtype=torch.float64)
        y = torch.randn(4, 2, dtype=torch.float64)
        w = torch.rand(2, dtype=torch.float64, requires_grad=True)

        dist = minkowski_distance(x, y, p=2.0, weight=w)
        loss = dist.sum()
        loss.backward()

        assert w.grad is not None
        assert torch.isfinite(w.grad).all()


class TestMinkowskiDistanceDtype:
    """Tests for different data type support."""

    def test_float32(self):
        """Works correctly with float32."""
        x = torch.randn(5, 3, dtype=torch.float32)
        y = torch.randn(4, 3, dtype=torch.float32)

        dist = minkowski_distance(x, y, p=2.0)

        assert dist.dtype == torch.float32
        assert dist.shape == (5, 4)

    def test_float64(self):
        """Works correctly with float64."""
        x = torch.randn(5, 3, dtype=torch.float64)
        y = torch.randn(4, 3, dtype=torch.float64)

        dist = minkowski_distance(x, y, p=2.0)

        assert dist.dtype == torch.float64
        assert dist.shape == (5, 4)

    def test_mixed_dtype_x_y(self):
        """Handles mixed dtypes for x and y."""
        x = torch.randn(5, 3, dtype=torch.float32)
        y = torch.randn(4, 3, dtype=torch.float64)

        # Should work due to PyTorch's type promotion
        dist = minkowski_distance(x, y, p=2.0)

        assert dist.shape == (5, 4)

    def test_weight_dtype_promotion(self):
        """Weight dtype is promoted to match inputs."""
        x = torch.randn(5, 3, dtype=torch.float64)
        y = torch.randn(4, 3, dtype=torch.float64)
        w = torch.ones(3, dtype=torch.float32)

        dist = minkowski_distance(x, y, p=2.0, weight=w)

        assert dist.dtype == torch.float64


class TestMinkowskiDistanceDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Works correctly on CPU."""
        device = torch.device("cpu")
        x = torch.randn(5, 3, device=device)
        y = torch.randn(4, 3, device=device)

        dist = minkowski_distance(x, y, p=2.0)

        assert dist.device.type == "cpu"
        assert dist.shape == (5, 4)

    def test_preserves_device(self):
        """Output device matches input device."""
        device = torch.device("cpu")
        x = torch.randn(5, 3, device=device)

        dist = minkowski_distance(x, p=2.0)

        assert dist.device == x.device

    def test_weight_device_matches(self):
        """Weight device should match input device."""
        device = torch.device("cpu")
        x = torch.randn(5, 3, device=device)
        y = torch.randn(4, 3, device=device)
        w = torch.ones(3, device=device)

        dist = minkowski_distance(x, y, p=2.0, weight=w)

        assert dist.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMinkowskiDistanceCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Basic functionality works on CUDA."""
        device = torch.device("cuda")
        x = torch.randn(5, 3, device=device)
        y = torch.randn(4, 3, device=device)

        dist = minkowski_distance(x, y, p=2.0)

        assert dist.device.type == "cuda"
        assert dist.shape == (5, 4)

    def test_cuda_matches_cpu(self):
        """CUDA results match CPU results."""
        x_cpu = torch.randn(10, 5, dtype=torch.float64)
        y_cpu = torch.randn(7, 5, dtype=torch.float64)

        dist_cpu = minkowski_distance(x_cpu, y_cpu, p=2.0)

        x_cuda = x_cpu.cuda()
        y_cuda = y_cpu.cuda()
        dist_cuda = minkowski_distance(x_cuda, y_cuda, p=2.0)

        torch.testing.assert_close(
            dist_cpu, dist_cuda.cpu(), rtol=1e-10, atol=1e-10
        )

    def test_cuda_gradient(self):
        """Gradient computation works on CUDA."""
        device = torch.device("cuda")
        x = torch.randn(
            5, 3, dtype=torch.float64, device=device, requires_grad=True
        )
        y = torch.randn(
            4, 3, dtype=torch.float64, device=device, requires_grad=True
        )

        dist = minkowski_distance(x, y, p=2.0)
        loss = dist.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.device.type == "cuda"
        assert y.grad.device.type == "cuda"
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()

    def test_cuda_weighted(self):
        """Weighted distance works on CUDA."""
        device = torch.device("cuda")
        x = torch.randn(5, 3, device=device)
        y = torch.randn(4, 3, device=device)
        w = torch.ones(3, device=device)

        dist = minkowski_distance(x, y, p=2.0, weight=w)

        assert dist.device.type == "cuda"
        assert dist.shape == (5, 4)
