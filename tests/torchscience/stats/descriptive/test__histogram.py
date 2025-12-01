"""Tests for torchscience.statistics.descriptive.histogram."""

import pytest
import torch

import torchscience.statistics.descriptive


class TestHistogramBasic:
    """Basic functionality tests."""

    def test_1d_histogram_shape(self):
        """Test 1D histogram returns correct shapes."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.shape == (10,)
        assert edges.shape == (11,)

    def test_1d_histogram_counts_sum(self):
        """Test histogram counts sum to input size."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.sum().item() == 1000

    def test_1d_histogram_edges_monotonic(self):
        """Test that bin edges are monotonically increasing."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert torch.all(edges[1:] > edges[:-1])

    def test_explicit_bin_edges(self):
        """Test histogram with explicit bin edges."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        counts, returned_edges = torchscience.statistics.descriptive.histogram(
            x, bins=edges
        )
        assert counts.shape == (5,)
        torch.testing.assert_close(returned_edges, edges)
        # Each value falls in its own bin
        torch.testing.assert_close(counts, torch.ones(5))


class TestHistogramRange:
    """Tests for range parameter."""

    def test_explicit_range(self):
        """Test histogram with explicit range."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10, range=(-3.0, 3.0)
        )
        assert edges[0].item() == -3.0
        assert edges[-1].item() == 3.0

    def test_auto_range(self):
        """Test histogram auto-computes range from data."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=4
        )
        assert edges[0].item() == pytest.approx(1.0)
        assert edges[-1].item() == pytest.approx(5.0)


class TestHistogramWeights:
    """Tests for weights parameter."""

    def test_uniform_weights(self):
        """Test that uniform weights give same result as unweighted."""
        x = torch.randn(1000)
        weights = torch.ones(1000)

        counts_unweighted, _ = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        counts_weighted, _ = torchscience.statistics.descriptive.histogram(
            x, bins=10, weight=weights
        )

        torch.testing.assert_close(counts_unweighted, counts_weighted)

    def test_zero_weights(self):
        """Test that zero weights exclude samples."""
        x = torch.tensor([0.5, 1.5, 2.5])
        weights = torch.tensor([1.0, 0.0, 1.0])
        edges = torch.tensor([0.0, 1.0, 2.0, 3.0])

        counts, _ = torchscience.statistics.descriptive.histogram(
            x, bins=edges, weight=weights
        )
        # Middle sample has zero weight
        torch.testing.assert_close(counts, torch.tensor([1.0, 0.0, 1.0]))


class TestHistogramDensity:
    """Tests for density parameter."""

    def test_density_integrates_to_one(self):
        """Test that density histogram integrates to 1."""
        x = torch.randn(10000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=50, density=True
        )
        bin_widths = edges[1:] - edges[:-1]
        integral = (counts * bin_widths).sum()
        assert integral.item() == pytest.approx(1.0, rel=1e-3)


class TestHistogramDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(100, dtype=dtype)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.dtype == dtype
        assert edges.dtype == dtype


class TestHistogramDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(100, device="cpu")
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.device.type == "cpu"
        assert edges.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(100, device="cuda")
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.device.type == "cuda"
        assert edges.device.type == "cuda"


class TestHistogramNotDifferentiable:
    """Tests confirming histogram is not differentiable."""

    def test_no_gradient(self):
        """Test that histogram does not propagate gradients to input."""
        x = torch.randn(100, requires_grad=True)
        with pytest.warns(
            UserWarning, match="autograd kernel was not registered"
        ):
            counts, edges = torchscience.statistics.descriptive.histogram(
                x, bins=10
            )
            # Histogram computation is not differentiable. The C++ operator uses
            # PyTorch's autograd fallback (WarnNotImplemented) which allows
            # backward pass but doesn't compute meaningful gradients.
            if counts.requires_grad:
                counts.sum().backward()
        # Input gradient should remain None since histogram is not differentiable
        assert x.grad is None


class TestHistogramSciPyCompatibility:
    """Tests for SciPy/NumPy compatibility."""

    def test_matches_numpy(self):
        """Test that results match NumPy histogram."""
        np = pytest.importorskip("numpy")

        torch.manual_seed(42)
        x_torch = torch.randn(1000)
        x_np = x_torch.numpy()

        counts_torch, edges_torch = (
            torchscience.statistics.descriptive.histogram(x_torch, bins=20)
        )
        counts_np, edges_np = np.histogram(x_np, bins=20)

        torch.testing.assert_close(
            counts_torch,
            torch.from_numpy(counts_np).float(),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            edges_torch,
            torch.from_numpy(edges_np).float(),
            rtol=1e-5,
            atol=1e-5,
        )


class TestHistogramEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor_raises(self):
        """Test that empty tensor raises ValueError."""
        x = torch.tensor([])
        with pytest.raises(ValueError, match="non-empty"):
            torchscience.statistics.descriptive.histogram(x, bins=10)

    def test_single_element(self):
        """Test histogram of single element."""
        x = torch.tensor([1.0])
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=1
        )
        assert counts.shape == (1,)
        assert counts[0].item() == 1.0

    def test_all_same_value(self):
        """Test histogram when all values are identical."""
        x = torch.ones(100)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        # All values should be in one bin
        assert counts.sum().item() == 100
        assert (counts > 0).sum().item() == 1
