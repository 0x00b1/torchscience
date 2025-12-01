"""Tests for torchscience.statistics.descriptive.kurtosis."""

import math

import pytest
import torch

import torchscience.statistics.descriptive


class TestKurtosisBasic:
    """Basic functionality tests."""

    def test_scalar_output(self):
        """Test kurtosis over all elements returns scalar."""
        x = torch.randn(100)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.dim() == 0

    def test_1d_tensor(self):
        """Test kurtosis of 1D tensor."""
        x = torch.randn(1000)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.shape == ()
        assert torch.isfinite(result)

    def test_2d_tensor_all_dims(self):
        """Test kurtosis over all dimensions of 2D tensor."""
        x = torch.randn(10, 20)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.shape == ()

    def test_2d_tensor_dim0(self):
        """Test kurtosis along dim 0."""
        x = torch.randn(10, 20)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=0)
        assert result.shape == (20,)

    def test_2d_tensor_dim1(self):
        """Test kurtosis along dim 1."""
        x = torch.randn(10, 20)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1)
        assert result.shape == (10,)

    def test_3d_tensor_multiple_dims(self):
        """Test kurtosis along multiple dimensions."""
        x = torch.randn(5, 10, 20)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=(0, 2))
        assert result.shape == (10,)

    def test_negative_dim(self):
        """Test kurtosis with negative dimension."""
        x = torch.randn(10, 20, 30)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=-1)
        assert result.shape == (10, 20)


class TestKurtosisKeepdim:
    """Tests for keepdim parameter."""

    def test_keepdim_true_single_dim(self):
        """Test keepdim=True with single dimension."""
        x = torch.randn(10, 20)
        result = torchscience.statistics.descriptive.kurtosis(
            x, dim=1, keepdim=True
        )
        assert result.shape == (10, 1)

    def test_keepdim_true_multiple_dims(self):
        """Test keepdim=True with multiple dimensions."""
        x = torch.randn(5, 10, 20)
        result = torchscience.statistics.descriptive.kurtosis(
            x, dim=(0, 2), keepdim=True
        )
        assert result.shape == (1, 10, 1)

    def test_keepdim_true_all_dims(self):
        """Test keepdim=True with all dimensions."""
        x = torch.randn(5, 10, 20)
        result = torchscience.statistics.descriptive.kurtosis(x, keepdim=True)
        assert result.shape == (1, 1, 1)


class TestKurtosisCorrectness:
    """Tests for numerical correctness against known values."""

    def test_normal_distribution_excess(self):
        """Normal distribution should have excess kurtosis near 0."""
        torch.manual_seed(42)
        x = torch.randn(10000)
        result = torchscience.statistics.descriptive.kurtosis(x, fisher=True)
        # Excess kurtosis of normal is 0, with some sampling error
        assert abs(result.item()) < 0.3

    def test_normal_distribution_pearson(self):
        """Normal distribution should have Pearson's kurtosis near 3."""
        torch.manual_seed(42)
        x = torch.randn(10000)
        result = torchscience.statistics.descriptive.kurtosis(x, fisher=False)
        # Pearson's kurtosis of normal is 3
        assert abs(result.item() - 3.0) < 0.3

    def test_uniform_distribution(self):
        """Uniform distribution should have negative excess kurtosis."""
        torch.manual_seed(42)
        x = torch.rand(10000)
        result = torchscience.statistics.descriptive.kurtosis(x, fisher=True)
        # Uniform has excess kurtosis of -1.2
        assert result.item() < 0
        assert abs(result.item() + 1.2) < 0.3

    def test_known_values(self):
        """Test against hand-computed values."""
        # Simple sequence [1, 2, 3, 4, 5]
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # Manual computation:
        # mean = 3.0
        # m2 = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
        #    = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
        # m4 = ((1-3)^4 + (2-3)^4 + (3-3)^4 + (4-3)^4 + (5-3)^4) / 5
        #    = (16 + 1 + 0 + 1 + 16) / 5 = 6.8
        # g2 = m4 / m2^2 - 3 = 6.8 / 4 - 3 = 1.7 - 3 = -1.3

        result = torchscience.statistics.descriptive.kurtosis(
            x, fisher=True, bias=True
        )
        torch.testing.assert_close(
            result, torch.tensor(-1.3), rtol=1e-5, atol=1e-5
        )


class TestKurtosisFisherParameter:
    """Tests for fisher parameter."""

    def test_fisher_true_vs_false_difference(self):
        """Fisher=True and Fisher=False should differ by 3."""
        x = torch.randn(100)
        excess = torchscience.statistics.descriptive.kurtosis(x, fisher=True)
        pearson = torchscience.statistics.descriptive.kurtosis(x, fisher=False)
        torch.testing.assert_close(
            pearson - excess, torch.tensor(3.0), rtol=1e-5, atol=1e-5
        )


class TestKurtosisBiasParameter:
    """Tests for bias parameter."""

    def test_biased_vs_unbiased(self):
        """Biased and unbiased estimates should differ."""
        torch.manual_seed(42)
        x = torch.randn(20)
        biased = torchscience.statistics.descriptive.kurtosis(x, bias=True)
        unbiased = torchscience.statistics.descriptive.kurtosis(x, bias=False)
        # They should be different (unless by chance)
        assert biased.item() != unbiased.item()

    def test_unbiased_needs_4_elements(self):
        """Unbiased kurtosis requires n > 3."""
        x = torch.randn(3)
        result = torchscience.statistics.descriptive.kurtosis(x, bias=False)
        assert torch.isnan(result)

    def test_unbiased_with_4_elements(self):
        """Unbiased kurtosis should work with exactly 4 elements."""
        x = torch.randn(4)
        result = torchscience.statistics.descriptive.kurtosis(x, bias=False)
        assert torch.isfinite(result)


class TestKurtosisEdgeCases:
    """Tests for edge cases."""

    def test_constant_tensor(self):
        """Constant tensor (zero variance) should return NaN."""
        x = torch.ones(10)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert torch.isnan(result)

    def test_single_element(self):
        """Single element should return NaN."""
        x = torch.tensor([1.0])
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert torch.isnan(result)

    def test_two_elements(self):
        """Two elements should work for biased kurtosis."""
        x = torch.tensor([1.0, 2.0])
        result = torchscience.statistics.descriptive.kurtosis(x, bias=True)
        assert torch.isfinite(result)


class TestKurtosisDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(100, dtype=dtype)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.dtype == dtype
        assert torch.isfinite(result)

    def test_float16_support(self):
        """Test float16 support."""
        x = torch.randn(100, dtype=torch.float16)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.dtype == torch.float16


class TestKurtosisGradient:
    """Tests for gradient computation."""

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        x = torch.randn(50, requires_grad=True, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x)
        result.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        x = torch.randn(50, requires_grad=True, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x)
        result.backward()
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_dim(self):
        """Test gradient with dimension specified."""
        x = torch.randn(10, 20, requires_grad=True, dtype=torch.float64)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.parametrize("fisher", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    def test_gradcheck_scalar_reduction(self, fisher, bias):
        """Test gradient correctness with gradcheck."""
        x = torch.randn(20, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.statistics.descriptive.kurtosis(
                input_tensor, fisher=fisher, bias=bias
            )

        # Skip unbiased if not enough elements
        if not bias and x.numel() <= 3:
            pytest.skip("Not enough elements for unbiased kurtosis")

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-3, rtol=0.05
        )

    def test_gradcheck_dim_reduction(self):
        """Test gradient correctness with dimension reduction."""
        x = torch.randn(5, 20, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.statistics.descriptive.kurtosis(
                input_tensor, dim=1
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-3, rtol=0.05
        )


class TestKurtosisComplex:
    """Tests for complex tensor support."""

    def test_complex_input_real_output(self):
        """Complex input should produce real output."""
        x = torch.randn(100, dtype=torch.complex64)
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.dtype == torch.float32

    def test_complex_uses_magnitudes(self):
        """Kurtosis of complex should use magnitudes."""
        # Create complex tensor with known magnitudes
        magnitudes = torch.randn(100).abs()
        phases = torch.rand(100) * 2 * math.pi
        x = magnitudes * torch.exp(1j * phases)

        result_complex = torchscience.statistics.descriptive.kurtosis(x)
        result_magnitudes = torchscience.statistics.descriptive.kurtosis(
            magnitudes
        )

        torch.testing.assert_close(
            result_complex, result_magnitudes, rtol=1e-4, atol=1e-4
        )


class TestKurtosisDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(100, device=torch.device("cpu"))
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(100, device=torch.device("cuda"))
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_cpu_consistency(self):
        """Test that CPU and CUDA give same results."""
        torch.manual_seed(42)
        x_cpu = torch.randn(100, dtype=torch.float64)
        x_cuda = x_cpu.cuda()

        result_cpu = torchscience.statistics.descriptive.kurtosis(x_cpu)
        result_cuda = torchscience.statistics.descriptive.kurtosis(x_cuda)

        torch.testing.assert_close(
            result_cpu, result_cuda.cpu(), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_gradient(self):
        """Test gradient computation on CUDA."""
        x = torch.randn(
            50, requires_grad=True, dtype=torch.float64, device="cuda"
        )
        result = torchscience.statistics.descriptive.kurtosis(x)
        result.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestKurtosisTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""

        def fn(x):
            return torchscience.statistics.descriptive.kurtosis(x)

        compiled_fn = torch.compile(fn)
        x = torch.randn(100)
        result = compiled_fn(x)
        assert torch.isfinite(result)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_with_dim(self):
        """Test torch.compile with dimension argument."""

        def fn(x):
            return torchscience.statistics.descriptive.kurtosis(x, dim=1)

        compiled_fn = torch.compile(fn)
        x = torch.randn(10, 100)
        result = compiled_fn(x)
        assert result.shape == (10,)


class TestKurtosisSciPyCompatibility:
    """Tests for SciPy compatibility."""

    def test_matches_scipy_biased_excess(self):
        """Test that results match SciPy with bias=True, fisher=True."""
        pytest.importorskip("scipy")
        from scipy import stats

        torch.manual_seed(42)
        x_np = torch.randn(100).numpy()
        x_torch = torch.from_numpy(x_np)

        scipy_result = stats.kurtosis(x_np, fisher=True, bias=True)
        torch_result = torchscience.statistics.descriptive.kurtosis(
            x_torch, fisher=True, bias=True
        ).item()

        assert abs(scipy_result - torch_result) < 1e-5

    def test_matches_scipy_unbiased_excess(self):
        """Test that results match SciPy with bias=False, fisher=True."""
        pytest.importorskip("scipy")
        from scipy import stats

        torch.manual_seed(42)
        x_np = torch.randn(100).numpy()
        x_torch = torch.from_numpy(x_np)

        scipy_result = stats.kurtosis(x_np, fisher=True, bias=False)
        torch_result = torchscience.statistics.descriptive.kurtosis(
            x_torch, fisher=True, bias=False
        ).item()

        assert abs(scipy_result - torch_result) < 1e-5

    def test_matches_scipy_pearson(self):
        """Test that results match SciPy with fisher=False."""
        pytest.importorskip("scipy")
        from scipy import stats

        torch.manual_seed(42)
        x_np = torch.randn(100).numpy()
        x_torch = torch.from_numpy(x_np)

        scipy_result = stats.kurtosis(x_np, fisher=False, bias=True)
        torch_result = torchscience.statistics.descriptive.kurtosis(
            x_torch, fisher=False, bias=True
        ).item()

        assert abs(scipy_result - torch_result) < 1e-5


class TestKurtosisBatched:
    """Tests for batched computation."""

    def test_batched_matches_loop(self):
        """Test that batched computation matches looped computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 100)

        # Batched
        batched_result = torchscience.statistics.descriptive.kurtosis(x, dim=1)

        # Looped
        looped_results = []
        for i in range(5):
            looped_results.append(
                torchscience.statistics.descriptive.kurtosis(x[i])
            )
        looped_result = torch.stack(looped_results)

        torch.testing.assert_close(
            batched_result, looped_result, rtol=1e-5, atol=1e-5
        )

    def test_large_batch(self):
        """Test with large batch size."""
        x = torch.randn(1000, 50)
        result = torchscience.statistics.descriptive.kurtosis(x, dim=1)
        assert result.shape == (1000,)
        assert torch.all(torch.isfinite(result))
