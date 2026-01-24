import pytest
import torch
import torch.testing
from torchscience.signal_processing.noise import impulse_noise


class TestImpulseNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = impulse_noise([100], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = impulse_noise([4, 100], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = impulse_noise([2, 3, 100], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = impulse_noise([10, 0], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = impulse_noise([0, 100], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0


class TestImpulseNoiseDtype:
    """Tests for dtype handling."""

    def test_default_dtype_float32(self):
        """Test default dtype is float32."""
        result = impulse_noise([100], p_salt=0.1)
        assert result.dtype == torch.float32

    def test_explicit_float64(self):
        """Test float64 dtype."""
        result = impulse_noise([100], p_salt=0.1, dtype=torch.float64)
        assert result.dtype == torch.float64

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_half_precision_dtypes(self, dtype):
        """Test half-precision dtypes."""
        result = impulse_noise([100], p_salt=0.1, dtype=dtype)
        assert result.dtype == dtype


class TestImpulseNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = impulse_noise([100], p_salt=0.1, device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA device."""
        result = impulse_noise([100], p_salt=0.1, device="cuda")
        assert result.device.type == "cuda"


class TestImpulseNoiseStatistics:
    """Tests for statistical properties."""

    def test_salt_proportion(self):
        """Test salt noise proportion matches p_salt."""
        torch.manual_seed(42)
        p_salt = 0.1
        result = impulse_noise([10000], p_salt=p_salt, p_pepper=0.0)
        actual_prop = (result == 1.0).float().mean().item()
        # Allow 2% absolute deviation
        assert abs(actual_prop - p_salt) < 0.02, (
            f"Salt proportion {actual_prop} not near {p_salt}"
        )

    def test_pepper_proportion(self):
        """Test pepper noise proportion matches p_pepper."""
        torch.manual_seed(42)
        p_pepper = 0.1
        result = impulse_noise([10000], p_salt=0.0, p_pepper=p_pepper)
        actual_prop = (result == -1.0).float().mean().item()
        # Allow 2% absolute deviation
        assert abs(actual_prop - p_pepper) < 0.02, (
            f"Pepper proportion {actual_prop} not near {p_pepper}"
        )

    def test_zero_proportion(self):
        """Test zero proportion matches expected."""
        torch.manual_seed(42)
        p_salt = 0.1
        p_pepper = 0.1
        result = impulse_noise([10000], p_salt=p_salt, p_pepper=p_pepper)
        actual_zero = (result == 0.0).float().mean().item()
        expected_zero = 1.0 - p_salt - p_pepper
        # Allow 2% absolute deviation
        assert abs(actual_zero - expected_zero) < 0.02

    def test_only_three_values(self):
        """Test output contains only salt, pepper, or zero values."""
        result = impulse_noise(
            [1000], p_salt=0.2, p_pepper=0.2, salt_value=1.0, pepper_value=-1.0
        )
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in [-1.0, 0.0, 1.0]

    def test_zero_probabilities_all_zeros(self):
        """Test zero probabilities give all zeros."""
        result = impulse_noise([100], p_salt=0.0, p_pepper=0.0)
        assert (result == 0.0).all()

    def test_full_salt_probability(self):
        """Test p_salt=1 gives all salt."""
        result = impulse_noise([100], p_salt=1.0, p_pepper=0.0)
        assert (result == 1.0).all()

    def test_full_pepper_probability(self):
        """Test p_pepper=1 gives all pepper."""
        result = impulse_noise([100], p_salt=0.0, p_pepper=1.0)
        assert (result == -1.0).all()


class TestImpulseNoiseCustomValues:
    """Tests for custom salt/pepper values."""

    def test_custom_salt_value(self):
        """Test custom salt value."""
        torch.manual_seed(42)
        result = impulse_noise([100], p_salt=1.0, salt_value=255.0)
        assert (result == 255.0).all()

    def test_custom_pepper_value(self):
        """Test custom pepper value."""
        torch.manual_seed(42)
        result = impulse_noise([100], p_pepper=1.0, pepper_value=0.0)
        assert (result == 0.0).all()

    def test_image_style_values(self):
        """Test uint8-style values (0 and 255)."""
        torch.manual_seed(42)
        result = impulse_noise(
            [1000],
            p_salt=0.1,
            p_pepper=0.1,
            salt_value=255.0,
            pepper_value=0.0,
        )
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in [0.0, 255.0]


class TestImpulseNoiseTensorProbabilities:
    """Tests for tensor probability parameters."""

    def test_tensor_p_salt_broadcasting(self):
        """Test p_salt tensor broadcasts with size."""
        p_salt = torch.tensor([[0.0], [1.0]])  # [2, 1]
        result = impulse_noise([2, 100], p_salt=p_salt, p_pepper=0.0)
        assert result.shape == torch.Size([2, 100])
        # First row should have no salt
        assert (result[0] == 0.0).all()
        # Second row should have all salt
        assert (result[1] == 1.0).all()

    def test_tensor_p_pepper_broadcasting(self):
        """Test p_pepper tensor broadcasts with size."""
        p_pepper = torch.tensor([[0.0], [1.0]])  # [2, 1]
        result = impulse_noise([2, 100], p_salt=0.0, p_pepper=p_pepper)
        assert result.shape == torch.Size([2, 100])
        # First row should have no pepper
        assert (result[0] == 0.0).all()
        # Second row should have all pepper
        assert (result[1] == -1.0).all()

    def test_spatially_varying_corruption(self):
        """Test spatially varying corruption rates."""
        torch.manual_seed(42)
        # Higher corruption probability in corners
        p = torch.zeros(10, 10)
        p[0, 0] = p[0, -1] = p[-1, 0] = p[-1, -1] = 1.0
        result = impulse_noise([10, 10], p_salt=p)
        # Corners should be salt
        assert result[0, 0] == 1.0
        assert result[0, -1] == 1.0
        assert result[-1, 0] == 1.0
        assert result[-1, -1] == 1.0
        # Other positions should be zero
        assert result[5, 5] == 0.0


class TestImpulseNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = impulse_noise([100], p_salt=0.1, p_pepper=0.1, generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = impulse_noise([100], p_salt=0.1, p_pepper=0.1, generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = impulse_noise([100], p_salt=0.3, p_pepper=0.3, generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = impulse_noise([100], p_salt=0.3, p_pepper=0.3, generator=g2)

        assert not torch.equal(result1, result2)


class TestImpulseNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(impulse_noise)
        result = compiled([100], p_salt=0.1, p_pepper=0.1)
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = impulse_noise([100], p_salt=0.1, p_pepper=0.1, generator=g1)

        compiled = torch.compile(impulse_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled(
            [100], p_salt=0.1, p_pepper=0.1, generator=g2
        )

        torch.testing.assert_close(eager, compiled_result)


class TestImpulseNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = impulse_noise([100000], p_salt=0.05, p_pepper=0.05)
        assert result.shape == torch.Size([100000])

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = impulse_noise([100], p_salt=0.1, p_pepper=0.1)
        assert result.is_contiguous()

    def test_single_element(self):
        """Test single element output."""
        result = impulse_noise([1], p_salt=0.5)
        assert result.shape == torch.Size([1])
        assert result.item() in [-1.0, 0.0, 1.0]

    def test_overlapping_probabilities(self):
        """Test behavior when p_salt + p_pepper > 1."""
        # Salt takes precedence (checked second)
        result = impulse_noise([100], p_salt=1.0, p_pepper=1.0)
        # All should be salt because salt mask is applied last
        assert (result == 1.0).all()
