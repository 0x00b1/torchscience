"""Tests for RotationMatrix tensorclass."""

import pytest
import torch

from torchscience.geometry.transform import RotationMatrix, rotation_matrix


class TestRotationMatrixConstruction:
    """Tests for RotationMatrix construction."""

    def test_from_tensor(self):
        """Create RotationMatrix from tensor."""
        mat = torch.eye(3)
        R = RotationMatrix(matrix=mat)
        assert R.matrix.shape == (3, 3)
        assert torch.allclose(R.matrix, mat)

    def test_batch(self):
        """Batch of rotation matrices."""
        mat = torch.randn(10, 3, 3)
        R = RotationMatrix(matrix=mat)
        assert R.matrix.shape == (10, 3, 3)

    def test_multidim_batch(self):
        """Multi-dimensional batch of rotation matrices."""
        mat = torch.randn(5, 7, 3, 3)
        R = RotationMatrix(matrix=mat)
        assert R.matrix.shape == (5, 7, 3, 3)

    def test_factory_function(self):
        """Create via rotation_matrix() factory."""
        mat = torch.eye(3)
        R = rotation_matrix(mat)
        assert isinstance(R, RotationMatrix)
        assert torch.allclose(R.matrix, mat)

    def test_invalid_shape_wrong_last_dims(self):
        """Raise error for wrong last dimensions."""
        with pytest.raises(ValueError, match="last two dimensions"):
            rotation_matrix(torch.randn(4, 4))

    def test_invalid_shape_2x2(self):
        """Raise error for 2x2 matrix."""
        with pytest.raises(ValueError, match="last two dimensions"):
            rotation_matrix(torch.randn(2, 2))

    def test_invalid_shape_1d(self):
        """Raise error for 1D tensor."""
        with pytest.raises(ValueError, match="last two dimensions"):
            rotation_matrix(torch.randn(9))


class TestRotationMatrixDtypes:
    """Tests for RotationMatrix with different dtypes."""

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """RotationMatrix supports various dtypes."""
        mat = torch.eye(3, dtype=dtype)
        R = rotation_matrix(mat)
        assert R.matrix.dtype == dtype


class TestRotationMatrixDevice:
    """Tests for RotationMatrix device handling."""

    def test_cpu(self):
        """RotationMatrix on CPU."""
        mat = torch.eye(3, device="cpu")
        R = rotation_matrix(mat)
        assert R.matrix.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """RotationMatrix on CUDA."""
        mat = torch.eye(3, device="cuda")
        R = rotation_matrix(mat)
        assert R.matrix.device.type == "cuda"
