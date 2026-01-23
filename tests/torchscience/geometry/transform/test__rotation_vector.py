"""Tests for RotationVector tensorclass and conversion functions."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    Quaternion,
    RotationVector,
    matrix_to_rotation_vector,
    quaternion,
    quaternion_normalize,
    quaternion_to_rotation_vector,
    rotation_vector,
    rotation_vector_to_matrix,
    rotation_vector_to_quaternion,
)


class TestRotationVectorConstruction:
    """Tests for RotationVector construction."""

    def test_from_tensor(self):
        """Create RotationVector from tensor."""
        vector = torch.tensor([0.0, 0.0, math.pi / 2])
        rv = RotationVector(vector=vector)
        assert rv.vector.shape == (3,)
        assert torch.allclose(rv.vector, vector)

    def test_batch(self):
        """Batch of rotation vectors."""
        vector = torch.randn(10, 3)
        rv = RotationVector(vector=vector)
        assert rv.vector.shape == (10, 3)

    def test_factory_function(self):
        """Create via rotation_vector() factory."""
        vector = torch.tensor([0.0, 0.0, math.pi / 2])
        rv = rotation_vector(vector)
        assert isinstance(rv, RotationVector)
        assert torch.allclose(rv.vector, vector)

    def test_invalid_shape(self):
        """Raise error for wrong last dimension."""
        with pytest.raises(ValueError, match="last dimension 3"):
            rotation_vector(torch.randn(4))


class TestRotationVectorToQuaternion:
    """Tests for rotation_vector_to_quaternion."""

    def test_zero_vector_identity(self):
        """Zero rotation vector gives identity quaternion."""
        rv = rotation_vector(torch.tensor([0.0, 0.0, 0.0]))
        q = rotation_vector_to_quaternion(rv)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
        q = rotation_vector_to_quaternion(rv)
        # q = [cos(45), 0, 0, sin(45)]
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_x(self):
        """90-degree rotation around x-axis."""
        rv = rotation_vector(torch.tensor([math.pi / 2, 0.0, 0.0]))
        q = rotation_vector_to_quaternion(rv)
        expected = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_y(self):
        """90-degree rotation around y-axis."""
        rv = rotation_vector(torch.tensor([0.0, math.pi / 2, 0.0]))
        q = rotation_vector_to_quaternion(rv)
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation.from_rotvec()."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Random rotation vector
        vector = torch.randn(3, dtype=torch.float64)
        rv = rotation_vector(vector)

        # Convert to quaternion
        q = rotation_vector_to_quaternion(rv)

        # scipy comparison
        r_scipy = R.from_rotvec(vector.numpy())
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        # Either q matches or -q matches (both represent same rotation)
        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_batch(self):
        """Batched conversion."""
        vector = torch.randn(10, 3)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.shape == (10, 4)

    def test_output_is_unit_quaternion(self):
        """Output should be a unit quaternion."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        norms = torch.linalg.norm(q.wxyz, dim=-1)
        assert torch.allclose(
            norms, torch.ones(10, dtype=torch.float64), atol=1e-5
        )

    def test_gradcheck(self):
        """Gradient check."""
        vector = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(vec):
            rv = RotationVector(vector=vec)
            return rotation_vector_to_quaternion(rv).wxyz

        assert gradcheck(fn, (vector,), eps=1e-6, atol=1e-4)


class TestQuaternionToRotationVector:
    """Tests for quaternion_to_rotation_vector."""

    def test_identity_quaternion(self):
        """Identity quaternion gives zero rotation vector."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        rv = quaternion_to_rotation_vector(q)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(rv.vector, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        rv = quaternion_to_rotation_vector(q)
        expected = torch.tensor([0.0, 0.0, math.pi / 2])
        assert torch.allclose(rv.vector, expected, atol=1e-5)

    def test_roundtrip(self):
        """rotation_vector -> quaternion -> rotation_vector roundtrip."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)

        q = rotation_vector_to_quaternion(rv)
        rv_back = quaternion_to_rotation_vector(q)

        # For rotation vectors representing the same rotation
        # (which may differ by multiples of 2*pi in norm)
        # we check via the rotation matrix

        R1 = rotation_vector_to_matrix(rv)
        R2 = rotation_vector_to_matrix(rv_back)
        assert torch.allclose(R1, R2, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        rv = quaternion_to_rotation_vector(q)
        assert rv.vector.shape == (10, 3)

    def test_gradcheck(self):
        """Gradient check."""
        q_raw = torch.randn(5, 4, dtype=torch.float64)
        q_raw = q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True)
        q = q_raw.clone().detach().requires_grad_(True)

        def fn(wxyz):
            qq = Quaternion(wxyz=wxyz)
            rv = quaternion_to_rotation_vector(qq)
            return rv.vector

        assert gradcheck(fn, (q,), eps=1e-6, atol=1e-4)


class TestRotationVectorToMatrix:
    """Tests for rotation_vector_to_matrix."""

    def test_zero_vector_identity(self):
        """Zero rotation vector gives identity matrix."""
        rv = rotation_vector(torch.tensor([0.0, 0.0, 0.0]))
        R = rotation_vector_to_matrix(rv)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
        R = rotation_vector_to_matrix(rv)
        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_x(self):
        """180-degree rotation around x-axis."""
        rv = rotation_vector(torch.tensor([math.pi, 0.0, 0.0]))
        R = rotation_vector_to_matrix(rv)
        expected = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation.from_rotvec().as_matrix()."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Random rotation vector
        vector = torch.randn(3, dtype=torch.float64)
        rv = rotation_vector(vector)

        # Convert to matrix
        mat = rotation_vector_to_matrix(rv)

        # scipy comparison
        r_scipy = R.from_rotvec(vector.numpy())
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_batch(self):
        """Batched conversion."""
        vector = torch.randn(10, 3)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self):
        """R @ R.T = I for valid rotation vectors."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        )
        assert torch.allclose(RRT, expected, atol=1e-5)

    def test_determinant_one(self):
        """det(R) = 1 for valid rotation vectors."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)
        dets = torch.linalg.det(R)
        expected = torch.ones(10, dtype=torch.float64)
        assert torch.allclose(dets, expected, atol=1e-5)

    def test_gradcheck(self):
        """Gradient check."""
        vector = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(vec):
            rv = RotationVector(vector=vec)
            return rotation_vector_to_matrix(rv)

        assert gradcheck(fn, (vector,), eps=1e-6, atol=1e-4)


class TestMatrixToRotationVector:
    """Tests for matrix_to_rotation_vector."""

    def test_identity_matrix(self):
        """Identity matrix gives zero rotation vector."""
        R = torch.eye(3)
        rv = matrix_to_rotation_vector(R)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(rv.vector, expected, atol=1e-5)

    def test_roundtrip(self):
        """rotation_vector -> matrix -> rotation_vector roundtrip."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)

        rv_back = matrix_to_rotation_vector(R)
        R_back = rotation_vector_to_matrix(rv_back)

        assert torch.allclose(R_back, R, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        # Create valid rotation matrices
        vector = torch.randn(10, 3)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)

        rv_back = matrix_to_rotation_vector(R)
        assert rv_back.vector.shape == (10, 3)

    def test_gradcheck(self):
        """Gradient check."""
        # Create valid rotation matrices
        vector = torch.randn(5, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        R = rotation_vector_to_matrix(rv)
        R = R.clone().detach().requires_grad_(True)

        def fn(mat):
            rv = matrix_to_rotation_vector(mat)
            return rv.vector

        assert gradcheck(fn, (R,), eps=1e-6, atol=1e-4)


class TestRotationVectorShape:
    """Tests for shape handling in rotation vector functions."""

    def test_single_rotation_vector(self):
        """Single rotation vector (3,) shape."""
        vector = torch.tensor([0.0, 0.0, math.pi / 2])
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.shape == (4,)
        R = rotation_vector_to_matrix(rv)
        assert R.shape == (3, 3)

    def test_batch(self):
        """Batch of rotation vectors (B, 3) shape."""
        vector = torch.randn(10, 3)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.shape == (10, 4)
        R = rotation_vector_to_matrix(rv)
        assert R.shape == (10, 3, 3)

    def test_multi_batch(self):
        """Multi-batch rotation vectors (B, C, 3) shape."""
        vector = torch.randn(5, 3, 3)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.shape == (5, 3, 4)
        R = rotation_vector_to_matrix(rv)
        assert R.shape == (5, 3, 3, 3)


class TestRotationVectorDtypes:
    """Tests for dtype handling."""

    def test_float32(self):
        """Works with float32."""
        vector = torch.randn(10, 3, dtype=torch.float32)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.dtype == torch.float32
        R = rotation_vector_to_matrix(rv)
        assert R.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        vector = torch.randn(10, 3, dtype=torch.float64)
        rv = rotation_vector(vector)
        q = rotation_vector_to_quaternion(rv)
        assert q.wxyz.dtype == torch.float64
        R = rotation_vector_to_matrix(rv)
        assert R.dtype == torch.float64


class TestRotationVectorIntegration:
    """Integration tests for rotation vector conversions."""

    def test_consistency_with_axis_angle(self):
        """RotationVector should be consistent with AxisAngle."""
        from torchscience.geometry.transform import (
            axis_angle,
            axis_angle_to_quaternion,
        )

        # Create rotation vector
        vector = torch.randn(3, dtype=torch.float64)
        rv = rotation_vector(vector)

        # Equivalent axis-angle
        angle = torch.linalg.norm(vector)
        eps = torch.finfo(vector.dtype).eps
        axis = vector / torch.clamp(angle, min=eps)
        aa = axis_angle(axis, angle)

        # Compare quaternions
        q_rv = rotation_vector_to_quaternion(rv)
        q_aa = axis_angle_to_quaternion(aa)

        # Either q matches or -q matches
        matches = torch.allclose(
            q_rv.wxyz, q_aa.wxyz, atol=1e-5
        ) or torch.allclose(q_rv.wxyz, -q_aa.wxyz, atol=1e-5)
        assert matches

    def test_scipy_full_roundtrip(self):
        """Full roundtrip with scipy validation."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Random rotation vector
        vector = torch.randn(3, dtype=torch.float64)
        rv = rotation_vector(vector)

        # Convert to matrix and back
        mat = rotation_vector_to_matrix(rv)
        rv_back = matrix_to_rotation_vector(mat)

        # scipy validation
        r_scipy = R.from_rotvec(vector.numpy())
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)
        assert torch.allclose(
            rotation_vector_to_matrix(rv_back), mat_scipy, atol=1e-6
        )

    def test_negative_rotation_vector(self):
        """Negative rotation vector should give inverse rotation."""
        vector = torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64)
        rv_pos = rotation_vector(vector)
        rv_neg = rotation_vector(-vector)

        R_pos = rotation_vector_to_matrix(rv_pos)
        R_neg = rotation_vector_to_matrix(rv_neg)

        # R_neg should be the transpose (inverse) of R_pos
        assert torch.allclose(R_neg, R_pos.T, atol=1e-5)
