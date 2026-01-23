"""Tests for Rotation6D tensorclass and conversion functions.

Reference: Zhou et al., "On the Continuity of Rotation Representations in
Neural Networks", CVPR 2019.
"""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    Quaternion,
    Rotation6D,
    matrix_to_rotation_6d,
    quaternion,
    quaternion_normalize,
    quaternion_to_matrix,
    quaternion_to_rotation_6d,
    rotation_6d,
    rotation_6d_to_matrix,
    rotation_6d_to_quaternion,
)


class TestRotation6DConstruction:
    """Tests for Rotation6D construction."""

    def test_basic_creation(self):
        """Create Rotation6D from tensor."""
        # Identity rotation: first two columns of I are [1,0,0] and [0,1,0]
        vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        r6d = Rotation6D(vectors=vectors)
        assert r6d.vectors.shape == (6,)
        assert torch.allclose(r6d.vectors, vectors)

    def test_batch_creation(self):
        """Batch of 6D rotation vectors."""
        vectors = torch.randn(10, 6)
        r6d = Rotation6D(vectors=vectors)
        assert r6d.vectors.shape == (10, 6)

    def test_factory_function(self):
        """Create via rotation_6d() factory."""
        vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        r6d = rotation_6d(vectors)
        assert isinstance(r6d, Rotation6D)
        assert torch.allclose(r6d.vectors, vectors)

    def test_invalid_shape_error(self):
        """Raise error for wrong last dimension."""
        with pytest.raises(ValueError, match="last dimension 6"):
            rotation_6d(torch.randn(4))

        with pytest.raises(ValueError, match="last dimension 6"):
            rotation_6d(torch.randn(10, 3))

    def test_multi_batch_creation(self):
        """Multi-batch 6D rotation vectors."""
        vectors = torch.randn(5, 3, 6)
        r6d = rotation_6d(vectors)
        assert r6d.vectors.shape == (5, 3, 6)


class TestRotation6DToMatrix:
    """Tests for rotation_6d_to_matrix (Gram-Schmidt orthonormalization)."""

    def test_identity_input(self):
        """Identity rotation: first two columns of I."""
        vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        expected = torch.eye(3)
        assert R.shape == (3, 3)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_orthonormalization_non_orthonormal_input(self):
        """Non-orthonormal input should be orthonormalized."""
        # Two non-orthogonal, non-unit vectors
        a1 = torch.tensor([2.0, 0.0, 0.0])  # Not unit
        a2 = torch.tensor([1.0, 1.0, 0.0])  # Not orthogonal to a1

        vectors = torch.cat([a1, a2])
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        # Check orthogonality: R @ R.T = I
        RRT = R @ R.T
        assert torch.allclose(RRT, torch.eye(3), atol=1e-5)

        # Check determinant = 1 (proper rotation)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        vectors = torch.randn(10, 6)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.shape == (10, 3, 3)

    def test_orthogonality_random(self):
        """R @ R.T = I for random 6D inputs."""
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        )
        assert torch.allclose(RRT, expected, atol=1e-5)

    def test_determinant_one(self):
        """det(R) = 1 for random 6D inputs."""
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        dets = torch.linalg.det(R)
        expected = torch.ones(10, dtype=torch.float64)
        assert torch.allclose(dets, expected, atol=1e-5)

    def test_gradient(self):
        """Gradient check for rotation_6d_to_matrix."""
        vectors = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)

        def fn(vec):
            r6d = Rotation6D(vectors=vec)
            return rotation_6d_to_matrix(r6d)

        assert gradcheck(fn, (vectors,), eps=1e-6, atol=1e-4)

    def test_90_deg_rotation_around_z(self):
        """90-degree rotation around z-axis."""
        # R_z(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        # First two columns: [0, 1, 0] and [-1, 0, 0]
        vectors = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_multi_batch(self):
        """Multi-batch shape handling."""
        vectors = torch.randn(5, 3, 6)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.shape == (5, 3, 3, 3)


class TestMatrixToRotation6D:
    """Tests for matrix_to_rotation_6d."""

    def test_identity_matrix(self):
        """Identity matrix extracts [1,0,0,0,1,0]."""
        R = torch.eye(3)
        r6d = matrix_to_rotation_6d(R)

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert r6d.vectors.shape == (6,)
        assert torch.allclose(r6d.vectors, expected, atol=1e-5)

    def test_roundtrip(self):
        """matrix -> 6d -> matrix roundtrip."""
        # Create valid rotation matrices
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        # Convert to 6D and back
        r6d_back = matrix_to_rotation_6d(R)
        R_back = rotation_6d_to_matrix(r6d_back)

        assert torch.allclose(R_back, R, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        # Create valid rotation matrices
        vectors = torch.randn(10, 6)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        r6d_back = matrix_to_rotation_6d(R)
        assert r6d_back.vectors.shape == (10, 6)

    def test_gradient(self):
        """Gradient check for matrix_to_rotation_6d."""
        # Create valid rotation matrices
        vectors = torch.randn(5, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        R = R.clone().detach().requires_grad_(True)

        def fn(mat):
            r6d = matrix_to_rotation_6d(mat)
            return r6d.vectors

        assert gradcheck(fn, (R,), eps=1e-6, atol=1e-4)


class TestRotation6DToQuaternion:
    """Tests for rotation_6d_to_quaternion (via matrix)."""

    def test_identity(self):
        """Identity 6D gives identity quaternion."""
        vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        r6d = rotation_6d(vectors)
        q = rotation_6d_to_quaternion(r6d)

        # Identity quaternion is [1, 0, 0, 0]
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        vectors = torch.randn(10, 6)
        r6d = rotation_6d(vectors)
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.shape == (10, 4)

    def test_output_is_unit_quaternion(self):
        """Output should be a unit quaternion."""
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        q = rotation_6d_to_quaternion(r6d)

        norms = torch.linalg.norm(q.wxyz, dim=-1)
        assert torch.allclose(
            norms, torch.ones(10, dtype=torch.float64), atol=1e-5
        )

    def test_gradient(self):
        """Gradient check for rotation_6d_to_quaternion."""
        vectors = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)

        def fn(vec):
            r6d = Rotation6D(vectors=vec)
            return rotation_6d_to_quaternion(r6d).wxyz

        assert gradcheck(fn, (vectors,), eps=1e-6, atol=1e-4)


class TestQuaternionToRotation6D:
    """Tests for quaternion_to_rotation_6d (via matrix)."""

    def test_identity_quaternion(self):
        """Identity quaternion gives identity 6D."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        r6d = quaternion_to_rotation_6d(q)

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert r6d.vectors.shape == (6,)
        assert torch.allclose(r6d.vectors, expected, atol=1e-5)

    def test_roundtrip_via_quaternion(self):
        """6d -> quaternion -> 6d roundtrip (via matrix)."""
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)

        # Convert to quaternion and back
        q = rotation_6d_to_quaternion(r6d)
        r6d_back = quaternion_to_rotation_6d(q)

        # Compare via rotation matrices (since 6D representation is not unique)
        R1 = rotation_6d_to_matrix(r6d)
        R2 = rotation_6d_to_matrix(r6d_back)
        assert torch.allclose(R1, R2, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        r6d = quaternion_to_rotation_6d(q)
        assert r6d.vectors.shape == (10, 6)

    def test_gradient(self):
        """Gradient check for quaternion_to_rotation_6d."""
        q_raw = torch.randn(5, 4, dtype=torch.float64)
        q_raw = q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True)
        q = q_raw.clone().detach().requires_grad_(True)

        def fn(wxyz):
            qq = Quaternion(wxyz=wxyz)
            r6d = quaternion_to_rotation_6d(qq)
            return r6d.vectors

        assert gradcheck(fn, (q,), eps=1e-6, atol=1e-4)


class TestRotation6DShape:
    """Tests for shape handling in 6D rotation functions."""

    def test_single_rotation(self):
        """Single rotation (6,) shape."""
        vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.shape == (3, 3)
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.shape == (4,)

    def test_batch(self):
        """Batch of rotations (B, 6) shape."""
        vectors = torch.randn(10, 6)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.shape == (10, 3, 3)
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.shape == (10, 4)

    def test_multi_batch(self):
        """Multi-batch rotations (B, C, 6) shape."""
        vectors = torch.randn(5, 3, 6)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.shape == (5, 3, 3, 3)
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.shape == (5, 3, 4)


class TestRotation6DDtypes:
    """Tests for dtype handling."""

    def test_float32(self):
        """Works with float32."""
        vectors = torch.randn(10, 6, dtype=torch.float32)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.dtype == torch.float32
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        vectors = torch.randn(10, 6, dtype=torch.float64)
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)
        assert R.dtype == torch.float64
        q = rotation_6d_to_quaternion(r6d)
        assert q.wxyz.dtype == torch.float64


class TestRotation6DIntegration:
    """Integration tests for 6D rotation conversions."""

    def test_consistency_with_quaternion(self):
        """6D should be consistent with quaternion representation."""
        # Start with a quaternion
        q_orig = quaternion_normalize(
            quaternion(torch.randn(4, dtype=torch.float64))
        )

        # Convert to 6D and back
        r6d = quaternion_to_rotation_6d(q_orig)
        q_back = rotation_6d_to_quaternion(r6d)

        # Quaternions should represent the same rotation
        # (may differ by sign since q and -q represent the same rotation)
        R_orig = quaternion_to_matrix(q_orig)
        R_back = quaternion_to_matrix(q_back)
        assert torch.allclose(R_orig, R_back, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Create a random rotation matrix via scipy
        r_scipy = R.random()
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        # Convert to 6D (extract first two columns)
        r6d = matrix_to_rotation_6d(mat_scipy)

        # Convert back to matrix
        mat_back = rotation_6d_to_matrix(r6d)

        # Should match the original rotation matrix
        assert torch.allclose(mat_back, mat_scipy, atol=1e-6)

    def test_continuity_property(self):
        """Test that 6D representation is continuous.

        The key advantage of 6D representation is continuity: small changes
        in rotation should result in small changes in representation.
        """
        # Two nearby rotations
        angle1 = torch.tensor(0.1, dtype=torch.float64)
        angle2 = torch.tensor(0.11, dtype=torch.float64)  # Close to angle1

        # Create rotation matrices around z-axis
        c1, s1 = torch.cos(angle1), torch.sin(angle1)
        c2, s2 = torch.cos(angle2), torch.sin(angle2)

        R1 = torch.tensor(
            [
                [c1, -s1, 0],
                [s1, c1, 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        R2 = torch.tensor(
            [
                [c2, -s2, 0],
                [s2, c2, 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )

        # Convert to 6D
        r6d_1 = matrix_to_rotation_6d(R1)
        r6d_2 = matrix_to_rotation_6d(R2)

        # Small change in rotation should result in small change in 6D
        diff = torch.linalg.norm(r6d_2.vectors - r6d_1.vectors)
        assert diff < 0.1  # Should be small since rotations are close

    def test_numerical_stability_small_vectors(self):
        """Test numerical stability with small input vectors.

        Small vectors (but not degenerate) should still produce valid
        rotation matrices after normalization.
        """
        # Small but reasonable vectors (well above the epsilon threshold)
        vectors = torch.tensor([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0])
        r6d = rotation_6d(vectors)
        R = rotation_6d_to_matrix(r6d)

        # Should still produce a valid rotation matrix
        RRT = R @ R.T
        assert torch.allclose(RRT, torch.eye(3), atol=1e-5)

        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)
