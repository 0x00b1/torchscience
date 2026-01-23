"""Tests for SE(3) rigid body transforms."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    Quaternion,
    quaternion,
    quaternion_normalize,
)
from torchscience.geometry.transform._rigid_transform import (
    RigidTransform,
    dual_quaternion_to_rigid_transform,
    rigid_transform,
    rigid_transform_apply,
    rigid_transform_apply_vector,
    rigid_transform_compose,
    rigid_transform_from_matrix,
    rigid_transform_identity,
    rigid_transform_inverse,
    rigid_transform_slerp,
    rigid_transform_to_dual_quaternion,
    rigid_transform_to_matrix,
)


class TestRigidTransformConstruction:
    """Tests for RigidTransform construction."""

    def test_from_quaternion_and_translation(self):
        """Create RigidTransform from quaternion and translation."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = RigidTransform(rotation=q, translation=t)
        assert transform.rotation.wxyz.shape == (4,)
        assert transform.translation.shape == (3,)

    def test_batch(self):
        """Batch of rigid transforms."""
        q = quaternion(torch.randn(10, 4))
        t = torch.randn(10, 3)
        transform = RigidTransform(rotation=q, translation=t)
        assert transform.rotation.wxyz.shape == (10, 4)
        assert transform.translation.shape == (10, 3)

    def test_factory_function(self):
        """Create via rigid_transform() factory."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)
        assert isinstance(transform, RigidTransform)
        assert torch.allclose(transform.rotation.wxyz, q.wxyz)
        assert torch.allclose(transform.translation, t)

    def test_invalid_rotation_shape(self):
        """Raise error for wrong rotation dimension."""
        q = quaternion(torch.randn(3, 4))
        t = torch.randn(5, 3)  # Different batch size
        with pytest.raises(ValueError, match="broadcast"):
            rigid_transform(q, t)

    def test_invalid_translation_shape(self):
        """Raise error for wrong translation dimension."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="last dimension 3"):
            rigid_transform(q, torch.randn(4))


class TestRigidTransformIdentity:
    """Tests for rigid_transform_identity."""

    def test_creates_identity(self):
        """Creates identity transform."""
        transform = rigid_transform_identity()
        # Identity rotation
        assert torch.allclose(
            transform.rotation.wxyz, torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        # Zero translation
        assert torch.allclose(transform.translation, torch.zeros(3))

    def test_batch_shape(self):
        """Creates batch of identity transforms."""
        transform = rigid_transform_identity(batch_shape=(5, 3))
        assert transform.rotation.wxyz.shape == (5, 3, 4)
        assert transform.translation.shape == (5, 3, 3)
        # All should be identity
        expected_q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        expected_t = torch.zeros(3)
        for i in range(5):
            for j in range(3):
                assert torch.allclose(
                    transform.rotation.wxyz[i, j], expected_q
                )
                assert torch.allclose(transform.translation[i, j], expected_t)

    def test_device(self):
        """Creates on specified device."""
        # Test CPU explicitly
        transform = rigid_transform_identity(device="cpu")
        assert transform.rotation.wxyz.device.type == "cpu"
        assert transform.translation.device.type == "cpu"

    def test_dtype(self):
        """Creates with specified dtype."""
        transform = rigid_transform_identity(dtype=torch.float64)
        assert transform.rotation.wxyz.dtype == torch.float64
        assert transform.translation.dtype == torch.float64


class TestRigidTransformCompose:
    """Tests for rigid_transform_compose."""

    def test_identity_compose_left(self):
        """Identity * T = T."""
        identity = rigid_transform_identity()
        q = quaternion_normalize(
            quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        result = rigid_transform_compose(identity, transform)
        assert torch.allclose(result.rotation.wxyz, q.wxyz, atol=1e-5)
        assert torch.allclose(result.translation, t, atol=1e-5)

    def test_identity_compose_right(self):
        """T * Identity = T."""
        identity = rigid_transform_identity()
        q = quaternion_normalize(
            quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        result = rigid_transform_compose(transform, identity)
        assert torch.allclose(result.rotation.wxyz, q.wxyz, atol=1e-5)
        assert torch.allclose(result.translation, t, atol=1e-5)

    def test_pure_translations_compose_to_sum(self):
        """Two pure translations compose to their sum."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t1 = torch.tensor([1.0, 0.0, 0.0])
        t2 = torch.tensor([0.0, 2.0, 0.0])

        transform1 = rigid_transform(identity_rot, t1)
        transform2 = rigid_transform(identity_rot, t2)

        result = rigid_transform_compose(transform1, transform2)
        expected_t = torch.tensor([1.0, 2.0, 0.0])
        assert torch.allclose(result.translation, expected_t, atol=1e-5)
        assert torch.allclose(
            result.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )

    def test_batch(self):
        """Batched composition."""
        q1 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t1 = torch.randn(10, 3)
        q2 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t2 = torch.randn(10, 3)

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_compose(transform1, transform2)
        assert result.rotation.wxyz.shape == (10, 4)
        assert result.translation.shape == (10, 3)

    def test_composition_formula(self):
        """Test that composition follows T1 * T2 = (R1*R2, R1*t2 + t1)."""
        # Create a 90-degree rotation around z
        q1 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t1 = torch.tensor([1.0, 0.0, 0.0])

        # Second transform: no rotation, translate in x
        q2 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t2 = torch.tensor([1.0, 0.0, 0.0])

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_compose(transform1, transform2)

        # T1 * T2: first apply T2 (translate x by 1), then T1 (rotate 90 deg + translate x by 1)
        # t_result = R1 * t2 + t1 = rotate(1,0,0) by 90 deg around z + (1,0,0)
        # = (0,1,0) + (1,0,0) = (1,1,0)
        expected_t = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(result.translation, expected_t, atol=1e-5)


class TestRigidTransformInverse:
    """Tests for rigid_transform_inverse."""

    def test_inverse_gives_identity_when_composed(self):
        """T * T^(-1) = Identity."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        inverse = rigid_transform_inverse(transform)
        result = rigid_transform_compose(transform, inverse)

        # Should be identity
        assert torch.allclose(
            result.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(result.translation, torch.zeros(3), atol=1e-5)

    def test_inverse_compose_left(self):
        """T^(-1) * T = Identity."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        inverse = rigid_transform_inverse(transform)
        result = rigid_transform_compose(inverse, transform)

        # Should be identity
        assert torch.allclose(
            result.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(result.translation, torch.zeros(3), atol=1e-5)

    def test_double_inverse(self):
        """(T^(-1))^(-1) = T."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        double_inverse = rigid_transform_inverse(
            rigid_transform_inverse(transform)
        )
        assert torch.allclose(
            double_inverse.rotation.wxyz, transform.rotation.wxyz, atol=1e-5
        )
        assert torch.allclose(
            double_inverse.translation, transform.translation, atol=1e-5
        )

    def test_identity_inverse_is_identity(self):
        """Inverse of identity is identity."""
        identity = rigid_transform_identity()
        inverse = rigid_transform_inverse(identity)

        assert torch.allclose(
            inverse.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(inverse.translation, torch.zeros(3), atol=1e-5)

    def test_batch(self):
        """Batched inverse."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        transform = rigid_transform(q, t)

        inverse = rigid_transform_inverse(transform)
        assert inverse.rotation.wxyz.shape == (10, 4)
        assert inverse.translation.shape == (10, 3)


class TestRigidTransformApply:
    """Tests for rigid_transform_apply."""

    def test_identity_returns_original(self):
        """Identity transform returns original point."""
        identity = rigid_transform_identity()
        point = torch.tensor([1.0, 2.0, 3.0])
        result = rigid_transform_apply(identity, point)
        assert torch.allclose(result, point, atol=1e-5)

    def test_pure_translation(self):
        """Pure translation applies correctly."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(identity_rot, translation)

        point = torch.tensor([0.0, 0.0, 0.0])
        result = rigid_transform_apply(transform, point)
        assert torch.allclose(result, translation, atol=1e-5)

    def test_90_deg_rotation_around_z(self):
        """Pure 90-degree rotation around z maps (1,0,0) to (0,1,0)."""
        # 90 degrees around z: [cos(45), 0, 0, sin(45)]
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t = torch.zeros(3)
        transform = rigid_transform(q, t)

        point = torch.tensor([1.0, 0.0, 0.0])
        result = rigid_transform_apply(transform, point)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_rotation_and_translation(self):
        """Combined rotation and translation."""
        # 90 degrees around z
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t = torch.tensor([1.0, 0.0, 0.0])
        transform = rigid_transform(q, t)

        point = torch.tensor([1.0, 0.0, 0.0])
        result = rigid_transform_apply(transform, point)
        # R * (1,0,0) + (1,0,0) = (0,1,0) + (1,0,0) = (1,1,0)
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_transforms_single_point(self):
        """Batch of transforms applied to single point."""
        # All identity rotations with different translations
        q = quaternion(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5))
        t = torch.arange(15).reshape(5, 3).float()
        transform = rigid_transform(q, t)

        point = torch.zeros(3)
        result = rigid_transform_apply(transform, point)
        assert result.shape == (5, 3)
        assert torch.allclose(result, t, atol=1e-5)

    def test_single_transform_batch_points(self):
        """Single transform applied to batch of points."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(identity_rot, translation)

        points = torch.randn(10, 3)
        result = rigid_transform_apply(transform, points)
        expected = points + translation
        assert result.shape == (10, 3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_inverse_apply(self):
        """Applying inverse transform reverses the transformation."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        point = torch.randn(3)
        transformed = rigid_transform_apply(transform, point)
        inverse = rigid_transform_inverse(transform)
        recovered = rigid_transform_apply(inverse, transformed)
        assert torch.allclose(recovered, point, atol=1e-5)


class TestRigidTransformApplyVector:
    """Tests for rigid_transform_apply_vector."""

    def test_identity_returns_original(self):
        """Identity transform returns original vector."""
        identity = rigid_transform_identity()
        vector = torch.tensor([1.0, 2.0, 3.0])
        result = rigid_transform_apply_vector(identity, vector)
        assert torch.allclose(result, vector, atol=1e-5)

    def test_translation_ignored(self):
        """Translation is ignored for vectors."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(identity_rot, translation)

        vector = torch.tensor([1.0, 0.0, 0.0])
        result = rigid_transform_apply_vector(transform, vector)
        # No translation should be applied
        assert torch.allclose(result, vector, atol=1e-5)

    def test_90_deg_rotation_around_z(self):
        """90-degree rotation around z maps (1,0,0) to (0,1,0)."""
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t = torch.tensor(
            [100.0, 200.0, 300.0]
        )  # Large translation - should be ignored
        transform = rigid_transform(q, t)

        vector = torch.tensor([1.0, 0.0, 0.0])
        result = rigid_transform_apply_vector(transform, vector)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_preserves_length(self):
        """Rotation preserves vector length."""
        q = quaternion_normalize(quaternion(torch.randn(4)))
        t = torch.randn(3)
        transform = rigid_transform(q, t)

        vector = torch.randn(3)
        result = rigid_transform_apply_vector(transform, vector)
        assert torch.allclose(
            torch.linalg.norm(result), torch.linalg.norm(vector), atol=1e-5
        )

    def test_batch(self):
        """Batched vector transformation."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        transform = rigid_transform(q, t)

        vectors = torch.randn(10, 3)
        result = rigid_transform_apply_vector(transform, vectors)
        assert result.shape == (10, 3)


class TestRigidTransformToMatrix:
    """Tests for rigid_transform_to_matrix."""

    def test_identity_gives_identity_matrix(self):
        """Identity transform gives 4x4 identity matrix."""
        identity = rigid_transform_identity()
        matrix = rigid_transform_to_matrix(identity)
        expected = torch.eye(4)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix, expected, atol=1e-5)

    def test_pure_translation(self):
        """Pure translation gives expected 4x4 matrix."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(identity_rot, translation)

        matrix = rigid_transform_to_matrix(transform)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(matrix, expected, atol=1e-5)

    def test_pure_rotation(self):
        """Pure rotation gives expected 4x4 matrix."""
        # 180 degrees around z: [0, 0, 0, 1]
        q = quaternion(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        t = torch.zeros(3)
        transform = rigid_transform(q, t)

        matrix = rigid_transform_to_matrix(transform)
        # Rotation part should be diag([-1, -1, 1])
        expected = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(matrix, expected, atol=1e-5)

    def test_batch(self):
        """Batched matrix conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        transform = rigid_transform(q, t)

        matrix = rigid_transform_to_matrix(transform)
        assert matrix.shape == (10, 4, 4)
        # Check last row is [0, 0, 0, 1]
        expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for i in range(10):
            assert torch.allclose(
                matrix[i, 3, :], expected_last_row, atol=1e-5
            )


class TestRigidTransformFromMatrix:
    """Tests for rigid_transform_from_matrix."""

    def test_identity_matrix(self):
        """Identity matrix gives identity transform."""
        matrix = torch.eye(4)
        transform = rigid_transform_from_matrix(matrix)
        assert torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        ) or torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(transform.translation, torch.zeros(3), atol=1e-5)

    def test_pure_translation_matrix(self):
        """Pure translation matrix gives expected transform."""
        matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        transform = rigid_transform_from_matrix(matrix)
        expected_t = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(transform.translation, expected_t, atol=1e-5)

    def test_matrix_roundtrip(self):
        """Matrix roundtrip preserves transform."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, t)

        matrix = rigid_transform_to_matrix(transform)
        recovered = rigid_transform_from_matrix(matrix)

        # Check translation matches
        assert torch.allclose(recovered.translation, t, atol=1e-5)
        # Check rotation matches (or negation - both represent same rotation)
        assert torch.allclose(
            recovered.rotation.wxyz, q.wxyz, atol=1e-5
        ) or torch.allclose(recovered.rotation.wxyz, -q.wxyz, atol=1e-5)

    def test_batch(self):
        """Batched from_matrix."""
        # Create valid transform matrices from transforms
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        transform = rigid_transform(q, t)
        matrices = rigid_transform_to_matrix(transform)

        recovered = rigid_transform_from_matrix(matrices)
        assert recovered.rotation.wxyz.shape == (10, 4)
        assert recovered.translation.shape == (10, 3)

    def test_invalid_shape(self):
        """Raise error for wrong matrix shape."""
        with pytest.raises(ValueError, match="4, 4"):
            rigid_transform_from_matrix(torch.randn(3, 3))


class TestRigidTransformSlerp:
    """Tests for rigid_transform_slerp."""

    def test_t_equals_zero_returns_t1(self):
        """At t=0, slerp returns t1."""
        q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t1 = torch.tensor([0.0, 0.0, 0.0])
        q2 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t2 = torch.tensor([1.0, 2.0, 3.0])

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_slerp(
            transform1, transform2, torch.tensor(0.0)
        )
        assert torch.allclose(result.rotation.wxyz, q1.wxyz, atol=1e-5)
        assert torch.allclose(result.translation, t1, atol=1e-5)

    def test_t_equals_one_returns_t2(self):
        """At t=1, slerp returns t2."""
        q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t1 = torch.tensor([0.0, 0.0, 0.0])
        q2 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t2 = torch.tensor([1.0, 2.0, 3.0])

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_slerp(
            transform1, transform2, torch.tensor(1.0)
        )
        assert torch.allclose(result.rotation.wxyz, q2.wxyz, atol=1e-5)
        assert torch.allclose(result.translation, t2, atol=1e-5)

    def test_midpoint(self):
        """At t=0.5, returns midpoint (SLERP for rotation, linear for translation)."""
        q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t1 = torch.tensor([0.0, 0.0, 0.0])
        q2 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t2 = torch.tensor([2.0, 4.0, 6.0])

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_slerp(
            transform1, transform2, torch.tensor(0.5)
        )

        # Translation should be linear midpoint
        expected_t = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result.translation, expected_t, atol=1e-5)

        # Rotation should be SLERP midpoint (45 deg = pi/4 / 2)
        expected_q = torch.tensor(
            [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]
        )
        assert torch.allclose(result.rotation.wxyz, expected_q, atol=1e-5)

    def test_batch(self):
        """Batched slerp."""
        q1 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t1 = torch.randn(10, 3)
        q2 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t2 = torch.randn(10, 3)

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        alpha = torch.rand(10)
        result = rigid_transform_slerp(transform1, transform2, alpha)
        assert result.rotation.wxyz.shape == (10, 4)
        assert result.translation.shape == (10, 3)

    def test_scalar_alpha_broadcast(self):
        """Scalar alpha broadcasts to batch."""
        q1 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t1 = torch.randn(10, 3)
        q2 = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t2 = torch.randn(10, 3)

        transform1 = rigid_transform(q1, t1)
        transform2 = rigid_transform(q2, t2)

        result = rigid_transform_slerp(
            transform1, transform2, torch.tensor(0.5)
        )
        assert result.rotation.wxyz.shape == (10, 4)
        assert result.translation.shape == (10, 3)


class TestRigidTransformGradients:
    """Tests for rigid transform gradient computation."""

    def test_compose_gradcheck(self):
        """Gradient check for compose."""
        q1 = torch.randn(4, dtype=torch.float64)
        q1 = q1 / torch.linalg.norm(q1)
        q1 = q1.clone().detach().requires_grad_(True)
        t1 = torch.randn(3, dtype=torch.float64, requires_grad=True)

        q2 = torch.randn(4, dtype=torch.float64)
        q2 = q2 / torch.linalg.norm(q2)
        q2 = q2.clone().detach().requires_grad_(True)
        t2 = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def compose_fn(q1_val, t1_val, q2_val, t2_val):
            transform1 = rigid_transform(Quaternion(wxyz=q1_val), t1_val)
            transform2 = rigid_transform(Quaternion(wxyz=q2_val), t2_val)
            result = rigid_transform_compose(transform1, transform2)
            return result.rotation.wxyz, result.translation

        assert gradcheck(
            lambda q1_v, t1_v, q2_v, t2_v: compose_fn(q1_v, t1_v, q2_v, t2_v)[
                0
            ],
            (q1, t1, q2, t2),
            eps=1e-6,
            atol=1e-4,
        )
        assert gradcheck(
            lambda q1_v, t1_v, q2_v, t2_v: compose_fn(q1_v, t1_v, q2_v, t2_v)[
                1
            ],
            (q1, t1, q2, t2),
            eps=1e-6,
            atol=1e-4,
        )

    def test_inverse_gradcheck(self):
        """Gradient check for inverse."""
        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def inverse_fn(q_val, t_val):
            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            result = rigid_transform_inverse(transform)
            return result.rotation.wxyz, result.translation

        assert gradcheck(
            lambda q_v, t_v: inverse_fn(q_v, t_v)[0],
            (q, t),
            eps=1e-6,
            atol=1e-4,
        )
        assert gradcheck(
            lambda q_v, t_v: inverse_fn(q_v, t_v)[1],
            (q, t),
            eps=1e-6,
            atol=1e-4,
        )

    def test_apply_gradcheck(self):
        """Gradient check for apply."""
        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)
        point = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def apply_fn(q_val, t_val, p_val):
            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            return rigid_transform_apply(transform, p_val)

        assert gradcheck(apply_fn, (q, t, point), eps=1e-6, atol=1e-4)

    def test_to_matrix_gradcheck(self):
        """Gradient check for to_matrix."""
        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def to_matrix_fn(q_val, t_val):
            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            return rigid_transform_to_matrix(transform)

        assert gradcheck(to_matrix_fn, (q, t), eps=1e-6, atol=1e-4)


class TestRigidTransformDtypes:
    """Tests for rigid transforms with different data types."""

    def test_float32(self):
        """Works with float32."""
        q = quaternion(torch.randn(4, dtype=torch.float32))
        t = torch.randn(3, dtype=torch.float32)
        transform = rigid_transform(q, t)
        assert transform.rotation.wxyz.dtype == torch.float32
        assert transform.translation.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q = quaternion(torch.randn(4, dtype=torch.float64))
        t = torch.randn(3, dtype=torch.float64)
        transform = rigid_transform(q, t)
        assert transform.rotation.wxyz.dtype == torch.float64
        assert transform.translation.dtype == torch.float64

    def test_identity_dtype(self):
        """Identity respects dtype parameter."""
        transform = rigid_transform_identity(dtype=torch.float64)
        assert transform.rotation.wxyz.dtype == torch.float64
        assert transform.translation.dtype == torch.float64


class TestRigidTransformIntegration:
    """Integration tests for rigid transforms."""

    def test_chain_of_transforms(self):
        """Multiple composed transforms work correctly."""
        # T1: translate (1, 0, 0)
        # T2: rotate 90 deg around z
        # T3: translate (0, 1, 0)
        # Result: T1 * T2 * T3 applied to origin

        q_identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q_90z = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )

        t1 = rigid_transform(q_identity, torch.tensor([1.0, 0.0, 0.0]))
        t2 = rigid_transform(q_90z, torch.zeros(3))
        t3 = rigid_transform(q_identity, torch.tensor([0.0, 1.0, 0.0]))

        # Compose: T1 * T2 * T3
        composed = rigid_transform_compose(rigid_transform_compose(t1, t2), t3)

        # Apply to origin
        origin = torch.zeros(3)
        result = rigid_transform_apply(composed, origin)

        # Expected: T3 translates to (0,1,0), T2 rotates to (-1,0,0), T1 adds (1,0,0) -> (0,0,0)
        # Actually, composition is T1 * (T2 * T3)
        # T2 * T3: rotate T3's translation, so (0,1,0) rotated by 90 deg z = (-1,0,0), total trans = (-1,0,0)
        # T1 * (T2*T3): add T1's translation = (-1,0,0) + (1,0,0) = (0,0,0)
        # But applied to origin...
        # T3 on origin: (0,1,0)
        # T2 on (0,1,0): rotate = (-1,0,0)
        # T1 on (-1,0,0): translate = (-1,0,0) + (1,0,0) = (0,0,0)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy if available."""
        scipy = pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Create random transform
        r_scipy = R.random()
        xyzw = r_scipy.as_quat()
        q_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )
        q = quaternion(q_wxyz)
        t = torch.randn(3, dtype=torch.float64)
        transform = rigid_transform(q, t)

        # Apply to point
        point = torch.randn(3, dtype=torch.float64)
        result_torch = rigid_transform_apply(transform, point)

        # Compare with scipy
        rotated = torch.tensor(
            r_scipy.apply(point.numpy()), dtype=torch.float64
        )
        expected = rotated + t
        assert torch.allclose(result_torch, expected, atol=1e-6)


class TestRigidTransformToDualQuaternion:
    """Tests for rigid_transform_to_dual_quaternion."""

    def test_identity_dual_quaternion(self):
        """Identity transform gives dual quaternion [1,0,0,0, 0,0,0,0]."""
        identity = rigid_transform_identity()
        dq = rigid_transform_to_dual_quaternion(identity)

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert dq.shape == (8,)
        assert torch.allclose(dq, expected, atol=1e-5)

    def test_pure_translation(self):
        """Pure translation: q_r = identity, q_d = (0, t/2)."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([2.0, 4.0, 6.0])
        transform = rigid_transform(identity_rot, translation)

        dq = rigid_transform_to_dual_quaternion(transform)

        # Real part should be identity
        assert torch.allclose(
            dq[..., :4], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5
        )

        # Dual part: q_d = (1/2) * q_t * q_r where q_t = (0, t)
        # For identity rotation: q_d = (0, t/2) = (0, 1, 2, 3)
        expected_dual = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert torch.allclose(dq[..., 4:], expected_dual, atol=1e-5)

    def test_pure_rotation(self):
        """Pure rotation: q_d should be zero (no translation)."""
        # 90 degrees around z-axis
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        transform = rigid_transform(q, torch.zeros(3))

        dq = rigid_transform_to_dual_quaternion(transform)

        # Real part should be the rotation quaternion
        assert torch.allclose(dq[..., :4], q.wxyz, atol=1e-5)

        # Dual part should be zero (no translation)
        assert torch.allclose(dq[..., 4:], torch.zeros(4), atol=1e-5)

    def test_rotation_and_translation(self):
        """Combined rotation and translation."""
        # 90 degrees around z-axis
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        t = torch.tensor([2.0, 4.0, 6.0])
        transform = rigid_transform(q, t)

        dq = rigid_transform_to_dual_quaternion(transform)

        # Real part should be the rotation quaternion
        assert torch.allclose(dq[..., :4], q.wxyz, atol=1e-5)

        # Verify the output has correct shape
        assert dq.shape == (8,)

    def test_batch(self):
        """Batched dual quaternion conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        transform = rigid_transform(q, t)

        dq = rigid_transform_to_dual_quaternion(transform)

        assert dq.shape == (10, 8)

    def test_multi_batch(self):
        """Multi-dimensional batch conversion."""
        q = quaternion_normalize(quaternion(torch.randn(5, 3, 4)))
        t = torch.randn(5, 3, 3)
        transform = rigid_transform(q, t)

        dq = rigid_transform_to_dual_quaternion(transform)

        assert dq.shape == (5, 3, 8)


class TestDualQuaternionToRigidTransform:
    """Tests for dual_quaternion_to_rigid_transform."""

    def test_identity_dual_quaternion(self):
        """Identity dual quaternion [1,0,0,0, 0,0,0,0] gives identity transform."""
        dq = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transform = dual_quaternion_to_rigid_transform(dq)

        assert torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(transform.translation, torch.zeros(3), atol=1e-5)

    def test_pure_translation(self):
        """Dual quaternion with identity rotation and translation."""
        # q_r = identity, q_d = (0, t/2) for translation t = (2, 4, 6)
        dq = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        transform = dual_quaternion_to_rigid_transform(dq)

        assert torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        expected_translation = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(
            transform.translation, expected_translation, atol=1e-5
        )

    def test_pure_rotation(self):
        """Dual quaternion with rotation only (zero dual part)."""
        # 90 degrees around z-axis
        q_r = torch.tensor(
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        )
        q_d = torch.zeros(4)
        dq = torch.cat([q_r, q_d])

        transform = dual_quaternion_to_rigid_transform(dq)

        assert torch.allclose(transform.rotation.wxyz, q_r, atol=1e-5)
        assert torch.allclose(transform.translation, torch.zeros(3), atol=1e-5)

    def test_batch(self):
        """Batched dual quaternion conversion."""
        # Create batch of dual quaternions
        q_r = torch.randn(10, 4)
        q_r = q_r / torch.linalg.norm(q_r, dim=-1, keepdim=True)
        q_d = torch.randn(10, 4)
        dq = torch.cat([q_r, q_d], dim=-1)

        transform = dual_quaternion_to_rigid_transform(dq)

        assert transform.rotation.wxyz.shape == (10, 4)
        assert transform.translation.shape == (10, 3)

    def test_invalid_shape(self):
        """Raise error for wrong input shape."""
        with pytest.raises(ValueError, match="8"):
            dual_quaternion_to_rigid_transform(torch.randn(6))

        with pytest.raises(ValueError, match="8"):
            dual_quaternion_to_rigid_transform(torch.randn(10, 4))


class TestDualQuaternionRoundtrip:
    """Tests for roundtrip conversions between rigid transform and dual quaternion."""

    def test_transform_to_dq_to_transform(self):
        """Roundtrip: transform -> dual quaternion -> transform."""
        q = quaternion_normalize(
            quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        )
        t = torch.tensor([1.0, 2.0, 3.0])
        original = rigid_transform(q, t)

        # Convert to dual quaternion and back
        dq = rigid_transform_to_dual_quaternion(original)
        recovered = dual_quaternion_to_rigid_transform(dq)

        # Check rotation matches (or negation)
        assert torch.allclose(
            recovered.rotation.wxyz, q.wxyz, atol=1e-5
        ) or torch.allclose(recovered.rotation.wxyz, -q.wxyz, atol=1e-5)

        # Check translation matches
        assert torch.allclose(recovered.translation, t, atol=1e-5)

    def test_batch_roundtrip(self):
        """Batched roundtrip conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        t = torch.randn(10, 3)
        original = rigid_transform(q, t)

        dq = rigid_transform_to_dual_quaternion(original)
        recovered = dual_quaternion_to_rigid_transform(dq)

        # Check translations match
        assert torch.allclose(recovered.translation, t, atol=1e-5)

        # Check rotations produce same effect
        test_point = torch.randn(3)
        original_applied = rigid_transform_apply(original, test_point)
        recovered_applied = rigid_transform_apply(recovered, test_point)
        assert torch.allclose(original_applied, recovered_applied, atol=1e-5)

    def test_identity_roundtrip(self):
        """Identity transform roundtrip."""
        identity = rigid_transform_identity()
        dq = rigid_transform_to_dual_quaternion(identity)
        recovered = dual_quaternion_to_rigid_transform(dq)

        assert torch.allclose(
            recovered.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-5,
        )
        assert torch.allclose(recovered.translation, torch.zeros(3), atol=1e-5)

    def test_pure_translation_roundtrip(self):
        """Pure translation roundtrip."""
        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        t = torch.tensor([1.0, 2.0, 3.0])
        original = rigid_transform(identity_rot, t)

        dq = rigid_transform_to_dual_quaternion(original)
        recovered = dual_quaternion_to_rigid_transform(dq)

        assert torch.allclose(recovered.translation, t, atol=1e-5)

    def test_various_rotations_roundtrip(self):
        """Roundtrip for various rotation angles."""
        angles = [
            0.0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2,
            math.pi,
        ]

        for angle in angles:
            # Rotation around z-axis
            q = quaternion(
                torch.tensor(
                    [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
                )
            )
            t = torch.tensor([1.0, 2.0, 3.0])
            original = rigid_transform(q, t)

            dq = rigid_transform_to_dual_quaternion(original)
            recovered = dual_quaternion_to_rigid_transform(dq)

            # Verify transformation effect is preserved
            test_point = torch.tensor([1.0, 0.0, 0.0])
            original_result = rigid_transform_apply(original, test_point)
            recovered_result = rigid_transform_apply(recovered, test_point)
            assert torch.allclose(
                original_result, recovered_result, atol=1e-5
            ), f"Failed for angle {angle}"


class TestDualQuaternionGradients:
    """Tests for dual quaternion gradient computation."""

    def test_to_dual_quaternion_gradcheck(self):
        """Gradient check for rigid_transform_to_dual_quaternion."""
        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def to_dq_fn(q_val, t_val):
            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            return rigid_transform_to_dual_quaternion(transform)

        assert gradcheck(to_dq_fn, (q, t), eps=1e-6, atol=1e-4)

    def test_from_dual_quaternion_gradcheck(self):
        """Gradient check for dual_quaternion_to_rigid_transform."""
        # Create a valid dual quaternion
        q_r = torch.randn(4, dtype=torch.float64)
        q_r = q_r / torch.linalg.norm(q_r)
        q_d = torch.randn(4, dtype=torch.float64)
        dq = torch.cat([q_r, q_d]).requires_grad_(True)

        def from_dq_fn(dq_val):
            transform = dual_quaternion_to_rigid_transform(dq_val)
            return transform.translation

        assert gradcheck(from_dq_fn, (dq,), eps=1e-6, atol=1e-4)

    def test_roundtrip_gradcheck(self):
        """Gradient check for roundtrip conversion."""
        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def roundtrip_fn(q_val, t_val):
            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            dq = rigid_transform_to_dual_quaternion(transform)
            recovered = dual_quaternion_to_rigid_transform(dq)
            return recovered.translation

        assert gradcheck(roundtrip_fn, (q, t), eps=1e-6, atol=1e-4)
