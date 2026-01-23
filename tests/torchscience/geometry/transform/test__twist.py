"""Tests for Twist (screw/se(3)) representation and SE(3) exponential/logarithm maps."""

import math

import pytest
import torch
from torch.autograd import gradcheck


class TestTwistConstruction:
    """Tests for Twist tensorclass construction."""

    def test_create_from_components(self):
        """Create Twist from angular and linear components."""
        from torchscience.geometry.transform import Twist, twist

        angular = torch.tensor([0.1, 0.2, 0.3])
        linear = torch.tensor([1.0, 2.0, 3.0])
        t = twist(angular, linear)
        assert isinstance(t, Twist)
        assert t.angular.shape == (3,)
        assert t.linear.shape == (3,)
        assert torch.allclose(t.angular, angular)
        assert torch.allclose(t.linear, linear)

    def test_create_from_vector(self):
        """Create Twist from 6D vector [angular; linear]."""
        from torchscience.geometry.transform import twist_from_vector

        vector = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        t = twist_from_vector(vector)
        assert torch.allclose(t.angular, vector[:3])
        assert torch.allclose(t.linear, vector[3:])

    def test_batch_construction(self):
        """Batch of Twists."""
        from torchscience.geometry.transform import twist

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        assert t.angular.shape == (10, 3)
        assert t.linear.shape == (10, 3)

    def test_batch_from_vector(self):
        """Batch from 6D vectors."""
        from torchscience.geometry.transform import twist_from_vector

        vectors = torch.randn(10, 6)
        t = twist_from_vector(vectors)
        assert t.angular.shape == (10, 3)
        assert t.linear.shape == (10, 3)

    def test_multi_batch(self):
        """Multi-batch construction."""
        from torchscience.geometry.transform import twist

        angular = torch.randn(5, 3, 3)
        linear = torch.randn(5, 3, 3)
        t = twist(angular, linear)
        assert t.angular.shape == (5, 3, 3)
        assert t.linear.shape == (5, 3, 3)

    def test_invalid_angular_shape(self):
        """Raise error for wrong angular dimension."""
        from torchscience.geometry.transform import twist

        with pytest.raises(ValueError, match="last dimension 3"):
            twist(torch.randn(4), torch.randn(3))

    def test_invalid_linear_shape(self):
        """Raise error for wrong linear dimension."""
        from torchscience.geometry.transform import twist

        with pytest.raises(ValueError, match="last dimension 3"):
            twist(torch.randn(3), torch.randn(4))

    def test_invalid_vector_shape(self):
        """Raise error for wrong vector dimension."""
        from torchscience.geometry.transform import twist_from_vector

        with pytest.raises(ValueError, match="last dimension 6"):
            twist_from_vector(torch.randn(5))

    def test_mismatched_batch_shapes(self):
        """Raise error for incompatible batch shapes."""
        from torchscience.geometry.transform import twist

        with pytest.raises(ValueError, match="broadcast"):
            twist(torch.randn(5, 3), torch.randn(7, 3))


class TestTwistToVector:
    """Tests for twist_to_vector conversion."""

    def test_single_twist(self):
        """Convert single twist to vector."""
        from torchscience.geometry.transform import twist, twist_to_vector

        angular = torch.tensor([0.1, 0.2, 0.3])
        linear = torch.tensor([1.0, 2.0, 3.0])
        t = twist(angular, linear)
        vector = twist_to_vector(t)
        expected = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        assert vector.shape == (6,)
        assert torch.allclose(vector, expected)

    def test_batch_twist(self):
        """Convert batch of twists to vectors."""
        from torchscience.geometry.transform import twist, twist_to_vector

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        vector = twist_to_vector(t)
        assert vector.shape == (10, 6)
        assert torch.allclose(vector[..., :3], angular)
        assert torch.allclose(vector[..., 3:], linear)

    def test_roundtrip_vector(self):
        """Roundtrip: vector -> twist -> vector."""
        from torchscience.geometry.transform import (
            twist_from_vector,
            twist_to_vector,
        )

        vector = torch.randn(10, 6)
        t = twist_from_vector(vector)
        recovered = twist_to_vector(t)
        assert torch.allclose(recovered, vector)


class TestTwistToMatrix:
    """Tests for twist_to_matrix (hat operator)."""

    def test_zero_twist(self):
        """Zero twist gives zero matrix."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        t = twist(torch.zeros(3), torch.zeros(3))
        matrix = twist_to_matrix(t)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix, torch.zeros(4, 4))

    def test_pure_angular(self):
        """Pure angular twist gives skew-symmetric upper-left."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        omega = torch.tensor([1.0, 0.0, 0.0])
        t = twist(omega, torch.zeros(3))
        matrix = twist_to_matrix(t)

        # Check structure:
        # [  [omega]_x   0  ]
        # [     0       0  ]
        expected_skew = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        assert torch.allclose(matrix[:3, :3], expected_skew, atol=1e-6)
        assert torch.allclose(matrix[:3, 3], torch.zeros(3))
        assert torch.allclose(matrix[3, :], torch.zeros(4))

    def test_pure_linear(self):
        """Pure linear twist gives translation in last column."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        v = torch.tensor([1.0, 2.0, 3.0])
        t = twist(torch.zeros(3), v)
        matrix = twist_to_matrix(t)

        assert torch.allclose(matrix[:3, :3], torch.zeros(3, 3))
        assert torch.allclose(matrix[:3, 3], v)
        assert torch.allclose(matrix[3, :], torch.zeros(4))

    def test_batch(self):
        """Batched twist_to_matrix."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        matrix = twist_to_matrix(t)
        assert matrix.shape == (10, 4, 4)

    def test_skew_symmetric_structure(self):
        """Upper-left 3x3 is skew-symmetric."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        angular = torch.randn(3)
        linear = torch.randn(3)
        t = twist(angular, linear)
        matrix = twist_to_matrix(t)

        skew = matrix[:3, :3]
        assert torch.allclose(skew, -skew.T, atol=1e-6)


class TestMatrixToTwist:
    """Tests for matrix_to_twist (vee operator)."""

    def test_roundtrip(self):
        """Roundtrip: twist -> matrix -> twist."""
        from torchscience.geometry.transform import (
            matrix_to_twist,
            twist,
            twist_to_matrix,
        )

        angular = torch.randn(3)
        linear = torch.randn(3)
        t = twist(angular, linear)
        matrix = twist_to_matrix(t)
        recovered = matrix_to_twist(matrix)

        assert torch.allclose(recovered.angular, angular, atol=1e-6)
        assert torch.allclose(recovered.linear, linear, atol=1e-6)

    def test_batch_roundtrip(self):
        """Batch roundtrip."""
        from torchscience.geometry.transform import (
            matrix_to_twist,
            twist,
            twist_to_matrix,
        )

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        matrix = twist_to_matrix(t)
        recovered = matrix_to_twist(matrix)

        assert torch.allclose(recovered.angular, angular, atol=1e-6)
        assert torch.allclose(recovered.linear, linear, atol=1e-6)

    def test_invalid_shape(self):
        """Raise error for wrong matrix shape."""
        from torchscience.geometry.transform import matrix_to_twist

        with pytest.raises(ValueError, match="4, 4"):
            matrix_to_twist(torch.randn(3, 3))


class TestSE3Exp:
    """Tests for se3_exp (SE(3) exponential map)."""

    def test_zero_twist_gives_identity(self):
        """Zero twist maps to identity transform."""
        from torchscience.geometry.transform import se3_exp, twist

        t = twist(torch.zeros(3), torch.zeros(3))
        transform = se3_exp(t)

        # Identity rotation
        assert torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-6,
        )
        # Zero translation
        assert torch.allclose(transform.translation, torch.zeros(3), atol=1e-6)

    def test_pure_translation(self):
        """Pure linear twist gives pure translation."""
        from torchscience.geometry.transform import se3_exp, twist

        v = torch.tensor([1.0, 2.0, 3.0])
        t = twist(torch.zeros(3), v)
        transform = se3_exp(t)

        # Identity rotation
        assert torch.allclose(
            transform.rotation.wxyz,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            atol=1e-6,
        )
        # Translation equals linear velocity
        assert torch.allclose(transform.translation, v, atol=1e-6)

    def test_pure_rotation(self):
        """Pure angular twist gives pure rotation."""
        from torchscience.geometry.transform import se3_exp, so3_exp, twist

        omega = torch.tensor([0.0, 0.0, math.pi / 2])  # 90 deg around z
        t = twist(omega, torch.zeros(3))
        transform = se3_exp(t)

        # Check rotation matches SO(3) exp
        R = so3_exp(omega)
        # For quaternion, check the rotation matrix
        from torchscience.geometry.transform import quaternion_to_matrix

        R_from_q = quaternion_to_matrix(transform.rotation)
        assert torch.allclose(R_from_q, R, atol=1e-5)

        # Zero translation
        assert torch.allclose(transform.translation, torch.zeros(3), atol=1e-6)

    def test_small_angle_taylor(self):
        """Small angle uses Taylor expansion correctly."""
        from torchscience.geometry.transform import se3_exp, twist

        # Very small rotation
        omega = torch.tensor([1e-8, 2e-8, 3e-8], dtype=torch.float64)
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        t = twist(omega, v)
        transform = se3_exp(t)

        # For small angles, translation should be approximately v
        assert torch.allclose(transform.translation, v, atol=1e-6)

    def test_screw_motion(self):
        """Screw motion (combined rotation + translation along axis)."""
        from torchscience.geometry.transform import se3_exp, twist

        # Rotation around z-axis with translation along z
        angle = math.pi / 4  # 45 degrees
        omega = torch.tensor([0.0, 0.0, angle])
        v = torch.tensor([0.0, 0.0, 1.0])  # Translation along axis
        t = twist(omega, v)
        transform = se3_exp(t)

        # Translation along the rotation axis should be preserved exactly
        # For screw motion along the axis: t_z = v_z (when omega is axis-aligned)
        assert torch.allclose(
            transform.translation[2], torch.tensor(1.0), atol=1e-5
        )

    def test_batch(self):
        """Batched se3_exp."""
        from torchscience.geometry.transform import se3_exp, twist

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        transform = se3_exp(t)

        assert transform.rotation.wxyz.shape == (10, 4)
        assert transform.translation.shape == (10, 3)

    def test_scipy_comparison_rotation(self):
        """Compare rotation part with scipy."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as Rot

        from torchscience.geometry.transform import (
            quaternion_to_matrix,
            se3_exp,
            twist,
        )

        omega = torch.randn(3, dtype=torch.float64) * 0.5
        v = torch.zeros(3, dtype=torch.float64)
        t = twist(omega, v)
        transform = se3_exp(t)

        # Compare rotation matrix
        R_scipy = torch.tensor(
            Rot.from_rotvec(omega.numpy()).as_matrix(), dtype=torch.float64
        )
        R_ours = quaternion_to_matrix(transform.rotation)
        assert torch.allclose(R_ours, R_scipy, atol=1e-10)


class TestSE3Log:
    """Tests for se3_log (SE(3) logarithm map)."""

    def test_identity_gives_zero(self):
        """Identity transform maps to zero twist."""
        from torchscience.geometry.transform import (
            rigid_transform_identity,
            se3_log,
        )

        transform = rigid_transform_identity(dtype=torch.float64)
        t = se3_log(transform)

        assert torch.allclose(
            t.angular, torch.zeros(3, dtype=torch.float64), atol=1e-6
        )
        assert torch.allclose(
            t.linear, torch.zeros(3, dtype=torch.float64), atol=1e-6
        )

    def test_pure_translation(self):
        """Pure translation gives pure linear twist."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            se3_log,
        )

        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 2.0, 3.0])
        transform = rigid_transform(q, translation)
        t = se3_log(transform)

        assert torch.allclose(t.angular, torch.zeros(3), atol=1e-6)
        assert torch.allclose(t.linear, translation, atol=1e-6)

    def test_pure_rotation(self):
        """Pure rotation gives pure angular twist."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            se3_log,
            so3_log,
        )

        # 90 deg rotation around z
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        transform = rigid_transform(q, torch.zeros(3))
        t = se3_log(transform)

        # Angular should match so3_log of the rotation matrix
        from torchscience.geometry.transform import quaternion_to_matrix

        R = quaternion_to_matrix(q)
        expected_omega = so3_log(R)
        assert torch.allclose(t.angular, expected_omega, atol=1e-5)
        assert torch.allclose(t.linear, torch.zeros(3), atol=1e-5)

    def test_batch(self):
        """Batched se3_log."""
        from torchscience.geometry.transform import (
            quaternion,
            quaternion_normalize,
            rigid_transform,
            se3_log,
        )

        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        translation = torch.randn(10, 3)
        transform = rigid_transform(q, translation)
        t = se3_log(transform)

        assert t.angular.shape == (10, 3)
        assert t.linear.shape == (10, 3)


class TestSE3ExpLogRoundtrip:
    """Tests for SE(3) exp/log roundtrip."""

    def test_exp_log_roundtrip(self):
        """se3_log(se3_exp(twist)) = twist for reasonable twists."""
        from torchscience.geometry.transform import (
            se3_exp,
            se3_log,
            twist,
        )

        # Keep rotation angle small enough (< pi)
        angular = torch.randn(20, 3, dtype=torch.float64) * 0.5
        linear = torch.randn(20, 3, dtype=torch.float64)
        t = twist(angular, linear)

        transform = se3_exp(t)
        t_back = se3_log(transform)

        assert torch.allclose(t_back.angular, angular, atol=1e-8)
        assert torch.allclose(t_back.linear, linear, atol=1e-8)

    def test_log_exp_roundtrip(self):
        """se3_exp(se3_log(transform)) = transform."""
        from torchscience.geometry.transform import (
            quaternion,
            quaternion_normalize,
            rigid_transform,
            rigid_transform_to_matrix,
            se3_exp,
            se3_log,
        )

        q = quaternion_normalize(
            quaternion(torch.randn(20, 4, dtype=torch.float64))
        )
        translation = torch.randn(20, 3, dtype=torch.float64)
        transform = rigid_transform(q, translation)

        t = se3_log(transform)
        transform_back = se3_exp(t)

        # Compare via transformation matrices
        M1 = rigid_transform_to_matrix(transform)
        M2 = rigid_transform_to_matrix(transform_back)
        assert torch.allclose(M1, M2, atol=1e-8)

    def test_roundtrip_small_rotation(self):
        """Roundtrip for very small rotations."""
        from torchscience.geometry.transform import (
            se3_exp,
            se3_log,
            twist,
        )

        angular = torch.randn(10, 3, dtype=torch.float64) * 1e-8
        linear = torch.randn(10, 3, dtype=torch.float64)
        t = twist(angular, linear)

        transform = se3_exp(t)
        t_back = se3_log(transform)

        assert torch.allclose(t_back.angular, angular, atol=1e-10)
        assert torch.allclose(t_back.linear, linear, atol=1e-10)


class TestTwistApply:
    """Tests for twist_apply."""

    def test_zero_twist_returns_original(self):
        """Zero twist with any dt returns original point."""
        from torchscience.geometry.transform import twist, twist_apply

        t = twist(torch.zeros(3), torch.zeros(3))
        point = torch.tensor([1.0, 2.0, 3.0])
        dt = torch.tensor(1.0)
        result = twist_apply(t, point, dt)
        assert torch.allclose(result, point, atol=1e-6)

    def test_zero_dt_returns_original(self):
        """Any twist with dt=0 returns original point."""
        from torchscience.geometry.transform import twist, twist_apply

        t = twist(torch.randn(3), torch.randn(3))
        point = torch.tensor([1.0, 2.0, 3.0])
        dt = torch.tensor(0.0)
        result = twist_apply(t, point, dt)
        assert torch.allclose(result, point, atol=1e-6)

    def test_pure_linear_motion(self):
        """Pure linear velocity moves point linearly."""
        from torchscience.geometry.transform import twist, twist_apply

        v = torch.tensor([1.0, 0.0, 0.0])
        t = twist(torch.zeros(3), v)
        point = torch.tensor([0.0, 0.0, 0.0])
        dt = torch.tensor(2.0)
        result = twist_apply(t, point, dt)
        expected = torch.tensor([2.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_pure_rotation_around_z(self):
        """Pure rotation around z rotates point in xy-plane."""
        from torchscience.geometry.transform import twist, twist_apply

        omega = torch.tensor([0.0, 0.0, math.pi / 2])  # 90 deg/unit_time
        t = twist(omega, torch.zeros(3))
        point = torch.tensor([1.0, 0.0, 0.0])
        dt = torch.tensor(1.0)
        result = twist_apply(t, point, dt)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_consistency_with_se3_exp(self):
        """twist_apply should match applying se3_exp(dt * twist) to point."""
        from torchscience.geometry.transform import (
            rigid_transform_apply,
            se3_exp,
            twist,
            twist_apply,
        )

        angular = torch.randn(3) * 0.5
        linear = torch.randn(3)
        t = twist(angular, linear)
        point = torch.randn(3)
        dt = torch.tensor(0.5)

        # Apply twist directly
        result1 = twist_apply(t, point, dt)

        # Apply via se3_exp
        scaled_twist = twist(angular * dt, linear * dt)
        transform = se3_exp(scaled_twist)
        result2 = rigid_transform_apply(transform, point)

        assert torch.allclose(result1, result2, atol=1e-5)

    def test_batch(self):
        """Batched twist application."""
        from torchscience.geometry.transform import twist, twist_apply

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)
        points = torch.randn(10, 3)
        dt = torch.randn(10)

        result = twist_apply(t, points, dt)
        assert result.shape == (10, 3)


class TestTwistGradients:
    """Tests for twist gradient computation."""

    def test_twist_to_matrix_gradcheck(self):
        """Gradient check for twist_to_matrix."""
        from torchscience.geometry.transform import twist, twist_to_matrix

        angular = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        linear = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(ang, lin):
            t = twist(ang, lin)
            return twist_to_matrix(t)

        assert gradcheck(fn, (angular, linear), eps=1e-6, atol=1e-4)

    def test_se3_exp_gradcheck(self):
        """Gradient check for se3_exp."""
        from torchscience.geometry.transform import (
            rigid_transform_to_matrix,
            se3_exp,
            twist,
        )

        angular = (
            torch.randn(5, 3, dtype=torch.float64, requires_grad=True) * 0.5
        )
        linear = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(ang, lin):
            t = twist(ang, lin)
            transform = se3_exp(t)
            # Return matrix representation for scalar output
            return rigid_transform_to_matrix(transform)

        assert gradcheck(fn, (angular, linear), eps=1e-6, atol=1e-4)

    def test_se3_log_gradcheck(self):
        """Gradient check for se3_log."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            se3_log,
        )

        # Create valid transforms with small rotations
        q_data = torch.randn(5, 4, dtype=torch.float64)
        q_data = q_data / torch.linalg.norm(q_data, dim=-1, keepdim=True)
        q_data = q_data.clone().detach().requires_grad_(True)
        t_data = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(q_val, t_val):
            q = quaternion(q_val)
            transform = rigid_transform(q, t_val)
            result = se3_log(transform)
            return result.angular, result.linear

        assert gradcheck(
            lambda q, t: fn(q, t)[0], (q_data, t_data), eps=1e-6, atol=1e-4
        )
        assert gradcheck(
            lambda q, t: fn(q, t)[1], (q_data, t_data), eps=1e-6, atol=1e-4
        )

    def test_twist_apply_gradcheck(self):
        """Gradient check for twist_apply."""
        from torchscience.geometry.transform import twist, twist_apply

        angular = (
            torch.randn(5, 3, dtype=torch.float64, requires_grad=True) * 0.5
        )
        linear = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        points = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        dt = torch.rand(5, dtype=torch.float64, requires_grad=True)

        def fn(ang, lin, pts, dt_val):
            t = twist(ang, lin)
            return twist_apply(t, pts, dt_val)

        assert gradcheck(
            fn, (angular, linear, points, dt), eps=1e-6, atol=1e-4
        )


class TestTwistDtypes:
    """Tests for twist with different data types."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.geometry.transform import twist

        angular = torch.randn(3, dtype=torch.float32)
        linear = torch.randn(3, dtype=torch.float32)
        t = twist(angular, linear)
        assert t.angular.dtype == torch.float32
        assert t.linear.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.geometry.transform import twist

        angular = torch.randn(3, dtype=torch.float64)
        linear = torch.randn(3, dtype=torch.float64)
        t = twist(angular, linear)
        assert t.angular.dtype == torch.float64
        assert t.linear.dtype == torch.float64


class TestSE3NumericalStability:
    """Tests for SE(3) numerical stability edge cases."""

    def test_very_small_rotation(self):
        """Test with rotation near machine epsilon."""
        from torchscience.geometry.transform import se3_exp, se3_log, twist

        for scale in [1e-10, 1e-12, 1e-14]:
            angular = torch.tensor([scale, 0.0, 0.0], dtype=torch.float64)
            linear = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
            t = twist(angular, linear)

            transform = se3_exp(t)
            t_back = se3_log(transform)

            # Should roundtrip correctly
            assert torch.allclose(t_back.angular, angular, atol=1e-10)
            assert torch.allclose(t_back.linear, linear, atol=1e-10)

    def test_mixed_scales_batch(self):
        """Batch with mixed small and normal rotations."""
        from torchscience.geometry.transform import (
            rigid_transform_to_matrix,
            se3_exp,
            se3_log,
            twist,
        )

        angular = torch.tensor(
            [
                [1e-10, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1e-12],
                [0.3, 0.3, 0.3],
            ],
            dtype=torch.float64,
        )
        linear = torch.randn(5, 3, dtype=torch.float64)
        t = twist(angular, linear)

        transform = se3_exp(t)
        t_back = se3_log(transform)
        transform_back = se3_exp(t_back)

        # Transform matrices should match
        M1 = rigid_transform_to_matrix(transform)
        M2 = rigid_transform_to_matrix(transform_back)
        assert torch.allclose(M1, M2, atol=1e-8)


class TestTwistTransform:
    """Tests for twist_transform (adjoint action on twists)."""

    def test_identity_transform_returns_same_twist(self):
        """Transforming a twist by identity returns the same twist."""
        from torchscience.geometry.transform import (
            rigid_transform_identity,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        angular = torch.tensor([0.1, 0.2, 0.3])
        linear = torch.tensor([1.0, 2.0, 3.0])
        t = twist(angular, linear)

        identity = rigid_transform_identity()
        transformed = twist_transform(t, identity)

        assert torch.allclose(transformed.angular, angular, atol=1e-5)
        assert torch.allclose(transformed.linear, linear, atol=1e-5)

    def test_pure_rotation_transform(self):
        """Pure rotation rotates both angular and linear components."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        # 90 degrees around z-axis
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        transform = rigid_transform(q, torch.zeros(3))

        # Twist in x direction
        angular = torch.tensor([1.0, 0.0, 0.0])
        linear = torch.tensor([1.0, 0.0, 0.0])
        t = twist(angular, linear)

        transformed = twist_transform(t, transform)

        # After 90 deg rotation around z:
        # angular' = R @ angular = (0, 1, 0)
        # linear' = R @ linear + [t]x @ R @ angular = (0, 1, 0) + 0 = (0, 1, 0)
        expected_angular = torch.tensor([0.0, 1.0, 0.0])
        expected_linear = torch.tensor([0.0, 1.0, 0.0])

        assert torch.allclose(transformed.angular, expected_angular, atol=1e-5)
        assert torch.allclose(transformed.linear, expected_linear, atol=1e-5)

    def test_pure_translation_transform(self):
        """Pure translation only affects linear component via cross product."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([0.0, 0.0, 1.0])  # z-translation
        transform = rigid_transform(identity_rot, translation)

        # Twist with angular velocity around z
        angular = torch.tensor([0.0, 0.0, 1.0])
        linear = torch.tensor([0.0, 0.0, 0.0])
        t = twist(angular, linear)

        transformed = twist_transform(t, transform)

        # angular' = R @ angular = angular (identity rotation)
        # linear' = R @ linear + t x (R @ angular) = 0 + (0,0,1) x (0,0,1) = 0
        assert torch.allclose(transformed.angular, angular, atol=1e-5)
        assert torch.allclose(transformed.linear, torch.zeros(3), atol=1e-5)

    def test_translation_with_non_parallel_angular(self):
        """Translation produces cross product when angular is non-parallel."""
        from torchscience.geometry.transform import (
            quaternion,
            rigid_transform,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        identity_rot = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        translation = torch.tensor([1.0, 0.0, 0.0])  # x-translation
        transform = rigid_transform(identity_rot, translation)

        # Twist with angular velocity around z
        angular = torch.tensor([0.0, 0.0, 1.0])
        linear = torch.tensor([0.0, 0.0, 0.0])
        t = twist(angular, linear)

        transformed = twist_transform(t, transform)

        # angular' = angular (identity rotation)
        # linear' = 0 + (1,0,0) x (0,0,1) = (0, -1, 0)
        expected_linear = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(transformed.angular, angular, atol=1e-5)
        assert torch.allclose(transformed.linear, expected_linear, atol=1e-5)

    def test_consistency_with_adjoint_matrix(self):
        """twist_transform should match Ad_T @ twist_vector."""
        from torchscience.geometry.transform import (
            quaternion,
            quaternion_normalize,
            rigid_transform,
            twist,
            twist_from_vector,
            twist_to_vector,
        )
        from torchscience.geometry.transform._rigid_transform import (
            rigid_transform_adjoint,
        )
        from torchscience.geometry.transform._twist import twist_transform

        q = quaternion_normalize(quaternion(torch.randn(4)))
        translation = torch.randn(3)
        transform = rigid_transform(q, translation)

        angular = torch.randn(3)
        linear = torch.randn(3)
        t = twist(angular, linear)

        # Method 1: twist_transform
        transformed1 = twist_transform(t, transform)

        # Method 2: adjoint matrix multiplication
        adjoint = rigid_transform_adjoint(transform)
        vector = twist_to_vector(t)
        transformed_vector = adjoint @ vector
        transformed2 = twist_from_vector(transformed_vector)

        assert torch.allclose(
            transformed1.angular, transformed2.angular, atol=1e-5
        )
        assert torch.allclose(
            transformed1.linear, transformed2.linear, atol=1e-5
        )

    def test_twist_transform_batch(self):
        """Batched twist transform."""
        from torchscience.geometry.transform import (
            quaternion,
            quaternion_normalize,
            rigid_transform,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        translation = torch.randn(10, 3)
        transform = rigid_transform(q, translation)

        angular = torch.randn(10, 3)
        linear = torch.randn(10, 3)
        t = twist(angular, linear)

        transformed = twist_transform(t, transform)

        assert transformed.angular.shape == (10, 3)
        assert transformed.linear.shape == (10, 3)

    def test_twist_transform_composition(self):
        """twist_transform(twist_transform(t, T1), T2) = twist_transform(t, compose(T2, T1))."""
        from torchscience.geometry.transform import (
            quaternion,
            quaternion_normalize,
            rigid_transform,
            rigid_transform_compose,
            twist,
        )
        from torchscience.geometry.transform._twist import twist_transform

        q1 = quaternion_normalize(quaternion(torch.randn(4)))
        t1 = torch.randn(3)
        transform1 = rigid_transform(q1, t1)

        q2 = quaternion_normalize(quaternion(torch.randn(4)))
        t2 = torch.randn(3)
        transform2 = rigid_transform(q2, t2)

        angular = torch.randn(3)
        linear = torch.randn(3)
        tw = twist(angular, linear)

        # Method 1: Sequential application
        temp = twist_transform(tw, transform1)
        result1 = twist_transform(temp, transform2)

        # Method 2: Compose and apply
        # Note: compose(T2, T1) applies T1 first, then T2
        composed = rigid_transform_compose(transform2, transform1)
        result2 = twist_transform(tw, composed)

        assert torch.allclose(result1.angular, result2.angular, atol=1e-5)
        assert torch.allclose(result1.linear, result2.linear, atol=1e-5)

    def test_twist_transform_gradcheck(self):
        """Gradient check for twist_transform."""
        from torchscience.geometry.transform._twist import twist_transform

        q = torch.randn(4, dtype=torch.float64)
        q = q / torch.linalg.norm(q)
        q = q.clone().detach().requires_grad_(True)
        t_trans = torch.randn(3, dtype=torch.float64, requires_grad=True)

        angular = torch.randn(3, dtype=torch.float64, requires_grad=True)
        linear = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def transform_fn(q_val, t_val, ang, lin):
            from torchscience.geometry.transform import (
                Quaternion,
                rigid_transform,
                twist,
            )

            transform = rigid_transform(Quaternion(wxyz=q_val), t_val)
            tw = twist(ang, lin)
            result = twist_transform(tw, transform)
            return result.angular, result.linear

        assert gradcheck(
            lambda q, t, a, l: transform_fn(q, t, a, l)[0],
            (q, t_trans, angular, linear),
            eps=1e-6,
            atol=1e-4,
        )
        assert gradcheck(
            lambda q, t, a, l: transform_fn(q, t, a, l)[1],
            (q, t_trans, angular, linear),
            eps=1e-6,
            atol=1e-4,
        )


class TestTwistIntegration:
    """Integration tests for twist operations."""

    def test_consistency_with_rigid_transform(self):
        """Twist operations should be consistent with rigid transform operations."""
        from torchscience.geometry.transform import (
            rigid_transform_apply,
            rigid_transform_inverse,
            se3_exp,
            twist,
        )

        # Create a twist and its exponential
        angular = torch.randn(3, dtype=torch.float64) * 0.5
        linear = torch.randn(3, dtype=torch.float64)
        t = twist(angular, linear)
        transform = se3_exp(t)

        # The inverse transform should correspond to negative twist
        inverse_transform = rigid_transform_inverse(transform)
        negative_twist = twist(-angular, -linear)
        inverse_from_twist = se3_exp(negative_twist)

        # Apply to point and check
        point = torch.randn(3, dtype=torch.float64)
        p1 = rigid_transform_apply(inverse_transform, point)
        p2 = rigid_transform_apply(inverse_from_twist, point)
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_adjoint_property(self):
        """Test that se3_exp(-twist) = inverse(se3_exp(twist))."""
        from torchscience.geometry.transform import (
            rigid_transform_inverse,
            rigid_transform_to_matrix,
            se3_exp,
            twist,
        )

        angular = torch.randn(10, 3, dtype=torch.float64) * 0.5
        linear = torch.randn(10, 3, dtype=torch.float64)
        t = twist(angular, linear)
        neg_t = twist(-angular, -linear)

        exp_t = se3_exp(t)
        exp_neg_t = se3_exp(neg_t)
        inv_exp_t = rigid_transform_inverse(exp_t)

        # exp(-t) should equal inverse(exp(t))
        M1 = rigid_transform_to_matrix(exp_neg_t)
        M2 = rigid_transform_to_matrix(inv_exp_t)
        assert torch.allclose(M1, M2, atol=1e-8)
