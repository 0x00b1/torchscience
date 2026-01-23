"""Tests for AxisAngle tensorclass and conversion functions."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    AxisAngle,
    Quaternion,
    axis_angle,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion,
    quaternion_normalize,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


class TestAxisAngleConstruction:
    """Tests for AxisAngle construction."""

    def test_from_tensor(self):
        """Create AxisAngle from tensors."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        aa = AxisAngle(axis=axis, angle=angle)
        assert aa.axis.shape == (3,)
        assert aa.angle.shape == ()
        assert torch.allclose(aa.axis, axis)
        assert torch.allclose(aa.angle, angle)

    def test_batch(self):
        """Batch of axis-angles."""
        axis = torch.randn(10, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10)
        aa = AxisAngle(axis=axis, angle=angle)
        assert aa.axis.shape == (10, 3)
        assert aa.angle.shape == (10,)

    def test_factory_function(self):
        """Create via axis_angle() factory."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        assert isinstance(aa, AxisAngle)
        assert torch.allclose(aa.axis, axis)
        assert torch.allclose(aa.angle, angle)

    def test_invalid_axis_shape(self):
        """Raise error for wrong last dimension on axis."""
        with pytest.raises(ValueError, match="last dimension 3"):
            axis_angle(torch.randn(4), torch.tensor(0.5))


class TestAxisAngleToQuaternion:
    """Tests for axis_angle_to_quaternion."""

    def test_identity_rotation(self):
        """Zero angle gives identity quaternion."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(0.0)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        # q = [cos(45), 0, 0, sin(45)]
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_x(self):
        """90-degree rotation around x-axis."""
        axis = torch.tensor([1.0, 0.0, 0.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        # q = [cos(45), sin(45), 0, 0]
        expected = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_around_y(self):
        """90-degree rotation around y-axis."""
        axis = torch.tensor([0.0, 1.0, 0.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        # q = [cos(45), 0, sin(45), 0]
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_180_deg_around_z(self):
        """180-degree rotation around z-axis."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        # q = [0, 0, 0, 1]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Random axis-angle
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis)
        angle = torch.tensor(1.23, dtype=torch.float64)
        aa = axis_angle(axis, angle)

        # Convert to quaternion
        q = axis_angle_to_quaternion(aa)

        # scipy uses rotvec (axis * angle)
        rotvec = (axis * angle).numpy()
        r_scipy = R.from_rotvec(rotvec)
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
        axis = torch.randn(10, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.shape == (10, 4)

    def test_output_is_unit_quaternion(self):
        """Output should be a unit quaternion."""
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        norms = torch.linalg.norm(q.wxyz, dim=-1)
        assert torch.allclose(
            norms, torch.ones(10, dtype=torch.float64), atol=1e-5
        )

    def test_gradcheck(self):
        """Gradient check."""
        axis = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        axis_normalized = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(5, dtype=torch.float64, requires_grad=True)

        def fn(ax, ang):
            ax_norm = ax / torch.linalg.norm(ax, dim=-1, keepdim=True)
            aa = AxisAngle(axis=ax_norm, angle=ang)
            return axis_angle_to_quaternion(aa).wxyz

        assert gradcheck(fn, (axis, angle), eps=1e-6, atol=1e-4)


class TestQuaternionToAxisAngle:
    """Tests for quaternion_to_axis_angle."""

    def test_identity_rotation(self):
        """Identity quaternion gives zero angle."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        aa = quaternion_to_axis_angle(q)
        assert torch.allclose(aa.angle, torch.tensor(0.0), atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        aa = quaternion_to_axis_angle(q)
        expected_axis = torch.tensor([0.0, 0.0, 1.0])
        expected_angle = torch.tensor(math.pi / 2)
        assert torch.allclose(aa.axis, expected_axis, atol=1e-5)
        assert torch.allclose(aa.angle, expected_angle, atol=1e-5)

    def test_roundtrip_axis_angle_to_quaternion_to_axis_angle(self):
        """AxisAngle -> Quaternion -> AxisAngle roundtrip."""
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        # Use positive angles in [0, pi] for cleaner roundtrip
        angle = torch.rand(10, dtype=torch.float64) * math.pi
        aa = axis_angle(axis, angle)

        q = axis_angle_to_quaternion(aa)
        aa_back = quaternion_to_axis_angle(q)

        # For angles near 0, the axis is not well-defined
        # For other angles, axis and angle should match (or axis flipped with angle negated)
        for i in range(10):
            if angle[i] > 0.1:  # Skip near-zero angles
                # Check if axis matches (or is flipped with negated angle)
                axis_match = torch.allclose(
                    aa_back.axis[i], axis[i], atol=1e-5
                )
                axis_flip = torch.allclose(
                    aa_back.axis[i], -axis[i], atol=1e-5
                )
                if axis_match:
                    assert torch.allclose(
                        aa_back.angle[i], angle[i], atol=1e-5
                    )
                elif axis_flip:
                    assert torch.allclose(
                        aa_back.angle[i], -angle[i], atol=1e-5
                    )
                else:
                    pytest.fail(f"Axis mismatch at index {i}")

    def test_batch(self):
        """Batched conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        aa = quaternion_to_axis_angle(q)
        assert aa.axis.shape == (10, 3)
        assert aa.angle.shape == (10,)

    def test_gradcheck(self):
        """Gradient check."""
        q_raw = torch.randn(5, 4, dtype=torch.float64)
        q_raw = q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True)
        q = q_raw.clone().detach().requires_grad_(True)

        def fn(wxyz):
            qq = Quaternion(wxyz=wxyz)
            aa = quaternion_to_axis_angle(qq)
            # Return angle only since axis has singularity at identity
            return aa.angle

        assert gradcheck(fn, (q,), eps=1e-6, atol=1e-4)


class TestAxisAngleToMatrix:
    """Tests for axis_angle_to_matrix."""

    def test_identity_rotation(self):
        """Zero angle gives identity matrix."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(0.0)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        # Expected: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
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
        axis = torch.tensor([1.0, 0.0, 0.0])
        angle = torch.tensor(math.pi)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        # Expected: diag([1, -1, -1])
        expected = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Random axis-angle
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis)
        angle = torch.tensor(1.23, dtype=torch.float64)
        aa = axis_angle(axis, angle)

        # Convert to matrix
        mat = axis_angle_to_matrix(aa)

        # scipy uses rotvec (axis * angle)
        rotvec = (axis * angle).numpy()
        r_scipy = R.from_rotvec(rotvec)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_batch(self):
        """Batched conversion."""
        axis = torch.randn(10, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self):
        """R @ R.T = I for valid axis-angles."""
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        )
        assert torch.allclose(RRT, expected, atol=1e-5)

    def test_determinant_one(self):
        """det(R) = 1 for valid axis-angles."""
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        dets = torch.linalg.det(R)
        expected = torch.ones(10, dtype=torch.float64)
        assert torch.allclose(dets, expected, atol=1e-5)

    def test_consistency_with_quaternion_path(self):
        """axis_angle_to_matrix should match going through quaternion."""
        axis = torch.randn(5, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(5, dtype=torch.float64)
        aa = axis_angle(axis, angle)

        # Direct conversion
        R_direct = axis_angle_to_matrix(aa)

        # Via quaternion
        q = axis_angle_to_quaternion(aa)
        R_via_quat = quaternion_to_matrix(q)

        assert torch.allclose(R_direct, R_via_quat, atol=1e-5)

    def test_gradcheck(self):
        """Gradient check."""
        axis = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        angle = torch.randn(5, dtype=torch.float64, requires_grad=True)

        def fn(ax, ang):
            ax_norm = ax / torch.linalg.norm(ax, dim=-1, keepdim=True)
            aa = AxisAngle(axis=ax_norm, angle=ang)
            return axis_angle_to_matrix(aa)

        assert gradcheck(fn, (axis, angle), eps=1e-6, atol=1e-4)


class TestMatrixToAxisAngle:
    """Tests for matrix_to_axis_angle."""

    def test_identity_matrix(self):
        """Identity matrix gives zero angle."""
        R = torch.eye(3)
        aa = matrix_to_axis_angle(R)
        assert torch.allclose(aa.angle, torch.tensor(0.0), atol=1e-5)

    def test_roundtrip_matrix_to_axis_angle_to_matrix(self):
        """Matrix -> AxisAngle -> Matrix roundtrip."""
        # Create valid rotation matrices from axis-angle
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        R_original = axis_angle_to_matrix(aa)

        # Convert back
        aa_back = matrix_to_axis_angle(R_original)
        R_back = axis_angle_to_matrix(aa_back)

        assert torch.allclose(R_back, R_original, atol=1e-5)

    def test_batch(self):
        """Batched conversion."""
        # Create valid rotation matrices
        axis = torch.randn(10, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)

        aa_back = matrix_to_axis_angle(R)
        assert aa_back.axis.shape == (10, 3)
        assert aa_back.angle.shape == (10,)

    def test_gradcheck(self):
        """Gradient check."""
        # Use valid rotation matrices
        axis = torch.randn(5, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(5, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)
        R = R.clone().detach().requires_grad_(True)

        def fn(mat):
            aa = matrix_to_axis_angle(mat)
            return aa.angle

        assert gradcheck(fn, (R,), eps=1e-6, atol=1e-4)


class TestAxisAngleShape:
    """Tests for shape handling in axis-angle functions."""

    def test_single_axis_angle(self):
        """Single axis-angle (3,) and () shapes."""
        axis = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.shape == (4,)
        R = axis_angle_to_matrix(aa)
        assert R.shape == (3, 3)

    def test_batch(self):
        """Batch of axis-angles (B, 3) and (B,) shapes."""
        axis = torch.randn(10, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.shape == (10, 4)
        R = axis_angle_to_matrix(aa)
        assert R.shape == (10, 3, 3)

    def test_multi_batch(self):
        """Multi-batch axis-angles (B, C, 3) and (B, C) shapes."""
        axis = torch.randn(5, 3, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(5, 3)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.shape == (5, 3, 4)
        R = axis_angle_to_matrix(aa)
        assert R.shape == (5, 3, 3, 3)


class TestAxisAngleDtypes:
    """Tests for dtype handling."""

    def test_float32(self):
        """Works with float32."""
        axis = torch.randn(10, 3, dtype=torch.float32)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float32)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.dtype == torch.float32
        R = axis_angle_to_matrix(aa)
        assert R.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        axis = torch.randn(10, 3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn(10, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        q = axis_angle_to_quaternion(aa)
        assert q.wxyz.dtype == torch.float64
        R = axis_angle_to_matrix(aa)
        assert R.dtype == torch.float64


class TestAxisAngleIntegration:
    """Integration tests for axis-angle conversions."""

    def test_full_roundtrip_axis_angle_quaternion_matrix(self):
        """axis_angle -> quaternion -> matrix -> quaternion -> axis_angle."""
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis)
        angle = torch.tensor(1.23, dtype=torch.float64)
        aa_original = axis_angle(axis, angle)

        # axis_angle -> quaternion
        q1 = axis_angle_to_quaternion(aa_original)

        # quaternion -> matrix
        R = quaternion_to_matrix(q1)

        # matrix -> axis_angle
        aa_back = matrix_to_axis_angle(R)

        # axis_angle -> matrix (to verify)
        R_back = axis_angle_to_matrix(aa_back)

        assert torch.allclose(R_back, R, atol=1e-5)

    def test_scipy_rotation_equivalence(self):
        """Full equivalence with scipy's Rotation class."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        # Create random axis-angle
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis)
        angle = torch.tensor(1.5, dtype=torch.float64)
        aa = axis_angle(axis, angle)

        # scipy rotation from rotvec
        rotvec = (axis * angle).numpy()
        r_scipy = R.from_rotvec(rotvec)

        # Compare quaternion
        q = axis_angle_to_quaternion(aa)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )
        q_match = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert q_match

        # Compare matrix
        mat = axis_angle_to_matrix(aa)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)
        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_negative_angle(self):
        """Negative angles should work correctly."""
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        angle_pos = torch.tensor(math.pi / 2, dtype=torch.float64)
        angle_neg = torch.tensor(-math.pi / 2, dtype=torch.float64)

        aa_pos = axis_angle(axis, angle_pos)
        aa_neg = axis_angle(axis, angle_neg)

        R_pos = axis_angle_to_matrix(aa_pos)
        R_neg = axis_angle_to_matrix(aa_neg)

        # R_neg should be the transpose of R_pos
        assert torch.allclose(R_neg, R_pos.T, atol=1e-5)

    def test_angle_greater_than_pi(self):
        """Angles greater than pi should work."""
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        angle = torch.tensor(3 * math.pi / 2, dtype=torch.float64)
        aa = axis_angle(axis, angle)
        R = axis_angle_to_matrix(aa)

        # 3pi/2 = -pi/2 rotation
        expected = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(R, expected, atol=1e-5)
