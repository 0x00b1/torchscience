"""Tests for EulerAngles tensorclass and conversion functions."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    EulerAngles,
    Quaternion,
    euler_angles,
    euler_angles_to_matrix,
    euler_angles_to_quaternion,
    matrix_to_euler_angles,
    quaternion,
    quaternion_normalize,
    quaternion_to_euler_angles,
    quaternion_to_matrix,
)
from torchscience.geometry.transform._conventions import (
    ALL_CONVENTIONS,
    PROPER_EULER_CONVENTIONS,
    TAIT_BRYAN_CONVENTIONS,
    get_axis_indices,
    validate_convention,
)


class TestConventions:
    """Tests for Euler angle conventions module."""

    def test_tait_bryan_conventions(self):
        """Tait-Bryan conventions should have three different axes."""
        expected = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}
        assert TAIT_BRYAN_CONVENTIONS == frozenset(expected)

    def test_proper_euler_conventions(self):
        """Proper Euler conventions should have first and third axes same."""
        expected = {"XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"}
        assert PROPER_EULER_CONVENTIONS == frozenset(expected)

    def test_all_conventions(self):
        """All conventions is the union of Tait-Bryan and proper Euler."""
        assert (
            ALL_CONVENTIONS
            == TAIT_BRYAN_CONVENTIONS | PROPER_EULER_CONVENTIONS
        )
        assert len(ALL_CONVENTIONS) == 12

    def test_validate_convention_valid(self):
        """Valid conventions should not raise."""
        for conv in ALL_CONVENTIONS:
            validate_convention(conv)  # Should not raise

    def test_validate_convention_invalid(self):
        """Invalid conventions should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Euler angle convention"):
            validate_convention("ABC")
        with pytest.raises(ValueError, match="Invalid Euler angle convention"):
            validate_convention("xyz")  # Lowercase
        with pytest.raises(ValueError, match="Invalid Euler angle convention"):
            validate_convention("XXX")  # All same

    def test_get_axis_indices(self):
        """Get axis indices from convention string."""
        assert get_axis_indices("XYZ") == (0, 1, 2)
        assert get_axis_indices("ZYX") == (2, 1, 0)
        assert get_axis_indices("XZX") == (0, 2, 0)
        assert get_axis_indices("ZXZ") == (2, 0, 2)


class TestEulerAnglesConstruction:
    """Tests for EulerAngles construction."""

    def test_from_tensor(self):
        """Create EulerAngles from tensor."""
        angles = torch.tensor([0.1, 0.2, 0.3])
        ea = EulerAngles(angles=angles, convention="XYZ")
        assert ea.angles.shape == (3,)
        assert ea.convention == "XYZ"
        assert torch.allclose(ea.angles, angles)

    def test_batch(self):
        """Batch of Euler angles."""
        angles = torch.randn(10, 3)
        ea = EulerAngles(angles=angles, convention="ZYX")
        assert ea.angles.shape == (10, 3)
        assert ea.convention == "ZYX"

    def test_factory_function(self):
        """Create via euler_angles() factory."""
        angles = torch.tensor([0.1, 0.2, 0.3])
        ea = euler_angles(angles, convention="XYZ")
        assert isinstance(ea, EulerAngles)
        assert torch.allclose(ea.angles, angles)
        assert ea.convention == "XYZ"

    def test_factory_default_convention(self):
        """Factory uses XYZ convention by default."""
        angles = torch.tensor([0.1, 0.2, 0.3])
        ea = euler_angles(angles)
        assert ea.convention == "XYZ"

    def test_invalid_shape(self):
        """Raise error for wrong last dimension."""
        with pytest.raises(ValueError, match="last dimension 3"):
            euler_angles(torch.randn(4))

    def test_invalid_convention(self):
        """Raise error for invalid convention."""
        with pytest.raises(ValueError, match="Invalid Euler angle convention"):
            euler_angles(torch.randn(3), convention="ABC")


class TestEulerAnglesToQuaternion:
    """Tests for euler_angles_to_quaternion."""

    def test_zero_angles_identity(self):
        """Zero Euler angles give identity quaternion."""
        ea = euler_angles(torch.tensor([0.0, 0.0, 0.0]), convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_x_only(self):
        """90-degree rotation around x-axis only."""
        ea = euler_angles(
            torch.tensor([math.pi / 2, 0.0, 0.0]), convention="XYZ"
        )
        q = euler_angles_to_quaternion(ea)
        expected = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_y_only(self):
        """90-degree rotation around y-axis only."""
        ea = euler_angles(
            torch.tensor([0.0, math.pi / 2, 0.0]), convention="XYZ"
        )
        q = euler_angles_to_quaternion(ea)
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_90_deg_z_only(self):
        """90-degree rotation around z-axis only."""
        ea = euler_angles(
            torch.tensor([0.0, 0.0, math.pi / 2]), convention="XYZ"
        )
        q = euler_angles_to_quaternion(ea)
        expected = torch.tensor(
            [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        )
        assert torch.allclose(q.wxyz, expected, atol=1e-5)

    def test_scipy_comparison_xyz(self):
        """Compare with scipy for XYZ convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="XYZ"
        )
        q = euler_angles_to_quaternion(ea)

        # scipy uses lowercase 'xyz' for intrinsic
        r_scipy = R.from_euler("xyz", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_zyx(self):
        """Compare with scipy for ZYX convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="ZYX"
        )
        q = euler_angles_to_quaternion(ea)

        # scipy uses lowercase for intrinsic rotations
        r_scipy = R.from_euler("zyx", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_yxz(self):
        """Compare with scipy for YXZ convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="YXZ"
        )
        q = euler_angles_to_quaternion(ea)

        r_scipy = R.from_euler("yxz", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_zxy(self):
        """Compare with scipy for ZXY convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="ZXY"
        )
        q = euler_angles_to_quaternion(ea)

        r_scipy = R.from_euler("zxy", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_xzy(self):
        """Compare with scipy for XZY convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="XZY"
        )
        q = euler_angles_to_quaternion(ea)

        r_scipy = R.from_euler("xzy", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_yzx(self):
        """Compare with scipy for YZX convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="YZX"
        )
        q = euler_angles_to_quaternion(ea)

        r_scipy = R.from_euler("yzx", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_scipy_comparison_proper_euler_zxz(self):
        """Compare with scipy for ZXZ proper Euler convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="ZXZ"
        )
        q = euler_angles_to_quaternion(ea)

        r_scipy = R.from_euler("zxz", angles_np)
        xyzw = r_scipy.as_quat()
        q_scipy_wxyz = torch.tensor(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float64
        )

        matches = torch.allclose(
            q.wxyz, q_scipy_wxyz, atol=1e-6
        ) or torch.allclose(q.wxyz, -q_scipy_wxyz, atol=1e-6)
        assert matches, (
            f"Got {q.wxyz}, expected {q_scipy_wxyz} or {-q_scipy_wxyz}"
        )

    def test_batch(self):
        """Batched conversion."""
        angles = torch.randn(10, 3)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.shape == (10, 4)

    def test_output_is_unit_quaternion(self):
        """Output should be a unit quaternion."""
        angles = torch.randn(10, 3, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        norms = torch.linalg.norm(q.wxyz, dim=-1)
        assert torch.allclose(
            norms, torch.ones(10, dtype=torch.float64), atol=1e-5
        )

    def test_gradcheck(self):
        """Gradient check."""
        angles = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(ang):
            ea = EulerAngles(angles=ang, convention="XYZ")
            return euler_angles_to_quaternion(ea).wxyz

        assert gradcheck(fn, (angles,), eps=1e-6, atol=1e-4)


class TestQuaternionToEulerAngles:
    """Tests for quaternion_to_euler_angles."""

    def test_identity_quaternion(self):
        """Identity quaternion gives zero Euler angles."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        ea = quaternion_to_euler_angles(q, convention="XYZ")
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(ea.angles, expected, atol=1e-5)

    def test_roundtrip_avoid_gimbal_lock(self):
        """euler_angles -> quaternion -> euler_angles roundtrip."""
        # Avoid gimbal lock: middle angle away from +/- pi/2
        angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")

        q = euler_angles_to_quaternion(ea)
        ea_back = quaternion_to_euler_angles(q, convention="XYZ")

        # The angles may differ but should represent the same rotation
        # Check via rotation matrix
        R1 = euler_angles_to_matrix(ea)
        R2 = euler_angles_to_matrix(ea_back)
        assert torch.allclose(R1, R2, atol=1e-5)

    def test_roundtrip_multiple_conventions(self):
        """Roundtrip for multiple conventions."""
        for convention in ["XYZ", "ZYX", "YXZ", "ZXY", "XZY", "YZX"]:
            # Avoid gimbal lock
            angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
            ea = euler_angles(angles, convention=convention)

            q = euler_angles_to_quaternion(ea)
            ea_back = quaternion_to_euler_angles(q, convention=convention)

            R1 = euler_angles_to_matrix(ea)
            R2 = euler_angles_to_matrix(ea_back)
            assert torch.allclose(R1, R2, atol=1e-5), (
                f"Failed for {convention}"
            )

    def test_batch(self):
        """Batched conversion."""
        q = quaternion_normalize(quaternion(torch.randn(10, 4)))
        ea = quaternion_to_euler_angles(q, convention="XYZ")
        assert ea.angles.shape == (10, 3)

    def test_gradcheck(self):
        """Gradient check (avoid gimbal lock in the test data)."""
        # Create quaternions that won't hit gimbal lock
        angles = torch.tensor([[0.3, 0.4, 0.5]] * 5, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        wxyz = q.wxyz.clone().detach().requires_grad_(True)

        def fn(w):
            qq = Quaternion(wxyz=w)
            ea = quaternion_to_euler_angles(qq, convention="XYZ")
            return ea.angles

        assert gradcheck(fn, (wxyz,), eps=1e-6, atol=1e-4)


class TestEulerAnglesToMatrix:
    """Tests for euler_angles_to_matrix."""

    def test_zero_angles_identity(self):
        """Zero Euler angles give identity matrix."""
        ea = euler_angles(torch.tensor([0.0, 0.0, 0.0]), convention="XYZ")
        R = euler_angles_to_matrix(ea)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_x_only(self):
        """90-degree rotation around x-axis."""
        ea = euler_angles(
            torch.tensor([math.pi / 2, 0.0, 0.0]), convention="XYZ"
        )
        R = euler_angles_to_matrix(ea)
        # Expected rotation matrix for 90 deg around x
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_z_only(self):
        """90-degree rotation around z-axis."""
        ea = euler_angles(
            torch.tensor([0.0, 0.0, math.pi / 2]), convention="XYZ"
        )
        R = euler_angles_to_matrix(ea)
        # Expected rotation matrix for 90 deg around z
        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_scipy_comparison_xyz(self):
        """Compare with scipy for XYZ convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="XYZ"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("xyz", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_scipy_comparison_zyx(self):
        """Compare with scipy for ZYX convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="ZYX"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("zyx", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_scipy_comparison_yxz(self):
        """Compare with scipy for YXZ convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="YXZ"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("yxz", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_scipy_comparison_zxy(self):
        """Compare with scipy for ZXY convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="ZXY"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("zxy", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_scipy_comparison_xzy(self):
        """Compare with scipy for XZY convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="XZY"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("xzy", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_scipy_comparison_yzx(self):
        """Compare with scipy for YZX convention."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]
        ea = euler_angles(
            torch.tensor(angles_np, dtype=torch.float64), convention="YZX"
        )
        mat = euler_angles_to_matrix(ea)

        r_scipy = R.from_euler("yzx", angles_np)
        mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

        assert torch.allclose(mat, mat_scipy, atol=1e-6)

    def test_batch(self):
        """Batched conversion."""
        angles = torch.randn(10, 3)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self):
        """R @ R.T = I for valid Euler angles."""
        angles = torch.randn(10, 3, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        )
        assert torch.allclose(RRT, expected, atol=1e-5)

    def test_determinant_one(self):
        """det(R) = 1 for valid Euler angles."""
        angles = torch.randn(10, 3, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)
        dets = torch.linalg.det(R)
        expected = torch.ones(10, dtype=torch.float64)
        assert torch.allclose(dets, expected, atol=1e-5)

    def test_gradcheck(self):
        """Gradient check."""
        angles = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        def fn(ang):
            ea = EulerAngles(angles=ang, convention="XYZ")
            return euler_angles_to_matrix(ea)

        assert gradcheck(fn, (angles,), eps=1e-6, atol=1e-4)


class TestMatrixToEulerAngles:
    """Tests for matrix_to_euler_angles."""

    def test_identity_matrix(self):
        """Identity matrix gives zero Euler angles."""
        R = torch.eye(3)
        ea = matrix_to_euler_angles(R, convention="XYZ")
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(ea.angles, expected, atol=1e-5)

    def test_roundtrip_avoid_gimbal_lock(self):
        """euler_angles -> matrix -> euler_angles roundtrip."""
        # Avoid gimbal lock
        angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)

        ea_back = matrix_to_euler_angles(R, convention="XYZ")
        R_back = euler_angles_to_matrix(ea_back)

        assert torch.allclose(R_back, R, atol=1e-5)

    def test_roundtrip_multiple_conventions(self):
        """Roundtrip for multiple conventions."""
        for convention in ["XYZ", "ZYX", "YXZ", "ZXY", "XZY", "YZX"]:
            # Avoid gimbal lock
            angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
            ea = euler_angles(angles, convention=convention)
            R = euler_angles_to_matrix(ea)

            ea_back = matrix_to_euler_angles(R, convention=convention)
            R_back = euler_angles_to_matrix(ea_back)

            assert torch.allclose(R_back, R, atol=1e-5), (
                f"Failed for {convention}"
            )

    def test_batch(self):
        """Batched conversion."""
        # Create valid rotation matrices
        angles = torch.randn(10, 3)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)

        ea_back = matrix_to_euler_angles(R, convention="XYZ")
        assert ea_back.angles.shape == (10, 3)

    def test_gradcheck(self):
        """Gradient check (avoid gimbal lock)."""
        # Create valid rotation matrices
        angles = torch.tensor([[0.3, 0.4, 0.5]] * 5, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)
        R = R.clone().detach().requires_grad_(True)

        def fn(mat):
            ea = matrix_to_euler_angles(mat, convention="XYZ")
            return ea.angles

        assert gradcheck(fn, (R,), eps=1e-6, atol=1e-4)


class TestEulerAnglesShape:
    """Tests for shape handling in Euler angles functions."""

    def test_single_euler_angles(self):
        """Single Euler angles (3,) shape."""
        angles = torch.tensor([0.1, 0.2, 0.3])
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.shape == (4,)
        R = euler_angles_to_matrix(ea)
        assert R.shape == (3, 3)

    def test_batch(self):
        """Batch of Euler angles (B, 3) shape."""
        angles = torch.randn(10, 3)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.shape == (10, 4)
        R = euler_angles_to_matrix(ea)
        assert R.shape == (10, 3, 3)

    def test_multi_batch(self):
        """Multi-batch Euler angles (B, C, 3) shape."""
        angles = torch.randn(5, 3, 3)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.shape == (5, 3, 4)
        R = euler_angles_to_matrix(ea)
        assert R.shape == (5, 3, 3, 3)


class TestEulerAnglesDtypes:
    """Tests for dtype handling."""

    def test_float32(self):
        """Works with float32."""
        angles = torch.randn(10, 3, dtype=torch.float32)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.dtype == torch.float32
        R = euler_angles_to_matrix(ea)
        assert R.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        angles = torch.randn(10, 3, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")
        q = euler_angles_to_quaternion(ea)
        assert q.wxyz.dtype == torch.float64
        R = euler_angles_to_matrix(ea)
        assert R.dtype == torch.float64


class TestEulerAnglesIntegration:
    """Integration tests for Euler angles conversions."""

    def test_consistency_with_quaternion_path(self):
        """euler_angles_to_matrix should match going through quaternion."""
        angles = torch.randn(5, 3, dtype=torch.float64)
        ea = euler_angles(angles, convention="XYZ")

        # Direct conversion
        R_direct = euler_angles_to_matrix(ea)

        # Via quaternion
        q = euler_angles_to_quaternion(ea)
        R_via_quat = quaternion_to_matrix(q)

        assert torch.allclose(R_direct, R_via_quat, atol=1e-5)

    def test_all_tait_bryan_conventions(self):
        """Test all 6 Tait-Bryan conventions."""
        angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)

        for convention in TAIT_BRYAN_CONVENTIONS:
            ea = euler_angles(angles, convention=convention)
            q = euler_angles_to_quaternion(ea)
            R = euler_angles_to_matrix(ea)

            # Verify quaternion gives same matrix
            R_from_q = quaternion_to_matrix(q)
            assert torch.allclose(R, R_from_q, atol=1e-5), (
                f"Failed for {convention}"
            )

    def test_all_proper_euler_conventions(self):
        """Test all 6 proper Euler conventions."""
        angles = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)

        for convention in PROPER_EULER_CONVENTIONS:
            ea = euler_angles(angles, convention=convention)
            q = euler_angles_to_quaternion(ea)
            R = euler_angles_to_matrix(ea)

            # Verify quaternion gives same matrix
            R_from_q = quaternion_to_matrix(q)
            assert torch.allclose(R, R_from_q, atol=1e-5), (
                f"Failed for {convention}"
            )

    def test_scipy_all_conventions(self):
        """Compare all conventions with scipy."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as R

        angles_np = [0.3, 0.5, 0.7]

        for convention in ALL_CONVENTIONS:
            ea = euler_angles(
                torch.tensor(angles_np, dtype=torch.float64),
                convention=convention,
            )
            mat = euler_angles_to_matrix(ea)

            # scipy uses lowercase for intrinsic rotations
            r_scipy = R.from_euler(convention.lower(), angles_np)
            mat_scipy = torch.tensor(r_scipy.as_matrix(), dtype=torch.float64)

            assert torch.allclose(mat, mat_scipy, atol=1e-6), (
                f"Failed for {convention}"
            )

    def test_gimbal_lock_near_singularity(self):
        """Test near gimbal lock (middle angle near pi/2) still produces valid rotation."""
        # Near gimbal lock for Tait-Bryan XYZ: middle angle near pi/2
        angles = torch.tensor(
            [0.3, math.pi / 2 - 0.01, 0.5], dtype=torch.float64
        )
        ea = euler_angles(angles, convention="XYZ")
        R = euler_angles_to_matrix(ea)

        # Should still be a valid rotation matrix
        RRT = R @ R.T
        assert torch.allclose(
            RRT, torch.eye(3, dtype=torch.float64), atol=1e-5
        )
        assert torch.allclose(
            torch.linalg.det(R),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-5,
        )

    def test_negative_angles(self):
        """Negative angles should work correctly."""
        angles_pos = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        angles_neg = torch.tensor([-0.3, -0.4, -0.5], dtype=torch.float64)

        ea_pos = euler_angles(angles_pos, convention="XYZ")
        ea_neg = euler_angles(angles_neg, convention="XYZ")

        R_pos = euler_angles_to_matrix(ea_pos)
        R_neg = euler_angles_to_matrix(ea_neg)

        # Product should give identity (inverses)
        # Not exact inverses due to composition order, but both should be valid
        assert torch.allclose(
            torch.linalg.det(R_pos),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-5,
        )
        assert torch.allclose(
            torch.linalg.det(R_neg),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-5,
        )
