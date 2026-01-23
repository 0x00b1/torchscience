"""Tests for SO(3) exponential and logarithm maps."""

import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck


class TestSO3Exp:
    """Tests for so3_exp (exponential map from so(3) to SO(3))."""

    def test_zero_vector_identity(self):
        """Zero rotation vector gives identity matrix."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([0.0, 0.0, 0.0])
        R = so3_exp(omega)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-6)

    def test_small_angle_taylor_expansion(self):
        """Small angles should match first-order Taylor expansion.

        For small theta: R = I + [omega]_x
        """
        from torchscience.geometry.transform import so3_exp

        # Very small rotation vector
        omega = torch.tensor([1e-8, 2e-8, 3e-8], dtype=torch.float64)
        R = so3_exp(omega)

        # First-order approximation: R = I + [omega]_x
        expected = torch.eye(3, dtype=torch.float64)
        expected[0, 1] -= omega[2]
        expected[0, 2] += omega[1]
        expected[1, 0] += omega[2]
        expected[1, 2] -= omega[0]
        expected[2, 0] -= omega[1]
        expected[2, 1] += omega[0]

        assert torch.allclose(R, expected, atol=1e-10)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([0.0, 0.0, math.pi / 2])
        R = so3_exp(omega)
        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_around_x(self):
        """90-degree rotation around x-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([math.pi / 2, 0.0, 0.0])
        R = so3_exp(omega)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_around_y(self):
        """90-degree rotation around y-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([0.0, math.pi / 2, 0.0])
        R = so3_exp(omega)
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_x(self):
        """180-degree rotation around x-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([math.pi, 0.0, 0.0])
        R = so3_exp(omega)
        expected = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_y(self):
        """180-degree rotation around y-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([0.0, math.pi, 0.0])
        R = so3_exp(omega)
        expected = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_z(self):
        """180-degree rotation around z-axis."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.tensor([0.0, 0.0, math.pi])
        R = so3_exp(omega)
        expected = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation.from_rotvec()."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as Rot

        from torchscience.geometry.transform import so3_exp

        # Random rotation vector
        omega = torch.randn(3, dtype=torch.float64)
        R = so3_exp(omega)

        # scipy comparison
        R_scipy = torch.tensor(
            Rot.from_rotvec(omega.numpy()).as_matrix(), dtype=torch.float64
        )
        assert torch.allclose(R, R_scipy, atol=1e-10)

    def test_scipy_comparison_batch(self):
        """Compare batch with scipy."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as Rot

        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float64)
        R = so3_exp(omega)

        for i in range(10):
            R_scipy = torch.tensor(
                Rot.from_rotvec(omega[i].numpy()).as_matrix(),
                dtype=torch.float64,
            )
            assert torch.allclose(R[i], R_scipy, atol=1e-10)

    def test_batch(self):
        """Batched conversion."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3)
        R = so3_exp(omega)
        assert R.shape == (10, 3, 3)

    def test_multi_batch(self):
        """Multi-batch conversion."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(5, 3, 3)
        R = so3_exp(omega)
        assert R.shape == (5, 3, 3, 3)

    def test_orthogonality(self):
        """Output should be orthogonal: R @ R.T = I."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float64)
        R = so3_exp(omega)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = (
            torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        )
        assert torch.allclose(RRT, expected, atol=1e-10)

    def test_determinant_one(self):
        """Output should have det = 1."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float64)
        R = so3_exp(omega)
        dets = torch.linalg.det(R)
        expected = torch.ones(10, dtype=torch.float64)
        assert torch.allclose(dets, expected, atol=1e-10)

    def test_gradcheck(self):
        """Gradient check."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(so3_exp, (omega,), eps=1e-6, atol=1e-4)

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(so3_exp, (omega,), eps=1e-6, atol=1e-4)

    def test_input_validation(self):
        """Should raise error for wrong shape."""
        from torchscience.geometry.transform import so3_exp

        with pytest.raises(ValueError, match="last dimension 3"):
            so3_exp(torch.randn(4))

    def test_float32(self):
        """Works with float32."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float32)
        R = so3_exp(omega)
        assert R.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float64)
        R = so3_exp(omega)
        assert R.dtype == torch.float64


class TestSO3Log:
    """Tests for so3_log (logarithm map from SO(3) to so(3))."""

    def test_identity_matrix(self):
        """Identity matrix gives zero rotation vector."""
        from torchscience.geometry.transform import so3_log

        R = torch.eye(3)
        omega = so3_log(R)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(omega, expected, atol=1e-6)

    def test_near_identity(self):
        """Near-identity rotation should give small rotation vector."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Very small rotation
        omega_in = torch.tensor([1e-8, 2e-8, 3e-8], dtype=torch.float64)
        R = so3_exp(omega_in)
        omega_out = so3_log(R)
        assert torch.allclose(omega_out, omega_in, atol=1e-10)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        from torchscience.geometry.transform import so3_log

        R = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        omega = so3_log(R)
        expected = torch.tensor([0.0, 0.0, math.pi / 2])
        assert torch.allclose(omega, expected, atol=1e-5)

    def test_180_deg_around_x(self):
        """180-degree rotation around x-axis."""
        from torchscience.geometry.transform import so3_log

        R = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        omega = so3_log(R)
        # Axis should be [1, 0, 0] (or [-1, 0, 0])
        assert torch.allclose(
            torch.abs(omega), torch.tensor([math.pi, 0.0, 0.0]), atol=1e-5
        )

    def test_180_deg_around_y(self):
        """180-degree rotation around y-axis."""
        from torchscience.geometry.transform import so3_log

        R = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
        omega = so3_log(R)
        # Axis should be [0, 1, 0] (or [0, -1, 0])
        assert torch.allclose(
            torch.abs(omega), torch.tensor([0.0, math.pi, 0.0]), atol=1e-5
        )

    def test_180_deg_around_z(self):
        """180-degree rotation around z-axis."""
        from torchscience.geometry.transform import so3_log

        R = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        omega = so3_log(R)
        # Axis should be [0, 0, 1] (or [0, 0, -1])
        assert torch.allclose(
            torch.abs(omega), torch.tensor([0.0, 0.0, math.pi]), atol=1e-5
        )

    def test_180_deg_arbitrary_axis(self):
        """180-degree rotation around arbitrary axis."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # 180-degree rotation around normalized axis [1, 1, 0] / sqrt(2)
        axis = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        axis = axis / torch.linalg.norm(axis)
        omega_in = axis * math.pi
        R = so3_exp(omega_in)
        omega_out = so3_log(R)

        # The returned omega should have magnitude pi
        angle = torch.linalg.norm(omega_out)
        assert torch.allclose(
            angle, torch.tensor(math.pi, dtype=torch.float64), atol=1e-6
        )

        # And applying it should give the same rotation
        R_back = so3_exp(omega_out)
        assert torch.allclose(R_back, R, atol=1e-6)

    def test_scipy_comparison(self):
        """Compare with scipy's Rotation.as_rotvec()."""
        pytest.importorskip("scipy")
        from scipy.spatial.transform import Rotation as Rot

        from torchscience.geometry.transform import so3_log

        # Random rotation matrix (via random rotation vector)
        omega_in = torch.randn(3, dtype=torch.float64) * 0.5  # Keep angle < pi
        R = torch.tensor(
            Rot.from_rotvec(omega_in.numpy()).as_matrix(), dtype=torch.float64
        )

        omega_out = so3_log(R)

        # scipy comparison
        omega_scipy = torch.tensor(
            Rot.from_matrix(R.numpy()).as_rotvec(), dtype=torch.float64
        )
        assert torch.allclose(omega_out, omega_scipy, atol=1e-10)

    def test_batch(self):
        """Batched conversion."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega_in = torch.randn(10, 3)
        R = so3_exp(omega_in)
        omega_out = so3_log(R)
        assert omega_out.shape == (10, 3)

    def test_multi_batch(self):
        """Multi-batch conversion."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega_in = torch.randn(5, 3, 3)
        R = so3_exp(omega_in)
        omega_out = so3_log(R)
        assert omega_out.shape == (5, 3, 3)

    def test_input_validation(self):
        """Should raise error for wrong shape."""
        from torchscience.geometry.transform import so3_log

        with pytest.raises(ValueError, match="last two dimensions"):
            so3_log(torch.randn(3, 4))

    def test_float32(self):
        """Works with float32."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega = torch.randn(10, 3, dtype=torch.float32)
        R = so3_exp(omega)
        omega_out = so3_log(R)
        assert omega_out.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega = torch.randn(10, 3, dtype=torch.float64)
        R = so3_exp(omega)
        omega_out = so3_log(R)
        assert omega_out.dtype == torch.float64

    def test_gradcheck(self):
        """Gradient check."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Create valid rotation matrices
        omega = (
            torch.randn(5, 3, dtype=torch.float64) * 0.5
        )  # Keep away from pi
        R = so3_exp(omega)
        R = R.clone().detach().requires_grad_(True)

        assert gradcheck(so3_log, (R,), eps=1e-6, atol=1e-4)

    def test_gradcheck_near_identity(self):
        """Gradient check near identity."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Very small rotation
        omega = torch.randn(5, 3, dtype=torch.float64) * 1e-4
        R = so3_exp(omega)
        R = R.clone().detach().requires_grad_(True)

        assert gradcheck(so3_log, (R,), eps=1e-7, atol=1e-4)


class TestSO3ExpLogRoundtrip:
    """Tests for round-trip consistency between exp and log."""

    def test_exp_log_roundtrip(self):
        """so3_log(so3_exp(omega)) = omega for |omega| < pi."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Random rotation vectors with angle < pi
        omega = torch.randn(100, 3, dtype=torch.float64)
        # Normalize to have angle < pi/2 (safer range)
        norms = torch.linalg.norm(omega, dim=-1, keepdim=True)
        omega = omega / norms * (torch.rand(100, 1, dtype=torch.float64) * 2.5)

        R = so3_exp(omega)
        omega_back = so3_log(R)

        assert torch.allclose(omega_back, omega, atol=1e-10)

    def test_log_exp_roundtrip(self):
        """so3_exp(so3_log(R)) = R for valid rotation matrices."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Create valid rotation matrices
        omega = torch.randn(100, 3, dtype=torch.float64)
        R = so3_exp(omega)

        omega_out = so3_log(R)
        R_back = so3_exp(omega_out)

        assert torch.allclose(R_back, R, atol=1e-10)

    def test_roundtrip_small_angles(self):
        """Round-trip for very small angles."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega = torch.randn(50, 3, dtype=torch.float64) * 1e-8
        R = so3_exp(omega)
        omega_back = so3_log(R)
        assert torch.allclose(omega_back, omega, atol=1e-12)

    def test_roundtrip_near_pi(self):
        """Round-trip for angles near pi."""
        from torchscience.geometry.transform import so3_exp, so3_log

        # Angles close to pi (but not exactly pi)
        for angle in [0.95 * math.pi, 0.99 * math.pi]:
            axes = torch.randn(10, 3, dtype=torch.float64)
            axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
            omega = axes * angle

            R = so3_exp(omega)
            omega_back = so3_log(R)
            R_back = so3_exp(omega_back)

            # The rotation matrices should match
            assert torch.allclose(R_back, R, atol=1e-8)


class TestSO3ExpLogIntegration:
    """Integration tests with other rotation representations."""

    def test_consistency_with_rotation_vector(self):
        """so3_exp should be consistent with rotation_vector_to_matrix."""
        from torchscience.geometry.transform import (
            rotation_vector,
            rotation_vector_to_matrix,
            so3_exp,
        )

        omega = torch.randn(10, 3, dtype=torch.float64)
        R_exp = so3_exp(omega)
        R_rv = rotation_vector_to_matrix(rotation_vector(omega))

        assert torch.allclose(R_exp, R_rv, atol=1e-10)

    def test_consistency_with_matrix_to_rotation_vector(self):
        """so3_log should be consistent with matrix_to_rotation_vector."""
        from torchscience.geometry.transform import (
            matrix_to_rotation_vector,
            so3_exp,
            so3_log,
        )

        # Create rotation matrices
        omega_in = torch.randn(10, 3, dtype=torch.float64) * 0.5
        R = so3_exp(omega_in)

        omega_log = so3_log(R)
        rv = matrix_to_rotation_vector(R)

        # Both should give the same result
        assert torch.allclose(omega_log, rv.vector, atol=1e-10)

    def test_negative_rotation_vector(self):
        """exp(-omega) = exp(omega).T (inverse rotation)."""
        from torchscience.geometry.transform import so3_exp

        omega = torch.randn(10, 3, dtype=torch.float64)
        R_pos = so3_exp(omega)
        R_neg = so3_exp(-omega)

        # R_neg should be the transpose (inverse) of R_pos
        assert torch.allclose(R_neg, R_pos.transpose(-1, -2), atol=1e-10)


class TestSO3NumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_angle(self):
        """Test with angles close to machine epsilon."""
        from torchscience.geometry.transform import so3_exp, so3_log

        for scale in [1e-10, 1e-12, 1e-14]:
            omega = torch.tensor([scale, 0.0, 0.0], dtype=torch.float64)
            R = so3_exp(omega)

            # Should still be orthogonal
            RRT = R @ R.T
            assert torch.allclose(
                RRT, torch.eye(3, dtype=torch.float64), atol=1e-10
            )

            # Round-trip
            omega_back = so3_log(R)
            assert torch.allclose(omega_back, omega, atol=1e-10)

    def test_angle_exactly_pi(self):
        """Test with angle exactly pi."""
        from torchscience.geometry.transform import so3_exp, so3_log

        for axis in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]:
            axis_t = torch.tensor(axis, dtype=torch.float64)
            axis_t = axis_t / torch.linalg.norm(axis_t)
            omega = axis_t * math.pi

            R = so3_exp(omega)

            # Check orthogonality
            RRT = R @ R.T
            assert torch.allclose(
                RRT, torch.eye(3, dtype=torch.float64), atol=1e-10
            )

            # Log should give angle pi (direction may flip)
            omega_back = so3_log(R)
            angle_back = torch.linalg.norm(omega_back)
            assert torch.allclose(
                angle_back,
                torch.tensor(math.pi, dtype=torch.float64),
                atol=1e-6,
            )

            # exp of log should give same rotation
            R_back = so3_exp(omega_back)
            assert torch.allclose(R_back, R, atol=1e-6)

    def test_mixed_small_and_large_angles_batch(self):
        """Batch with mixed small and large angles."""
        from torchscience.geometry.transform import so3_exp, so3_log

        omega = torch.tensor(
            [
                [1e-10, 0.0, 0.0],  # Very small
                [1.0, 0.0, 0.0],  # Normal
                [3.0, 0.0, 0.0],  # Near pi
                [0.0, 1e-12, 0.0],  # Very small
                [0.0, 2.5, 0.0],  # Normal
            ],
            dtype=torch.float64,
        )

        R = so3_exp(omega)
        omega_back = so3_log(R)
        R_back = so3_exp(omega_back)

        # All should round-trip correctly
        assert torch.allclose(R_back, R, atol=1e-8)
