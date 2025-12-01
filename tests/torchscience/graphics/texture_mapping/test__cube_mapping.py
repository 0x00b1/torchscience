"""Tests for cube_mapping texture mapping operator."""

import torch


class TestCubeMappingBasic:
    """Tests for basic shape and property verification."""

    def test_output_shapes(self):
        """Output shapes are correct."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(10, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.shape == (10,)
        assert u.shape == (10,)
        assert v.shape == (10,)

    def test_face_indices_valid(self):
        """Face indices are in range [0, 5]."""
        from torchscience.graphics.texture_mapping import cube_mapping

        torch.manual_seed(42)
        direction = torch.randn(100, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, _, _ = cube_mapping(direction)

        assert (face >= 0).all()
        assert (face <= 5).all()

    def test_uv_range(self):
        """UV coordinates are in range [0, 1]."""
        from torchscience.graphics.texture_mapping import cube_mapping

        torch.manual_seed(42)
        direction = torch.randn(100, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        _, u, v = cube_mapping(direction)

        assert (u >= 0).all()
        assert (u <= 1).all()
        assert (v >= 0).all()
        assert (v <= 1).all()


class TestCubeMappingCorrectness:
    """Tests for numerical correctness."""

    def test_positive_x_axis(self):
        """Direction along +X maps to face 0."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[1.0, 0.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 0  # +X face
        torch.testing.assert_close(
            u, torch.tensor([0.5]), rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            v, torch.tensor([0.5]), rtol=1e-5, atol=1e-7
        )

    def test_negative_x_axis(self):
        """Direction along -X maps to face 1."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[-1.0, 0.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 1  # -X face

    def test_positive_y_axis(self):
        """Direction along +Y maps to face 2."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 1.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 2  # +Y face

    def test_negative_y_axis(self):
        """Direction along -Y maps to face 3."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, -1.0, 0.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 3  # -Y face

    def test_positive_z_axis(self):
        """Direction along +Z maps to face 4."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 0.0, 1.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 4  # +Z face

    def test_negative_z_axis(self):
        """Direction along -Z maps to face 5."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.tensor([[0.0, 0.0, -1.0]])

        face, u, v = cube_mapping(direction)

        assert face.item() == 5  # -Z face


class TestCubeMappingDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(5, 3, dtype=torch.float32)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.dtype == torch.int64  # Face index is int
        assert u.dtype == torch.float32
        assert v.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        from torchscience.graphics.texture_mapping import cube_mapping

        direction = torch.randn(5, 3, dtype=torch.float64)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        face, u, v = cube_mapping(direction)

        assert face.dtype == torch.int64
        assert u.dtype == torch.float64
        assert v.dtype == torch.float64
