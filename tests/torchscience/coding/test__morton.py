import pytest
import torch
import torch.testing

from torchscience.coding import morton_decode, morton_encode


class TestMortonEncode3D:
    """Tests for 3D Morton encoding."""

    def test_basic_encoding(self):
        """Test basic 3D Morton encoding."""
        coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        codes = morton_encode(coords)
        expected = torch.tensor([0, 1, 2, 4])
        torch.testing.assert_close(codes, expected)

    def test_unit_cube_corners(self):
        """Test encoding of unit cube corners."""
        coords = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        codes = morton_encode(coords)
        expected = torch.tensor([3, 5, 6, 7])
        torch.testing.assert_close(codes, expected)

    def test_larger_coordinates(self):
        """Test encoding with larger coordinate values."""
        coords = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        codes = morton_encode(coords)
        # 2 in binary is 10, spread to every 3rd bit:
        # x=2: bits at positions 3 -> code 8
        # y=2: bits at positions 4 -> code 16
        # z=2: bits at positions 5 -> code 32
        expected = torch.tensor([8, 16, 32])
        torch.testing.assert_close(codes, expected)

    def test_interleaving_pattern(self):
        """Test that interleaving follows z-y-x pattern in low bits."""
        # Code 7 = 111 in binary = z=1, y=1, x=1
        coords = torch.tensor([[1, 1, 1]])
        codes = morton_encode(coords)
        assert codes.item() == 7

        # Code 56 = 111000 in binary = z=2, y=2, x=2
        coords = torch.tensor([[2, 2, 2]])
        codes = morton_encode(coords)
        assert codes.item() == 56

    def test_batched_encoding(self):
        """Test batched encoding with multiple dimensions."""
        coords = torch.tensor([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 1]]])
        codes = morton_encode(coords)
        assert codes.shape == (2, 2)
        expected = torch.tensor([[0, 1], [2, 7]])
        torch.testing.assert_close(codes, expected)

    def test_int32_input(self):
        """Test encoding with int32 input."""
        coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        codes = morton_encode(coords)
        assert codes.dtype == torch.int64
        # Verify round-trip
        decoded = morton_decode(codes, dimensions=3)
        expected = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        torch.testing.assert_close(decoded, expected)


class TestMortonEncode2D:
    """Tests for 2D Morton encoding."""

    def test_basic_encoding(self):
        """Test basic 2D Morton encoding."""
        coords = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        codes = morton_encode(coords)
        expected = torch.tensor([0, 1, 2, 3])
        torch.testing.assert_close(codes, expected)

    def test_larger_coordinates(self):
        """Test 2D encoding with larger coordinate values."""
        coords = torch.tensor([[2, 0], [0, 2], [2, 2]])
        codes = morton_encode(coords)
        # 2D interleaving: x at even bits, y at odd bits
        # x=2 (10 binary) -> bits at 2 -> 4
        # y=2 (10 binary) -> bits at 3 -> 8
        # x=2, y=2 -> 4 + 8 = 12
        expected = torch.tensor([4, 8, 12])
        torch.testing.assert_close(codes, expected)


class TestMortonDecode3D:
    """Tests for 3D Morton decoding."""

    def test_basic_decoding(self):
        """Test basic 3D Morton decoding."""
        codes = torch.tensor([0, 1, 2, 4, 7])
        coords = morton_decode(codes, dimensions=3)
        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
        )
        torch.testing.assert_close(coords, expected)

    def test_batched_decoding(self):
        """Test batched decoding."""
        codes = torch.tensor([[0, 7], [56, 63]])
        coords = morton_decode(codes, dimensions=3)
        assert coords.shape == (2, 2, 3)
        expected = torch.tensor(
            [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]
        )
        torch.testing.assert_close(coords, expected)


class TestMortonDecode2D:
    """Tests for 2D Morton decoding."""

    def test_basic_decoding(self):
        """Test basic 2D Morton decoding."""
        codes = torch.tensor([0, 1, 2, 3])
        coords = morton_decode(codes, dimensions=2)
        expected = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        torch.testing.assert_close(coords, expected)


class TestMortonRoundTrip:
    """Tests for encode-decode round trips."""

    def test_round_trip_3d(self):
        """Test that decode(encode(x)) == x for 3D."""
        coords = torch.tensor(
            [
                [0, 0, 0],
                [1, 2, 3],
                [10, 20, 30],
                [100, 200, 50],
                [1000, 500, 750],
            ]
        )
        codes = morton_encode(coords)
        decoded = morton_decode(codes, dimensions=3)
        torch.testing.assert_close(decoded, coords.to(torch.int64))

    def test_round_trip_2d(self):
        """Test that decode(encode(x)) == x for 2D."""
        coords = torch.tensor([[0, 0], [1, 2], [100, 200], [1000, 500]])
        codes = morton_encode(coords)
        decoded = morton_decode(codes, dimensions=2)
        torch.testing.assert_close(decoded, coords.to(torch.int64))

    def test_round_trip_random_3d(self):
        """Test round trip with random 3D coordinates."""
        torch.manual_seed(42)
        # Keep coordinates small enough to avoid overflow (< 2^20 for 3D)
        coords = torch.randint(0, 1000, (100, 3), dtype=torch.int64)
        codes = morton_encode(coords)
        decoded = morton_decode(codes, dimensions=3)
        torch.testing.assert_close(decoded, coords)

    def test_round_trip_random_2d(self):
        """Test round trip with random 2D coordinates."""
        torch.manual_seed(42)
        coords = torch.randint(0, 10000, (100, 2), dtype=torch.int64)
        codes = morton_encode(coords)
        decoded = morton_decode(codes, dimensions=2)
        torch.testing.assert_close(decoded, coords)


class TestMortonSpatialLocality:
    """Tests for spatial locality properties."""

    def test_adjacent_points_similar_codes(self):
        """Verify that adjacent points have similar Morton codes."""
        # Adjacent points in a grid should have codes that differ by at most 7
        coords = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],  # x-adjacent
                [0, 0, 0],
                [0, 1, 0],  # y-adjacent
                [0, 0, 0],
                [0, 0, 1],  # z-adjacent
            ]
        )
        codes = morton_encode(coords)

        # Check x-adjacent: codes differ by 1
        assert abs(codes[0].item() - codes[1].item()) == 1

        # Check y-adjacent: codes differ by 2
        assert abs(codes[2].item() - codes[3].item()) == 2

        # Check z-adjacent: codes differ by 4
        assert abs(codes[4].item() - codes[5].item()) == 4


class TestMortonEdgeCases:
    """Tests for edge cases."""

    def test_single_point(self):
        """Test single point encoding/decoding."""
        coords = torch.tensor([[5, 10, 15]])
        codes = morton_encode(coords)
        assert codes.shape == (1,)
        decoded = morton_decode(codes, dimensions=3)
        torch.testing.assert_close(decoded, coords.to(torch.int64))

    def test_zero_coordinates(self):
        """Test all-zero coordinates."""
        coords = torch.tensor([[0, 0, 0]])
        codes = morton_encode(coords)
        assert codes.item() == 0
        decoded = morton_decode(codes, dimensions=3)
        torch.testing.assert_close(decoded, coords.to(torch.int64))

    def test_large_coordinates(self):
        """Test coordinates near the upper limit for 3D."""
        # Max 3D coordinate is 2^20 - 1 = 1048575
        max_coord = (1 << 20) - 1
        coords = torch.tensor([[max_coord, 0, 0]], dtype=torch.int64)
        codes = morton_encode(coords)
        decoded = morton_decode(codes, dimensions=3)
        torch.testing.assert_close(decoded, coords)

    def test_empty_batch(self):
        """Test empty batch."""
        coords = torch.zeros((0, 3), dtype=torch.int64)
        codes = morton_encode(coords)
        assert codes.shape == (0,)
        decoded = morton_decode(codes, dimensions=3)
        assert decoded.shape == (0, 3)

    def test_invalid_dimensions_decode(self):
        """Test that invalid dimensions raises error."""
        codes = torch.tensor([0, 1, 2])
        with pytest.raises(RuntimeError, match="dimensions must be 2 or 3"):
            morton_decode(codes, dimensions=4)
        with pytest.raises(RuntimeError, match="dimensions must be 2 or 3"):
            morton_decode(codes, dimensions=1)


class TestMortonOctreeHelpers:
    """Tests for octree-specific Morton code operations."""

    def test_octree_parent_child_relationship(self):
        """Verify parent-child Morton code relationships."""
        # In an octree, child code = (parent_morton << 3) | octant
        # This test verifies the bit layout matches expectations

        # Parent at depth 2, coordinates (1, 1, 1)
        # Child at depth 3, coordinates (2, 2, 2) which is octant 0 of parent's region
        parent_coords = torch.tensor([[1, 1, 1]])
        child_coords = torch.tensor([[2, 2, 2]])

        parent_code = morton_encode(parent_coords).item()
        child_code = morton_encode(child_coords).item()

        # Child morton should be parent morton shifted left by 3 bits
        assert child_code == (parent_code << 3)

    def test_z_order_curve_traversal(self):
        """Verify Z-order curve traversal order."""
        # For a 2x2x2 cube, Z-order should visit in this sequence
        expected_order = [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [0, 1, 1],  # 6
            [1, 1, 1],  # 7
        ]
        coords = torch.tensor(expected_order)
        codes = morton_encode(coords)

        # Codes should be sequential 0-7
        expected_codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        torch.testing.assert_close(codes, expected_codes)
