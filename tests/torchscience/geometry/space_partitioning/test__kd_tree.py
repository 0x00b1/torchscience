# tests/torchscience/geometry/space_partitioning/test__kd_tree.py
import pytest
import torch

from torchscience.geometry.space_partitioning import KdTree, kd_tree


class TestKdTreeBasic:
    """Tests for kd_tree build function."""

    def test_returns_kdtree_instance(self):
        """kd_tree returns a KdTree instance."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        assert isinstance(tree, KdTree)

    def test_stores_original_points(self):
        """KdTree stores original points."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        torch.testing.assert_close(tree.points, points)

    def test_has_required_attributes(self):
        """KdTree has all required tree structure attributes."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)

        # Check all attributes exist
        assert hasattr(tree, "points")
        assert hasattr(tree, "split_dim")
        assert hasattr(tree, "split_val")
        assert hasattr(tree, "left")
        assert hasattr(tree, "right")
        assert hasattr(tree, "indices")
        assert hasattr(tree, "leaf_starts")
        assert hasattr(tree, "leaf_counts")

    def test_leaf_size_parameter(self):
        """leaf_size parameter controls max points per leaf."""
        points = torch.randn(100, 3)
        tree = kd_tree(points, leaf_size=5)

        # All indices should be present
        assert tree.indices.shape[0] == 100

    def test_preserves_dtype(self):
        """Output tensors preserve input dtype."""
        points = torch.randn(100, 3, dtype=torch.float64)
        tree = kd_tree(points)
        assert tree.points.dtype == torch.float64
        # split_val matches input dtype for precision
        assert tree.split_val.dtype == torch.float64

    def test_preserves_device(self):
        """Output tensors preserve input device."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        assert tree.points.device == points.device


class TestKdTreeBatched:
    """Tests for batched kd_tree construction."""

    def test_batched_input(self):
        """Supports [..., n, d] batched input."""
        points = torch.randn(4, 100, 3)
        tree = kd_tree(points)

        assert isinstance(tree, KdTree)
        assert tree.batch_size == torch.Size([4])
        assert tree.points.shape == (4, 100, 3)
        assert tree.indices.shape == (4, 100)

    def test_batched_preserves_dtype(self):
        """Batched construction preserves dtype."""
        points = torch.randn(2, 50, 3, dtype=torch.float64)
        tree = kd_tree(points)

        assert tree.points.dtype == torch.float64
        assert tree.split_val.dtype == torch.float64

    def test_batched_indexing(self):
        """Batched KdTree supports indexing via tensorclass."""
        points = torch.randn(4, 100, 3)
        tree = kd_tree(points)

        # Index single tree
        tree0 = tree[0]
        assert tree0.points.shape == (100, 3)

        # Slice multiple trees
        tree_slice = tree[:2]
        assert tree_slice.batch_size == torch.Size([2])

    def test_multi_batch_dims(self):
        """Supports multiple batch dimensions."""
        points = torch.randn(2, 3, 100, 3)
        tree = kd_tree(points)

        assert tree.batch_size == torch.Size([2, 3])
        assert tree.points.shape == (2, 3, 100, 3)
        assert tree[0, 1].points.shape == (100, 3)


class TestKdTreeValidation:
    """Tests for input validation."""

    def test_requires_at_least_2d_input(self):
        """Raises RuntimeError for 1D input."""
        points = torch.randn(100)
        with pytest.raises(RuntimeError, match="at least 2D"):
            kd_tree(points)

    def test_handles_empty_input(self):
        """Handles empty point set gracefully."""
        points = torch.empty(0, 3)
        tree = kd_tree(points)
        assert tree.points.shape == (0, 3)

    def test_handles_single_point(self):
        """Handles single point correctly."""
        points = torch.randn(1, 3)
        tree = kd_tree(points)
        assert tree.points.shape == (1, 3)


class TestKdTreeCorrectness:
    """Tests for tree structure correctness."""

    def test_indices_cover_all_points(self):
        """All point indices appear exactly once in tree."""
        points = torch.randn(50, 3)
        tree = kd_tree(points)

        sorted_indices = torch.sort(tree.indices)[0]
        expected = torch.arange(50)
        torch.testing.assert_close(sorted_indices, expected)

    def test_split_dims_are_valid(self):
        """Split dimensions are in valid range [0, d) for non-leaf nodes."""
        points = torch.randn(100, 5)
        tree = kd_tree(points)

        # Non-leaf nodes have valid split_dim
        non_leaf_mask = (tree.left != -1) | (tree.right != -1)
        if non_leaf_mask.any():
            non_leaf_split_dims = tree.split_dim[non_leaf_mask]
            assert (non_leaf_split_dims >= 0).all()
            assert (non_leaf_split_dims < 5).all()

    def test_produces_balanced_tree(self):
        """L1 extent heuristic produces reasonably balanced trees."""
        # Create clustered data where L1 extent heuristic should help
        torch.manual_seed(42)
        cluster1 = torch.randn(50, 3) + torch.tensor([0.0, 0.0, 0.0])
        cluster2 = torch.randn(50, 3) + torch.tensor([10.0, 0.0, 0.0])
        points = torch.cat([cluster1, cluster2], dim=0)

        tree = kd_tree(points, leaf_size=10)

        # Check tree depth is reasonable (not degenerate)
        n_nodes = tree.split_dim.shape[0]
        # For 100 points with leaf_size=10, expect ~20 nodes max
        assert n_nodes < 50, f"Tree has {n_nodes} nodes, may be degenerate"


class TestKdTreeDtypes:
    """Tests for various input dtypes."""

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Supports float16, bfloat16, float32, float64."""
        points = torch.randn(50, 3, dtype=dtype)
        tree = kd_tree(points)
        assert tree.points.dtype == dtype
        # split_val matches input dtype for precision
        assert tree.split_val.dtype == dtype

    def test_float64_precision_preserved(self):
        """Float64 precision is preserved in split values near boundaries."""
        # Create points with values that require float64 precision
        # These values differ only in lower bits that float32 would lose
        torch.manual_seed(42)
        base = 1e10
        epsilon = 1e-6  # This difference is lost in float32 at this magnitude
        points = torch.tensor(
            [
                [base, 0.0, 0.0],
                [base + epsilon, 0.0, 0.0],
                [base + 2 * epsilon, 0.0, 0.0],
                [base + 3 * epsilon, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )

        tree = kd_tree(points, leaf_size=2)

        # Verify split_val can distinguish between points
        # In float32, base and base+epsilon would be identical
        assert tree.split_val.dtype == torch.float64
        # The split value should be between some of the point values
        non_leaf_mask = tree.split_dim >= 0
        if non_leaf_mask.any():
            split_vals = tree.split_val[non_leaf_mask]
            # At least one split should be in the range (base, base + 3*epsilon)
            in_range = (split_vals > base) & (split_vals < base + 3 * epsilon)
            assert in_range.any(), (
                "Float64 precision not preserved in split values"
            )


class TestKdTreeEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_nan_in_input(self):
        """NaN in input propagates through (PyTorch semantics)."""
        points = torch.randn(50, 3)
        points[10, 1] = float("nan")
        tree = kd_tree(points)
        # NaN should propagate through unchanged
        assert torch.isnan(tree.points).any()
        assert torch.isnan(tree.points[10, 1])

    def test_inf_in_input(self):
        """Inf in input propagates through (PyTorch semantics)."""
        points = torch.randn(50, 3)
        points[10, 1] = float("inf")
        tree = kd_tree(points)
        # Inf should propagate through unchanged
        assert torch.isinf(tree.points).any()
        assert torch.isinf(tree.points[10, 1])

    def test_high_dimensional(self):
        """Handles high-dimensional points (d=64)."""
        points = torch.randn(100, 64)
        tree = kd_tree(points)
        assert tree.points.shape == (100, 64)
        # Split dims should be in [0, 64)
        non_leaf_mask = tree.left != -1
        if non_leaf_mask.any():
            assert (tree.split_dim[non_leaf_mask] < 64).all()

    def test_duplicate_points(self):
        """Handles duplicate points correctly."""
        points = torch.zeros(50, 3)  # All same point
        tree = kd_tree(points)
        # Should still build a valid tree (all in one leaf)
        assert tree.indices.shape[0] == 50

    def test_collinear_points(self):
        """Handles collinear points (1D manifold in 3D)."""
        t = torch.linspace(0, 1, 100).unsqueeze(1)
        points = torch.cat([t, torch.zeros(100, 2)], dim=1)
        tree = kd_tree(points)
        # Split dimensions should be valid (in range [0, d))
        non_leaf_mask = tree.left != -1
        if non_leaf_mask.any():
            assert (tree.split_dim[non_leaf_mask] >= 0).all()
            assert (tree.split_dim[non_leaf_mask] < 3).all()


class TestKdTreeSerialization:
    """Tests for KdTree serialization (tensorclass provides this automatically)."""

    def test_save_load(self, tmp_path):
        """KdTree can be saved and loaded with torch.save/load."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)

        path = tmp_path / "tree.pt"
        torch.save(tree, path)
        tree2 = torch.load(path, weights_only=False)

        torch.testing.assert_close(tree.points, tree2.points)
        torch.testing.assert_close(tree.split_dim, tree2.split_dim)

    def test_save_load_batched(self, tmp_path):
        """Batched KdTree can be saved and loaded."""
        points = torch.randn(4, 100, 3)
        tree = kd_tree(points)

        path = tmp_path / "batched_tree.pt"
        torch.save(tree, path)
        tree2 = torch.load(path, weights_only=False)

        assert tree2.batch_size == torch.Size([4])
        torch.testing.assert_close(tree.points, tree2.points)

    def test_to_tensordict(self):
        """KdTree can be converted to TensorDict."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        td = tree.to_tensordict()
        assert "points" in td.keys()
        assert "split_dim" in td.keys()

    def test_from_tensordict(self):
        """KdTree can be reconstructed from TensorDict."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        td = tree.to_tensordict()
        tree2 = KdTree.from_tensordict(td)
        torch.testing.assert_close(tree.points, tree2.points)


class TestKdTreeCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.xfail(
        reason="test module 'statistics' shadows stdlib; run from project root"
    )
    def test_compile_builds_tree(self):
        """kd_tree works with torch.compile."""

        @torch.compile
        def build_tree(points):
            return kd_tree(points)

        points = torch.randn(100, 3)
        tree = build_tree(points)
        assert isinstance(tree, KdTree)
        assert tree.indices.shape[0] == 100

    @pytest.mark.xfail(
        reason="test module 'statistics' shadows stdlib; run from project root"
    )
    def test_compile_batched(self):
        """Batched kd_tree works with torch.compile."""

        @torch.compile
        def build_batched(points):
            return kd_tree(points)

        points = torch.randn(4, 100, 3)
        tree = build_batched(points)
        assert tree.batch_size == torch.Size([4])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_compile_cuda(self):
        """torch.compile works with CUDA backend."""

        @torch.compile
        def build_tree(points):
            return kd_tree(points)

        points = torch.randn(100, 3, device="cuda")
        tree = build_tree(points)
        assert tree.points.device.type == "cuda"

    def test_meta_tensor_shape_inference(self):
        """Meta tensors produce correct shapes for torch.compile tracing."""
        points = torch.randn(100, 3, device="meta")
        tree = kd_tree(points)

        # Meta tensors should have symbolic shapes
        assert tree.points.shape == (100, 3)
        assert tree.indices.shape[0] == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestKdTreeCUDA:
    """Tests for CUDA kd_tree implementation."""

    def test_cuda_basic(self):
        """kd_tree works on CUDA tensors."""
        points = torch.randn(100, 3, device="cuda")
        tree = kd_tree(points)
        assert isinstance(tree, KdTree)
        assert tree.points.device.type == "cuda"

    def test_cuda_indices_coverage(self):
        """CUDA tree contains all point indices."""
        torch.manual_seed(42)
        points_cuda = torch.randn(100, 3, device="cuda")
        tree_cuda = kd_tree(points_cuda)

        # All indices should be present
        sorted_indices = torch.sort(tree_cuda.indices)[0]
        expected = torch.arange(100, device="cuda")
        torch.testing.assert_close(sorted_indices, expected)

    # NOTE: CPU and CUDA use different algorithms (L1 extent heuristic vs Morton code radix tree)
    # so tree *structures* will differ. Query equivalence tests in Phase 1B/1C will verify
    # that both produce correct nearest neighbor and range search results.

    def test_cuda_split_val_dtype(self):
        """CUDA split_val matches input dtype."""
        points = torch.randn(100, 3, dtype=torch.float16, device="cuda")
        tree = kd_tree(points)
        assert tree.split_val.dtype == torch.float16

    def test_cuda_batched(self):
        """Batched CUDA construction works."""
        points = torch.randn(4, 100, 3, device="cuda")
        tree = kd_tree(points)
        assert tree.batch_size == torch.Size([4])
        assert tree.points.device.type == "cuda"

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    def test_cuda_dtypes(self, dtype):
        """CUDA supports float16, bfloat16, float32."""
        points = torch.randn(50, 3, dtype=dtype, device="cuda")
        tree = kd_tree(points)
        assert tree.points.dtype == dtype
        assert tree.split_val.dtype == dtype

    def test_cuda_large_pointcloud(self):
        """CUDA handles large point clouds efficiently."""
        points = torch.randn(100000, 3, device="cuda")
        tree = kd_tree(points, leaf_size=32)
        assert tree.indices.shape[0] == 100000
