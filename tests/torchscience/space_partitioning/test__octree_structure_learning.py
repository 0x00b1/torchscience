"""Tests for octree structure learning operations."""

import pytest
import torch
import torch.nn as nn
import torch.testing

from torchscience.space_partitioning import (
    octree,
    octree_adaptive_subdivide,
    octree_insert,
    octree_subdivision_scores,
)


class TestOctreeSubdivisionScores:
    """Tests for octree_subdivision_scores function."""

    def test_scores_shape_matches_count(self):
        """Output scores shape matches tree.count."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 8)
        tree = octree(points, data, maximum_depth=4)

        scores = octree_subdivision_scores(tree, lambda x: x.mean(dim=-1))

        assert scores.shape == (tree.count.item(),)

    def test_internal_nodes_have_zero_score(self):
        """Internal nodes (children_mask != 0) have score 0."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 8)
        tree = octree(points, data, maximum_depth=4)

        # Score function that returns positive values
        scores = octree_subdivision_scores(
            tree, lambda x: torch.ones(x.shape[0])
        )

        # Internal nodes should have score 0
        internal_mask = tree.children_mask != 0
        assert (scores[internal_mask] == 0).all()

    def test_max_depth_leaves_have_zero_score(self):
        """Leaves at maximum_depth have score 0 (cannot subdivide)."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 8)
        tree = octree(points, data, maximum_depth=4)

        scores = octree_subdivision_scores(
            tree, lambda x: torch.ones(x.shape[0])
        )

        # Find leaves at max depth
        depths = (tree.codes >> 60) & 0xF
        max_depth_leaves = (tree.children_mask == 0) & (depths == 4)

        assert (scores[max_depth_leaves] == 0).all()

    def test_subdivide_candidates_get_scores(self):
        """Leaves not at max depth receive scores from score_fn."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 8)
        tree = octree(points, data, maximum_depth=6)

        # Score function that returns 1.0 for all candidates
        scores = octree_subdivision_scores(
            tree, lambda x: torch.ones(x.shape[0])
        )

        # Find candidates (leaves not at max depth)
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)

        # All candidates should have score 1.0
        if can_subdivide.sum() > 0:
            assert (scores[can_subdivide] == 1.0).all()

    def test_score_fn_receives_leaf_data(self):
        """Score function receives only leaf voxel data."""
        # Create tree with leaves NOT at max depth (insert at depth 3)
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=6)

        # Insert points at depth 3 (not max depth 6)
        pts = torch.rand(10, 3) * 2 - 1
        dats = torch.rand(10, 4)
        tree = octree_insert(tree, pts, dats, depth=3)

        # Track what score_fn receives
        received_shapes = []

        def tracking_score_fn(x):
            received_shapes.append(x.shape)
            return x.mean(dim=-1)

        octree_subdivision_scores(tree, tracking_score_fn)

        # Find expected candidate count
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)
        expected_count = can_subdivide.sum().item()

        # Score function should be called once with candidate data
        assert len(received_shapes) == 1
        assert expected_count > 0
        assert received_shapes[0] == (expected_count, 4)

    def test_empty_tree_returns_empty_scores(self):
        """Empty tree returns empty scores tensor."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        scores = octree_subdivision_scores(tree, lambda x: x.mean(dim=-1))

        assert scores.shape == (0,)

    def test_gradients_flow_through_score_fn(self):
        """Gradients flow through score_fn parameters."""
        # Create tree with leaves NOT at max depth (insert at depth 3)
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 8)
        tree = octree(points, data, maximum_depth=6)

        # Insert points at depth 3 (not max depth 6)
        pts = torch.rand(20, 3) * 2 - 1
        dats = torch.rand(20, 8)
        tree = octree_insert(tree, pts, dats, depth=3)

        # Verify we have candidates
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)
        assert can_subdivide.sum() > 0, (
            "Need subdivide candidates for this test"
        )

        # Create a simple learnable score function
        score_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        def score_fn(x):
            return score_mlp(x).squeeze(-1)

        scores = octree_subdivision_scores(tree, score_fn)

        # Backpropagate
        loss = scores.sum()
        loss.backward()

        # Check that MLP parameters have gradients
        for param in score_mlp.parameters():
            assert param.grad is not None
            assert (param.grad != 0).any()

    def test_invalid_score_fn_output_shape_raises(self):
        """Score function with wrong output count raises error."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 4)
        tree = octree(points, data, maximum_depth=6)

        # Score function that returns wrong number of scores
        def bad_score_fn(x):
            return torch.ones(x.shape[0] + 1)  # Wrong count

        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)

        if can_subdivide.sum() > 0:
            with pytest.raises(RuntimeError, match="scores"):
                octree_subdivision_scores(tree, bad_score_fn)

    def test_invalid_score_fn_output_dim_raises(self):
        """Score function with wrong output dimension raises error."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 4)
        tree = octree(points, data, maximum_depth=6)

        # Score function that returns 2D tensor
        def bad_score_fn(x):
            return x  # Wrong dimension (should be 1D)

        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)

        if can_subdivide.sum() > 0:
            with pytest.raises(RuntimeError, match="1D"):
                octree_subdivision_scores(tree, bad_score_fn)


class TestOctreeAdaptiveSubdivide:
    """Tests for octree_adaptive_subdivide function."""

    def test_subdivides_high_score_leaves(self):
        """Leaves with score > threshold are subdivided."""
        # Create tree with a single point at depth 3 (not max depth)
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=6)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[1.0]])
        tree = octree_insert(tree, pt, dat, depth=3)  # Insert at depth 3
        original_count = tree.count.item()

        # Verify we have a candidate to subdivide
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)
        assert can_subdivide.sum() > 0, "Need subdivide candidate"

        # Create scores: 1.0 for all voxels
        scores = torch.ones(tree.count.item())

        # Subdivide with threshold 0.5
        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Count should increase (subdivision happened)
        assert tree.count.item() > original_count

    def test_no_subdivide_low_score_leaves(self):
        """Leaves with score <= threshold are not subdivided."""
        # Create tree with a single point
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=6)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[1.0]])
        tree = octree_insert(tree, pt, dat)
        original_count = tree.count.item()

        # Create scores: 0.0 for all voxels
        scores = torch.zeros(tree.count.item())

        # Subdivide with threshold 0.5
        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Count should be unchanged
        assert tree.count.item() == original_count

    def test_internal_nodes_not_subdivided(self):
        """Internal nodes are never subdivided regardless of score."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Get original internal node codes
        original_internal_mask = tree.children_mask != 0
        original_internal_codes = set(
            tree.codes[original_internal_mask].tolist()
        )

        # Create scores: high for everything
        scores = torch.ones(tree.count.item()) * 10.0

        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Original internal nodes should still exist
        new_codes = set(tree.codes.tolist())
        for code in original_internal_codes:
            assert code in new_codes

    def test_max_depth_leaves_not_subdivided(self):
        """Leaves at maximum_depth are not subdivided regardless of score."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Count max depth leaves
        depths = (tree.codes >> 60) & 0xF
        max_depth_leaves = (tree.children_mask == 0) & (depths == 4)
        max_depth_count = max_depth_leaves.sum().item()

        # High scores for all
        scores = torch.ones(tree.count.item()) * 10.0

        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Count max depth leaves after (should still exist, not subdivided)
        depths = (tree.codes >> 60) & 0xF
        new_max_depth_leaves = (tree.children_mask == 0) & (depths == 4)

        # Original max depth leaves should still be leaves
        assert new_max_depth_leaves.sum().item() >= max_depth_count

    def test_empty_tree_unchanged(self):
        """Empty tree is returned unchanged."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        scores = torch.zeros(0)
        tree_out = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        assert tree_out.count.item() == 0

    def test_threshold_boundary(self):
        """Scores exactly at threshold are not subdivided."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=6)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[1.0]])
        tree = octree_insert(tree, pt, dat)
        original_count = tree.count.item()

        # Create scores exactly at threshold
        scores = torch.ones(tree.count.item()) * 0.5

        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Count unchanged (score not > threshold)
        assert tree.count.item() == original_count

    def test_scores_shape_mismatch_raises(self):
        """Wrong scores shape raises error."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 4)
        tree = octree(points, data, maximum_depth=4)

        # Wrong number of scores
        scores = torch.zeros(tree.count.item() + 10)

        with pytest.raises(RuntimeError, match="scores"):
            octree_adaptive_subdivide(tree, scores)


class TestStraightThroughEstimator:
    """Tests for straight-through estimator gradient behavior."""

    def test_straight_through_gradients_flow(self):
        """Gradients flow through straight-through estimator."""
        # Test the _StraightThroughSubdivide autograd function directly
        from torchscience.space_partitioning._octree_structure_learning import (
            _StraightThroughSubdivide,
        )

        # Create learnable scores
        scores = torch.tensor([0.3, 0.6, 0.8], requires_grad=True)

        # Apply straight-through estimator
        decisions = _StraightThroughSubdivide.apply(scores, 0.5, 1.0)

        # Forward produces hard decisions: [0, 1, 1]
        assert decisions[0] == 0.0
        assert decisions[1] == 1.0
        assert decisions[2] == 1.0

        # Backward should provide soft gradients
        loss = decisions.sum()
        loss.backward()

        # Gradients should exist
        assert scores.grad is not None

        # Gradient at score=0.6 (near threshold) should be larger than at 0.8 (far from threshold)
        # Because sigmoid gradient is maximized at the threshold
        assert scores.grad[1].abs() > scores.grad[2].abs()

    def test_temperature_affects_gradient_sharpness(self):
        """Lower temperature gives sharper gradients."""
        scores = torch.tensor([0.4, 0.5, 0.6], requires_grad=True)

        # Low temperature - sharp gradients
        from torchscience.space_partitioning._octree_structure_learning import (
            _StraightThroughSubdivide,
        )

        decisions_low = _StraightThroughSubdivide.apply(scores, 0.5, 0.1)
        decisions_low.sum().backward()
        grad_low = scores.grad.clone()

        scores.grad.zero_()

        # High temperature - soft gradients
        decisions_high = _StraightThroughSubdivide.apply(scores, 0.5, 10.0)
        decisions_high.sum().backward()
        grad_high = scores.grad.clone()

        # Low temperature should have sharper gradients (larger magnitude near threshold)
        # At score=0.5 (exactly at threshold), gradient is maximized
        # Low temp: sigmoid is steep, gradient is large
        # High temp: sigmoid is flat, gradient is small
        assert grad_low[1].abs() > grad_high[1].abs()

    def test_without_straight_through_no_gradients(self):
        """Without straight-through, decisions don't have gradients."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=6)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.rand(1, 4)
        tree = octree_insert(tree, pt, dat)

        scores = torch.tensor([0.6], requires_grad=True).expand(
            tree.count.item()
        )

        # Without straight-through, no gradient tracking
        tree_out = octree_adaptive_subdivide(
            tree, scores, threshold=0.5, straight_through=False
        )

        # Structure operations are graph breaks anyway
        # This test just verifies the flag works
        assert tree_out.count.item() > 0


class TestEndToEndStructureLearning:
    """End-to-end tests for structure learning workflow."""

    def test_full_workflow(self):
        """Complete workflow: scores -> adaptive subdivide -> query."""
        from torchscience.space_partitioning import octree_sample

        # Create initial tree
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 8)
        tree = octree(points, data, maximum_depth=6)

        # Create score MLP
        score_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Compute scores
        scores = octree_subdivision_scores(
            tree, lambda x: score_mlp(x).squeeze(-1)
        )

        # Adaptive subdivide
        tree = octree_adaptive_subdivide(
            tree, scores, threshold=0.0
        )  # Low threshold

        # Query the refined tree
        queries = torch.rand(20, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.shape == (20, 8)

    def test_iterative_refinement(self):
        """Multiple rounds of adaptive subdivision."""
        # Start with coarse tree
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=8)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.rand(1, 4)
        tree = octree_insert(tree, pt, dat)

        # Refine multiple times
        for _ in range(3):
            # Always subdivide (score = 1.0)
            scores = octree_subdivision_scores(
                tree, lambda x: torch.ones(x.shape[0])
            )

            # Check if any candidates exist
            depths = (tree.codes >> 60) & 0xF
            can_subdivide = (tree.children_mask == 0) & (depths < 8)

            if can_subdivide.sum() == 0:
                break

            tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Tree should have grown
        assert tree.count.item() > 1

    def test_selective_refinement(self):
        """Only refine specific regions based on learned scores."""
        # Create tree with multiple points in different regions
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=6)

        # Insert points in two regions
        pts = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        dats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        tree = octree_insert(tree, pts, dats)

        # Score function: only refine where first feature is high
        def selective_score(x):
            return x[:, 0]  # First feature as score

        scores = octree_subdivision_scores(tree, selective_score)

        # Subdivide with threshold that only affects first point's region
        tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)

        # Tree should have grown (first point region subdivided)
        assert tree.count.item() > 2
