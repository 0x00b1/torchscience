"""Tests for maximum_bipartite_matching."""

import pytest
import torch

from torchscience.graph import maximum_bipartite_matching


class TestMaximumBipartiteMatchingBasic:
    """Basic functionality tests."""

    def test_perfect_matching_3x3(self):
        """3x3 with perfect matching possible."""
        biadj = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 3
        # Verify matching is valid
        for i in range(3):
            j = left[i].item()
            assert j >= 0  # All matched
            assert right[j].item() == i  # Consistent

    def test_perfect_matching_2x2(self):
        """2x2 complete bipartite graph."""
        biadj = torch.ones(2, 2)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 2
        # Verify matching
        for i in range(2):
            j = left[i].item()
            assert 0 <= j < 2
            assert right[j].item() == i

    def test_no_matching_possible(self):
        """Empty graph - no edges."""
        biadj = torch.zeros(3, 3)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 0
        assert (left == -1).all()
        assert (right == -1).all()

    def test_partial_matching(self):
        """Not all nodes can be matched."""
        biadj = torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],  # Both left 0 and 1 can only match right 0
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Maximum matching is 2: one of (left 0 or 1) -> right 0, left 2 -> right 2
        assert size.item() == 2

    def test_single_edge(self):
        """Single edge graph."""
        biadj = torch.tensor([[1.0]])
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 1
        assert left[0].item() == 0
        assert right[0].item() == 0

    def test_rectangular_more_left(self):
        """More left nodes than right."""
        biadj = torch.tensor(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Maximum matching is 2 (can only match 2 right nodes)
        assert size.item() == 2

    def test_rectangular_more_right(self):
        """More right nodes than left."""
        biadj = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Maximum matching is 2 (can only match 2 left nodes)
        assert size.item() == 2


class TestMaximumBipartiteMatchingEmpty:
    """Tests for empty and edge cases."""

    def test_empty_left(self):
        """Empty left partition."""
        biadj = torch.zeros(0, 3)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 0
        assert left.shape == (0,)
        assert right.shape == (3,)
        assert (right == -1).all()

    def test_empty_right(self):
        """Empty right partition."""
        biadj = torch.zeros(3, 0)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 0
        assert left.shape == (3,)
        assert right.shape == (0,)
        assert (left == -1).all()

    def test_empty_both(self):
        """Empty graph (0x0)."""
        biadj = torch.zeros(0, 0)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.item() == 0
        assert left.shape == (0,)
        assert right.shape == (0,)


class TestMaximumBipartiteMatchingDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
    )
    def test_dtype(self, dtype):
        """Test different data types."""
        biadj = torch.tensor(
            [
                [1, 1],
                [1, 1],
            ],
            dtype=dtype,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.dtype == torch.int64
        assert left.dtype == torch.int64
        assert right.dtype == torch.int64
        assert size.item() == 2


class TestMaximumBipartiteMatchingValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            maximum_bipartite_matching(torch.tensor([1.0, 2.0, 3.0]))


class TestMaximumBipartiteMatchingConsistency:
    """Tests for matching consistency."""

    def test_matching_is_consistent(self):
        """Left and right matches should be consistent."""
        biadj = torch.tensor(
            [
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Check consistency: if left[i] = j, then right[j] = i
        for i in range(3):
            j = left[i].item()
            if j >= 0:
                assert right[j].item() == i, (
                    f"Inconsistent: left[{i}]={j}, right[{j}]={right[j].item()}"
                )

        # Check reverse consistency
        for j in range(4):
            i = right[j].item()
            if i >= 0:
                assert left[i].item() == j, (
                    f"Inconsistent: right[{j}]={i}, left[{i}]={left[i].item()}"
                )

    def test_matching_uses_valid_edges(self):
        """Matching should only use edges that exist."""
        biadj = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Check that matched edges exist
        for i in range(3):
            j = left[i].item()
            if j >= 0:
                assert biadj[i, j] > 0, f"Edge ({i}, {j}) doesn't exist"


class TestMaximumBipartiteMatchingReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("M,N", [(5, 5), (10, 8), (8, 10)])
    def test_matches_scipy(self, M, N):
        """Compare to scipy.sparse.csgraph.maximum_bipartite_matching."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random bipartite graph with ~50% edge density
        biadj_np = (numpy.random.rand(M, N) > 0.5).astype(numpy.float32)

        biadj = torch.from_numpy(biadj_np)

        # Our implementation
        size_ours, left_ours, right_ours = maximum_bipartite_matching(biadj)

        # Scipy implementation
        # scipy expects sparse matrix
        biadj_sparse = scipy_sparse.csr_matrix(biadj_np)
        matching_scipy = scipy_sparse.csgraph.maximum_bipartite_matching(
            biadj_sparse, perm_type="column"
        )
        size_scipy = (matching_scipy >= 0).sum()

        # Compare matching sizes (the matching itself may differ)
        assert size_ours.item() == size_scipy, (
            f"Size mismatch: ours={size_ours.item()}, scipy={size_scipy}"
        )


class TestMaximumBipartiteMatchingMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        biadj = torch.rand(4, 5, device="meta")
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.shape == ()
        assert left.shape == (4,)
        assert right.shape == (5,)
        assert size.device.type == "meta"
        assert left.device.type == "meta"
        assert right.device.type == "meta"

    def test_meta_rectangular(self):
        """Meta tensor with rectangular input."""
        biadj = torch.rand(3, 7, device="meta")
        size, left, right = maximum_bipartite_matching(biadj)

        assert left.shape == (3,)
        assert right.shape == (7,)


class TestMaximumBipartiteMatchingClassicExamples:
    """Classic bipartite matching examples."""

    def test_hall_condition_satisfied(self):
        """Hall's marriage theorem condition satisfied."""
        # Each left node has distinct neighbors
        biadj = torch.tensor(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Perfect matching exists
        assert size.item() == 4

    def test_hall_condition_not_satisfied(self):
        """Hall's condition violated - no perfect matching."""
        # First 3 left nodes only connect to first 2 right nodes
        biadj = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        size, left, right = maximum_bipartite_matching(biadj)

        # Can match at most 3: 2 from first 3 + 1 from last
        assert size.item() == 3

    def test_job_assignment(self):
        """Classic job assignment problem."""
        # 5 workers, 5 jobs
        # Worker i can do job j if capable[i,j] = 1
        capable = torch.tensor(
            [
                [1, 1, 0, 0, 0],  # Worker 0
                [0, 1, 1, 0, 0],  # Worker 1
                [0, 0, 1, 1, 0],  # Worker 2
                [0, 0, 0, 1, 1],  # Worker 3
                [1, 0, 0, 0, 1],  # Worker 4
            ],
            dtype=torch.float32,
        )
        size, assignment, _ = maximum_bipartite_matching(capable)

        # All 5 workers can be assigned unique jobs
        assert size.item() == 5

        # Verify valid assignment
        for worker in range(5):
            job = assignment[worker].item()
            assert capable[worker, job] == 1


class TestMaximumBipartiteMatchingBatched:
    """Tests for batched input support."""

    def test_batch_2d(self):
        """Batched with single batch dimension."""
        biadj = torch.rand(3, 4, 5)
        biadj = (biadj > 0.5).float()

        size, left, right = maximum_bipartite_matching(biadj)
        assert size.shape == (3,)
        assert left.shape == (3, 4)
        assert right.shape == (3, 5)

    def test_batch_consistency(self):
        """Batched result matches individual computations."""
        biadj1 = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        biadj2 = torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        )

        s1, l1, r1 = maximum_bipartite_matching(biadj1)
        s2, l2, r2 = maximum_bipartite_matching(biadj2)

        batch_biadj = torch.stack([biadj1, biadj2])
        s_batch, l_batch, r_batch = maximum_bipartite_matching(batch_biadj)

        assert s_batch[0].item() == s1.item()
        assert s_batch[1].item() == s2.item()
        assert torch.equal(l_batch[0], l1)
        assert torch.equal(l_batch[1], l2)
        assert torch.equal(r_batch[0], r1)
        assert torch.equal(r_batch[1], r2)

    def test_batch_multi_dimensional(self):
        """Multiple batch dimensions."""
        biadj = torch.rand(2, 3, 4, 5)
        biadj = (biadj > 0.5).float()

        size, left, right = maximum_bipartite_matching(biadj)
        assert size.shape == (2, 3)
        assert left.shape == (2, 3, 4)
        assert right.shape == (2, 3, 5)

    def test_batch_empty_partition(self):
        """Batched with empty partitions."""
        biadj = torch.zeros(2, 0, 3)
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.shape == (2,)
        assert left.shape == (2, 0)
        assert right.shape == (2, 3)
        assert (size == 0).all()
        assert (right == -1).all()

    def test_batch_meta_shape(self):
        """Meta tensor with batched input."""
        biadj = torch.rand(3, 4, 5, device="meta")
        size, left, right = maximum_bipartite_matching(biadj)

        assert size.shape == (3,)
        assert left.shape == (3, 4)
        assert right.shape == (3, 5)
        assert size.device.type == "meta"
        assert left.device.type == "meta"
        assert right.device.type == "meta"

    def test_batch_consistency_matching_valid(self):
        """Verify batched matching consistency."""
        biadj = torch.tensor(
            [
                [
                    [1, 1, 0],
                    [0, 1, 1],
                    [0, 0, 1],
                ],
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
            ],
            dtype=torch.float32,
        )

        size, left, right = maximum_bipartite_matching(biadj)

        # Both should have perfect matching
        assert size[0].item() == 3
        assert size[1].item() == 3

        # Verify consistency for each batch
        for b in range(2):
            for i in range(3):
                j = left[b, i].item()
                if j >= 0:
                    assert right[b, j].item() == i
                    assert biadj[b, i, j] > 0
