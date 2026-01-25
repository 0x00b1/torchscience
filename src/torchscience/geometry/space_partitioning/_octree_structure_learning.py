"""Structure learning operations for octree adaptive refinement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._octree import Octree


def octree_subdivision_scores(
    tree: "Octree",
    score_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Compute refinement scores for leaf voxels.

    Parameters
    ----------
    tree : Octree
        Sparse voxel structure.
    score_fn : Callable[[Tensor], Tensor]
        Function mapping voxel data (n_leaves, *value_shape) -> scores (n_leaves,).
        Higher score = should subdivide.
        Typically a small MLP.

    Returns
    -------
    scores : Tensor, shape (count,)
        Refinement score per voxel. Non-leaf voxels have score 0.
        Supports autograd through score_fn parameters.

    Notes
    -----
    Only leaf voxels (children_mask == 0) receive scores from score_fn.
    Internal nodes always have score 0 since they already have children.

    Leaves at maximum_depth also receive score 0 since they cannot be
    subdivided further.

    Examples
    --------
    >>> score_mlp = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    >>> scores = octree_subdivision_scores(tree, lambda x: score_mlp(x).squeeze(-1))
    """
    count = tree.count.item()
    if count == 0:
        return torch.zeros(0, device=tree.data.device, dtype=tree.data.dtype)

    # Find leaf voxels that can be subdivided
    # Leaves are voxels with children_mask == 0
    # But leaves at maximum_depth cannot be subdivided
    depths = (tree.codes >> 60) & 0xF
    maximum_depth = tree.maximum_depth.item()

    # Subdivide candidates: leaves not at max depth
    can_subdivide = (tree.children_mask == 0) & (depths < maximum_depth)

    # Initialize scores to zero
    scores = torch.zeros(count, device=tree.data.device, dtype=tree.data.dtype)

    if can_subdivide.sum() == 0:
        return scores

    # Extract data for subdivide candidates
    candidate_data = tree.data[can_subdivide]

    # Apply score function
    candidate_scores = score_fn(candidate_data)

    # Validate output shape
    expected_count = can_subdivide.sum().item()
    if candidate_scores.shape[0] != expected_count:
        raise RuntimeError(
            f"score_fn returned {candidate_scores.shape[0]} scores but expected "
            f"{expected_count} (number of subdivide candidates)"
        )
    if candidate_scores.dim() != 1:
        raise RuntimeError(
            f"score_fn must return 1D tensor of scores, got {candidate_scores.dim()}D"
        )

    # Place scores in output tensor
    scores[can_subdivide] = candidate_scores

    return scores


class _StraightThroughSubdivide(torch.autograd.Function):
    """Straight-through estimator for discrete subdivision decisions."""

    @staticmethod
    def forward(
        ctx, scores: Tensor, threshold: float, temperature: float
    ) -> Tensor:
        """Forward: hard threshold decision."""
        # Hard decision: subdivide if score > threshold
        decisions = (scores > threshold).float()

        # Save for backward
        ctx.save_for_backward(scores)
        ctx.threshold = threshold
        ctx.temperature = temperature

        return decisions

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """Backward: soft sigmoid gradient."""
        (scores,) = ctx.saved_tensors
        temperature = ctx.temperature
        threshold = ctx.threshold

        # Soft gradient using sigmoid
        # sigmoid((score - threshold) / temperature)
        soft_decisions = torch.sigmoid((scores - threshold) / temperature)

        # Gradient of sigmoid: soft_decisions * (1 - soft_decisions) / temperature
        sigmoid_grad = soft_decisions * (1 - soft_decisions) / temperature

        # Chain rule
        grad_scores = grad_output * sigmoid_grad

        return grad_scores, None, None


def octree_adaptive_subdivide(
    tree: "Octree",
    scores: Tensor,
    *,
    threshold: float = 0.5,
    temperature: float = 1.0,
    straight_through: bool = True,
) -> "Octree":
    """Refine leaf voxels based on learned scores.

    Parameters
    ----------
    tree : Octree
        Sparse voxel structure.
    scores : Tensor, shape (count,)
        Refinement scores (e.g., from octree_subdivision_scores).
    threshold : float, default=0.5
        Leaf voxels with score > threshold are subdivided.
    temperature : float, default=1.0
        Softmax temperature for soft decisions during training.
        Lower = sharper decisions, higher = softer.
    straight_through : bool, default=True
        Use straight-through estimator: hard decisions forward,
        soft gradients backward.

    Returns
    -------
    Octree
        Refined octree with adaptive resolution.

    Notes
    -----
    **Leaf nodes only:** Only leaf voxels (children_mask == 0) are candidates
    for subdivision. Internal nodes are ignored regardless of their scores.

    **Max depth:** Leaves at maximum_depth cannot be subdivided and are
    ignored even if their scores exceed the threshold.

    **Training:** Soft scores flow gradients to score_fn parameters via the
    straight-through estimator.

    **Inference:** Hard threshold produces discrete structure.

    The straight-through estimator allows gradients to flow through
    the discrete subdivision decision, enabling end-to-end learning
    of adaptive resolution.

    Examples
    --------
    >>> score_mlp = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    >>> scores = octree_subdivision_scores(tree, lambda x: score_mlp(x).squeeze(-1))
    >>> tree = octree_adaptive_subdivide(tree, scores, threshold=0.5)
    """
    from ._octree_dynamic import octree_subdivide

    count = tree.count.item()
    if count == 0:
        return tree

    if scores.shape[0] != count:
        raise RuntimeError(
            f"scores must have shape ({count},), got {tuple(scores.shape)}"
        )

    # Find subdivide candidates (leaves not at max depth)
    depths = (tree.codes >> 60) & 0xF
    maximum_depth = tree.maximum_depth.item()
    can_subdivide = (tree.children_mask == 0) & (depths < maximum_depth)

    if can_subdivide.sum() == 0:
        return tree

    # Apply threshold with straight-through estimator
    if straight_through and scores.requires_grad:
        decisions = _StraightThroughSubdivide.apply(
            scores, threshold, temperature
        )
    else:
        decisions = (scores > threshold).float()

    # Get codes of voxels to subdivide
    # decisions[i] > 0 and can_subdivide[i] means subdivide
    should_subdivide = (decisions > 0) & can_subdivide

    if should_subdivide.sum() == 0:
        return tree

    subdivide_codes = tree.codes[should_subdivide]

    # Perform subdivision
    return octree_subdivide(tree, subdivide_codes)
