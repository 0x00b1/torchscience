"""Minkowski distance implementation."""

from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def minkowski_distance(
    x: Tensor,
    y: Optional[Tensor] = None,
    *,
    p: float = 2.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute pairwise Minkowski distances.

    Computes the weighted Minkowski distance between each pair of row vectors
    from two sets of points, or self-pairwise distances if only one set is given.

    Mathematical Definition
    -----------------------
    For points :math:`x \in \mathbb{R}^d` and :math:`y \in \mathbb{R}^d` with
    optional weights :math:`w \in \mathbb{R}^d`:

    .. math::
        d_p(x, y; w) = \left( \sum_{i=1}^{d} w_i |x_i - y_i|^p \right)^{1/p}

    Special cases:

    - :math:`p = 1`: Manhattan (taxicab) distance
    - :math:`p = 2`: Euclidean distance
    - :math:`p \to \infty`: Chebyshev distance (max absolute difference)

    Parameters
    ----------
    x : Tensor, shape (m, d)
        First set of m points in d-dimensional space.
    y : Tensor, shape (n, d), optional
        Second set of n points in d-dimensional space.
        If ``None``, computes self-pairwise distances from x to x.
    p : float, default=2.0
        Order of the Minkowski norm. Must be > 0.
        For 0 < p < 1, this is a quasi-metric (triangle inequality doesn't hold).
    weight : Tensor, shape (d,), optional
        Non-negative weights for each dimension.
        If ``None``, all weights are 1 (unweighted distance).

    Returns
    -------
    Tensor
        Pairwise distance matrix.

        - If ``y`` is provided: shape ``(m, n)`` where ``[i, j]`` is the
          distance from ``x[i]`` to ``y[j]``.
        - If ``y`` is ``None``: shape ``(m, m)`` where ``[i, j]`` is the
          distance from ``x[i]`` to ``x[j]``.

    Examples
    --------
    Compute Euclidean distances between two sets of points:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> torchscience.similarity_measure.minkowski_distance(x, y, p=2.0)
    tensor([[1.0000, 1.0000],
            [1.0000, 1.0000]])

    Compute Manhattan distances (p=1):

    >>> torchscience.similarity_measure.minkowski_distance(x, y, p=1.0)
    tensor([[1., 1.],
            [1., 1.]])

    Self-pairwise distances:

    >>> x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    >>> torchscience.similarity_measure.minkowski_distance(x, p=2.0)
    tensor([[0., 5.],
            [5., 0.]])

    Weighted distance:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = torch.tensor([[1.0, 1.0]])
    >>> w = torch.tensor([1.0, 4.0])  # Weight second dimension more
    >>> torchscience.similarity_measure.minkowski_distance(x, y, p=2.0, weight=w)
    tensor([[2.2361],
            [0.0000]])

    Notes
    -----
    - For ``p >= 1``, the Minkowski distance is a true metric satisfying the
      triangle inequality.
    - For ``0 < p < 1``, the result is a quasi-metric.
    - Gradients are computed analytically and support higher-order derivatives.
    - The diagonal of self-pairwise distances is always 0.

    See Also
    --------
    torch.cdist : PyTorch's built-in pairwise distance function.
    scipy.spatial.distance.cdist : SciPy's pairwise distance function.

    References
    ----------
    .. [1] Wikipedia, "Minkowski distance",
           https://en.wikipedia.org/wiki/Minkowski_distance
    """
    # Input validation
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (m, d), got {x.dim()}D")

    if y is None:
        y = x

    if y.dim() != 2:
        raise ValueError(f"y must be 2D (n, d), got {y.dim()}D")

    if x.size(1) != y.size(1):
        raise ValueError(
            f"Feature dimensions must match: x has {x.size(1)}, y has {y.size(1)}"
        )

    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")

    if weight is not None:
        if weight.dim() != 1:
            raise ValueError(f"weight must be 1D (d,), got {weight.dim()}D")
        if weight.size(0) != x.size(1):
            raise ValueError(
                f"weight size {weight.size(0)} must match feature dim {x.size(1)}"
            )
        if (weight < 0).any():
            raise ValueError("weight must be non-negative")

    # Dtype promotion: promote x, y, and weight to common dtype
    target_dtype = torch.promote_types(x.dtype, y.dtype)
    if x.dtype != target_dtype:
        x = x.to(target_dtype)
    if y.dtype != target_dtype:
        y = y.to(target_dtype)
    if weight is not None and weight.dtype != target_dtype:
        weight = weight.to(target_dtype)

    return torch.ops.torchscience.minkowski_distance(x, y, p, weight)
