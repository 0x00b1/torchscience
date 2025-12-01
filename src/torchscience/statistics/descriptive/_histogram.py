"""Histogram implementation."""

from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Valid values for out_of_bounds parameter
_VALID_OUT_OF_BOUNDS = {"ignore", "clamp", "error"}


def histogram(
    input: Tensor,
    bins: Union[int, Tensor] = 10,
    *,
    range: Optional[Tuple[float, float]] = None,
    weight: Optional[Tensor] = None,
    density: bool = False,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    closed: Literal["left", "right"] = "right",
    out_of_bounds: str = "ignore",
) -> Tuple[Tensor, Tensor]:
    r"""Compute a histogram of the values in a tensor.

    Computes a histogram by binning values into equal-width bins. By default,
    the bin range is determined by the minimum and maximum values of the input.

    Mathematical Definition
    -----------------------
    For an input tensor :math:`x` with values :math:`x_1, x_2, \ldots, x_n` and
    :math:`B` bins with edges :math:`e_0, e_1, \ldots, e_B`:

    **Count histogram** (``density=False``):

    .. math::
        h_i = \sum_{j=1}^{n} \mathbf{1}_{[e_{i-1}, e_i)}(x_j)

    where :math:`\mathbf{1}_{[a,b)}(x)` is the indicator function for the
    half-open interval :math:`[a, b)`.

    **Weighted histogram**:

    .. math::
        h_i = \sum_{j=1}^{n} w_j \cdot \mathbf{1}_{[e_{i-1}, e_i)}(x_j)

    where :math:`w_j` is the weight associated with :math:`x_j`.

    **Density histogram** (``density=True``):

    .. math::
        h_i = \frac{h_i^{\text{count}}}{\sum_k h_k^{\text{count}} \cdot (e_k - e_{k-1})}

    so that the integral over the bins equals 1.

    Parameters
    ----------
    input : Tensor
        Input tensor. For the current implementation, must be 1D.
    bins : int or Tensor, optional
        If ``int``, defines the number of equal-width bins. Default: 10.
        If ``Tensor``, defines the sequence of bin edges including the
        rightmost edge. Must contain at least 2 elements and be increasing.
    range : tuple of float, optional
        The lower and upper range of the bins. If not provided, the range
        is ``(input.min(), input.max())``. Values outside the range are
        ignored. Default: ``None``.
    weight : Tensor, optional
        A tensor of weights with the same shape as ``input``. Each value in
        ``input`` contributes its associated weight towards its bin's result.
        Default: ``None``.
    density : bool, optional
        If ``False`` (default), the result contains the count (or total weight)
        in each bin. If ``True``, the result is the value of the probability
        density function over the bins, normalized such that the integral
        over the range of the bins equals 1.
    dim : int or tuple of ints, optional
        The dimension(s) along which to compute the histogram. If ``None``
        (default), computes over all elements (flattened). Currently only
        ``None`` is supported.
    closed : {"left", "right"}, optional
        Which side of the bin intervals is closed. Must be one of:
        - ``"left"``: Bins are half-open ``[a, b)``
        - ``"right"`` (default): Bins are half-open ``(a, b]``
        Currently only ``"right"`` is supported.
    out_of_bounds : str, optional
        How to handle values outside the bin range. Must be one of:
        - ``"ignore"`` (default): Values outside the range are not counted
        - ``"clip"``: Values outside the range are clipped to the nearest edge
        - ``"error"``: Raise an error if any values are outside the range
        Currently only ``"ignore"`` is supported.

    Returns
    -------
    counts : Tensor
        The histogram counts (or densities if ``density=True``).
        Shape: ``(bins,)`` for 1D input.
    edges : Tensor
        The bin edges. Shape: ``(bins + 1,)`` for 1D input.

    Examples
    --------
    Compute a basic histogram:

    >>> x = torch.randn(1000)
    >>> counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
    >>> counts.shape
    torch.Size([10])
    >>> edges.shape
    torch.Size([11])

    Histogram with a specified range:

    >>> x = torch.randn(1000)
    >>> counts, edges = torchscience.statistics.descriptive.histogram(
    ...     x, bins=5, range=(-2.0, 2.0)
    ... )
    >>> edges
    tensor([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0])

    Weighted histogram:

    >>> x = torch.tensor([1.0, 2.0, 1.0])
    >>> w = torch.tensor([1.0, 2.0, 4.0])
    >>> counts, edges = torchscience.statistics.descriptive.histogram(
    ...     x, bins=4, range=(0.0, 3.0), weight=w
    ... )
    >>> counts
    tensor([0., 5., 2., 0.])

    Density histogram (integrates to 1):

    >>> x = torch.randn(10000)
    >>> counts, edges = torchscience.statistics.descriptive.histogram(
    ...     x, bins=50, density=True
    ... )
    >>> # Approximate integral using trapezoidal rule
    >>> bin_widths = edges[1:] - edges[:-1]
    >>> (counts * bin_widths).sum()  # Close to 1.0
    tensor(1.0000)

    Notes
    -----
    - This function currently delegates to ``torch.histogram`` for 1D cases.

    - For **batched** or **N-D** histograms (using the ``dim`` parameter),
      a full C++ backend will be implemented in a future version.

    - The ``closed`` and ``out_of_bounds`` parameters are reserved for
      future functionality and currently only support their default values.

    - Unlike NumPy's ``histogram``, this function returns both counts and
      edges as a tuple, similar to ``torch.histogram``.

    - **Gradients** are not currently supported for histogram computation.

    Warnings
    --------
    - Histogram computation does not support gradients. Attempting to
      compute gradients through this function will raise an error.

    - For very large tensors, consider using a smaller number of bins
      to reduce memory usage.

    References
    ----------
    .. [1] NumPy documentation for numpy.histogram:
           https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

    .. [2] PyTorch documentation for torch.histogram:
           https://pytorch.org/docs/stable/generated/torch.histogram.html

    See Also
    --------
    torch.histogram : PyTorch's histogram function.
    numpy.histogram : NumPy's histogram function.
    """
    # Validate empty input
    if input.numel() == 0:
        raise ValueError("histogram: input tensor must be non-empty")

    # Validate dim parameter - only None is currently supported
    if dim is not None:
        raise NotImplementedError(
            "Batched and N-D histograms (dim parameter) are not yet implemented. "
            "Only dim=None (histogram over all elements) is currently supported."
        )

    # Validate closed parameter
    if closed not in ("left", "right"):
        raise ValueError(
            f"histogram: closed must be 'left' or 'right', got '{closed}'"
        )

    # Validate out_of_bounds parameter
    if out_of_bounds not in _VALID_OUT_OF_BOUNDS:
        raise ValueError(
            f"histogram: out_of_bounds must be one of {_VALID_OUT_OF_BOUNDS}, "
            f"got '{out_of_bounds}'"
        )

    # Flatten input for 1D histogram computation
    flat_input = input.flatten()
    flat_weight = weight.flatten() if weight is not None else None

    # Dispatch to appropriate C++ operator based on bins type
    if isinstance(bins, Tensor):
        # Tensor bins: use histogram_edges operator
        return torch.ops.torchscience.histogram_edges(
            flat_input,
            bins,
            flat_weight,
            density,
            closed,
            out_of_bounds,
        )
    else:
        # Integer bins: use histogram operator
        return torch.ops.torchscience.histogram(
            flat_input,
            bins,
            list(range) if range is not None else None,
            flat_weight,
            density,
            closed,
            out_of_bounds,
        )
