from typing import Literal

import torch
from torch import Tensor

PaddingMode = Literal[
    "constant",
    "replicate",
    "reflect",
    "reflect_odd",
    "circular",
    "linear",
    "polynomial",
    "spline",
    "smooth",
]


def pad(
    input: Tensor,
    padding: int
    | tuple[int, int]
    | tuple[int, ...]
    | tuple[tuple[int, int], ...],
    mode: PaddingMode = "constant",
    value: float = 0.0,
    dim: int | tuple[int, ...] | None = None,
    order: int = 1,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Pad a tensor along specified dimensions.

    Provides N-dimensional padding with multiple modes including extrapolation,
    supporting full autograd (first and second order derivatives).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    padding : int, tuple of int, or tuple of tuple of int
        Padding amounts. Accepts multiple formats:

        - ``int``: Same padding on all sides of all dimensions
        - ``(before, after)``: Same padding for all dimensions
        - PyTorch-style ``(left_n, right_n, ..., left_0, right_0)``: Pairs for
          trailing dimensions in reverse order
        - NumPy-style ``((before_0, after_0), (before_1, after_1), ...)``:
          Per-dimension pairs

    mode : str, default "constant"
        Padding mode. One of:

        - ``"constant"``: Fill with ``value``
        - ``"replicate"``: Repeat edge values
        - ``"reflect"``: Symmetric reflection (edge-inclusive)
        - ``"reflect_odd"``: Antisymmetric reflection
        - ``"circular"``: Wrap around (periodic boundary)
        - ``"linear"``: Linear extrapolation from edge
        - ``"polynomial"``: Polynomial extrapolation of degree ``order``
        - ``"spline"``: Cubic spline extrapolation
        - ``"smooth"``: C1-continuous extension (matches value and derivative)

    value : float, default 0.0
        Fill value for ``mode="constant"``.
    dim : int, tuple of int, or None, default None
        Dimensions to pad. If None, pads trailing dimensions based on padding
        length (PyTorch convention). Negative dimensions are supported.
    order : int, default 1
        Polynomial order for extrapolation modes (1=linear, 2=quadratic, etc.).
        Only used when mode is "polynomial".
    out : Tensor, optional
        Output tensor. Must have the correct shape. If provided, the result
        is written to this tensor.

    Returns
    -------
    Tensor
        Padded tensor with shape adjusted according to padding amounts.

    Examples
    --------
    Constant padding (default):

    >>> x = torch.tensor([1, 2, 3])
    >>> pad(x, (2, 1), mode="constant", value=0)
    tensor([0, 0, 1, 2, 3, 0])

    Reflect padding:

    >>> x = torch.tensor([1, 2, 3, 4])
    >>> pad(x, (2, 2), mode="reflect")
    tensor([3, 2, 1, 2, 3, 4, 3, 2])

    Circular (periodic) padding:

    >>> x = torch.tensor([1, 2, 3])
    >>> pad(x, (1, 2), mode="circular")
    tensor([3, 1, 2, 3, 1, 2])

    Linear extrapolation:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> pad(x, (1, 1), mode="linear")
    tensor([0., 1., 2., 3., 4.])

    Multi-dimensional padding:

    >>> x = torch.randn(3, 4)
    >>> y = pad(x, (1, 1, 2, 2), mode="replicate")
    >>> y.shape
    torch.Size([7, 6])

    Explicit dimension selection:

    >>> x = torch.randn(2, 3, 4)
    >>> y = pad(x, (1, 1), mode="reflect", dim=1)
    >>> y.shape
    torch.Size([2, 5, 4])

    Gradient support:

    >>> x = torch.randn(5, requires_grad=True)
    >>> y = pad(x, (1, 1), mode="reflect")
    >>> y.sum().backward()
    >>> x.grad is not None
    True

    Notes
    -----
    - For ``mode="reflect"``, the reflection is edge-inclusive (the edge value
      is not repeated). This matches NumPy's ``symmetric`` mode.
    - For ``mode="reflect_odd"`` (antisymmetric), values are reflected with
      sign change: ``f(-x) = 2*f(0) - f(x)`` at the left edge.
    - Extrapolation modes (linear, polynomial, spline, smooth) fit a curve
      to edge values and extend it into the padding region.
    - All modes support autograd with first and second-order gradients.

    See Also
    --------
    torch.nn.functional.pad : PyTorch's built-in padding (limited to 3D)
    """
    # Normalize padding to list
    if isinstance(padding, int):
        padding_list = [padding]
    elif isinstance(padding, tuple):
        if padding and isinstance(padding[0], tuple):
            # NumPy-style: flatten to our format
            padding_list = []
            for before, after in padding:
                padding_list.extend([before, after])
        else:
            padding_list = list(padding)
    else:
        padding_list = list(padding)

    # Normalize dim to list or None
    if dim is not None:
        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
    else:
        dim_list = None

    return torch.ops.torchscience.pad(
        input, padding_list, mode, value, dim_list, order, out
    )
