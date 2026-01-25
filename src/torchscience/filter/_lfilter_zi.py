"""Compute initial conditions for lfilter for step response steady-state."""

import torch
from torch import Tensor


def lfilter_zi(b: Tensor, a: Tensor) -> Tensor:
    """Compute initial conditions for lfilter for step response steady-state.

    Computes the initial state such that the output of the filter is the
    steady-state value when the input is a step function (all ones).

    This is used to initialize lfilter to avoid transients when the input
    signal doesn't start at zero.

    Parameters
    ----------
    b : Tensor
        Numerator polynomial coefficients (1-D).
    a : Tensor
        Denominator polynomial coefficients (1-D).

    Returns
    -------
    zi : Tensor
        Initial conditions, shape ``(max(len(a), len(b)) - 1,)``.

    Raises
    ------
    ValueError
        If b or a is not 1-D, or if a is empty or has no nonzero leading
        coefficient.

    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::

        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)

    where z(n) is a vector of length m. lfilter_zi solves::

        zi = A*zi + B

    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.

    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter
    are::

        A = companion(a).T
        B = b[1:] - a[1:]*b[0]

    assuming ``a[0]`` is 1.0; if ``a[0]`` is not 1, `a` and `b` are first
    divided by ``a[0]``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import lfilter_zi
    >>> # Simple first-order lowpass filter
    >>> b = torch.tensor([0.5, 0.5], dtype=torch.float64)
    >>> a = torch.tensor([1.0, -0.3], dtype=torch.float64)
    >>> zi = lfilter_zi(b, a)
    """
    # Ensure 1-D inputs
    b = torch.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")

    a = torch.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")

    # Remove leading zeros from a
    while a.numel() > 1 and a[0] == 0.0:
        a = a[1:]

    if a.numel() < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")

    # Normalize coefficients so a[0] == 1
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Pad a or b with zeros so they are the same length
    n = max(a.numel(), b.numel())

    if a.numel() < n:
        a = torch.cat(
            [a, torch.zeros(n - a.numel(), dtype=a.dtype, device=a.device)]
        )
    elif b.numel() < n:
        b = torch.cat(
            [b, torch.zeros(n - b.numel(), dtype=b.dtype, device=b.device)]
        )

    # Ensure the result dtype can handle division
    dt = torch.result_type(a, b)

    # Build I - A where A is the companion matrix transpose
    # The companion matrix for polynomial with coefficients [1, a1, a2, ..., an]
    # has first row [-a1, -a2, ..., -an] and sub-diagonal of ones.
    # Taking the transpose: first column is [-a1, -a2, ..., -an]^T
    # and the super-diagonal is all ones.
    #
    # So A^T has:
    # - First column: -a[1:]/a[0] = -a[1:]  (since a[0]=1)
    # - Super-diagonal: ones
    #
    # I - A^T has:
    # - First column: 1 + a[1], a[2], ..., a[n]
    # - Diagonal: all ones
    # - Super-diagonal: all negative ones

    IminusA = _companion(a).T
    IminusA = torch.eye(n - 1, dtype=dt, device=a.device) - IminusA

    # Build B vector
    B = b[1:] - a[1:] * b[0]

    # Solve zi = A*zi + B, equivalently (I - A)*zi = B
    zi = torch.linalg.solve(IminusA, B.to(dt))

    return zi


def _companion(a: Tensor) -> Tensor:
    """Create a companion matrix.

    Create the companion matrix associated with the polynomial whose
    coefficients are given in `a`.

    Parameters
    ----------
    a : Tensor
        1-D tensor of polynomial coefficients. The length of `a` must be
        at least two, and ``a[0]`` must not be zero.

    Returns
    -------
    c : Tensor
        Companion matrix of shape ``(n-1, n-1)`` where n is the length of `a`.
        The first row of `c` is ``-a[1:]/a[0]``, and the first sub-diagonal
        is all ones.

    Raises
    ------
    ValueError
        If ``len(a) < 2`` or ``a[0] == 0``.
    """
    n = a.numel()

    if n < 2:
        raise ValueError("The length of `a` must be at least 2.")

    if a[0] == 0:
        raise ValueError("The first coefficient of `a` must not be zero.")

    # First row is -a[1:] / a[0]
    first_row = -a[1:] / a[0]

    # Create companion matrix
    c = torch.zeros(n - 1, n - 1, dtype=first_row.dtype, device=a.device)
    c[0, :] = first_row

    # Set sub-diagonal to ones
    if n > 2:
        c[torch.arange(1, n - 1), torch.arange(0, n - 2)] = 1.0

    return c
