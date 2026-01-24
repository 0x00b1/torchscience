from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _dpss_tridiagonal_eigenvector(
    n: int,
    nw: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Compute the principal DPSS eigenvector using tridiagonal matrix method.

    The DPSS windows are eigenvectors of a symmetric tridiagonal matrix T
    where:
        - Diagonal: d[i] = ((N-1)/2 - i)^2 * cos(2*pi*W)
        - Off-diagonal: e[i] = i*(N-i)/2

    with W = NW/N being the half-bandwidth parameter.

    This implementation uses torch.linalg.eigh for the eigenvalue decomposition.
    """
    # Compute W from NW
    w = nw / n

    # Build the tridiagonal matrix
    # Using the formulation from Percival & Walden (1993)
    i = torch.arange(n, dtype=dtype, device=device)

    # Diagonal elements: ((N-1)/2 - i)^2 * cos(2*pi*W)
    center = (n - 1) / 2.0
    diagonal = ((center - i) ** 2) * torch.cos(2 * torch.pi * w)

    # Off-diagonal elements: i*(N-i)/2 for i = 1, ..., N-1
    # These are the elements e[0], e[1], ..., e[N-2] connecting
    # row i to row i+1
    j = torch.arange(1, n, dtype=dtype, device=device)
    off_diagonal = j * (n - j) / 2.0

    # Construct the full tridiagonal matrix
    matrix = torch.diag(diagonal)
    matrix += torch.diag(off_diagonal, diagonal=1)
    matrix += torch.diag(off_diagonal, diagonal=-1)

    # Compute eigenvalues and eigenvectors
    # The DPSS windows are the eigenvectors sorted by eigenvalue (largest first)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # eigh returns eigenvalues in ascending order, we want the largest
    # The principal DPSS (zeroth order) corresponds to the largest eigenvalue
    principal_eigenvector = eigenvectors[:, -1]

    # Ensure the window is positive at the center (conventional normalization)
    center_idx = n // 2
    if principal_eigenvector[center_idx] < 0:
        principal_eigenvector = -principal_eigenvector

    return principal_eigenvector


def discrete_prolate_spheroidal_sequence_window(
    n: int,
    nw: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Discrete prolate spheroidal sequence (DPSS) window function (symmetric).

    Computes the zeroth-order DPSS window (also known as the Slepian window)
    of length n. The DPSS windows are the optimal windows for maximizing
    energy concentration within a specified bandwidth, making them ideal
    for spectral analysis applications.

    Mathematical Definition
    -----------------------
    The DPSS windows are the eigenvectors of a symmetric tridiagonal matrix
    that arises from the discrete-time concentration problem. For a sequence
    of length N and half-bandwidth W, the DPSS windows maximize:

        λ = (∫_{-W}^{W} |X(f)|² df) / (∫_{-1/2}^{1/2} |X(f)|² df)

    where X(f) is the discrete-time Fourier transform of the window.

    The zeroth-order DPSS (returned by this function) has the highest
    concentration ratio λ₀ and is the most commonly used.

    Properties
    ----------
    - Optimal energy concentration within bandwidth [-W, W]
    - Smooth, bell-shaped curve similar to Gaussian
    - Kaiser window is an approximation to DPSS
    - Used in multitaper spectral estimation (Thomson's method)
    - Higher NW values give better frequency resolution but wider main lobe

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    nw : float or Tensor
        Time-bandwidth product (N × W where W is the half-bandwidth).
        Must be positive. Common values:
        - NW = 2.0: narrow bandwidth, good for resolving close frequencies
        - NW = 3.0: balanced trade-off (default in many applications)
        - NW = 4.0: wider bandwidth, better leakage suppression
        - NW should typically satisfy NW < N/2
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values, normalized
        so the maximum value is 1.

    Notes
    -----
    This function computes the principal (zeroth-order) DPSS window.
    For multitaper spectral estimation, multiple DPSS windows of different
    orders are typically used together.

    The implementation uses eigenvalue decomposition of the symmetric
    tridiagonal matrix formulation, which is numerically stable and
    efficient for moderate window lengths.

    Autograd is supported - gradients flow through the nw parameter.

    References
    ----------
    .. [1] D. Slepian, "Prolate spheroidal wave functions, Fourier analysis,
           and uncertainty - V: The discrete case," Bell System Technical
           Journal, vol. 57, pp. 1371-1430, 1978.
    .. [2] D. J. Thomson, "Spectrum estimation and harmonic analysis,"
           Proceedings of the IEEE, vol. 70, no. 9, pp. 1055-1096, 1982.
    .. [3] D. B. Percival and A. T. Walden, "Spectral Analysis for Physical
           Applications," Cambridge University Press, 1993.

    See Also
    --------
    periodic_discrete_prolate_spheroidal_sequence_window :
        Periodic version for spectral analysis.
    kaiser_window : Approximation to DPSS using Bessel functions.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.window_function import (
    ...     discrete_prolate_spheroidal_sequence_window
    ... )
    >>> window = discrete_prolate_spheroidal_sequence_window(64, 3.0)
    >>> window.shape
    torch.Size([64])
    """
    if n < 0:
        raise ValueError(
            f"discrete_prolate_spheroidal_sequence_window: n must be "
            f"non-negative, got {n}"
        )

    target_dtype = dtype or torch.float32

    if n == 0:
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    # Convert nw to tensor
    if not isinstance(nw, Tensor):
        nw = torch.tensor(nw, dtype=target_dtype, device=device)

    # Ensure consistent dtype
    if nw.dtype != target_dtype:
        nw = nw.to(dtype=target_dtype)

    target_device = device if device is not None else nw.device

    # Validate parameters
    if nw.item() <= 0:
        raise ValueError(
            f"discrete_prolate_spheroidal_sequence_window: nw must be "
            f"positive, got {nw.item()}"
        )

    return torch.ops.torchscience.discrete_prolate_spheroidal_sequence_window(
        n, nw, dtype, layout, device
    )
