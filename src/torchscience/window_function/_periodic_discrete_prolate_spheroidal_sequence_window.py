from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _dpss_tridiagonal_eigenvector_periodic(
    n: int,
    nw: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Compute the principal DPSS eigenvector for periodic window.

    The periodic version uses the full period n for bandwidth calculations,
    suitable for spectral analysis with FFT.
    """
    # Compute W from NW (using full period n)
    w = nw / n

    # Build the tridiagonal matrix
    i = torch.arange(n, dtype=dtype, device=device)

    # Diagonal elements: ((N-1)/2 - i)^2 * cos(2*pi*W)
    # For periodic, we use N/2 as center
    center = n / 2.0
    diagonal = ((center - i) ** 2) * torch.cos(2 * torch.pi * w)

    # Off-diagonal elements: i*(N-i)/2 for i = 1, ..., N-1
    j = torch.arange(1, n, dtype=dtype, device=device)
    off_diagonal = j * (n - j) / 2.0

    # Construct the full tridiagonal matrix
    matrix = torch.diag(diagonal)
    matrix += torch.diag(off_diagonal, diagonal=1)
    matrix += torch.diag(off_diagonal, diagonal=-1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # The principal DPSS corresponds to the largest eigenvalue
    principal_eigenvector = eigenvectors[:, -1]

    # Ensure the window is positive at the center
    center_idx = n // 2
    if principal_eigenvector[center_idx] < 0:
        principal_eigenvector = -principal_eigenvector

    return principal_eigenvector


def periodic_discrete_prolate_spheroidal_sequence_window(
    n: int,
    nw: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Discrete prolate spheroidal sequence (DPSS) window function (periodic).

    Computes the zeroth-order periodic DPSS window of length n. The periodic
    version is designed for spectral analysis where the window will be used
    with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic DPSS window uses the full period n for bandwidth calculations,
    making it suitable for spectral analysis applications. The window maximizes
    energy concentration within the specified bandwidth.

    Properties
    ----------
    - Optimal energy concentration within bandwidth [-W, W]
    - Designed for spectral analysis with FFT
    - Used in multitaper spectral estimation (Thomson's method)

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
    Autograd is supported - gradients flow through the nw parameter.

    References
    ----------
    .. [1] D. Slepian, "Prolate spheroidal wave functions, Fourier analysis,
           and uncertainty - V: The discrete case," Bell System Technical
           Journal, vol. 57, pp. 1371-1430, 1978.
    .. [2] D. J. Thomson, "Spectrum estimation and harmonic analysis,"
           Proceedings of the IEEE, vol. 70, no. 9, pp. 1055-1096, 1982.

    See Also
    --------
    discrete_prolate_spheroidal_sequence_window : Symmetric version.
    periodic_kaiser_window : Approximation using Bessel functions.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.window_function import (
    ...     periodic_discrete_prolate_spheroidal_sequence_window
    ... )
    >>> window = periodic_discrete_prolate_spheroidal_sequence_window(64, 3.0)
    >>> window.shape
    torch.Size([64])
    """
    if n < 0:
        raise ValueError(
            f"periodic_discrete_prolate_spheroidal_sequence_window: n must be "
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
            f"periodic_discrete_prolate_spheroidal_sequence_window: nw must be "
            f"positive, got {nw.item()}"
        )

    return torch.ops.torchscience.periodic_discrete_prolate_spheroidal_sequence_window(
        n, nw, dtype, layout, device
    )
