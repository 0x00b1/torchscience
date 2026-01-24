"""Adaptive filtering using the Recursive Least Squares (RLS) algorithm."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def rls(
    x: Tensor,
    d: Tensor,
    num_taps: int,
    lam: float = 0.99,
    delta: float = 1.0,
    w0: Optional[Tensor] = None,
    return_weights: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Adaptive filtering using the Recursive Least Squares (RLS) algorithm.

    The RLS algorithm is an adaptive filter that recursively minimizes a
    weighted least squares cost function, with exponentially decaying
    weights controlled by the forgetting factor.

    Parameters
    ----------
    x : Tensor
        Input signal, shape (..., n_samples).
    d : Tensor
        Desired signal, shape (..., n_samples).
    num_taps : int
        Number of filter taps (filter length).
    lam : float, optional
        Forgetting factor (0 < lam <= 1). Default is 0.99. Values closer
        to 1 give longer memory. Lambda = 1 corresponds to standard
        (non-forgetting) least squares.
    delta : float, optional
        Initial inverse correlation matrix scaling. Default is 1.0.
        Larger values give faster initial adaptation but may cause
        initial transients.
    w0 : Tensor, optional
        Initial filter weights, shape (num_taps,). Default is zeros.
    return_weights : bool, optional
        If True, returns (y, w) where w is the final filter weights.
        Default is False.

    Returns
    -------
    y : Tensor
        Filter output, shape (..., n_samples).
    w : Tensor, optional
        Final filter weights, shape (num_taps,) for 1D input or
        (..., num_taps) for batched input. Only returned if
        return_weights=True.

    Notes
    -----
    The RLS algorithm minimizes the weighted least squares cost function:

    .. math::

        J[n] = \\sum_{i=0}^{n} \\lambda^{n-i} |d[i] - \\mathbf{w}^T \\mathbf{x}[i]|^2

    The algorithm uses the matrix inversion lemma for efficient updates:

    .. math::

        \\mathbf{k}[n] = \\frac{\\mathbf{P}[n-1] \\mathbf{x}[n]}
            {\\lambda + \\mathbf{x}^T[n] \\mathbf{P}[n-1] \\mathbf{x}[n]}

        e[n] = d[n] - \\mathbf{w}^T[n-1] \\mathbf{x}[n]

        \\mathbf{w}[n] = \\mathbf{w}[n-1] + \\mathbf{k}[n] e[n]

        \\mathbf{P}[n] = \\frac{1}{\\lambda}
            (\\mathbf{P}[n-1] - \\mathbf{k}[n] \\mathbf{x}^T[n] \\mathbf{P}[n-1])

    where :math:`\\mathbf{x}[n] = [x[n], x[n-1], ..., x[n-L+1]]^T` is the
    input vector and :math:`L` is the number of taps.

    RLS typically converges faster than LMS but requires more computation
    (O(L^2) per sample vs O(L) for LMS).

    For batched inputs, each batch element has its own weight trajectory
    and inverse correlation matrix, allowing independent adaptation.

    This implementation is fully differentiable and can be used for training
    or optimization via automatic differentiation.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import rls
    >>> # System identification
    >>> h_true = torch.tensor([0.5, 0.3, 0.2])
    >>> x = torch.randn(500)
    >>> d = torch.conv1d(x.view(1, 1, -1), h_true.view(1, 1, -1).flip(-1),
    ...                  padding=2).squeeze()[:500]
    >>> y, w = rls(x, d, num_taps=3, lam=0.99, return_weights=True)
    >>> # w should be close to h_true after convergence

    See Also
    --------
    lms : LMS adaptive filter (simpler, O(L) per sample).
    nlms : Normalized LMS with automatic step size adaptation.

    References
    ----------
    .. [1] Haykin, S. (2014). Adaptive Filter Theory (5th ed.).
           Pearson Education.
    .. [2] Sayed, A. H. (2003). Fundamentals of Adaptive Filtering.
           Wiley-IEEE Press.
    """
    # Ensure we have at least 1D tensors
    x = torch.atleast_1d(x)
    d = torch.atleast_1d(d)

    if x.shape != d.shape:
        raise ValueError(
            f"x and d must have the same shape, got {x.shape} and {d.shape}"
        )

    # Get signal properties
    n_samples = x.shape[-1]
    batch_shape = x.shape[:-1]
    is_batched = len(batch_shape) > 0

    # Determine output dtype
    out_dtype = torch.promote_types(x.dtype, d.dtype)
    if w0 is not None:
        out_dtype = torch.promote_types(out_dtype, w0.dtype)

    # Flatten batch dimensions for processing
    x_flat = x.reshape(-1, n_samples).to(out_dtype)
    d_flat = d.reshape(-1, n_samples).to(out_dtype)
    n_batch = x_flat.shape[0]

    # Initialize weights - one set per batch element
    if w0 is not None:
        if w0.shape != (num_taps,):
            raise ValueError(
                f"w0 must have shape ({num_taps},), got {w0.shape}"
            )
        # Broadcast w0 to all batch elements
        w = w0.to(out_dtype).unsqueeze(0).expand(n_batch, -1).clone()
    else:
        w = torch.zeros(n_batch, num_taps, dtype=out_dtype, device=x.device)

    # Initialize inverse correlation matrix P = delta * I
    # Shape: (n_batch, num_taps, num_taps)
    P = (
        delta
        * torch.eye(num_taps, dtype=out_dtype, device=x.device)
        .unsqueeze(0)
        .expand(n_batch, -1, -1)
        .clone()
    )

    # Allocate output
    y_flat = torch.zeros(n_batch, n_samples, dtype=out_dtype, device=x.device)

    # RLS algorithm - process sample by sample
    for n in range(n_samples):
        # Build input vector x_n = [x[n], x[n-1], ..., x[n-num_taps+1]]
        # Handle boundary conditions with zero padding
        start_idx = max(0, n - num_taps + 1)
        end_idx = n + 1
        x_slice = x_flat[:, start_idx:end_idx]

        # Flip to get [x[n], x[n-1], ..., x[n-k]] ordering
        x_slice = x_slice.flip(-1)

        # Pad with zeros if we don't have enough samples
        pad_size = num_taps - x_slice.shape[-1]
        if pad_size > 0:
            x_slice = F.pad(x_slice, (0, pad_size))

        # x_n shape: (n_batch, num_taps)
        x_n = x_slice

        # Filter output: y[n] = w^T @ x_n (element-wise per batch)
        # w shape: (n_batch, num_taps), x_n shape: (n_batch, num_taps)
        y_n = (w * x_n).sum(dim=-1)
        y_flat[:, n] = y_n

        # Error: e[n] = d[n] - y[n]
        e_n = d_flat[:, n] - y_n

        # Compute gain vector k = P @ x_n / (lam + x_n^T @ P @ x_n)
        # P shape: (n_batch, num_taps, num_taps)
        # x_n shape: (n_batch, num_taps)

        # P @ x_n: (n_batch, num_taps)
        Px = torch.bmm(P, x_n.unsqueeze(-1)).squeeze(-1)

        # x_n^T @ P @ x_n: (n_batch,)
        xPx = (x_n * Px).sum(dim=-1)

        # k = Px / (lam + xPx): (n_batch, num_taps)
        k = Px / (lam + xPx).unsqueeze(-1)

        # Weight update: w[n] = w[n-1] + k * e[n]
        w = w + k * e_n.unsqueeze(-1)

        # Update inverse correlation matrix:
        # P[n] = (P[n-1] - k @ x_n^T @ P[n-1]) / lam
        # k shape: (n_batch, num_taps)
        # x_n shape: (n_batch, num_taps)

        # k @ x_n^T: (n_batch, num_taps, num_taps)
        k_xT = k.unsqueeze(-1) * x_n.unsqueeze(-2)

        # k @ x_n^T @ P: (n_batch, num_taps, num_taps)
        k_xT_P = torch.bmm(k_xT, P)

        P = (P - k_xT_P) / lam

    # Reshape output
    y = y_flat.reshape(batch_shape + (n_samples,))

    if return_weights:
        if is_batched:
            # Reshape weights to match batch shape
            w_out = w.reshape(batch_shape + (num_taps,))
        else:
            # For 1D input, return 1D weights
            w_out = w.squeeze(0)
        return y, w_out

    return y
