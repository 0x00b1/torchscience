"""Adaptive filtering using the Least Mean Squares (LMS) algorithm."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def lms(
    x: Tensor,
    d: Tensor,
    num_taps: int,
    mu: float = 0.01,
    w0: Optional[Tensor] = None,
    return_weights: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Adaptive filtering using the Least Mean Squares (LMS) algorithm.

    The LMS algorithm is a stochastic gradient descent method for adaptively
    finding the filter coefficients that minimize the mean squared error
    between the filter output and a desired signal.

    Parameters
    ----------
    x : Tensor
        Input signal, shape (..., n_samples).
    d : Tensor
        Desired signal, shape (..., n_samples).
    num_taps : int
        Number of filter taps (filter length).
    mu : float, optional
        Step size (learning rate). Default is 0.01. Larger values converge
        faster but may be unstable. For stability, mu should be less than
        2/(num_taps * input_power).
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
    The LMS algorithm updates filter weights at each sample:

    .. math::

        y[n] = \\mathbf{w}^T[n] \\mathbf{x}[n]

        e[n] = d[n] - y[n]

        \\mathbf{w}[n+1] = \\mathbf{w}[n] + \\mu \\cdot e[n] \\cdot \\mathbf{x}[n]

    where :math:`\\mathbf{x}[n] = [x[n], x[n-1], ..., x[n-L+1]]^T` is the
    input vector and :math:`L` is the number of taps.

    For batched inputs, each batch element has its own weight trajectory,
    allowing independent adaptation for each signal.

    This implementation is fully differentiable and can be used for training
    or optimization via automatic differentiation.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import lms
    >>> # System identification
    >>> h_true = torch.tensor([0.5, 0.3, 0.2])
    >>> x = torch.randn(1000)
    >>> d = torch.conv1d(x.view(1, 1, -1), h_true.view(1, 1, -1).flip(-1),
    ...                  padding=2).squeeze()[:1000]
    >>> y, w = lms(x, d, num_taps=3, mu=0.1, return_weights=True)
    >>> # w should be close to h_true after convergence

    See Also
    --------
    nlms : Normalized LMS with automatic step size adaptation.
    rls : Recursive Least Squares with faster convergence.

    References
    ----------
    .. [1] Widrow, B., & Hoff, M. E. (1960). "Adaptive switching circuits".
           IRE WESCON Convention Record, 4(1), 96-104.
    .. [2] Haykin, S. (2014). Adaptive Filter Theory (5th ed.).
           Pearson Education.
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

    # Allocate output
    y_flat = torch.zeros(n_batch, n_samples, dtype=out_dtype, device=x.device)

    # LMS algorithm - process sample by sample
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

        # Weight update: w[n+1] = w[n] + mu * e[n] * x_n
        # Each batch element updates its own weights independently
        w = w + mu * e_n.unsqueeze(-1) * x_n

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
