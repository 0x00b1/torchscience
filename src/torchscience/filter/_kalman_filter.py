"""Kalman filter for state estimation."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor


def kalman_filter(
    z: Tensor,
    H: Tensor,
    R: Tensor,
    F: Optional[Tensor] = None,
    Q: Optional[Tensor] = None,
    x0: Optional[Tensor] = None,
    P0: Optional[Tensor] = None,
    return_covariance: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Kalman filter for state estimation.

    The Kalman filter is an optimal recursive estimator for linear dynamic
    systems with Gaussian noise. It estimates the state of a system from
    noisy observations.

    Parameters
    ----------
    z : Tensor
        Observations, shape (..., n_samples, n_obs).
    H : Tensor
        Observation matrix, shape (n_obs, n_state). Maps state to observation
        space via z = H @ x + noise.
    R : Tensor
        Observation noise covariance, shape (n_obs, n_obs). Must be positive
        definite.
    F : Tensor, optional
        State transition matrix, shape (n_state, n_state). Default is identity.
        Models the state evolution: x_{k+1} = F @ x_k + process_noise.
    Q : Tensor, optional
        Process noise covariance, shape (n_state, n_state). Default is zeros.
        Must be positive semi-definite.
    x0 : Tensor, optional
        Initial state estimate, shape (n_state,). Default is zeros.
    P0 : Tensor, optional
        Initial error covariance, shape (n_state, n_state). Default is identity.
        Must be positive definite.
    return_covariance : bool, optional
        If True, returns (x_hat, P) where P is the final error covariance.
        Default is False.

    Returns
    -------
    x_hat : Tensor
        State estimates, shape (..., n_samples, n_state).
    P : Tensor, optional
        Final error covariance, shape (n_state, n_state). Only returned if
        return_covariance=True. For batched inputs, this is the covariance
        for the last batch element.

    Notes
    -----
    The standard Kalman filter equations are:

    **Predict step:**

    .. math::

        \\hat{x}_{k|k-1} = F \\hat{x}_{k-1|k-1}

        P_{k|k-1} = F P_{k-1|k-1} F^T + Q

    **Update step:**

    .. math::

        K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}

        \\hat{x}_{k|k} = \\hat{x}_{k|k-1} + K_k (z_k - H \\hat{x}_{k|k-1})

        P_{k|k} = (I - K_k H) P_{k|k-1}

    where :math:`K_k` is the Kalman gain and the innovation is
    :math:`z_k - H \\hat{x}_{k|k-1}`.

    The Kalman filter is optimal in the sense that it minimizes the mean
    squared error of the state estimate for linear Gaussian systems.

    For batched inputs, each batch element is processed with its own state
    trajectory, allowing independent estimation for multiple signals.

    This implementation is fully differentiable and can be used for training
    or optimization via automatic differentiation.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import kalman_filter
    >>> # Track a 1D position with noisy measurements
    >>> torch.manual_seed(42)
    >>> n_samples = 100
    >>> true_position = torch.arange(n_samples, dtype=torch.float64) * 0.1
    >>> z = true_position + 0.5 * torch.randn(n_samples, dtype=torch.float64)
    >>> z = z.unsqueeze(-1)  # Shape: (n_samples, 1)
    >>> H = torch.tensor([[1.0]], dtype=torch.float64)
    >>> R = torch.tensor([[0.25]], dtype=torch.float64)
    >>> x_hat = kalman_filter(z, H, R)
    >>> # x_hat should track true_position with reduced noise

    See Also
    --------
    lms : LMS adaptive filter for system identification.
    rls : RLS adaptive filter with faster convergence.

    References
    ----------
    .. [1] Kalman, R. E. (1960). "A New Approach to Linear Filtering and
           Prediction Problems". Journal of Basic Engineering.
    .. [2] Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman
           Filter". University of North Carolina at Chapel Hill.
    """
    # Ensure we have at least 2D observations (n_samples, n_obs)
    z = torch.atleast_2d(z)

    # Get dimensions
    n_obs = z.shape[-1]
    n_state = H.shape[-1]
    n_samples = z.shape[-2]
    batch_shape = z.shape[:-2]

    # Determine output dtype (promote all inputs)
    out_dtype = torch.promote_types(z.dtype, H.dtype)
    out_dtype = torch.promote_types(out_dtype, R.dtype)
    if F is not None:
        out_dtype = torch.promote_types(out_dtype, F.dtype)
    if Q is not None:
        out_dtype = torch.promote_types(out_dtype, Q.dtype)
    if x0 is not None:
        out_dtype = torch.promote_types(out_dtype, x0.dtype)
    if P0 is not None:
        out_dtype = torch.promote_types(out_dtype, P0.dtype)

    device = z.device

    # Convert inputs to output dtype
    z = z.to(out_dtype)
    H = H.to(out_dtype)
    R = R.to(out_dtype)

    # Set defaults for optional parameters
    if F is None:
        F = torch.eye(n_state, dtype=out_dtype, device=device)
    else:
        F = F.to(out_dtype)

    if Q is None:
        Q = torch.zeros(n_state, n_state, dtype=out_dtype, device=device)
    else:
        Q = Q.to(out_dtype)

    if x0 is None:
        x0 = torch.zeros(n_state, dtype=out_dtype, device=device)
    else:
        x0 = x0.to(out_dtype)

    if P0 is None:
        P0 = torch.eye(n_state, dtype=out_dtype, device=device)
    else:
        P0 = P0.to(out_dtype)

    # Validate shapes
    if H.shape != (n_obs, n_state):
        raise ValueError(
            f"H must have shape ({n_obs}, {n_state}), got {H.shape}"
        )
    if R.shape != (n_obs, n_obs):
        raise ValueError(
            f"R must have shape ({n_obs}, {n_obs}), got {R.shape}"
        )
    if F.shape != (n_state, n_state):
        raise ValueError(
            f"F must have shape ({n_state}, {n_state}), got {F.shape}"
        )
    if Q.shape != (n_state, n_state):
        raise ValueError(
            f"Q must have shape ({n_state}, {n_state}), got {Q.shape}"
        )
    if x0.shape != (n_state,):
        raise ValueError(f"x0 must have shape ({n_state},), got {x0.shape}")
    if P0.shape != (n_state, n_state):
        raise ValueError(
            f"P0 must have shape ({n_state}, {n_state}), got {P0.shape}"
        )

    # Flatten batch dimensions for processing
    is_batched = len(batch_shape) > 0
    if is_batched:
        n_batch = z.reshape(-1, n_samples, n_obs).shape[0]
        z_flat = z.reshape(n_batch, n_samples, n_obs)
    else:
        n_batch = 1
        z_flat = z.unsqueeze(0)

    # Initialize state and covariance for each batch element
    # x shape: (n_batch, n_state)
    x = x0.unsqueeze(0).expand(n_batch, -1).clone()
    # P shape: (n_batch, n_state, n_state)
    P = P0.unsqueeze(0).expand(n_batch, -1, -1).clone()

    # Allocate output: (n_batch, n_samples, n_state)
    x_hat = torch.zeros(
        n_batch, n_samples, n_state, dtype=out_dtype, device=device
    )

    # Identity matrix for update step
    I = torch.eye(n_state, dtype=out_dtype, device=device)

    # Precompute transposes
    H_T = H.T
    F_T = F.T

    # Kalman filter loop
    for k in range(n_samples):
        # Get current observation: (n_batch, n_obs)
        z_k = z_flat[:, k, :]

        # --- Predict step ---
        # x_pred = F @ x
        # (n_batch, n_state) = (n_state, n_state) @ (n_batch, n_state)^T -> (n_batch, n_state)
        x_pred = torch.matmul(x, F_T)  # x @ F^T = (F @ x^T)^T

        # P_pred = F @ P @ F^T + Q
        # (n_batch, n_state, n_state)
        P_pred = (
            torch.matmul(torch.matmul(F.unsqueeze(0), P), F_T.unsqueeze(0)) + Q
        )

        # --- Update step ---
        # Innovation covariance: S = H @ P_pred @ H^T + R
        # (n_batch, n_obs, n_obs)
        HP = torch.matmul(H.unsqueeze(0), P_pred)  # (n_batch, n_obs, n_state)
        S = torch.matmul(HP, H_T.unsqueeze(0)) + R  # (n_batch, n_obs, n_obs)

        # Kalman gain: K = P_pred @ H^T @ S^{-1}
        # (n_batch, n_state, n_obs)
        PH_T = torch.matmul(
            P_pred, H_T.unsqueeze(0)
        )  # (n_batch, n_state, n_obs)
        S_inv = torch.linalg.inv(S)  # (n_batch, n_obs, n_obs)
        K = torch.matmul(PH_T, S_inv)  # (n_batch, n_state, n_obs)

        # Innovation: y = z - H @ x_pred
        # (n_batch, n_obs)
        y = z_k - torch.matmul(x_pred, H_T)  # x_pred @ H^T = (H @ x_pred^T)^T

        # State update: x = x_pred + K @ y
        # (n_batch, n_state) = (n_batch, n_state) + (n_batch, n_state, n_obs) @ (n_batch, n_obs, 1)
        x = x_pred + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)

        # Covariance update: P = (I - K @ H) @ P_pred
        # (n_batch, n_state, n_state)
        KH = torch.matmul(K, H.unsqueeze(0))  # (n_batch, n_state, n_state)
        P = torch.matmul(I - KH, P_pred)

        # Store estimate
        x_hat[:, k, :] = x

    # Reshape output to match input batch shape
    if is_batched:
        x_hat = x_hat.reshape(batch_shape + (n_samples, n_state))
    else:
        x_hat = x_hat.squeeze(0)

    if return_covariance:
        # Return final covariance (for last batch element if batched)
        P_final = P[-1] if is_batched else P.squeeze(0)
        return x_hat, P_final

    return x_hat
