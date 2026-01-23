"""SO(3) exponential and logarithm maps.

This module provides the Lie algebra exponential and logarithm maps for the
rotation group SO(3):

- so3_exp: Maps rotation vectors (Lie algebra so(3)) to rotation matrices (SO(3))
- so3_log: Maps rotation matrices (SO(3)) to rotation vectors (so(3))

These are fundamental operations in robotics, computer vision, and physics
for working with 3D rotations using their tangent space representation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _skew_symmetric(omega: Tensor) -> Tensor:
    """Compute the skew-symmetric matrix [omega]_x from a 3-vector.

    Parameters
    ----------
    omega : Tensor
        3-vector, shape (..., 3).

    Returns
    -------
    Tensor
        Skew-symmetric matrix, shape (..., 3, 3).

    Notes
    -----
    The skew-symmetric matrix is defined as:

    .. math::

        [\\omega]_\\times = \\begin{bmatrix}
            0 & -\\omega_3 & \\omega_2 \\\\
            \\omega_3 & 0 & -\\omega_1 \\\\
            -\\omega_2 & \\omega_1 & 0
        \\end{bmatrix}

    such that :math:`[\\omega]_\\times v = \\omega \\times v` for any vector v.
    """
    batch_shape = omega.shape[:-1]
    omega_x = omega[..., 0]
    omega_y = omega[..., 1]
    omega_z = omega[..., 2]

    zeros = torch.zeros_like(omega_x)

    # Build the skew-symmetric matrix row by row
    row0 = torch.stack([zeros, -omega_z, omega_y], dim=-1)
    row1 = torch.stack([omega_z, zeros, -omega_x], dim=-1)
    row2 = torch.stack([-omega_y, omega_x, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def _vee(skew: Tensor) -> Tensor:
    """Extract the vector from a skew-symmetric matrix.

    Parameters
    ----------
    skew : Tensor
        Skew-symmetric matrix, shape (..., 3, 3).

    Returns
    -------
    Tensor
        3-vector, shape (..., 3).
    """
    return torch.stack(
        [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1
    )


def so3_exp(omega: Tensor) -> Tensor:
    """Compute the SO(3) exponential map (Rodrigues formula).

    Maps a rotation vector (axis * angle) to a rotation matrix using the
    Rodrigues formula.

    Parameters
    ----------
    omega : Tensor
        Rotation vector (axis * angle), shape (..., 3). The direction of the
        vector is the rotation axis, and the magnitude is the rotation angle
        in radians.

    Returns
    -------
    Tensor
        Rotation matrix, shape (..., 3, 3).

    Raises
    ------
    ValueError
        If omega does not have last dimension 3.

    Notes
    -----
    The Rodrigues formula computes the rotation matrix as:

    .. math::

        R = I + \\frac{\\sin\\theta}{\\theta}[\\omega]_\\times +
            \\frac{1 - \\cos\\theta}{\\theta^2}[\\omega]_\\times^2

    where :math:`\\theta = \\|\\omega\\|` is the rotation angle and
    :math:`[\\omega]_\\times` is the skew-symmetric matrix of :math:`\\omega`.

    For small angles (:math:`\\theta < 10^{-6}`), a Taylor expansion is used
    for numerical stability:

    .. math::

        R \\approx I + [\\omega]_\\times +
            \\frac{1}{2}[\\omega]_\\times^2

    See Also
    --------
    so3_log : Inverse operation (logarithm map).

    Examples
    --------
    Zero rotation vector gives identity:

    >>> omega = torch.tensor([0.0, 0.0, 0.0])
    >>> so3_exp(omega)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    90-degree rotation around z-axis:

    >>> import math
    >>> omega = torch.tensor([0.0, 0.0, math.pi / 2])
    >>> so3_exp(omega)
    tensor([[ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]])
    """
    if omega.shape[-1] != 3:
        raise ValueError(
            f"so3_exp: omega must have last dimension 3, got {omega.shape[-1]}"
        )

    batch_shape = omega.shape[:-1]
    device = omega.device
    dtype = omega.dtype

    # Compute angle (norm of omega)
    theta = torch.linalg.norm(omega, dim=-1)
    theta_sq = theta * theta

    # Build the skew-symmetric matrix [omega]_x
    omega_skew = _skew_symmetric(omega)

    # For numerical stability, use Taylor expansion for small angles
    # sin(theta)/theta = 1 - theta^2/6 + theta^4/120 - ...
    # (1 - cos(theta))/theta^2 = 1/2 - theta^2/24 + theta^4/720 - ...
    small_angle_threshold = 1e-6
    is_small = theta < small_angle_threshold

    # Compute coefficients
    # For normal angles: sin(theta)/theta and (1 - cos(theta))/theta^2
    # For small angles: Taylor expansions
    theta_safe = torch.where(is_small, torch.ones_like(theta), theta)
    theta_sq_safe = theta_safe * theta_safe

    sin_theta = torch.sin(theta_safe)
    cos_theta = torch.cos(theta_safe)

    # Coefficient for [omega]_x term: sin(theta)/theta
    # Taylor: 1 - theta^2/6 + theta^4/120
    coeff1_normal = sin_theta / theta_safe
    coeff1_taylor = 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0
    coeff1 = torch.where(is_small, coeff1_taylor, coeff1_normal)

    # Coefficient for [omega]_x^2 term: (1 - cos(theta))/theta^2
    # Taylor: 1/2 - theta^2/24 + theta^4/720
    coeff2_normal = (1.0 - cos_theta) / theta_sq_safe
    coeff2_taylor = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0
    coeff2 = torch.where(is_small, coeff2_taylor, coeff2_normal)

    # Expand coefficients for broadcasting with (3, 3) matrices
    coeff1 = coeff1[..., None, None]
    coeff2 = coeff2[..., None, None]

    # Compute omega_skew^2
    omega_skew_sq = torch.matmul(omega_skew, omega_skew)

    # Identity matrix with proper batch shape
    eye = torch.eye(3, device=device, dtype=dtype)
    if batch_shape:
        eye = eye.expand(*batch_shape, 3, 3)

    # Rodrigues formula: R = I + coeff1 * [omega]_x + coeff2 * [omega]_x^2
    R = eye + coeff1 * omega_skew + coeff2 * omega_skew_sq

    return R


def so3_log(matrix: Tensor) -> Tensor:
    """Compute the SO(3) logarithm map.

    Maps a rotation matrix to a rotation vector (axis * angle).

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3). Must be a valid rotation matrix
        (orthogonal with determinant 1).

    Returns
    -------
    Tensor
        Rotation vector (axis * angle), shape (..., 3).

    Raises
    ------
    ValueError
        If matrix does not have shape (..., 3, 3).

    Notes
    -----
    The rotation angle is computed from the trace:

    .. math::

        \\theta = \\arccos\\left(\\frac{\\text{tr}(R) - 1}{2}\\right)

    The rotation vector is then extracted differently depending on the angle:

    - **Near identity** (:math:`\\theta \\approx 0`): Use Taylor expansion
      :math:`\\omega \\approx \\text{vee}(R - R^T) / 2`
    - **Near 180 degrees** (:math:`\\theta \\approx \\pi`): Extract axis from
      the eigenvector corresponding to eigenvalue 1
    - **General case**: :math:`\\omega = \\frac{\\theta}{2\\sin\\theta}
      \\text{vee}(R - R^T)`

    See Also
    --------
    so3_exp : Inverse operation (exponential map).

    Examples
    --------
    Identity matrix gives zero vector:

    >>> R = torch.eye(3)
    >>> so3_log(R)
    tensor([0., 0., 0.])

    90-degree rotation around z-axis:

    >>> import math
    >>> R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    >>> so3_log(R)
    tensor([0.0000, 0.0000, 1.5708])
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"so3_log: matrix must have last two dimensions (3, 3), "
            f"got {matrix.shape[-2:]}"
        )

    batch_shape = matrix.shape[:-2]
    device = matrix.device
    dtype = matrix.dtype

    # Compute the trace of R
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    # Compute angle from trace: cos(theta) = (trace - 1) / 2
    # Clamp to [-1, 1] for numerical stability
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # Extract the skew-symmetric part: (R - R^T) / 2
    R_minus_RT = matrix - matrix.transpose(-1, -2)
    skew_part = R_minus_RT / 2.0

    # The vee of (R - R^T)/2 gives us sin(theta) * axis
    sin_theta_axis = _vee(skew_part)

    # Handle three cases based on theta
    small_threshold = 1e-6
    near_pi_threshold = 1e-6

    sin_theta = torch.sin(theta)
    is_small = theta < small_threshold
    is_near_pi = (torch.pi - theta) < near_pi_threshold

    # Case 1: Small angle (theta ~ 0)
    # omega = vee((R - R^T)/2) * (1 + theta^2/6 + ...)
    # For very small angles, just omega ≈ vee((R - R^T)/2)
    theta_sq = theta * theta
    small_angle_coeff = (
        1.0 + theta_sq / 6.0 + theta_sq * theta_sq * 7.0 / 360.0
    )
    omega_small = sin_theta_axis * small_angle_coeff[..., None]

    # Case 2: General case (0 < theta < pi - eps)
    # omega = theta / (2 * sin(theta)) * vee(R - R^T)
    # Note: vee(R - R^T) = 2 * vee((R - R^T)/2) = 2 * sin_theta_axis
    # So omega = theta / sin(theta) * sin_theta_axis
    sin_theta_safe = torch.where(
        is_small | is_near_pi, torch.ones_like(sin_theta), sin_theta
    )
    general_coeff = theta / sin_theta_safe
    omega_general = sin_theta_axis * general_coeff[..., None]

    # Case 3: Near pi (theta ~ pi)
    # At theta = pi, sin(theta) = 0, so we need a different approach.
    # For R = I + 2 * v * v^T - 2 * I (where v is the axis), we have:
    # R + I = 2 * (I + v * v^T - I) = 2 * v * v^T for theta = pi
    # Actually: R = -I + 2 * k * k^T for 180-degree rotation around axis k
    # So R + I = 2 * k * k^T, and the diagonal of R + I gives us 2 * k_i^2
    # We extract k from the column of (R + I) with the largest diagonal entry.

    # Compute R + I
    eye = torch.eye(3, device=device, dtype=dtype)
    if batch_shape:
        eye = eye.expand(*batch_shape, 3, 3)
    R_plus_I = matrix + eye

    # Find the diagonal element with maximum absolute value
    diag = torch.diagonal(R_plus_I, dim1=-2, dim2=-1)  # (..., 3)

    # Get the index of the max diagonal element
    # We'll compute all three possible axis extractions and select the best one
    k0 = R_plus_I[..., :, 0]  # (..., 3) - first column
    k1 = R_plus_I[..., :, 1]  # (..., 3) - second column
    k2 = R_plus_I[..., :, 2]  # (..., 3) - third column

    # Normalize each candidate axis
    k0_norm = torch.linalg.norm(k0, dim=-1, keepdim=True)
    k1_norm = torch.linalg.norm(k1, dim=-1, keepdim=True)
    k2_norm = torch.linalg.norm(k2, dim=-1, keepdim=True)

    eps = torch.finfo(dtype).eps
    k0_safe = k0 / torch.clamp(k0_norm, min=eps)
    k1_safe = k1 / torch.clamp(k1_norm, min=eps)
    k2_safe = k2 / torch.clamp(k2_norm, min=eps)

    # Select axis based on which diagonal is largest
    max_idx = torch.argmax(diag, dim=-1)  # (...,)

    # Use advanced indexing to select the right axis for each batch element
    # This is equivalent to: axis = k_{max_idx} for each element
    k_candidates = torch.stack(
        [k0_safe, k1_safe, k2_safe], dim=-2
    )  # (..., 3, 3)

    # Gather the axis corresponding to max_idx
    # k_candidates has shape (..., 3, 3) where dim -2 is the candidate index
    # We want to select along dim -2 using max_idx
    max_idx_expanded = (
        max_idx[..., None, None].expand(*batch_shape, 1, 3)
        if batch_shape
        else max_idx[None, None].expand(1, 3)
    )
    axis_pi = torch.gather(k_candidates, -2, max_idx_expanded).squeeze(-2)

    # Ensure the axis direction is consistent with the skew-symmetric part
    # The sign should match the sign of sin_theta_axis when theta is near pi
    # (sin(theta) is small but positive for theta slightly less than pi)
    # We use the sign from the anti-symmetric part where available
    dot_sign = torch.sum(axis_pi * sin_theta_axis, dim=-1, keepdim=True)
    axis_pi = torch.where(dot_sign >= 0, axis_pi, -axis_pi)

    omega_pi = axis_pi * theta[..., None]

    # Combine all cases
    omega = torch.where(
        is_small[..., None],
        omega_small,
        torch.where(is_near_pi[..., None], omega_pi, omega_general),
    )

    return omega
