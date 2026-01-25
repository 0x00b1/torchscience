"""Analog frequency response from zeros-poles-gain representation."""

from typing import Tuple, Union

import torch
from torch import Tensor


def freqs_zpk(
    z: Tensor,
    p: Tensor,
    k: Union[Tensor, float],
    worN: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute frequency response of an analog filter in ZPK form.

    Evaluates the transfer function H(s) = k * prod(s - z_i) / prod(s - p_i)
    at s = j*w for the given angular frequencies w.

    Parameters
    ----------
    z : Tensor
        Zeros of the transfer function (complex).
    p : Tensor
        Poles of the transfer function (complex).
    k : Tensor or float
        System gain.
    worN : Tensor
        Angular frequencies at which to evaluate (rad/s).

    Returns
    -------
    w : Tensor
        The angular frequencies (same as input worN).
    h : Tensor
        Complex frequency response H(jw).

    Notes
    -----
    The transfer function is evaluated as:

    .. math::
        H(s) = k \\frac{\\prod_i (s - z_i)}{\\prod_i (s - p_i)}

    at s = jw, where w is the angular frequency.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import freqs_zpk
    >>> # First-order lowpass: H(s) = 1/(s+1)
    >>> zeros = torch.empty(0, dtype=torch.complex128)
    >>> poles = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
    >>> gain = torch.tensor(1.0)
    >>> w = torch.tensor([0.0, 1.0, 10.0])
    >>> w_out, h = freqs_zpk(zeros, poles, gain, w)
    >>> torch.abs(h)
    tensor([1.0000, 0.7071, 0.0995], dtype=torch.float64)

    References
    ----------
    .. [1] scipy.signal.freqs_zpk
    """
    # Determine working dtype based on inputs
    if worN.dtype == torch.float32:
        complex_dtype = torch.complex64
        real_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        real_dtype = torch.float64

    device = worN.device

    # Handle empty frequency array
    if worN.numel() == 0:
        return worN, torch.empty(0, dtype=complex_dtype, device=device)

    # Convert gain to tensor if needed
    if not isinstance(k, Tensor):
        k = torch.tensor(k, dtype=real_dtype, device=device)

    # Compute s = j * w
    s = torch.complex(
        torch.zeros_like(worN, dtype=real_dtype),
        worN.to(dtype=real_dtype),
    )  # s = jw

    # Compute numerator: prod(s - z_i)
    if z.numel() > 0:
        # s has shape (N,), z has shape (M,)
        # We want (s - z) with shape (N, M), then prod over dim=-1
        z_complex = z.to(dtype=complex_dtype, device=device)
        diff_z = s.unsqueeze(-1) - z_complex  # (N, M)
        num = torch.prod(diff_z, dim=-1)  # (N,)
    else:
        num = torch.ones_like(s, dtype=complex_dtype)

    # Compute denominator: prod(s - p_i)
    if p.numel() > 0:
        p_complex = p.to(dtype=complex_dtype, device=device)
        diff_p = s.unsqueeze(-1) - p_complex  # (N, K)
        den = torch.prod(diff_p, dim=-1)  # (N,)
    else:
        den = torch.ones_like(s, dtype=complex_dtype)

    # Compute H(s) = k * num / den
    k_complex = k.to(dtype=complex_dtype, device=device)
    h = k_complex * num / den

    return worN, h
