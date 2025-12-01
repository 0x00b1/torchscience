from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def rectangular_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Rectangular window function.

    Computes a rectangular (boxcar) window of length n. The rectangular window
    is the simplest window function, consisting of all ones.

    Mathematical Definition
    -----------------------
    The rectangular window is defined as:

        w[k] = 1,  for k = 0, 1, ..., n-1

    This window provides no tapering and is equivalent to no windowing at all.
    It has the narrowest main lobe but the highest side lobes (-13 dB) in the
    frequency domain.

    Properties
    ----------
    - Main lobe width: 4*pi/n (narrowest of all windows)
    - Side lobe level: -13 dB (highest of all windows)
    - Coherent gain: n (highest possible)
    - Scalloping loss: 3.92 dB

    Use Cases
    ---------
    - When frequency resolution is critical and spectral leakage is acceptable
    - As a baseline for comparing other window functions
    - In applications where the signal is already periodic within the analysis frame

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
        If n=0, an empty tensor is returned.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor. If None, uses the
        default floating point type.
    layout : torch.layout, optional
        The desired layout of the returned tensor. If None, uses the
        default layout (torch.strided).
    device : torch.device, optional
        The desired device of the returned tensor. If None, uses the
        default device.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients. Default is False.
        Note that gradients through a constant window are typically not useful.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values, all equal to 1.

    Examples
    --------
    Create a rectangular window of length 5:

    >>> rectangular_window(5)
    tensor([1., 1., 1., 1., 1.])

    Create a window with specific dtype and device:

    >>> rectangular_window(4, dtype=torch.float64)
    tensor([1., 1., 1., 1.], dtype=torch.float64)

    Raises
    ------
    RuntimeError
        If n < 0.

    See Also
    --------
    torch.signal.windows.hann : Hann window (raised cosine)
    torch.signal.windows.hamming : Hamming window
    torch.signal.windows.blackman : Blackman window

    References
    ----------
    F.J. Harris, "On the use of windows for harmonic analysis with the
    discrete Fourier transform," Proceedings of the IEEE, vol. 66,
    no. 1, pp. 51-83, Jan. 1978.

    A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal Processing,"
    3rd ed., Prentice Hall, 2009.
    """
    return torch.ops.torchscience.rectangular_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
