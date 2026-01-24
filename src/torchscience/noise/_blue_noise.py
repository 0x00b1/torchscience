from typing import Sequence

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def blue_noise(
    size: Sequence[int],
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate blue noise with f power spectral density.

    Blue noise (also known as azure noise) has a power spectral density
    proportional to frequency. It emphasizes high frequencies, sounding
    hissy or bright. It is the spectral opposite of pink noise.

    Mathematical Definition
    -----------------------
    The power spectral density S(f) of blue noise follows:

        S(f) ~ f

    This is generated using spectral shaping:

    1. Generate white noise w
    2. Transform to frequency domain: W = FFT(w)
    3. Apply sqrt(f) amplitude scaling: B = W * sqrt(f)
    4. Transform back: b = IFFT(B)
    5. Normalize to unit variance

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. The last dimension is treated as the
        time/sample axis where the f spectrum is applied. Other dimensions
        are batch dimensions generating independent noise sequences.
    generator : torch.Generator, optional
        A pseudorandom number generator for sampling. If None, uses the default
        generator.
    out : Tensor, optional
    dtype : torch.dtype, optional
        The desired data type of the returned tensor. If None, uses the default
        floating point type.
    layout : torch.layout, optional
        The desired layout of the returned tensor. Default: torch.strided.
    device : torch.device, optional
        The desired device of the returned tensor. Default: CPU.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients. Default: False.
    pin_memory : bool, optional

    Returns
    -------
    Tensor
        A tensor of shape `size` containing blue noise samples with
        approximately zero mean and unit variance.

    Examples
    --------
    Generate 1D blue noise with 1000 samples:

    >>> noise = blue_noise([1000])
    >>> noise.shape
    torch.Size([1000])

    Generate batched blue noise (4 channels, 1000 samples each):

    >>> noise = blue_noise([4, 1000])
    >>> noise.shape
    torch.Size([4, 1000])

    Generate reproducible noise using a generator:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = blue_noise([100], generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = blue_noise([100], generator=g)
    >>> torch.allclose(noise1, noise2)
    True

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    white_noise : Generate noise with flat spectrum
    pink_noise : Generate noise with 1/f spectrum (opposite of blue)
    violet_noise : Generate noise with f^2 spectrum

    Notes
    -----
    Spectral Properties
    ^^^^^^^^^^^^^^^^^^^
    The output has a power spectrum proportional to f, meaning:

    - High frequencies have more power than low frequencies
    - Power increases by 3 dB per octave
    - The DC component is set to zero to ensure zero mean
    - Spectral opposite of pink noise

    Physical Interpretation
    ^^^^^^^^^^^^^^^^^^^^^^^
    Blue noise can be thought of as differentiated pink noise. It has
    a high-frequency emphasis that makes it sound "bright" or "hissy".
    In image processing, blue noise is valued for dithering because
    its high-frequency content is less perceptible.

    Normalization
    ^^^^^^^^^^^^^
    The output is normalized to have approximately unit variance,
    similar to torch.randn.

    Gradient Support
    ^^^^^^^^^^^^^^^^
    When requires_grad=True, gradients flow through the FFT operations.

    References
    ----------
    N. J. Kasdin, "Discrete simulation of colored noise and stochastic
    processes and 1/f^alpha power law noise generation," Proceedings of
    the IEEE, vol. 83, no. 5, pp. 802-827, 1995.
    """
    return torch.ops.torchscience.blue_noise(
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
