from typing import Sequence

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def brownian_noise(
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
    Generate brown noise with 1/f^2 power spectral density.

    Brown noise (also known as Brownian noise, red noise, or random walk noise)
    has a power spectral density inversely proportional to the square of frequency.
    It sounds deeper and more rumbling than pink noise, with more emphasis on
    low frequencies.

    Mathematical Definition
    -----------------------
    The power spectral density S(f) of brown noise follows:

        S(f) ~ 1/f^2

    This is generated using spectral shaping:

    1. Generate white noise w
    2. Transform to frequency domain: W = FFT(w)
    3. Apply 1/f amplitude scaling: B = W / f
    4. Transform back: b = IFFT(B)
    5. Normalize to unit variance

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. The last dimension is treated as the
        time/sample axis where the 1/f^2 spectrum is applied. Other dimensions
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
        A tensor of shape `size` containing brown noise samples with
        approximately zero mean and unit variance.

    Examples
    --------
    Generate 1D brown noise with 1000 samples:

    >>> noise = brownian_noise([1000])
    >>> noise.shape
    torch.Size([1000])

    Generate batched brown noise (4 channels, 1000 samples each):

    >>> noise = brownian_noise([4, 1000])
    >>> noise.shape
    torch.Size([4, 1000])

    Generate reproducible noise using a generator:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = brownian_noise([100], generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = brownian_noise([100], generator=g)
    >>> torch.allclose(noise1, noise2)
    True

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    white_noise : Generate noise with flat spectrum
    pink_noise : Generate noise with 1/f spectrum
    violet_noise : Generate noise with f^2 spectrum (opposite of brown)

    Notes
    -----
    Spectral Properties
    ^^^^^^^^^^^^^^^^^^^
    The output has a power spectrum proportional to 1/f^2, meaning:

    - Very low frequencies dominate
    - Power decreases by 6 dB per octave
    - The DC component is set to zero to ensure zero mean
    - Equivalent to integrating white noise

    Physical Interpretation
    ^^^^^^^^^^^^^^^^^^^^^^^
    Brown noise is named after Robert Brown (Brownian motion), not the color.
    It represents the random walk process - the position of a particle
    undergoing random displacement. Each sample is correlated with its
    neighbors, giving it a smooth, flowing character.

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
    return torch.ops.torchscience.brown_noise(
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
