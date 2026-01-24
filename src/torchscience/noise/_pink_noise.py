from typing import Sequence

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def pink_noise(
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
    Generate pink noise with 1/f power spectral density.

    Pink noise (also known as 1/f noise or flicker noise) has a power spectral
    density that is inversely proportional to frequency. This gives it equal
    energy per octave, making it useful for audio synthesis, testing, and
    scientific simulations.

    Mathematical Definition
    -----------------------
    The power spectral density S(f) of pink noise follows:

        S(f) ~ 1/f

    This is generated using spectral shaping:
    1. Generate white noise w
    2. Transform to frequency domain: W = FFT(w)
    3. Apply 1/sqrt(f) scaling: P = W / sqrt(f)
    4. Transform back: p = IFFT(P)
    5. Normalize to unit variance

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. The last dimension is treated as the
        time/sample axis where the 1/f spectrum is applied. Other dimensions
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
        A tensor of shape `size` containing pink noise samples with
        approximately zero mean and unit variance.

    Examples
    --------
    Generate 1D pink noise with 1000 samples:

    >>> noise = pink_noise([1000])
    >>> noise.shape
    torch.Size([1000])

    Generate batched pink noise (4 channels, 1000 samples each):

    >>> noise = pink_noise([4, 1000])
    >>> noise.shape
    torch.Size([4, 1000])

    Generate reproducible noise using a generator:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = pink_noise([100], generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = pink_noise([100], generator=g)
    >>> torch.allclose(noise1, noise2)
    True

    Generate on GPU with gradients:

    >>> noise = pink_noise([1000], device='cuda', requires_grad=True)  # doctest: +SKIP

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    torch.randn : Generate white (Gaussian) noise

    Notes
    -----
    Spectral Properties
    ^^^^^^^^^^^^^^^^^^^
    The output has a power spectrum proportional to 1/f, meaning:
    - Lower frequencies have more power than higher frequencies
    - Equal power per octave (logarithmic frequency bands)
    - The DC component (f=0) is set to zero to ensure zero mean

    Normalization
    ^^^^^^^^^^^^^
    The output is normalized to have approximately unit variance,
    similar to torch.randn. This is achieved by dividing by the
    theoretical standard deviation of the shaped noise.

    Gradient Support
    ^^^^^^^^^^^^^^^^
    When requires_grad=True, gradients flow through the FFT operations.
    This enables use in differentiable audio synthesis and learned noise
    models.

    References
    ----------
    N. J. Kasdin, "Discrete simulation of colored noise and stochastic
    processes and 1/f^alpha power law noise generation," Proceedings of
    the IEEE, vol. 83, no. 5, pp. 802-827, 1995.

    J. Timmer and M. Koenig, "On generating power law noise," Astronomy
    and Astrophysics, vol. 300, pp. 707-710, 1995.
    """
    return torch.ops.torchscience.pink_noise(
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
