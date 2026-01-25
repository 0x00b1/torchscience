from typing import Sequence, Union

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def poisson_noise(
    size: Sequence[int],
    rate: Union[float, Tensor],
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
    Generate Poisson-distributed noise.

    Poisson noise models discrete counting processes such as photon detection,
    radioactive decay, or packet arrivals. Each sample is an independent draw
    from a Poisson distribution with the specified rate parameter.

    Mathematical Definition
    -----------------------
    The probability mass function for Poisson distribution is:

        P(k; λ) = (λ^k * e^{-λ}) / k!

    where λ (lambda) is the rate parameter and k ∈ {0, 1, 2, ...}.

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. All dimensions are treated as independent
        sample dimensions.
    rate : float or Tensor
        Rate parameter (λ) of the Poisson distribution. Must be non-negative.
        If a tensor, it is broadcast with `size` to allow spatially-varying
        rates across the output.
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
        A tensor of shape `size` containing integer samples from Poisson(rate).
        Mean ≈ rate, variance ≈ rate.

    Examples
    --------
    Generate Poisson noise with rate 5:

    >>> counts = poisson_noise([1000], rate=5.0)
    >>> counts.dtype
    torch.int64
    >>> counts.float().mean()  # approximately 5.0
    tensor(5.0020)

    Generate with spatially-varying rates:

    >>> rates = torch.tensor([[1.0, 10.0], [5.0, 20.0]])
    >>> counts = poisson_noise([2, 2], rate=rates)
    >>> counts.shape
    torch.Size([2, 2])

    Generate reproducible noise:

    >>> g = torch.Generator().manual_seed(42)
    >>> counts1 = poisson_noise([100], rate=5.0, generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> counts2 = poisson_noise([100], rate=5.0, generator=g)
    >>> torch.equal(counts1, counts2)
    True

    Raises
    ------
    RuntimeError
        If size is empty, contains negative values, or rate is negative.

    See Also
    --------
    shot_noise : Differentiable continuous approximation to Poisson noise
    white_noise : Gaussian white noise

    Notes
    -----
    Discrete Nature
    ^^^^^^^^^^^^^^^
    Poisson noise produces integer outputs and is NOT differentiable.
    For differentiable photon-counting simulation, use `shot_noise` instead.

    Statistical Properties
    ^^^^^^^^^^^^^^^^^^^^^^
    - Mean = rate (λ)
    - Variance = rate (λ)
    - For large λ (>10), approaches Gaussian N(λ, √λ)

    Physical Applications
    ^^^^^^^^^^^^^^^^^^^^^
    - Photon counting in imaging
    - Radioactive decay detection
    - Network packet arrivals
    - Neuron spike generation

    References
    ----------
    A. Papoulis, "Probability, Random Variables, and Stochastic Processes,"
    McGraw-Hill, 4th edition, 2002.
    """
    # Convert scalar rate to tensor
    if not isinstance(rate, Tensor):
        rate_tensor = torch.tensor(rate, dtype=torch.float32)
    else:
        rate_tensor = rate

    return torch.ops.torchscience.poisson_noise(
        size,
        rate_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        generator=generator,
    )
