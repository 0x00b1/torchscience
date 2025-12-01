from typing import Sequence, Union

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def impulse_noise(
    size: Sequence[int],
    *,
    p_salt: Union[float, Tensor] = 0.0,
    p_pepper: Union[float, Tensor] = 0.0,
    salt_value: float = 1.0,
    pepper_value: float = -1.0,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate impulse noise (salt-and-pepper noise).

    Impulse noise corrupts signals with extreme values, modeling phenomena
    like dead pixels, bit errors, or transmission dropouts. Each sample
    independently takes one of three values: salt, pepper, or zero.

    Mathematical Definition
    -----------------------
    For each output position independently:

        X = { pepper_value  with probability p_pepper
            { 0             with probability 1 - p_salt - p_pepper
            { salt_value    with probability p_salt

    Note: If p_salt + p_pepper > 1, some positions may have both conditions
    true. In this case, salt takes precedence (checked second).

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. All dimensions are treated as independent
        sample dimensions.
    p_salt : float or Tensor, optional
        Probability of salt noise (high values). Default: 0.0.
        If a tensor, it is broadcast with `size` for spatially-varying
        corruption rates.
    p_pepper : float or Tensor, optional
        Probability of pepper noise (low values). Default: 0.0.
        If a tensor, it is broadcast with `size` for spatially-varying
        corruption rates.
    salt_value : float, optional
        Value used for salt noise. Default: 1.0 (suitable for normalized
        signals in [-1, 1] or [0, 1] range).
    pepper_value : float, optional
        Value used for pepper noise. Default: -1.0 (suitable for normalized
        signals in [-1, 1] range). Use 0.0 for [0, 1] normalized images.
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
        A tensor of shape `size` containing impulse noise. Values are
        either `salt_value`, `pepper_value`, or 0.

    Examples
    --------
    Generate salt-and-pepper noise with 5% each:

    >>> noise = impulse_noise([100, 100], p_salt=0.05, p_pepper=0.05)
    >>> (noise == 1.0).float().mean()  # approximately 0.05
    tensor(0.0503)
    >>> (noise == -1.0).float().mean()  # approximately 0.05
    tensor(0.0497)

    Generate noise for uint8 images (values 0 and 255):

    >>> noise = impulse_noise([256, 256], p_salt=0.01, p_pepper=0.01,
    ...                       salt_value=255.0, pepper_value=0.0)

    Spatially-varying corruption (more noise at edges):

    >>> p_edge = torch.zeros(10, 10)
    >>> p_edge[0, :] = p_edge[-1, :] = p_edge[:, 0] = p_edge[:, -1] = 0.3
    >>> noise = impulse_noise([10, 10], p_salt=p_edge, p_pepper=p_edge)

    Generate reproducible noise:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = impulse_noise([100], p_salt=0.1, generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = impulse_noise([100], p_salt=0.1, generator=g)
    >>> torch.equal(noise1, noise2)
    True

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    white_noise : Gaussian white noise
    poisson_noise : Discrete Poisson noise

    Notes
    -----
    Discrete Nature
    ^^^^^^^^^^^^^^^
    Impulse noise is discrete (values are exactly salt_value, pepper_value,
    or 0) and is NOT differentiable. For differentiable corruption, consider
    using a soft mask approach with shot_noise or custom implementations.

    Image Corruption
    ^^^^^^^^^^^^^^^^
    To corrupt an image with impulse noise, add the noise to the image:

        corrupted = image + noise

    Or use the noise as a mask:

        salt_mask = (noise == salt_value)
        pepper_mask = (noise == pepper_value)
        corrupted = torch.where(salt_mask, salt_value, image)
        corrupted = torch.where(pepper_mask, pepper_value, corrupted)

    Probability Interpretation
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    The probabilities p_salt and p_pepper are independent per-position.
    If their sum exceeds 1, the implementation handles overlap by giving
    salt precedence (it's checked after pepper).

    References
    ----------
    R. C. Gonzalez and R. E. Woods, "Digital Image Processing,"
    Pearson, 4th edition, 2017. (Salt-and-pepper noise model)
    """
    # Convert scalar probabilities to tensors
    if not isinstance(p_salt, Tensor):
        p_salt_tensor = torch.tensor(p_salt, dtype=torch.float32)
    else:
        p_salt_tensor = p_salt

    if not isinstance(p_pepper, Tensor):
        p_pepper_tensor = torch.tensor(p_pepper, dtype=torch.float32)
    else:
        p_pepper_tensor = p_pepper

    return torch.ops.torchscience.impulse_noise(
        size,
        p_salt_tensor,
        p_pepper_tensor,
        salt_value,
        pepper_value,
        dtype=dtype,
        layout=layout,
        device=device,
        generator=generator,
    )
