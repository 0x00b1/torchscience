import torch
from torch import Tensor


def trigamma(z: Tensor) -> Tensor:
    r"""
    Trigamma function.

    Computes the trigamma function evaluated at each element of the input
    tensor. The trigamma function is the second derivative of the log-gamma
    function, or equivalently, the first derivative of the digamma function.

    Mathematical Definition
    -----------------------
    The trigamma function is defined as:

    .. math::

       \psi_1(z) = \frac{d^2}{dz^2} \ln \Gamma(z) = \frac{d}{dz} \psi(z)

    where :math:`\psi(z)` is the digamma function.

    Special Values
    --------------
    - psi_1(1) = pi^2/6 (approximately 1.6449)
    - psi_1(1/2) = pi^2/2 (approximately 4.9348)
    - psi_1(n) = pi^2/6 - sum_{k=1}^{n-1} 1/k^2 for positive integers n

    Domain
    ------
    - z: any real or complex value except non-positive integers
    - Poles at z = 0, -1, -2, -3, ... where the function returns +inf

    Algorithm
    ---------
    - Uses recurrence relation psi_1(z+1) = psi_1(z) - 1/z^2 to shift argument
    - Asymptotic expansion for |z| >= 6
    - Reflection formula for Re(z) < 0.5 (complex)

    Recurrence Relations
    --------------------
    - psi_1(z+1) = psi_1(z) - 1/z^2
    - psi_1(1-z) + psi_1(z) = pi^2 / sin^2(pi*z)

    Applications
    ------------
    The trigamma function appears in:
    - Variance calculations for gamma and Dirichlet distributions
    - Fisher information for exponential family distributions
    - Gradients of the digamma function in optimization
    - Higher-order corrections in statistical mechanics

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the tetragamma function:

    .. math::

       \frac{d}{dz} \psi_1(z) = \psi_2(z) = \text{tetragamma}(z)

    Second-order derivatives are also supported using the pentagamma function.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The trigamma function evaluated at each element of z.

    Examples
    --------
    Evaluate at positive integers:

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> trigamma(z)
    tensor([1.6449, 0.6449, 0.3949, 0.2838])

    Verify recurrence relation psi_1(z+1) = psi_1(z) - 1/z^2:

    >>> z = torch.tensor([2.0])
    >>> trigamma(z) - trigamma(z + 1)
    tensor([0.2500])  # equals 1/z^2 = 0.25

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j])
    >>> trigamma(z)
    tensor([0.4038-0.6194j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = trigamma(z)
    >>> y.backward()
    >>> z.grad  # tetragamma(2)
    tensor([-0.4041])

    See Also
    --------
    torchscience.special_functions.digamma : Digamma function
    torchscience.special_functions.gamma : Gamma function
    torch.special.polygamma : General polygamma function
    """
    return torch.ops.torchscience.trigamma(z)
