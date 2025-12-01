import torch
from torch import Tensor


def digamma(z: Tensor) -> Tensor:
    r"""
    Digamma function.

    Computes the digamma (psi) function evaluated at each element of the input
    tensor. The digamma function is the logarithmic derivative of the gamma
    function.

    Mathematical Definition
    -----------------------
    The digamma function is defined as:

    .. math::

       \psi(z) = \frac{d}{dz} \ln \Gamma(z) = \frac{\Gamma'(z)}{\Gamma(z)}

    Special Values
    --------------
    - psi(1) = -gamma (Euler-Mascheroni constant, approximately -0.5772)
    - psi(n) = -gamma + H_{n-1} for positive integers n, where H_n is the
      n-th harmonic number
    - psi(1/2) = -gamma - 2*ln(2)

    Domain
    ------
    - z: any real or complex value except non-positive integers
    - Poles at z = 0, -1, -2, -3, ... where the function returns -inf

    Algorithm
    ---------
    - Uses recurrence relation psi(z+1) = psi(z) + 1/z to shift argument
    - Asymptotic expansion for |z| >= 6
    - Reflection formula for Re(z) < 0.5 (complex)

    Recurrence Relations
    --------------------
    - psi(z+1) = psi(z) + 1/z
    - psi(1-z) - psi(z) = pi * cot(pi*z)

    Applications
    ------------
    The digamma function appears in:
    - Maximum likelihood estimation for gamma and Dirichlet distributions
    - Gradients of the gamma function: d/dz Gamma(z) = Gamma(z) * psi(z)
    - Bayesian inference with conjugate priors
    - Renormalization in quantum field theory

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the trigamma function:

    .. math::

       \frac{d}{dz} \psi(z) = \psi'(z) = \text{trigamma}(z)

    Second-order derivatives are also supported using the tetragamma function.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The digamma function evaluated at each element of z.

    Examples
    --------
    Evaluate at positive integers:

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> digamma(z)
    tensor([-0.5772,  0.4228,  0.9228,  1.2561])

    Verify recurrence relation psi(z+1) = psi(z) + 1/z:

    >>> z = torch.tensor([2.0])
    >>> digamma(z + 1) - digamma(z)
    tensor([0.5000])  # equals 1/z = 0.5

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j])
    >>> digamma(z)
    tensor([0.0946+1.0762j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = digamma(z)
    >>> y.backward()
    >>> z.grad  # trigamma(2) = pi^2/6 - 1 ≈ 0.6449
    tensor([0.6449])

    See Also
    --------
    torchscience.special_functions.gamma : Gamma function
    torch.special.digamma : PyTorch's digamma implementation
    torch.special.polygamma : General polygamma function
    """
    return torch.ops.torchscience.digamma(z)
