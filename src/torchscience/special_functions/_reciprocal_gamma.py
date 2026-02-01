import torch
from torch import Tensor


def reciprocal_gamma(z: Tensor) -> Tensor:
    r"""
    Reciprocal gamma function.

    Computes the reciprocal of the gamma function evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The reciprocal gamma function is defined as:

    .. math::

       \frac{1}{\Gamma(z)}

    Unlike the gamma function which has poles at non-positive integers,
    1/Gamma(z) is an entire function (analytic everywhere in the complex plane).

    Special Values
    --------------
    - 1/Gamma(n) = 1/(n-1)! for positive integers n
    - 1/Gamma(1) = 1
    - 1/Gamma(0) = 0 (gamma has a pole, so reciprocal is zero)
    - 1/Gamma(-n) = 0 for non-positive integers n (gamma has poles)

    Domain
    ------
    - z: any real or complex value
    - The function is entire (analytic everywhere)
    - Returns 0 at z = 0, -1, -2, -3, ... where Gamma has poles

    Algorithm
    ---------
    - Computes gamma(z) and returns 1/gamma(z)
    - When gamma(z) is infinite (at poles), returns 0
    - Uses the same Lanczos approximation as the gamma function

    Applications
    ------------
    The reciprocal gamma function appears in:
    - Regularized incomplete gamma functions
    - Normalization constants for distributions
    - Series expansions where 1/Gamma appears naturally
    - Numerical stability when working near gamma poles

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed as:

    .. math::

       \frac{d}{dz} \frac{1}{\Gamma(z)} = -\frac{\psi(z)}{\Gamma(z)}

    where :math:`\psi(z)` is the digamma function.

    Second-order derivatives (gradgradcheck) are also supported, computed as:

    .. math::

       \frac{d^2}{dz^2} \frac{1}{\Gamma(z)} = \frac{1}{\Gamma(z)} \left[ \psi(z)^2 - \psi'(z) \right]

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The reciprocal gamma function evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Reciprocal of factorials via reciprocal gamma:

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> reciprocal_gamma(z)
    tensor([1.0000, 1.0000, 0.5000, 0.1667, 0.0417])

    Values at poles return zero:

    >>> z = torch.tensor([0.0, -1.0, -2.0])
    >>> reciprocal_gamma(z)
    tensor([0., 0., 0.])

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j])
    >>> reciprocal_gamma(z)
    tensor([...])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = reciprocal_gamma(z)
    >>> y.backward()
    >>> z.grad  # -digamma(2) * reciprocal_gamma(2)
    tensor([...])

    Notes
    -----
    - Unlike Gamma(z) which has poles at non-positive integers,
      1/Gamma(z) is entire and simply returns 0 at those points.
    - For numerical stability near gamma poles, prefer reciprocal_gamma
      over 1/gamma(z) to avoid infinity/zero issues.

    See Also
    --------
    gamma : The gamma function
    log_gamma : Natural logarithm of the gamma function
    digamma : Logarithmic derivative of the gamma function
    """
    return torch.ops.torchscience.reciprocal_gamma(z)
