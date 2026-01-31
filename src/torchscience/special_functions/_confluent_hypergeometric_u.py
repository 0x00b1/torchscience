import torch
from torch import Tensor


def confluent_hypergeometric_u(a: Tensor, b: Tensor, z: Tensor) -> Tensor:
    r"""
    Confluent hypergeometric function U(a, b, z), also known as Kummer's
    function of the second kind or Tricomi's function.

    Mathematical Definition
    -----------------------
    The confluent hypergeometric function U is the second linearly independent
    solution to Kummer's differential equation:

    .. math::

       z \frac{d^2w}{dz^2} + (b - z) \frac{dw}{dz} - a w = 0

    When b is not an integer, U can be expressed in terms of Kummer's M function:

    .. math::

       U(a, b, z) = \frac{\Gamma(1-b)}{\Gamma(a-b+1)} M(a, b, z)
                  + \frac{\Gamma(b-1)}{\Gamma(a)} z^{1-b} M(a-b+1, 2-b, z)

    For integer b, U is defined by a limiting process.

    Domain and Branch Structure
    ---------------------------
    - a: any real or complex value
    - b: any real or complex value
    - z: complex plane with branch cut along negative real axis

    The function has a branch cut along the negative real axis (z < 0) when
    the parameters are not special values. The principal branch is defined
    for -pi < arg(z) <= pi.

    Asymptotic Behavior
    -------------------
    For large |z| with |arg(z)| < 3*pi/2:

    .. math::

       U(a, b, z) \sim z^{-a} \quad \text{as } |z| \to \infty

    This decay property distinguishes U from M, which grows exponentially.

    Special Values
    --------------
    - U(a, b, z) has a logarithmic singularity at z = 0 when b is not an integer
    - When a = 0, -1, -2, ..., U reduces to a polynomial in z
    - U(0, b, z) = 1
    - U(a, a+1, z) = z^(-a)

    Parameters
    ----------
    a : Tensor
        First parameter. Broadcasting with b and z is supported.
    b : Tensor
        Second parameter. Broadcasting with a and z is supported.
    z : Tensor
        Input value. Typically z > 0 (positive real) or complex with
        positive real part. Broadcasting with a and b is supported.

    Returns
    -------
    Tensor
        The confluent hypergeometric function U(a, b, z) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([2.0])
    >>> z = torch.tensor([1.0])
    >>> confluent_hypergeometric_u(a, b, z)
    tensor([0.5963])

    Special case U(a, a+1, z) = z^(-a):

    >>> a = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = confluent_hypergeometric_u(a, a + 1, z)
    >>> expected = z ** (-a)
    >>> torch.allclose(result, expected)
    True

    Notes
    -----
    The function U is particularly useful in physics and engineering
    applications where solutions that decay at infinity are needed, such as:

    - Quantum mechanics (Coulomb wave functions)
    - Diffusion problems
    - Statistical distributions (e.g., Whittaker functions)

    See Also
    --------
    confluent_hypergeometric_m : Kummer's function of the first kind
    hypergeometric_2_f_1 : Gauss hypergeometric function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Chapter 13
           https://dlmf.nist.gov/13
    .. [2] Abramowitz and Stegun, Handbook of Mathematical Functions,
           Chapter 13
    """
    return torch.ops.torchscience.confluent_hypergeometric_u(a, b, z)
