import torch
from torch import Tensor


def struve_l_1(z: Tensor) -> Tensor:
    r"""
    Modified Struve function of order one.

    Computes the modified Struve function L_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Struve function of order one is defined as:

    .. math::

       \mathbf{L}_1(z) = \sum_{k=0}^\infty \frac{(z/2)^{2k+2}}{\Gamma(k + 3/2) \Gamma(k + 5/2)}

    This is related to the regular Struve function H_1 by:

    .. math::

       \mathbf{L}_1(z) = -i \mathbf{H}_1(iz)

    Special Values
    --------------
    - L_1(0) = 0
    - L_1(+inf) = +inf (grows exponentially like I_1(z))
    - L_1(-inf) = +inf (even function)
    - L_1(NaN) = NaN

    Symmetry
    --------
    L_1 is an even function: L_1(-z) = L_1(z)

    Domain
    ------
    - z: any real or complex value
    - L_1 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses power series expansion for small |z|
    - The power series has the same form as H_1 but without alternating signs
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Relation to Modified Bessel Function
    -------------------------------------
    For large positive z:

    .. math::

       \mathbf{L}_1(z) \sim I_1(z) - \frac{2}{\pi}

    where I_1(z) is the modified Bessel function of the first kind.

    Applications
    ------------
    The modified Struve function L_1 appears in many contexts:
    - Electromagnetics: radiation impedance calculations
    - Heat conduction: cylindrical heat flow problems
    - Potential theory: Green's functions
    - Plasma physics: electron kinetic equations

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} \mathbf{L}_1(z) = \mathbf{L}_0(z) - \frac{\mathbf{L}_1(z)}{z}

    This is the same recurrence relation as for H_1:
    d/dz H_1(z) = H_0(z) - H_1(z)/z.

    Second-order derivatives are also supported.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified Struve function L_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> struve_l_1(z)
    tensor([0.0000, 0.2879, 1.1052, 2.9310])

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(struve_l_1(-z), struve_l_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> struve_l_1(z)
    tensor([...])  # Complex output

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_l_1(z)
    >>> y.backward()
    >>> z.grad  # equals L_0(2.0) + L_1(2.0)/2.0
    tensor([...])

    Asymptotic behavior (approaches I_1(z) - 2/pi for large z):

    >>> import scipy.special
    >>> z_val = 10.0
    >>> l1 = struve_l_1(torch.tensor([z_val])).item()
    >>> i1 = scipy.special.i1(z_val)
    >>> two_over_pi = 2.0 / 3.14159265
    >>> # l1 is approximately i1 - two_over_pi for large z

    Notes
    -----
    - The modified Struve function is related to the regular Struve function
      by L_n(z) = -i^{-n-1} H_n(iz) for integer n.
    - L_1(z) grows exponentially for large positive z, unlike H_1 which oscillates.
    - Complex accuracy: approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.

    See Also
    --------
    struve_l_0 : Modified Struve function of order zero
    struve_h_1 : Struve function of order one
    modified_bessel_i_1 : Modified Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.struve_l_1(z)
