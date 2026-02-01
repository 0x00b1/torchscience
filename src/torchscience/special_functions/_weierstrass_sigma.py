import torch
from torch import Tensor


def weierstrass_sigma(z: Tensor, g2: Tensor, g3: Tensor) -> Tensor:
    r"""
    Weierstrass sigma function sigma(z; g2, g3).

    Computes the Weierstrass sigma function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Weierstrass sigma function is defined as an infinite product:

    .. math::

       \sigma(z; g_2, g_3) = z \prod_{\omega \in \Lambda \setminus \{0\}}
       \left(1 - \frac{z}{\omega}\right)
       \exp\left(\frac{z}{\omega} + \frac{z^2}{2\omega^2}\right)

    where the lattice Lambda is determined by the invariants g2 and g3.

    The sigma function is an entire function (no poles), unlike the
    Weierstrass P function which has double poles at lattice points.

    Fundamental Relations
    ---------------------
    The sigma function is related to the other Weierstrass functions:

    - Logarithmic derivative gives the zeta function:

      .. math::

         \frac{\sigma'(z)}{\sigma(z)} = \zeta(z)

    - Second logarithmic derivative gives P (with sign):

      .. math::

         -\frac{d^2}{dz^2} \log \sigma(z) = \wp(z)

    - Equivalently:

      .. math::

         \wp(z) = -\frac{\sigma''(z)}{\sigma(z)} + \left(\frac{\sigma'(z)}{\sigma(z)}\right)^2

    Taylor Series
    -------------
    Near z = 0, the sigma function has the Taylor expansion:

    .. math::

       \sigma(z) = z - \frac{g_2}{240} z^5 - \frac{g_3}{840} z^7
       - \frac{g_2^2}{161280} z^9 + O(z^{11})

    Note that sigma(z) has only odd powers of z because it is an odd function.

    Special Properties
    ------------------
    - sigma(0) = 0 (simple zero at the origin)
    - sigma(-z) = -sigma(z) (odd function)
    - sigma(z) is an entire function with simple zeros at all lattice points
    - For a lattice point omega: sigma(z + 2*omega) = -exp(2*eta*(z + omega)) * sigma(z)
      where eta is the quasi-period

    Quasi-periodicity
    -----------------
    Unlike the P function which is doubly periodic, sigma is quasi-periodic:

    .. math::

       \sigma(z + 2\omega_i) = -e^{2\eta_i(z + \omega_i)} \sigma(z)

    where eta_i are the quasi-periods associated with the half-periods omega_i.

    The lack of true periodicity makes sigma essential for constructing
    elliptic functions with prescribed zeros and poles.

    Relationship to Theta Functions
    --------------------------------
    The sigma function can be expressed in terms of Jacobi theta functions:

    .. math::

       \sigma(z) = \frac{2\omega_1}{\pi} \exp\left(\frac{\eta_1 z^2}{2\omega_1}\right)
       \frac{\theta_1\left(\frac{\pi z}{2\omega_1}, q\right)}{\theta_1'(0, q)}

    where q = exp(i*pi*omega_2/omega_1) is the nome.

    Applications
    ------------
    The Weierstrass sigma function appears in:
    - Construction of elliptic functions with prescribed zeros/poles
    - Solution of differential equations on elliptic curves
    - Representation theory and theta correspondences
    - Algebraic geometry and moduli spaces
    - Addition formulas for elliptic functions
    - Weierstrass factorization of elliptic functions

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to z is:

    .. math::

       \frac{\partial \sigma}{\partial z} = \sigma(z) \zeta(z)

    where zeta(z) is the Weierstrass zeta function.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    z : Tensor
        The argument. Can be floating-point or complex.
        Broadcasting with g2 and g3 is supported.
    g2 : Tensor
        The first invariant of the Weierstrass function.
        Broadcasting with z and g3 is supported.
    g3 : Tensor
        The second invariant of the Weierstrass function.
        Broadcasting with z and g2 is supported.

    Returns
    -------
    Tensor
        The Weierstrass sigma function sigma(z; g2, g3) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> weierstrass_sigma(z, g2, g3)
    tensor([0.4794])

    Verify sigma(0) = 0:

    >>> z = torch.tensor([0.0])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.5])
    >>> weierstrass_sigma(z, g2, g3)
    tensor([0.])

    Odd function property:

    >>> z = torch.tensor([0.3])
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.5])
    >>> sigma_pos = weierstrass_sigma(z, g2, g3)
    >>> sigma_neg = weierstrass_sigma(-z, g2, g3)
    >>> torch.allclose(sigma_pos, -sigma_neg)
    True

    Complex input:

    >>> z = torch.tensor([0.3 + 0.2j])
    >>> g2 = torch.tensor([1.0 + 0.0j])
    >>> g3 = torch.tensor([0.0 + 0.0j])
    >>> weierstrass_sigma(z, g2, g3)
    tensor([0.2916+0.2041j])

    Autograd:

    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> g2 = torch.tensor([1.0])
    >>> g3 = torch.tensor([0.0])
    >>> sigma = weierstrass_sigma(z, g2, g3)
    >>> sigma.backward()
    >>> z.grad  # d(sigma)/dz = sigma * zeta
    tensor([0.9375])

    Notes
    -----
    - The sigma function has simple zeros at all lattice points.
    - Unlike P which is doubly periodic, sigma is quasi-periodic.
    - The normalization is such that sigma'(0) = 1.
    - For numerical stability near zero, the Taylor series is used.

    See Also
    --------
    weierstrass_p : Weierstrass elliptic P function
    weierstrass_zeta : Weierstrass zeta function (sigma'/sigma)
    """
    return torch.ops.torchscience.weierstrass_sigma(z, g2, g3)
