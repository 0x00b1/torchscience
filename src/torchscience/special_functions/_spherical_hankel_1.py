import torch
from torch import Tensor


def spherical_hankel_1(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Spherical Hankel function of the first kind of general order n.

    Computes the spherical Hankel function of the first kind h_n^(1)(z) evaluated
    at each element of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The spherical Hankel function of the first kind of order n is defined as:

    .. math::

       h_n^{(1)}(z) = j_n(z) + i \cdot y_n(z)

    where j_n(z) is the spherical Bessel function of the first kind and
    y_n(z) is the spherical Bessel function of the second kind.

    Relation to the standard Hankel functions:

    .. math::

       h_n^{(1)}(z) = \sqrt{\frac{\pi}{2z}} H_{n+1/2}^{(1)}(z)

    Explicit formulas for low orders:

    .. math::

       h_0^{(1)}(z) &= \frac{e^{iz}}{iz} = -i \frac{e^{iz}}{z} \\
       h_1^{(1)}(z) &= -\frac{e^{iz}}{z}\left(1 + \frac{i}{z}\right)

    Special Values
    --------------
    - h_n^(1)(0) is singular (has a pole at z = 0)
    - h_n^(1)(NaN) = NaN + NaN*i

    Recurrence Relation
    -------------------
    Spherical Hankel functions satisfy the same recurrence as spherical Bessel functions:

    .. math::

       h_{n+1}^{(1)}(z) = \frac{2n+1}{z} h_n^{(1)}(z) - h_{n-1}^{(1)}(z)

    Derivative
    ----------
    The derivative with respect to z is:

    .. math::

       \frac{d}{dz} h_n^{(1)}(z) = \frac{n}{z} h_n^{(1)}(z) - h_{n+1}^{(1)}(z)

    Or equivalently:

    .. math::

       \frac{d}{dz} h_n^{(1)}(z) = h_{n-1}^{(1)}(z) - \frac{n+1}{z} h_n^{(1)}(z)

    Domain
    ------
    - n: any complex number (order)
    - z: any complex number except z = 0 (argument)
    - The function is singular at z = 0

    Algorithm
    ---------
    The function is computed using the definition h_n^(1)(z) = j_n(z) + i*y_n(z),
    where j_n and y_n are computed using the existing spherical Bessel function
    implementations.

    Since the spherical Hankel function always has an imaginary component,
    the inputs are automatically promoted to complex types if they are real.

    Applications
    ------------
    The spherical Hankel function of the first kind appears in many contexts:

    - **Scattering theory**: Represents outgoing spherical waves
    - **Electromagnetic radiation**: Multipole fields radiated by oscillating charges
    - **Quantum mechanics**: Asymptotic solutions for scattering problems
    - **Acoustics**: Outgoing spherical sound waves
    - **Green's functions**: Fundamental solutions in spherical geometry

    The spherical Hankel functions h_n^(1) and h_n^(2) represent outgoing and
    incoming spherical waves respectively, and form a complete basis for
    solutions to the Helmholtz equation in spherical coordinates.

    Dtype Promotion
    ---------------
    - Real inputs (float16, bfloat16, float32, float64) are automatically
      converted to complex types (complex64, complex128)
    - Complex inputs are supported directly
    - The output is always complex

    Autograd Support
    ----------------
    Gradients are fully supported for both n and z when they require grad.
    The gradient with respect to z is computed analytically using the
    derivative formula. The gradient with respect to n is computed numerically.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the spherical Hankel function. Can be any complex number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the spherical Hankel function.
        Can be any complex number except zero.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The spherical Hankel function h_n^(1)(z) evaluated at the input values.
        Output dtype is always complex (complex64 or complex128).

    Examples
    --------
    Basic usage with integer order:

    >>> n = torch.tensor([0.0])
    >>> z = torch.tensor([1.0 + 0j])
    >>> spherical_hankel_1(n, z)
    tensor([0.8415-0.5403j])  # j_0(1) + i*y_0(1)

    Verification against definition:

    >>> import torchscience.special_functions as sf
    >>> n = torch.tensor([1.0 + 0j])
    >>> z = torch.tensor([2.0 + 0j])
    >>> h1 = sf.spherical_hankel_1(n, z)
    >>> j1 = sf.spherical_bessel_j(n.real, z.real)
    >>> y1 = sf.spherical_bessel_y(n.real, z.real)
    >>> expected = j1 + 1j * y1
    >>> torch.allclose(h1.real, expected.real) and torch.allclose(h1.imag, expected.imag)
    True

    Recurrence relation verification:

    >>> n = torch.tensor([2.0 + 0j])
    >>> z = torch.tensor([3.0 + 0j])
    >>> h_nm1 = sf.spherical_hankel_1(n - 1, z)
    >>> h_n = sf.spherical_hankel_1(n, z)
    >>> h_np1 = sf.spherical_hankel_1(n + 1, z)
    >>> lhs = h_nm1 + h_np1
    >>> rhs = (2*n + 1) / z * h_n
    >>> torch.allclose(lhs, rhs)
    True

    Autograd:

    >>> n = torch.tensor([1.0 + 0j])
    >>> z = torch.tensor([2.0 + 0j], requires_grad=True)
    >>> y = sf.spherical_hankel_1(n, z)
    >>> y.abs().backward()
    >>> z.grad is not None
    True

    .. warning:: Singularity at origin

       The spherical Hankel function is singular at z = 0. Evaluating
       at or very near z = 0 will produce NaN values.

    Notes
    -----
    - For applications involving standing waves, use the spherical Bessel
      functions j_n and y_n directly.
    - For incoming spherical waves, use spherical_hankel_2 (h_n^(2) = j_n - i*y_n).
    - The spherical Hankel functions are related to the Riccati-Hankel functions by
      xi_n(z) = z * h_n^(1)(z).

    See Also
    --------
    spherical_bessel_j : Spherical Bessel function of the first kind j_n
    spherical_bessel_y : Spherical Bessel function of the second kind y_n
    bessel_j : Bessel function of the first kind J_n
    bessel_y : Bessel function of the second kind Y_n
    """
    # Promote real inputs to complex
    if not n.is_complex():
        if n.dtype == torch.float16:
            n = n.to(torch.complex32)
        elif n.dtype == torch.bfloat16:
            # bfloat16 doesn't have a complex counterpart, promote to complex64
            n = n.to(torch.float32).to(torch.complex64)
        elif n.dtype == torch.float32:
            n = n.to(torch.complex64)
        else:
            n = n.to(torch.complex128)

    if not z.is_complex():
        if z.dtype == torch.float16:
            z = z.to(torch.complex32)
        elif z.dtype == torch.bfloat16:
            # bfloat16 doesn't have a complex counterpart, promote to complex64
            z = z.to(torch.float32).to(torch.complex64)
        elif z.dtype == torch.float32:
            z = z.to(torch.complex64)
        else:
            z = z.to(torch.complex128)

    return torch.ops.torchscience.spherical_hankel_1(n, z)
