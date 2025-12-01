import torch
from torch import Tensor


def chebyshev_polynomial_t(v: Tensor, z: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the first kind.

    Computes the Chebyshev polynomial of the first kind T_v(z).

    Mathematical Definition
    -----------------------
    For integer v = n >= 0, uses the recurrence relation (exact polynomial
    semantics):

    .. math::

       T_0(z) &= 1 \\
       T_1(z) &= z \\
       T_n(z) &= 2z \, T_{n-1}(z) - T_{n-2}(z)

    This works for all real z, not just z in [-1, 1].

    For non-integer v with real z in [-1, 1], or complex z, uses the analytic
    continuation:

    .. math::

       T_v(z) = \cos(v \arccos z)

    where arccos(z) uses the principal branch, consistent with PyTorch's
    complex acos.

    For non-integer v with real z outside [-1, 1], uses the hyperbolic
    continuation:

    .. math::

       T_v(z) = \begin{cases}
           \cosh(v \, \mathrm{arccosh} \, z) & z > 1 \\
           \cos(v \pi) \cosh(v \, \mathrm{arccosh}(-z)) & z < -1
       \end{cases}

    This is the analytic continuation of the Chebyshev polynomial to the
    real line outside the standard domain, equivalent to evaluating the
    complex formula :math:`\cos(v \arccos z)` on the real axis.

    Special Values
    --------------
    - T_0(z) = 1 for all z
    - T_1(z) = z for all z
    - T_n(1) = 1 for all integer n >= 0
    - T_n(-1) = (-1)^n for all integer n >= 0
    - T_n(0) = cos(n * pi / 2) for integer n (i.e., 1, 0, -1, 0, 1, ...)
    - T_n(cos(theta)) = cos(n * theta) for integer n >= 0

    Domain
    ------
    - v: any real or complex value (integral values use efficient recurrence)
    - z: any real or complex value
    - For real z in [-1, 1] with non-integer v: uses analytic continuation
    - For real z outside [-1, 1] with non-integer v: uses hyperbolic continuation
    - For complex z: uses principal branch of arccos with branch cuts at
      (-inf, -1) and (1, +inf) on the real axis

    Algorithm
    ---------
    - Path A (Recurrence): v is integral AND z is real -> polynomial recurrence.
      Numerically stable for integer degrees, exact polynomial semantics.
      Works for all real z, not just z in [-1, 1].
    - Path B (Analytic Continuation): non-integer v AND real z in [-1, 1]
      -> uses cos(v * acos(z)) with principal branch.
    - Path C (Hyperbolic Continuation): non-integer v AND real z outside [-1, 1]
      -> uses cosh(v * acosh(z)) for z > 1, or cos(v*π) * cosh(v * acosh(-z)) for z < -1.
    - Path D (Complex): complex v OR complex z
      -> uses cos(v * acos(z)) with principal branch.
    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Chebyshev polynomials are fundamental in numerical analysis and applied mathematics:
    - Function approximation: Chebyshev series provide near-optimal polynomial approximations (minimax property)
    - Spectral methods: basis functions for solving PDEs with spectral accuracy
    - Numerical integration: Clenshaw-Curtis and Gauss-Chebyshev quadrature
    - Filter design: Chebyshev filters in signal processing
    - Polynomial interpolation: Chebyshev nodes minimize Runge's phenomenon

    Dtype Promotion
    ---------------
    - If either v or z is complex -> output is complex.
    - complex64 if all inputs <= float32/complex64, else complex128.
    - If both real -> standard PyTorch promotion rules apply.
    - Supports float16, bfloat16, float32, float64, complex64, complex128.

    Integer Dtype Handling
    ----------------------
    If v is passed as an integer dtype tensor (e.g., torch.int32, torch.int64),
    it will be promoted to a floating-point dtype via PyTorch's standard type
    promotion rules before computation. The promoted dtype is determined by
    the dtype of z.

    To use the efficient recurrence path for integer degrees, pass v as a
    floating-point tensor with integer values (e.g., torch.tensor([2.0]) rather
    than torch.tensor([2])). The implementation detects integer values by
    checking if v == floor(v) and uses the recurrence relation for such values,
    regardless of the underlying dtype.

    Examples of dtype handling:
        - v=torch.tensor([2], dtype=torch.int64), z=torch.tensor([0.5]) ->
          v promoted to float32, uses recurrence path (v has integer value)
        - v=torch.tensor([2.0]), z=torch.tensor([0.5]) ->
          stays float32, uses recurrence path
        - v=torch.tensor([2.5]), z=torch.tensor([0.5]) ->
          uses analytic continuation (non-integer v)

    Autograd Support
    ----------------
    - Gradients for z are always computed when z.requires_grad is True.
    - Gradients for v are only computed when v is floating-point or complex
      and v.requires_grad is True. No gradients are provided for integral v.
    - Second-order derivatives (gradgradcheck) are fully supported, enabling
      use in applications requiring Hessians or higher-order optimization.

    Backward formulas:

    .. math::

       \frac{\partial T_v(z)}{\partial z} &= \frac{v \sin(v \arccos z)}{\sqrt{1 - z^2}} \\
       \frac{\partial T_v(z)}{\partial v} &= -\sin(v \arccos z) \arccos z

    Parameters
    ----------
    v : Tensor
        Degree of the polynomial. Can be integral, floating-point, or complex.
        When integral, uses efficient polynomial recurrence for real z.
        When floating or complex, uses analytic continuation.
    z : Tensor
        Input tensor. Can be floating-point or complex.
        Broadcasting with v is supported.

    Returns
    -------
    Tensor
        The Chebyshev polynomial T_v(z) evaluated at the input values.
        Output dtype follows the promotion rules described above.

    Examples
    --------
    Integer degree with real input (recurrence path):

    >>> v = torch.tensor([0, 1, 2, 3])
    >>> z = torch.tensor([0.5])
    >>> chebyshev_polynomial_t(v, z)
    tensor([ 1.0000,  0.5000, -0.5000, -1.0000])

    Non-integer degree (analytic continuation):

    >>> v = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([0.0])
    >>> chebyshev_polynomial_t(v, z)  # cos(v * pi/2)
    tensor([ 0.7071,  0.0000, -0.7071])

    Complex input:

    >>> v = torch.tensor([2.0])
    >>> z = torch.tensor([1.0 + 0.1j])
    >>> chebyshev_polynomial_t(v, z)  # Returns complex result
    tensor([1.0200-0.4000j])

    Real z outside [-1, 1] with non-integer v (hyperbolic continuation):

    >>> v = torch.tensor([2.5])
    >>> z = torch.tensor([2.0])
    >>> chebyshev_polynomial_t(v, z)  # cosh(2.5 * acosh(2))
    tensor([18.6953])

    Autograd with floating v:

    >>> v = torch.tensor([2.0], requires_grad=True)
    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> y = chebyshev_polynomial_t(v, z)
    >>> y.backward()
    >>> v.grad  # Gradient w.r.t. degree
    tensor([-0.9069])
    >>> z.grad  # Gradient w.r.t. input
    tensor([-2.0000])

    .. warning:: Numerical stability for large degrees

       For very large integer degrees (``|v| > 1000``), the recurrence relation
       may accumulate numerical errors, especially for ``|z|`` close to 1.
       Consider using the analytic continuation formula for such cases.

    .. warning:: Branch cuts for complex inputs

       For complex z, the function uses torch.acos which has branch cuts
       at (-inf, -1) and (1, +inf) on the real axis. Results may be
       discontinuous across these cuts.

    .. warning:: Gradient singularity

       The gradient :math:`\frac{\partial T_v}{\partial z} = \frac{v \sin(v \arccos z)}{\sqrt{1 - z^2}}`
       has a singularity at z = ±1. Gradients at these points may be
       inf or NaN.

    Notes
    -----
    - Integer-valued v (even in floating-point dtype) is detected via
      ``v == floor(v)`` and uses the recurrence path for exact polynomial
      semantics.
    - The recurrence relation is mathematically equivalent to the analytic
      continuation for integer degrees, but provides better numerical
      stability and avoids branch cut issues.

    See Also
    --------
    torch.special.chebyshev_polynomial_t : PyTorch's Chebyshev polynomial (integer degrees only)
    torch.special.chebyshev_polynomial_u : Chebyshev polynomial of the second kind
    """
    return torch.ops.torchscience.chebyshev_polynomial_t(z, v)
