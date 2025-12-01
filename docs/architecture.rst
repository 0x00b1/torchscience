Architecture
============

This document describes the architectural patterns used in torchscience to enable maintainability and comprehensive PyTorch integration.

Layered Architecture
--------------------

Each operator is implemented across multiple layers:

.. code-block:: text

    Python API (src/torchscience/)
        │
        ▼
    torch.ops.torchscience.* (registered operators)
        │
        ▼
    Dispatcher (dispatch by device, dtype, layout)
        │
        ├─► CPU (csrc/cpu/)
        ├─► CUDA (csrc/cuda/)
        ├─► Autograd (csrc/autograd/)
        ├─► Autocast (csrc/autocast/)
        ├─► Meta (csrc/meta/)
        ├─► Sparse (csrc/sparse/)
        └─► Quantized (csrc/quantized/)
        │
        ▼
    Core Implementation (csrc/impl/)
        Device-agnostic, header-only C++

Core Implementation Pattern
---------------------------

Mathematical implementations live in ``csrc/impl/`` as header-only, device-agnostic C++:

.. code-block:: cpp

    // csrc/impl/special_functions/gamma.h
    namespace torchscience::impl::special_functions {

    template <typename T>
    C10_HOST_DEVICE T gamma(T z) {
        // Implementation using Lanczos approximation
        // Works on both CPU and CUDA
    }

    } // namespace torchscience::impl::special_functions

Key properties:

* ``C10_HOST_DEVICE`` enables compilation for both CPU and CUDA
* Template-based for dtype flexibility
* Header-only for inlining and optimization
* Extensively documented with mathematical derivations

Dispatcher Registration Pattern
-------------------------------

Operators register implementations for different dispatch keys:

.. code-block:: cpp

    // Register the operator schema
    TORCH_LIBRARY(torchscience, m) {
        m.def("gamma(Tensor z) -> Tensor");
        m.def("gamma_backward(Tensor grad, Tensor z) -> Tensor");
    }

    // CPU implementation
    TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
        m.impl("gamma", &cpu::special_functions::gamma);
    }

    // CUDA implementation
    TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
        m.impl("gamma", &cuda::special_functions::gamma);
    }

    // Autograd implementation
    TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
        m.impl("gamma", &autograd::special_functions::gamma);
    }

Macro-Driven Code Generation
-----------------------------

Boilerplate is reduced through macros for common patterns:

.. code-block:: cpp

    // Unary operator on CPU
    CPU_UNARY_OPERATOR(special_functions, gamma, z)

    // Generates forward pass, TensorIterator setup, dtype dispatch

Similar macros exist for:

* ``CPU_BINARY_OPERATOR``, ``CPU_TERNARY_OPERATOR``, etc.
* ``CUDA_UNARY_OPERATOR``, ``CUDA_BINARY_OPERATOR``, etc.
* ``AUTOGRAD_UNARY_OPERATOR``, ``AUTOGRAD_BINARY_OPERATOR``, etc.
* ``META_UNARY_OPERATOR``, ``AUTOCAST_UNARY_OPERATOR``, etc.

Autograd Pattern
----------------

Gradient computation uses PyTorch's autograd machinery:

.. code-block:: cpp

    class GammaFunction : public torch::autograd::Function<GammaFunction> {
    public:
        static Tensor forward(AutogradContext* ctx, Tensor z) {
            ctx->save_for_backward({z});
            return torch::ops::torchscience::gamma(z);
        }

        static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto z = saved[0];
            auto grad = grad_outputs[0];
            // d/dz Γ(z) = Γ(z) * ψ(z)
            return {torch::ops::torchscience::gamma_backward(grad, z)};
        }
    };

Composite Operation Pattern
---------------------------

Some operators are implemented purely in terms of existing PyTorch operations:

.. code-block:: python

    # Registered with CompositeImplicitAutograd dispatch key
    def sine_wave(
        frequency: Tensor,
        num_samples: int,
        sample_rate: float,
    ) -> Tensor:
        t = torch.arange(num_samples, dtype=frequency.dtype, device=frequency.device)
        t = t / sample_rate
        return torch.sin(2 * math.pi * frequency * t)

Benefits:

* Autograd comes automatically from composition
* Works on all devices PyTorch supports
* Automatically benefits from PyTorch compiler improvements

Python API Pattern
------------------

The Python layer provides documentation and ergonomic APIs:

.. code-block:: python

    def gamma(z: Tensor) -> Tensor:
        r"""
        Gamma function.

        .. math::
            \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt

        Parameters
        ----------
        z : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Gamma function values.

        Examples
        --------
        >>> torchscience.special_functions.gamma(torch.tensor([1.0, 2.0, 3.0]))
        tensor([1., 1., 2.])
        """
        return torch.ops.torchscience.gamma(z)

Testing Pattern
---------------

Tests use a mixin-based architecture for comprehensive coverage:

.. code-block:: python

    class TestGamma(
        AutogradMixin,
        TorchCompileMixin,
        VmapMixin,
        SparseMixin,
        QuantizedMixin,
        MetaTensorMixin,
        AutocastMixin,
        SpecialValueMixin,
        RecurrenceMixin,
        OpTestCase,
    ):
        """Gamma function tests."""

        def reference(self, z):
            return scipy.special.gamma(z)

        # Mixin methods automatically test:
        # - Gradient correctness (first and second order)
        # - torch.compile compatibility
        # - vmap batching
        # - Sparse tensor support
        # - Quantized tensor support
        # - Meta tensor shape inference
        # - Autocast behavior
        # - Special value accuracy
        # - Mathematical identities

Implementation Patterns Summary
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Pattern
     - Description
     - Example Operators
   * - Elementwise (1-4 ary)
     - Point-wise operations with broadcasting
     - ``gamma``, ``chebyshev_polynomial_t``, ``incomplete_beta``, ``hypergeometric_2_f_1``, ``rosenbrock``
   * - Reduction
     - Aggregate along dimensions
     - ``entropy``, ``kurtosis``
   * - Creation
     - Generate tensors from parameters
     - ``rectangular_window``, ``pink_noise``
   * - Transformation
     - Reshape or reorder data
     - ``hilbert_transform``, ``inverse_hilbert_transform``, ``discrete_wavelet_transform``
   * - Pairwise
     - Operate on pairs of elements
     - ``minkowski_distance``, ``kd_tree_query``
   * - Scatter/Gather
     - Index-based operations
     - ``histogram2d``, ``binned_statistic``
   * - Iterative
     - Algorithms requiring loops
     - ``newton_raphson``, ``shortest_path``
   * - Higher-order
     - Takes callable as argument
     - ``adaptive_quadrature``, ``brent_minimization``, ``brent_root``, ``runge_kutta_step``
   * - Composite
     - Built from PyTorch primitives
     - ``sine_wave``, ``butterworth_analog_bandpass_filter``

PyTorch Compatibility
---------------------

Every operator supports:

Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^

* **First-order gradients**: Full backward pass support with ``torch.autograd.grad`` and ``.backward()``
* **Higher-order gradients**: Second and higher derivatives via ``torch.autograd.grad`` with ``create_graph=True``
* **Complex gradients**: Wirtinger calculus for proper holomorphic and non-holomorphic function derivatives
* **Forward-mode AD**: Compatible with ``torch.func.jvp`` for efficient Jacobian-vector products

Compilation
^^^^^^^^^^^

* **torch.compile**: All operators work with ``torch.compile()`` for graph optimization and kernel fusion
* **Meta tensors**: Shape and dtype inference without computation for ahead-of-time compilation

Vectorization and Batching
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **torch.vmap**: Batched operations via ``torch.func.vmap`` for efficient parallelism
* **Broadcasting**: NumPy-style broadcasting semantics
* **TensorIterator**: Automatic vectorization on CPU

Data Types
^^^^^^^^^^

* **Floating point**: float16, bfloat16, float32, float64
* **Complex**: complex64, complex128
* **Integer promotion**: Automatic promotion to floating point where required
* **Quantized**: quint8, qint8 for inference optimization

Hardware Backends
^^^^^^^^^^^^^^^^^

* **CPU**: Optimized implementations with vectorization
* **CUDA**: GPU acceleration with the same implementation as CPU (device-agnostic kernels)
* **MPS**: Apple Silicon support (planned)
* **XLA**: TPU support via torch-xla (planned)

Tensor Layouts
^^^^^^^^^^^^^^

* **Strided**: Standard dense tensors
* **Sparse COO**: Coordinate format sparse tensors
* **Sparse CSR**: Compressed sparse row format
* **Nested tensors**: Variable-length sequences (planned)
* **Named tensors**: Dimension names for clarity (planned)

Mixed Precision
^^^^^^^^^^^^^^^

* **Autocast**: Automatic dtype management for mixed-precision training
* **Numerical stability**: Appropriate precision promotion for sensitive computations

Ecosystem Integration
^^^^^^^^^^^^^^^^^^^^^

* **tensordict**: Compatible with tensordict for structured data (planned)
* **functorch**: Full compatibility with functional transformations
* **torch.fx**: Graph transformable for custom compiler passes
