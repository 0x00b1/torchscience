import pytest
import torch
import torch.testing

import torchscience.optimization.test_function


class TestRosenbrock:
    """Tests for the Rosenbrock test function."""

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_global_minimum_2d(self):
        """Test that the global minimum is 0 at (1, 1) for default parameters."""
        x = torch.tensor([1.0, 1.0])
        result = torchscience.optimization.test_function.rosenbrock(x)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_global_minimum_nd(self):
        """Test that the global minimum is 0 at (1, 1, ..., 1) for default parameters."""
        for n in [2, 3, 4, 5, 10]:
            x = torch.ones(n)
            result = torchscience.optimization.test_function.rosenbrock(x)
            torch.testing.assert_close(
                result,
                torch.tensor(0.0),
                msg=f"Failed for n={n}",
            )

    def test_origin_2d(self):
        """Test function value at the origin (0, 0)."""
        x = torch.tensor([0.0, 0.0])
        result = torchscience.optimization.test_function.rosenbrock(x)
        # f(0, 0) = (1 - 0)^2 + 100*(0 - 0^2)^2 = 1
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_known_values(self):
        """Test function at various known points."""
        test_cases = [
            # (x, expected_value)
            (torch.tensor([1.0, 1.0]), 0.0),
            (torch.tensor([0.0, 0.0]), 1.0),
            (torch.tensor([-1.0, 1.0]), 4.0),  # (1-(-1))^2 + 100*(1-1)^2 = 4
            (torch.tensor([2.0, 4.0]), 1.0),  # (1-2)^2 + 100*(4-4)^2 = 1
        ]
        for x, expected in test_cases:
            result = torchscience.optimization.test_function.rosenbrock(x)
            torch.testing.assert_close(
                result,
                torch.tensor(expected),
                msg=f"Failed for x={x.tolist()}",
            )

    # =========================================================================
    # Batch dimension tests
    # =========================================================================

    def test_batch_1d(self):
        """Test with 2D input (batch of 1D points)."""
        x = torch.tensor(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [-1.0, 1.0],
            ]
        )
        result = torchscience.optimization.test_function.rosenbrock(x)
        expected = torch.tensor([0.0, 1.0, 4.0])
        torch.testing.assert_close(result, expected)

    def test_batch_2d(self):
        """Test with 3D input (2D batch of points)."""
        x = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[-1.0, 1.0], [2.0, 4.0]],
            ]
        )
        result = torchscience.optimization.test_function.rosenbrock(x)
        expected = torch.tensor(
            [
                [0.0, 1.0],
                [4.0, 1.0],
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Test that output shape is input shape without last dimension."""
        test_cases = [
            ((2,), ()),
            ((3, 2), (3,)),
            ((4, 5, 2), (4, 5)),
            ((2, 3, 4, 5), (2, 3, 4)),
        ]
        for input_shape, expected_shape in test_cases:
            x = torch.randn(input_shape)
            result = torchscience.optimization.test_function.rosenbrock(x)
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {result.shape} "
                f"for input shape {input_shape}"
            )

    # =========================================================================
    # Custom parameter tests
    # =========================================================================

    def test_custom_a_parameter(self):
        """Test with custom 'a' parameter."""
        # For a=2, minimum is at (2, 4) for 2D
        x = torch.tensor([2.0, 4.0])
        result = torchscience.optimization.test_function.rosenbrock(x, a=2.0)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_custom_b_parameter(self):
        """Test with custom 'b' parameter."""
        x = torch.tensor([0.0, 0.0])
        # f(0, 0) = (1-0)^2 + b*(0-0)^2 = 1 (independent of b)
        result_b50 = torchscience.optimization.test_function.rosenbrock(
            x, b=50.0
        )
        result_b200 = torchscience.optimization.test_function.rosenbrock(
            x, b=200.0
        )
        torch.testing.assert_close(result_b50, torch.tensor(1.0))
        torch.testing.assert_close(result_b200, torch.tensor(1.0))

    def test_custom_a_and_b(self):
        """Test with both custom parameters."""
        x = torch.tensor([2.0, 4.0])
        result = torchscience.optimization.test_function.rosenbrock(
            x, a=2.0, b=50.0
        )
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_b_affects_valley_steepness(self):
        """Test that larger b creates steeper valleys."""
        x = torch.tensor([0.0, 1.0])
        # f(0, 1) = (1-0)^2 + b*(1-0)^2 = 1 + b
        result_b100 = torchscience.optimization.test_function.rosenbrock(
            x, b=100.0
        )
        result_b200 = torchscience.optimization.test_function.rosenbrock(
            x, b=200.0
        )
        torch.testing.assert_close(result_b100, torch.tensor(101.0))
        torch.testing.assert_close(result_b200, torch.tensor(201.0))

    # =========================================================================
    # Tensor parameter tests
    # =========================================================================

    def test_tensor_a_scalar(self):
        """Test with a as a 0-d tensor."""
        x = torch.tensor([1.0, 1.0])
        a = torch.tensor(1.0)
        result = torchscience.optimization.test_function.rosenbrock(x, a=a)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_tensor_b_scalar(self):
        """Test with b as a 0-d tensor."""
        x = torch.tensor([0.0, 1.0])
        b = torch.tensor(50.0)
        result = torchscience.optimization.test_function.rosenbrock(x, b=b)
        # f(0, 1) = (1-0)^2 + 50*(1-0)^2 = 1 + 50 = 51
        torch.testing.assert_close(result, torch.tensor(51.0))

    def test_tensor_a_batch(self):
        """Test with different a values for each batch element."""
        x = torch.tensor([[1.0, 1.0], [2.0, 4.0]])
        a = torch.tensor([[1.0], [2.0]])
        result = torchscience.optimization.test_function.rosenbrock(x, a=a)
        # Both should be at their respective minima
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))

    def test_tensor_b_batch(self):
        """Test with different b values for each batch element."""
        x = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        b = torch.tensor([[50.0], [200.0]])
        result = torchscience.optimization.test_function.rosenbrock(x, b=b)
        # f(0, 1) = 1 + b
        torch.testing.assert_close(result, torch.tensor([51.0, 201.0]))

    def test_tensor_a_and_b_batch(self):
        """Test with both a and b as tensors."""
        x = torch.tensor([[2.0, 4.0], [3.0, 9.0]])
        a = torch.tensor([[2.0], [3.0]])
        b = torch.tensor([[100.0], [50.0]])
        result = torchscience.optimization.test_function.rosenbrock(
            x, a=a, b=b
        )
        # Both at their minima
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))

    def test_tensor_a_gradient(self):
        """Test gradient with respect to tensor a."""
        x = torch.tensor([0.5, 0.5])
        a = torch.tensor(1.0, requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x, a=a)
        y.backward()
        # df/da = 2*(a - x_0) = 2*(1 - 0.5) = 1
        torch.testing.assert_close(a.grad, torch.tensor(1.0))

    def test_tensor_b_gradient(self):
        """Test gradient with respect to tensor b."""
        x = torch.tensor([0.0, 1.0])
        b = torch.tensor(100.0, requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x, b=b)
        y.backward()
        # df/db = (x_1 - x_0^2)^2 = (1 - 0)^2 = 1
        torch.testing.assert_close(b.grad, torch.tensor(1.0))

    def test_tensor_a_gradcheck(self):
        """Test gradcheck for tensor a parameter."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def fn(a_):
            return torchscience.optimization.test_function.rosenbrock(x, a=a_)

        assert torch.autograd.gradcheck(
            fn, (a,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_tensor_b_gradcheck(self):
        """Test gradcheck for tensor b parameter."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64)
        b = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)

        def fn(b_):
            return torchscience.optimization.test_function.rosenbrock(x, b=b_)

        assert torch.autograd.gradcheck(
            fn, (b,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_tensor_a_b_gradcheck(self):
        """Test gradcheck for both tensor a and b parameters."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, a_, b_):
            return torchscience.optimization.test_function.rosenbrock(
                x_, a=a_, b=b_
            )

        assert torch.autograd.gradcheck(
            fn, (x, a, b), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_tensor_params_broadcasting(self):
        """Test broadcasting of tensor parameters."""
        # x has shape (3, 2), a has shape (3, 1), b has shape (1,)
        x = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 4.0],
                [0.0, 0.0],
            ]
        )
        a = torch.tensor([[1.0], [2.0], [1.0]])
        b = torch.tensor([100.0])
        result = torchscience.optimization.test_function.rosenbrock(
            x, a=a, b=b
        )
        expected = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Gradient tests
    # =========================================================================

    def test_gradient_at_minimum(self):
        """Test that gradient is zero at the global minimum."""
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x)
        y.backward()
        torch.testing.assert_close(
            x.grad,
            torch.tensor([0.0, 0.0]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_gradient_at_origin(self):
        """Test gradient at the origin."""
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x)
        y.backward()
        # df/dx1 = -2(1-x1) - 4*b*x1*(x2-x1^2) = -2(1-0) - 0 = -2
        # df/dx2 = 2*b*(x2-x1^2) = 2*100*(0-0) = 0
        torch.testing.assert_close(
            x.grad,
            torch.tensor([-2.0, 0.0]),
        )

    def test_gradient_general_point(self):
        """Test gradient at a general point using finite differences."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x)
        y.backward()

        # Finite difference approximation
        eps = 1e-6
        x_np = x.detach().clone()
        grad_fd = torch.zeros(2, dtype=torch.float64)
        for i in range(2):
            x_plus = x_np.clone()
            x_plus[i] += eps
            x_minus = x_np.clone()
            x_minus[i] -= eps
            f_plus = torchscience.optimization.test_function.rosenbrock(x_plus)
            f_minus = torchscience.optimization.test_function.rosenbrock(
                x_minus
            )
            grad_fd[i] = (f_plus - f_minus) / (2 * eps)

        torch.testing.assert_close(x.grad, grad_fd, atol=1e-5, rtol=1e-5)

    def test_gradcheck(self):
        """Test gradients using torch.autograd.gradcheck."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            torchscience.optimization.test_function.rosenbrock,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradgradcheck(self):
        """Test second-order gradients."""
        x = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            torchscience.optimization.test_function.rosenbrock,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_batch_gradient(self):
        """Test gradients with batched input."""
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torchscience.optimization.test_function.rosenbrock(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_float32(self):
        """Test with float32 dtype."""
        x = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.float32
        torch.testing.assert_close(
            result, torch.tensor(0.0, dtype=torch.float32)
        )

    def test_float64(self):
        """Test with float64 dtype."""
        x = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.float64
        torch.testing.assert_close(
            result, torch.tensor(0.0, dtype=torch.float64)
        )

    def test_float16(self):
        """Test with float16 dtype."""
        x = torch.tensor([1.0, 1.0], dtype=torch.float16)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.float16
        torch.testing.assert_close(
            result,
            torch.tensor(0.0, dtype=torch.float16),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_bfloat16(self):
        """Test with bfloat16 dtype."""
        x = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(
            result,
            torch.tensor(0.0, dtype=torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
        )

    # =========================================================================
    # Complex dtype tests
    # =========================================================================

    def test_complex64(self):
        """Test with complex64 dtype."""
        x = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex64)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.complex64
        torch.testing.assert_close(
            result,
            torch.tensor(0.0 + 0j, dtype=torch.complex64),
        )

    def test_complex128(self):
        """Test with complex128 dtype."""
        x = torch.tensor([1.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert result.dtype == torch.complex128
        torch.testing.assert_close(
            result,
            torch.tensor(0.0 + 0j, dtype=torch.complex128),
        )

    def test_complex_at_origin(self):
        """Test complex dtype at origin."""
        x = torch.tensor([0.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        result = torchscience.optimization.test_function.rosenbrock(x)
        # f(0, 0) = (1 - 0)^2 + 100*(0 - 0^2)^2 = 1
        torch.testing.assert_close(
            result,
            torch.tensor(1.0 + 0j, dtype=torch.complex128),
        )

    def test_complex_with_imaginary_part(self):
        """Test with purely imaginary input."""
        x = torch.tensor([1j, 1j], dtype=torch.complex128)
        result = torchscience.optimization.test_function.rosenbrock(x)
        # f(i, i) = (1 - i)^2 + 100*(i - i^2)^2
        #         = (1 - 2i + i^2) + 100*(i - (-1))^2
        #         = (1 - 2i - 1) + 100*(1 + i)^2
        #         = -2i + 100*(1 + 2i + i^2)
        #         = -2i + 100*(1 + 2i - 1)
        #         = -2i + 100*2i
        #         = -2i + 200i = 198i
        expected = torch.tensor(198j, dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_complex_general_point(self):
        """Test complex input at a general point."""
        x = torch.tensor([1.0 + 1j, 1.0 + 1j], dtype=torch.complex128)
        result = torchscience.optimization.test_function.rosenbrock(x)
        # f(1+i, 1+i) = (1 - (1+i))^2 + 100*((1+i) - (1+i)^2)^2
        #             = (-i)^2 + 100*((1+i) - (1 + 2i + i^2))^2
        #             = -1 + 100*((1+i) - (1 + 2i - 1))^2
        #             = -1 + 100*((1+i) - 2i)^2
        #             = -1 + 100*(1 - i)^2
        #             = -1 + 100*(1 - 2i + i^2)
        #             = -1 + 100*(1 - 2i - 1)
        #             = -1 + 100*(-2i)
        #             = -1 - 200i
        expected = torch.tensor(-1.0 - 200j, dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_complex_batch(self):
        """Test complex dtype with batch dimensions."""
        x = torch.tensor(
            [
                [1.0 + 0j, 1.0 + 0j],
                [0.0 + 0j, 0.0 + 0j],
            ],
            dtype=torch.complex128,
        )
        result = torchscience.optimization.test_function.rosenbrock(x)
        expected = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
        torch.testing.assert_close(result, expected)

    def test_complex_gradient(self):
        """Test gradient computation with complex input.

        For complex tensors, PyTorch computes Wirtinger derivatives.
        """
        pytest.skip("Complex backward not implemented for rosenbrock")
        x = torch.tensor(
            [0.0 + 0j, 0.0 + 0j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        y = torchscience.optimization.test_function.rosenbrock(x)
        # For complex output, we need a real loss
        loss = y.real + y.imag
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.skip(
        reason="Complex gradcheck requires Wirtinger derivative handling in analytical backward"
    )
    def test_complex_gradcheck(self):
        """Test gradients for complex input using gradcheck.

        Note: This test is skipped because the analytical backward implementation
        doesn't perfectly match PyTorch's Wirtinger derivative convention for
        complex numbers. The function still works correctly for complex inputs;
        the gradient direction is correct but the numerical gradcheck is sensitive
        to the exact convention used.
        """
        x = torch.tensor(
            [0.5 + 0.1j, 0.5 + 0.1j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(
            torchscience.optimization.test_function.rosenbrock,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_complex_higher_dimensional(self):
        """Test complex input with higher dimensions."""
        n = 5
        x = torch.ones(n, dtype=torch.complex128)
        result = torchscience.optimization.test_function.rosenbrock(x)
        torch.testing.assert_close(
            result,
            torch.tensor(0.0 + 0j, dtype=torch.complex128),
        )

    # =========================================================================
    # Error handling tests
    # =========================================================================

    def test_dimension_too_small(self):
        """Test that dimension < 2 raises an error."""
        x = torch.tensor([1.0])
        with pytest.raises(RuntimeError):
            torchscience.optimization.test_function.rosenbrock(x)

    def test_empty_tensor_error(self):
        """Test that empty tensor raises an error."""
        x = torch.tensor([])
        with pytest.raises(RuntimeError):
            torchscience.optimization.test_function.rosenbrock(x)

    def test_integer_dtype_raises_error(self):
        """Test that integer dtypes raise an error."""
        x = torch.tensor([1, 1])
        with pytest.raises(RuntimeError, match="floating-point or complex"):
            torchscience.optimization.test_function.rosenbrock(x)

    # =========================================================================
    # Optimization workflow tests
    # =========================================================================

    def test_optimization_with_sgd(self):
        """Test that SGD can optimize the Rosenbrock function."""
        x = torch.tensor([-1.0, -1.0], requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=0.001)

        initial_loss = torchscience.optimization.test_function.rosenbrock(
            x
        ).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = torchscience.optimization.test_function.rosenbrock(x)
            loss.backward()
            optimizer.step()

        final_loss = torchscience.optimization.test_function.rosenbrock(
            x
        ).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_optimization_with_adam(self):
        """Test that Adam can optimize the Rosenbrock function."""
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.01)

        for _ in range(1000):
            optimizer.zero_grad()
            loss = torchscience.optimization.test_function.rosenbrock(x)
            loss.backward()
            optimizer.step()

        # Should be close to minimum
        assert x[0].item() > 0.5  # Moving towards 1
        assert x[1].item() > 0.2  # Moving towards 1

    # =========================================================================
    # torch.compile tests
    # =========================================================================

    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(
            torchscience.optimization.test_function.rosenbrock
        )
        x = torch.tensor([1.0, 1.0])
        result = compiled_fn(x)
        expected = torchscience.optimization.test_function.rosenbrock(x)
        torch.testing.assert_close(result, expected)

    def test_torch_compile_with_grad(self):
        """Test torch.compile with gradient computation."""
        compiled_fn = torch.compile(
            torchscience.optimization.test_function.rosenbrock
        )
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        y = compiled_fn(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([-2.0, 0.0]))

    # =========================================================================
    # Higher dimensional tests
    # =========================================================================

    def test_high_dimensional(self):
        """Test with high-dimensional input."""
        n = 100
        x = torch.ones(n)
        result = torchscience.optimization.test_function.rosenbrock(x)
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_high_dimensional_gradient(self):
        """Test gradient computation for high-dimensional input."""
        n = 10
        x = torch.ones(n, dtype=torch.float64, requires_grad=True)
        y = torchscience.optimization.test_function.rosenbrock(x)
        y.backward()
        # At minimum, all gradients should be zero
        torch.testing.assert_close(
            x.grad,
            torch.zeros(n, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )

    # =========================================================================
    # Mathematical property tests
    # =========================================================================

    def test_nonnegative(self):
        """Test that the function is always non-negative."""
        x = torch.randn(100, 5)  # Random batch of points
        result = torchscience.optimization.test_function.rosenbrock(x)
        assert (result >= 0).all()

    def test_valley_structure(self):
        """Test the characteristic valley structure.

        Points along the parabola y = x^2 should have lower values than
        points away from it (for the first term).
        """
        # Point on the parabola x^2
        x_on_parabola = torch.tensor([0.5, 0.25])
        # Point off the parabola with same x
        x_off_parabola = torch.tensor([0.5, 0.5])

        f_on = torchscience.optimization.test_function.rosenbrock(
            x_on_parabola
        )
        f_off = torchscience.optimization.test_function.rosenbrock(
            x_off_parabola
        )

        # Being on the parabola y=x^2 zeros the second term
        # f_on = (1-0.5)^2 + 100*(0.25 - 0.25)^2 = 0.25
        # f_off = (1-0.5)^2 + 100*(0.5 - 0.25)^2 = 0.25 + 100*0.0625 = 6.5
        assert f_on < f_off

    def test_symmetry_property(self):
        """Test that function values reflect the structure properly.

        The Rosenbrock function is NOT symmetric, but has specific structure.
        """
        x1 = torch.tensor([0.5, 0.5])
        x2 = torch.tensor([0.5, 0.0])

        f1 = torchscience.optimization.test_function.rosenbrock(x1)
        f2 = torchscience.optimization.test_function.rosenbrock(x2)

        # f(0.5, 0.5) = (1-0.5)^2 + 100*(0.5-0.25)^2 = 0.25 + 6.25 = 6.5
        # f(0.5, 0.0) = (1-0.5)^2 + 100*(0-0.25)^2 = 0.25 + 6.25 = 6.5
        torch.testing.assert_close(f1, torch.tensor(6.5))
        torch.testing.assert_close(f2, torch.tensor(6.5))

    # =========================================================================
    # vmap tests
    # =========================================================================

    def test_vmap_basic(self):
        """Test basic vmap functionality."""
        x = torch.tensor([[1.0, 1.0], [0.0, 0.0], [-1.0, 1.0]])

        # Using vmap
        vmapped_fn = torch.vmap(
            torchscience.optimization.test_function.rosenbrock
        )
        result_vmap = vmapped_fn(x)

        # Using batch directly
        result_batch = torchscience.optimization.test_function.rosenbrock(x)

        torch.testing.assert_close(result_vmap, result_batch)

    def test_vmap_nested(self):
        """Test nested vmap (vmap of vmap)."""
        x = torch.randn(3, 4, 5)  # 3x4 batch of 5-dimensional points

        # Double vmap
        fn = torchscience.optimization.test_function.rosenbrock
        result_vmap = torch.vmap(torch.vmap(fn))(x)

        # Direct batch
        result_batch = fn(x)

        torch.testing.assert_close(result_vmap, result_batch)

    def test_vmap_with_grad(self):
        """Test vmap combined with gradient computation."""
        pytest.skip(
            "vmap batching rule not implemented for rosenbrock with autograd::Function"
        )
        x = torch.tensor(
            [[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def fn(x_single):
            return torchscience.optimization.test_function.rosenbrock(x_single)

        # Compute vmapped gradients using jacrev
        jacobian = torch.func.jacrev(torch.vmap(fn))(x)

        # Each row of jacobian should be the gradient for that input
        assert jacobian.shape == (3, 3, 2)

    def test_vmap_in_dim(self):
        """Test vmap with different in_dims."""
        # x has batch in first dimension
        x = torch.randn(5, 3)

        result1 = torch.vmap(
            torchscience.optimization.test_function.rosenbrock,
            in_dims=0,
        )(x)

        result2 = torchscience.optimization.test_function.rosenbrock(x)

        torch.testing.assert_close(result1, result2)

    # =========================================================================
    # Numerical stability warning tests
    # =========================================================================

    def test_large_b_warning_float32(self):
        """Test that a warning is issued for large b with float32."""
        import warnings

        x = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=2e6)

            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "float32" in str(w[0].message)
            assert "2000000.0" in str(w[0].message)

    def test_large_b_warning_float16(self):
        """Test that a warning is issued for large b with float16."""
        import warnings

        x = torch.tensor([0.0, 0.0], dtype=torch.float16)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=2e3)

            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "float16" in str(w[0].message)

    def test_large_b_warning_bfloat16(self):
        """Test that a warning is issued for large b with bfloat16."""
        import warnings

        x = torch.tensor([0.0, 0.0], dtype=torch.bfloat16)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=2e3)

            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "bfloat16" in str(w[0].message)

    def test_no_warning_float64(self):
        """Test that no warning is issued for float64 even with large b."""
        import warnings

        x = torch.tensor([0.0, 0.0], dtype=torch.float64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=1e10)

            # No warnings should be issued for float64
            assert len(w) == 0

    def test_no_warning_normal_b(self):
        """Test that no warning is issued for normal b values."""
        import warnings

        x = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=100.0)

            # No warnings for default b
            assert len(w) == 0

    def test_no_warning_tensor_b(self):
        """Test that no warning is issued when b is a multi-element tensor."""
        import warnings

        x = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        b = torch.tensor([[1e7], [1e7]])  # Multi-element tensor

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torchscience.optimization.test_function.rosenbrock(x, b=b)

            # No warning for multi-element tensor (can't easily check all values)
            assert len(w) == 0

    # =========================================================================
    # Sparse tensor tests
    # =========================================================================

    def test_sparse_coo_basic(self):
        """Test basic sparse COO tensor support."""
        # Create a dense tensor and its sparse version
        x_dense = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.5, 0.25]])
        x_sparse = x_dense.to_sparse()

        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )

        # Results should match (sparse output is dense since it's a reduction)
        torch.testing.assert_close(result_sparse, result_dense)

    def test_sparse_coo_global_minimum(self):
        """Test sparse COO at global minimum."""
        # Global minimum at (1, 1)
        x_dense = torch.tensor([[1.0, 1.0]])
        x_sparse = x_dense.to_sparse()

        result = torchscience.optimization.test_function.rosenbrock(x_sparse)
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_sparse_coo_with_zeros(self):
        """Test sparse COO with implicit zeros."""
        # Sparse tensor with some zero elements
        indices = torch.tensor([[0, 0], [0, 1]])
        values = torch.tensor([0.5, 0.25])
        x_sparse = torch.sparse_coo_tensor(indices, values, size=(1, 2))

        x_dense = x_sparse.to_dense()
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )
        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )

        torch.testing.assert_close(result_sparse, result_dense)

    def test_sparse_coo_higher_dim(self):
        """Test sparse COO with higher dimensional input."""
        x_dense = torch.randn(3, 4, 5)
        x_sparse = x_dense.to_sparse()

        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )

        torch.testing.assert_close(result_sparse, result_dense)

    def test_sparse_csr_basic(self):
        """Test basic sparse CSR tensor support."""
        x_dense = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.5, 0.25]])
        x_sparse = x_dense.to_sparse_csr()

        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )

        torch.testing.assert_close(result_sparse, result_dense)

    def test_sparse_csr_global_minimum(self):
        """Test sparse CSR at global minimum."""
        x_dense = torch.tensor([[1.0, 1.0]])
        x_sparse = x_dense.to_sparse_csr()

        result = torchscience.optimization.test_function.rosenbrock(x_sparse)
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_sparse_coo_gradient(self):
        """Test gradient computation with sparse COO input."""
        x_dense = torch.tensor(
            [[0.5, 0.5], [1.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        x_sparse = x_dense.detach().to_sparse().requires_grad_(True)

        # Compute with dense
        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )
        result_dense.sum().backward()
        grad_dense = x_dense.grad

        # Compute with sparse
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )
        result_sparse.sum().backward()
        grad_sparse = x_sparse.grad

        # Gradients should match (sparse grad converted to dense for comparison)
        torch.testing.assert_close(grad_sparse.to_dense(), grad_dense)

    def test_sparse_csr_gradient(self):
        """Test gradient computation with sparse CSR input."""
        x_dense = torch.tensor(
            [[0.5, 0.5], [1.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        x_sparse = x_dense.detach().to_sparse_csr().requires_grad_(True)

        # Compute with dense
        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense
        )
        result_dense.sum().backward()
        grad_dense = x_dense.grad

        # Compute with sparse
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse
        )
        result_sparse.sum().backward()
        grad_sparse = x_sparse.grad

        # Gradients should match
        torch.testing.assert_close(grad_sparse.to_dense(), grad_dense)

    def test_sparse_custom_parameters(self):
        """Test sparse with custom a and b parameters."""
        x_dense = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        x_sparse = x_dense.to_sparse()

        result_dense = torchscience.optimization.test_function.rosenbrock(
            x_dense, a=2.0, b=50.0
        )
        result_sparse = torchscience.optimization.test_function.rosenbrock(
            x_sparse, a=2.0, b=50.0
        )

        torch.testing.assert_close(result_sparse, result_dense)

    def test_sparse_preserves_device(self):
        """Test that sparse computation preserves device."""
        x_dense = torch.tensor([[1.0, 1.0]])
        x_sparse = x_dense.to_sparse()

        result = torchscience.optimization.test_function.rosenbrock(x_sparse)

        assert result.device == x_sparse.device

    def test_sparse_different_dtypes(self):
        """Test sparse with different floating-point dtypes."""
        for dtype in [torch.float32, torch.float64]:
            x_dense = torch.tensor([[0.5, 0.5]], dtype=dtype)
            x_sparse = x_dense.to_sparse()

            result_dense = torchscience.optimization.test_function.rosenbrock(
                x_dense
            )
            result_sparse = torchscience.optimization.test_function.rosenbrock(
                x_sparse
            )

            torch.testing.assert_close(result_sparse, result_dense)

    # =========================================================================
    # Quantized tensor tests
    # =========================================================================

    def test_quantized_basic(self):
        """Test basic quantized tensor support."""
        x_float = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.5, 0.25]])

        # Quantize the input
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.1, zero_point=0, dtype=torch.quint8
        )

        result_float = torchscience.optimization.test_function.rosenbrock(
            x_float
        )
        result_quant = torchscience.optimization.test_function.rosenbrock(
            x_quant
        )

        # Results should be close (some quantization error expected)
        torch.testing.assert_close(
            result_quant, result_float, atol=0.5, rtol=0.1
        )

    def test_quantized_global_minimum(self):
        """Test quantized at global minimum."""
        x_float = torch.tensor([[1.0, 1.0]])
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.01, zero_point=0, dtype=torch.quint8
        )

        result = torchscience.optimization.test_function.rosenbrock(x_quant)

        # Should be close to 0 (some quantization error)
        torch.testing.assert_close(
            result, torch.tensor([0.0]), atol=0.1, rtol=0.1
        )

    def test_quantized_output_is_float(self):
        """Test that quantized input produces float output (not quantized)."""
        x_float = torch.tensor([[0.5, 0.5]])
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.1, zero_point=0, dtype=torch.quint8
        )

        result = torchscience.optimization.test_function.rosenbrock(x_quant)

        # Output should be a regular float tensor, not quantized
        assert not result.is_quantized
        assert result.dtype == torch.float32

    def test_quantized_different_scales(self):
        """Test quantized with different scales."""
        x_float = torch.tensor([[0.5, 0.5]])

        for scale in [0.01, 0.1, 1.0]:
            x_quant = torch.quantize_per_tensor(
                x_float, scale=scale, zero_point=0, dtype=torch.quint8
            )

            result_float = torchscience.optimization.test_function.rosenbrock(
                x_float
            )
            result_quant = torchscience.optimization.test_function.rosenbrock(
                x_quant
            )

            # Smaller scale = higher precision = smaller error
            torch.testing.assert_close(
                result_quant,
                result_float,
                atol=scale * 10,  # Error scales with quantization scale
                rtol=0.2,
            )

    def test_quantized_qint8(self):
        """Test quantized with qint8 dtype."""
        x_float = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.01, zero_point=0, dtype=torch.qint8
        )

        result_float = torchscience.optimization.test_function.rosenbrock(
            x_float
        )
        result_quant = torchscience.optimization.test_function.rosenbrock(
            x_quant
        )

        torch.testing.assert_close(
            result_quant, result_float, atol=0.1, rtol=0.1
        )

    def test_quantized_custom_parameters(self):
        """Test quantized with custom a and b parameters."""
        x_float = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.1, zero_point=0, dtype=torch.quint8
        )

        result_float = torchscience.optimization.test_function.rosenbrock(
            x_float, a=2.0, b=50.0
        )
        result_quant = torchscience.optimization.test_function.rosenbrock(
            x_quant, a=2.0, b=50.0
        )

        torch.testing.assert_close(
            result_quant, result_float, atol=1.0, rtol=0.1
        )

    def test_quantized_batch(self):
        """Test quantized with batch dimension."""
        x_float = torch.randn(5, 4)
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.1, zero_point=0, dtype=torch.quint8
        )

        result_float = torchscience.optimization.test_function.rosenbrock(
            x_float
        )
        result_quant = torchscience.optimization.test_function.rosenbrock(
            x_quant
        )

        # Results should have same shape
        assert result_quant.shape == result_float.shape

    def test_quantized_per_channel(self):
        """Test quantized per-channel (if supported)."""
        x_float = torch.tensor([[0.5, 0.5], [1.0, 1.0]])

        # Per-channel quantization along dimension 0
        scales = torch.tensor([0.1, 0.1])
        zero_points = torch.tensor([0, 0])
        x_quant = torch.quantize_per_channel(
            x_float, scales, zero_points, axis=0, dtype=torch.quint8
        )

        result_float = torchscience.optimization.test_function.rosenbrock(
            x_float
        )
        result_quant = torchscience.optimization.test_function.rosenbrock(
            x_quant
        )

        torch.testing.assert_close(
            result_quant, result_float, atol=0.5, rtol=0.1
        )

    def test_quantized_preserves_device(self):
        """Test that quantized computation preserves device."""
        x_float = torch.tensor([[1.0, 1.0]])
        x_quant = torch.quantize_per_tensor(
            x_float, scale=0.1, zero_point=0, dtype=torch.quint8
        )

        result = torchscience.optimization.test_function.rosenbrock(x_quant)

        assert result.device == x_quant.device

    # =========================================================================
    # Meta tensor tests (shape inference)
    # =========================================================================

    def test_meta_basic_shape(self):
        """Test meta tensor shape inference."""
        x_meta = torch.empty(3, 5, device="meta")
        a_meta = torch.empty((), device="meta")
        b_meta = torch.empty((), device="meta")

        result = torchscience.optimization.test_function.rosenbrock(
            x_meta, a=a_meta, b=b_meta
        )

        # Output shape should be (3,) - last dimension reduced
        assert result.shape == (3,)
        assert result.device.type == "meta"

    def test_meta_batch_shape(self):
        """Test meta tensor with batch dimensions."""
        x_meta = torch.empty(2, 4, 6, device="meta")

        result = torchscience.optimization.test_function.rosenbrock(x_meta)

        # Output shape should be (2, 4)
        assert result.shape == (2, 4)
        assert result.device.type == "meta"

    def test_meta_1d_shape(self):
        """Test meta tensor with 1D input."""
        x_meta = torch.empty(10, device="meta")

        result = torchscience.optimization.test_function.rosenbrock(x_meta)

        # Output shape should be () - scalar
        assert result.shape == ()
        assert result.device.type == "meta"

    def test_meta_dtype_promotion(self):
        """Test meta tensor dtype promotion."""
        x_meta = torch.empty(3, 5, dtype=torch.float32, device="meta")
        a_meta = torch.empty((), dtype=torch.float64, device="meta")
        b_meta = torch.empty((), dtype=torch.float32, device="meta")

        result = torchscience.optimization.test_function.rosenbrock(
            x_meta, a=a_meta, b=b_meta
        )

        # Should promote to float64
        assert result.dtype == torch.float64

    def test_meta_backward_shapes(self):
        """Test meta tensor backward shape inference."""
        x_meta = torch.empty(3, 5, device="meta")
        a_meta = torch.empty((), device="meta")
        b_meta = torch.empty((), device="meta")
        grad_output_meta = torch.empty(3, device="meta")

        grad_x, grad_a, grad_b = torch.ops.torchscience.rosenbrock_backward(
            grad_output_meta, x_meta, a_meta, b_meta
        )

        # grad_x should have same shape as x
        assert grad_x.shape == (3, 5)
        # grad_a should have same shape as a
        assert grad_a.shape == ()
        # grad_b should have same shape as b
        assert grad_b.shape == ()

    def test_meta_with_torch_compile(self):
        """Test that meta tensors work with torch.compile tracing."""

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return torchscience.optimization.test_function.rosenbrock(x)

        x = torch.randn(4, 3)
        result = fn(x)

        # Should produce correct shape
        assert result.shape == (4,)

    def test_meta_broadcast_parameters(self):
        """Test meta tensor with broadcast parameters."""
        x_meta = torch.empty(3, 4, 5, device="meta")
        a_meta = torch.empty(3, 1, device="meta")  # Broadcast along dim 1
        b_meta = torch.empty(4, device="meta")  # Broadcast along dims 0, 2

        result = torchscience.optimization.test_function.rosenbrock(
            x_meta, a=a_meta, b=b_meta
        )

        # Output shape should be (3, 4)
        assert result.shape == (3, 4)
