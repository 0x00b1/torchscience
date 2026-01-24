import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestCUDAWindowParity:
    """Test that CUDA and CPU implementations produce identical results."""

    @pytest.mark.parametrize(
        "window_fn",
        [
            "hann_window",
            "hamming_window",
            "blackman_window",
            "bartlett_window",
            "cosine_window",
            "nuttall_window",
            "triangular_window",
            "welch_window",
            "parzen_window",
            "blackman_harris_window",
            "flat_top_window",
            "sine_window",
            "bartlett_hann_window",
            "lanczos_window",
        ],
    )
    @pytest.mark.parametrize("n", [0, 1, 10, 64, 256, 1000])
    def test_parameterless_window_parity(self, window_fn, n):
        import torchscience.window_function as wf

        fn = getattr(wf, window_fn)

        cpu_result = fn(n, device="cpu", dtype=torch.float64)
        cuda_result = fn(n, device="cuda", dtype=torch.float64)

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "window_fn",
        [
            "periodic_hann_window",
            "periodic_hamming_window",
            "periodic_blackman_window",
            "periodic_bartlett_window",
            "periodic_cosine_window",
            "periodic_nuttall_window",
            "periodic_triangular_window",
            "periodic_welch_window",
            "periodic_parzen_window",
            "periodic_blackman_harris_window",
            "periodic_flat_top_window",
            "periodic_sine_window",
            "periodic_bartlett_hann_window",
            "periodic_lanczos_window",
        ],
    )
    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_periodic_window_parity(self, window_fn, n):
        import torchscience.window_function as wf

        fn = getattr(wf, window_fn)

        cpu_result = fn(n, device="cpu", dtype=torch.float64)
        cuda_result = fn(n, device="cuda", dtype=torch.float64)

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_gaussian_window_parity(self, n):
        from torchscience.window_function import gaussian_window

        std = 0.5
        cpu_result = gaussian_window(n, std, device="cpu", dtype=torch.float64)
        cuda_result = gaussian_window(
            n, std, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_periodic_gaussian_window_parity(self, n):
        from torchscience.window_function import periodic_gaussian_window

        std = 0.5
        cpu_result = periodic_gaussian_window(
            n, std, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_gaussian_window(
            n, std, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_kaiser_window_parity(self, n):
        from torchscience.window_function import kaiser_window

        beta = 14.0
        cpu_result = kaiser_window(n, beta, device="cpu", dtype=torch.float64)
        cuda_result = kaiser_window(
            n, beta, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_periodic_kaiser_window_parity(self, n):
        from torchscience.window_function import periodic_kaiser_window

        beta = 14.0
        cpu_result = periodic_kaiser_window(
            n, beta, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_kaiser_window(
            n, beta, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_tukey_window_parity(self, n):
        from torchscience.window_function import tukey_window

        alpha = 0.5
        cpu_result = tukey_window(n, alpha, device="cpu", dtype=torch.float64)
        cuda_result = tukey_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64, 256])
    def test_periodic_tukey_window_parity(self, n):
        from torchscience.window_function import periodic_tukey_window

        alpha = 0.5
        cpu_result = periodic_tukey_window(
            n, alpha, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_tukey_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_general_cosine_window_parity(self, n):
        from torchscience.window_function import general_cosine_window

        # Hann coefficients
        coeffs = [0.5, 0.5]
        cpu_result = general_cosine_window(
            n, coeffs, device="cpu", dtype=torch.float64
        )
        cuda_result = general_cosine_window(
            n, coeffs, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_periodic_general_cosine_window_parity(self, n):
        from torchscience.window_function import periodic_general_cosine_window

        # Blackman coefficients
        coeffs = [0.42, 0.5, 0.08]
        cpu_result = periodic_general_cosine_window(
            n, coeffs, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_general_cosine_window(
            n, coeffs, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_general_hamming_window_parity(self, n):
        from torchscience.window_function import general_hamming_window

        alpha = 0.54
        cpu_result = general_hamming_window(
            n, alpha, device="cpu", dtype=torch.float64
        )
        cuda_result = general_hamming_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_periodic_general_hamming_window_parity(self, n):
        from torchscience.window_function import (
            periodic_general_hamming_window,
        )

        alpha = 0.54
        cpu_result = periodic_general_hamming_window(
            n, alpha, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_general_hamming_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_exponential_window_parity(self, n):
        from torchscience.window_function import exponential_window

        tau = 0.5
        cpu_result = exponential_window(
            n, tau, device="cpu", dtype=torch.float64
        )
        cuda_result = exponential_window(
            n, tau, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_periodic_exponential_window_parity(self, n):
        from torchscience.window_function import periodic_exponential_window

        tau = 0.5
        cpu_result = periodic_exponential_window(
            n, tau, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_exponential_window(
            n, tau, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_hann_poisson_window_parity(self, n):
        from torchscience.window_function import hann_poisson_window

        alpha = 2.0
        cpu_result = hann_poisson_window(
            n, alpha, device="cpu", dtype=torch.float64
        )
        cuda_result = hann_poisson_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 10, 64])
    def test_periodic_hann_poisson_window_parity(self, n):
        from torchscience.window_function import periodic_hann_poisson_window

        alpha = 2.0
        cpu_result = periodic_hann_poisson_window(
            n, alpha, device="cpu", dtype=torch.float64
        )
        cuda_result = periodic_hann_poisson_window(
            n, alpha, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )


class TestCUDADtypeSupport:
    """Test that CUDA windows work with various data types."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_hann_dtype_support(self, dtype):
        from torchscience.window_function import hann_window

        n = 64
        result = hann_window(n, device="cuda", dtype=dtype)

        assert result.dtype == dtype
        assert result.device.type == "cuda"
        assert result.shape == (n,)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_gaussian_dtype_support(self, dtype):
        from torchscience.window_function import gaussian_window

        n = 64
        std = 0.5
        result = gaussian_window(n, std, device="cuda", dtype=dtype)

        assert result.dtype == dtype
        assert result.device.type == "cuda"
        assert result.shape == (n,)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_kaiser_dtype_support(self, dtype):
        from torchscience.window_function import kaiser_window

        n = 64
        beta = 14.0
        result = kaiser_window(n, beta, device="cuda", dtype=dtype)

        assert result.dtype == dtype
        assert result.device.type == "cuda"
        assert result.shape == (n,)


class TestCUDALargeWindows:
    """Test CUDA performance with large window sizes."""

    @pytest.mark.parametrize("n", [10000, 100000])
    def test_large_hann_window(self, n):
        from torchscience.window_function import hann_window

        cpu_result = hann_window(n, device="cpu", dtype=torch.float64)
        cuda_result = hann_window(n, device="cuda", dtype=torch.float64)

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [10000, 100000])
    def test_large_gaussian_window(self, n):
        from torchscience.window_function import gaussian_window

        std = 0.5
        cpu_result = gaussian_window(n, std, device="cpu", dtype=torch.float64)
        cuda_result = gaussian_window(
            n, std, device="cuda", dtype=torch.float64
        )

        torch.testing.assert_close(
            cpu_result,
            cuda_result.cpu(),
            rtol=1e-10,
            atol=1e-10,
        )
