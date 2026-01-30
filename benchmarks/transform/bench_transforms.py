"""Benchmarks for transform functions.

This module benchmarks torchscience transform functions and compares against
scipy/numpy baselines where applicable.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import torch

# scipy imports - handle optional dependency
try:
    from scipy import fft as scipy_fft

    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False

try:
    from skimage.transform import radon as skimage_radon

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# torchscience imports
import torchscience.transform as T


def benchmark(
    func: Callable,
    *args: Any,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """Run a simple benchmark on a function.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args : Any
        Positional arguments to pass to func.
    warmup : int, optional
        Number of warmup iterations. Default is 3.
    iterations : int, optional
        Number of timed iterations. Default is 10.
    **kwargs : Any
        Keyword arguments to pass to func.

    Returns
    -------
    dict
        Dictionary with timing statistics:
        - 'mean': Mean time in seconds
        - 'std': Standard deviation in seconds
        - 'min': Minimum time in seconds
        - 'max': Maximum time in seconds
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.3f}us"
    elif seconds < 1:
        return f"{seconds * 1e3:.3f}ms"
    else:
        return f"{seconds:.3f}s"


def print_comparison(
    name: str,
    ts_time: dict[str, float],
    baseline_time: dict[str, float] | None = None,
    baseline_name: str = "scipy",
) -> None:
    """Print benchmark comparison results."""
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"  torchscience: {format_time(ts_time['mean'])} +/- {format_time(ts_time['std'])}"
    )
    if baseline_time is not None:
        print(
            f"  {baseline_name}:        {format_time(baseline_time['mean'])} +/- {format_time(baseline_time['std'])}"
        )
        speedup = baseline_time["mean"] / ts_time["mean"]
        if speedup >= 1:
            print(f"  Speedup:      {speedup:.2f}x faster")
        else:
            print(f"  Speedup:      {1 / speedup:.2f}x slower")


class BenchTransforms:
    """Benchmarks for transform functions."""

    def __init__(
        self, warmup: int = 3, iterations: int = 10, device: str = "cpu"
    ):
        """Initialize benchmark runner.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup iterations. Default is 3.
        iterations : int, optional
            Number of timed iterations. Default is 10.
        device : str, optional
            Device to run benchmarks on. Default is "cpu".
        """
        self.warmup = warmup
        self.iterations = iterations
        self.device = device

    def _bench(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Run benchmark with configured settings."""
        return benchmark(
            func,
            *args,
            warmup=self.warmup,
            iterations=self.iterations,
            **kwargs,
        )

    # ============== Fourier Transforms ==============

    def bench_cosine_transform(self, n: int = 1024) -> None:
        """Benchmark cosine_transform vs scipy.fft.dct.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 1024.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.cosine_transform, x, type=2)

        scipy_time = None
        if SCIPY_FFT_AVAILABLE and self.device == "cpu":
            x_np = x.numpy()
            scipy_time = self._bench(scipy_fft.dct, x_np, type=2)

        print_comparison(
            f"cosine_transform (DCT-II, n={n})", ts_time, scipy_time
        )

    def bench_sine_transform(self, n: int = 1024) -> None:
        """Benchmark sine_transform vs scipy.fft.dst.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 1024.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.sine_transform, x, type=2)

        scipy_time = None
        if SCIPY_FFT_AVAILABLE and self.device == "cpu":
            x_np = x.numpy()
            scipy_time = self._bench(scipy_fft.dst, x_np, type=2)

        print_comparison(
            f"sine_transform (DST-II, n={n})", ts_time, scipy_time
        )

    def bench_hartley_transform(self, n: int = 1024) -> None:
        """Benchmark hartley_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 1024.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.hartley_transform, x)

        print_comparison(f"hartley_transform (n={n})", ts_time)

    def bench_hilbert_transform(self, n: int = 1024) -> None:
        """Benchmark hilbert_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 1024.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.hilbert_transform, x)

        print_comparison(f"hilbert_transform (n={n})", ts_time)

    # ============== Wavelet Transforms ==============

    def bench_discrete_wavelet_transform(self, n: int = 1024) -> None:
        """Benchmark discrete_wavelet_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 1024.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.discrete_wavelet_transform, x, wavelet="haar")

        print_comparison(f"discrete_wavelet_transform (haar, n={n})", ts_time)

    def bench_continuous_wavelet_transform(
        self, n: int = 512, n_scales: int = 32
    ) -> None:
        """Benchmark continuous_wavelet_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 512.
        n_scales : int, optional
            Number of scales. Default is 32.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)
        scales = torch.logspace(
            0, 2, n_scales, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.continuous_wavelet_transform, x, scales)

        print_comparison(
            f"continuous_wavelet_transform (n={n}, scales={n_scales})", ts_time
        )

    # ============== Time-Frequency Transforms ==============

    def bench_short_time_fourier_transform(
        self, n: int = 4096, n_fft: int = 256
    ) -> None:
        """Benchmark short_time_fourier_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 4096.
        n_fft : int, optional
            FFT size. Default is 256.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.short_time_fourier_transform, x, n_fft=n_fft)

        print_comparison(
            f"short_time_fourier_transform (n={n}, n_fft={n_fft})", ts_time
        )

    def bench_gabor_transform(self, n: int = 4096, n_fft: int = 256) -> None:
        """Benchmark gabor_transform.

        Parameters
        ----------
        n : int, optional
            Signal length. Default is 4096.
        n_fft : int, optional
            FFT size. Default is 256.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)

        ts_time = self._bench(T.gabor_transform, x, n_fft=n_fft)

        print_comparison(f"gabor_transform (n={n}, n_fft={n_fft})", ts_time)

    # ============== Imaging Transforms ==============

    def bench_radon_transform(
        self, size: int = 64, n_angles: int = 90
    ) -> None:
        """Benchmark radon_transform vs skimage.

        Parameters
        ----------
        size : int, optional
            Image size (size x size). Default is 64.
        n_angles : int, optional
            Number of projection angles. Default is 90.
        """
        image = torch.randn(
            size, size, dtype=torch.float64, device=self.device
        )
        angles = torch.linspace(
            0, math.pi, n_angles, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.radon_transform, image, angles)

        skimage_time = None
        if SKIMAGE_AVAILABLE and self.device == "cpu":
            image_np = image.numpy()
            angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
            skimage_time = self._bench(
                skimage_radon, image_np, theta=angles_deg
            )

        print_comparison(
            f"radon_transform ({size}x{size}, angles={n_angles})",
            ts_time,
            skimage_time,
            "skimage",
        )

    def bench_inverse_radon_transform(
        self, size: int = 64, n_angles: int = 90
    ) -> None:
        """Benchmark inverse_radon_transform.

        Parameters
        ----------
        size : int, optional
            Output image size (size x size). Default is 64.
        n_angles : int, optional
            Number of projection angles. Default is 90.
        """
        # Create a sinogram
        image = torch.randn(
            size, size, dtype=torch.float64, device=self.device
        )
        angles = torch.linspace(
            0, math.pi, n_angles, dtype=torch.float64, device=self.device
        )
        sinogram = T.radon_transform(image, angles)

        ts_time = self._bench(
            T.inverse_radon_transform, sinogram, angles, size
        )

        print_comparison(
            f"inverse_radon_transform ({size}x{size}, angles={n_angles})",
            ts_time,
        )

    # ============== Integral Transforms ==============

    def bench_laplace_transform(self, n: int = 1000, n_s: int = 10) -> None:
        """Benchmark laplace_transform.

        Parameters
        ----------
        n : int, optional
            Number of sample points. Default is 1000.
        n_s : int, optional
            Number of s values. Default is 10.
        """
        t = torch.linspace(0, 10, n, dtype=torch.float64, device=self.device)
        f = torch.exp(-2 * t)
        s = torch.linspace(
            0.5, 5, n_s, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.laplace_transform, f, s, t)

        print_comparison(f"laplace_transform (n={n}, n_s={n_s})", ts_time)

    def bench_mellin_transform(self, n: int = 1000, n_s: int = 10) -> None:
        """Benchmark mellin_transform.

        Parameters
        ----------
        n : int, optional
            Number of sample points. Default is 1000.
        n_s : int, optional
            Number of s values. Default is 10.
        """
        t = torch.linspace(
            0.01, 10, n, dtype=torch.float64, device=self.device
        )
        f = torch.exp(-t)
        s = torch.linspace(
            0.5, 3, n_s, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.mellin_transform, f, s, t)

        print_comparison(f"mellin_transform (n={n}, n_s={n_s})", ts_time)

    def bench_abel_transform(self, n: int = 500, n_y: int = 50) -> None:
        """Benchmark abel_transform.

        Parameters
        ----------
        n : int, optional
            Number of sample points. Default is 500.
        n_y : int, optional
            Number of y values. Default is 50.
        """
        r = torch.linspace(
            0.01, 10, n, dtype=torch.float64, device=self.device
        )
        f = torch.exp(-(r**2))
        y = torch.linspace(
            0.01, 5, n_y, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.abel_transform, f, y, r)

        print_comparison(f"abel_transform (n={n}, n_y={n_y})", ts_time)

    def bench_hankel_transform(self, n: int = 500, n_k: int = 50) -> None:
        """Benchmark hankel_transform.

        Parameters
        ----------
        n : int, optional
            Number of sample points. Default is 500.
        n_k : int, optional
            Number of k values. Default is 50.
        """
        r = torch.linspace(
            0.01, 10, n, dtype=torch.float64, device=self.device
        )
        f = torch.exp(-r)
        k = torch.linspace(
            0.1, 5, n_k, dtype=torch.float64, device=self.device
        )

        ts_time = self._bench(T.hankel_transform, f, k, r)

        print_comparison(f"hankel_transform (n={n}, n_k={n_k})", ts_time)

    def bench_z_transform(self, n: int = 100, n_z: int = 50) -> None:
        """Benchmark z_transform.

        Parameters
        ----------
        n : int, optional
            Sequence length. Default is 100.
        n_z : int, optional
            Number of z values. Default is 50.
        """
        x = torch.randn(n, dtype=torch.float64, device=self.device)
        # Evaluate on unit circle
        theta = torch.linspace(
            0, 2 * math.pi, n_z, dtype=torch.float64, device=self.device
        )
        z = torch.exp(1j * theta.to(torch.complex128))

        ts_time = self._bench(T.z_transform, x, z)

        print_comparison(f"z_transform (n={n}, n_z={n_z})", ts_time)

    # ============== Run All ==============

    def run_all(self) -> None:
        """Run all transform benchmarks."""
        device_str = f" ({self.device})" if self.device != "cpu" else ""
        print("=" * 60)
        print(f"TRANSFORM BENCHMARKS{device_str}")
        print("=" * 60)

        # Fourier-family transforms
        print("\n--- Fourier-Family Transforms ---")
        self.bench_cosine_transform()
        self.bench_sine_transform()
        self.bench_hartley_transform()
        self.bench_hilbert_transform()

        # Wavelet transforms
        print("\n--- Wavelet Transforms ---")
        self.bench_discrete_wavelet_transform()
        self.bench_continuous_wavelet_transform()

        # Time-frequency transforms
        print("\n--- Time-Frequency Transforms ---")
        self.bench_short_time_fourier_transform()
        self.bench_gabor_transform()

        # Imaging transforms
        print("\n--- Imaging Transforms ---")
        self.bench_radon_transform()
        self.bench_inverse_radon_transform()

        # Integral transforms
        print("\n--- Integral Transforms ---")
        self.bench_laplace_transform()
        self.bench_mellin_transform()
        self.bench_abel_transform()
        self.bench_hankel_transform()
        self.bench_z_transform()

    def run_scaling(self) -> None:
        """Run scaling benchmarks with varying parameters."""
        print("=" * 60)
        print("SCALING BENCHMARKS")
        print("=" * 60)

        # DCT scaling with signal length
        print("\n--- DCT Signal Length Scaling ---")
        for n in [256, 512, 1024, 2048, 4096]:
            self.bench_cosine_transform(n=n)

        # DWT scaling with signal length
        print("\n--- DWT Signal Length Scaling ---")
        for n in [256, 512, 1024, 2048, 4096]:
            self.bench_discrete_wavelet_transform(n=n)

        # STFT scaling with signal length
        print("\n--- STFT Signal Length Scaling ---")
        for n in [1024, 2048, 4096, 8192, 16384]:
            self.bench_short_time_fourier_transform(n=n)

        # Radon scaling with image size
        print("\n--- Radon Image Size Scaling ---")
        for size in [32, 64, 128, 256]:
            self.bench_radon_transform(size=size)

        # Laplace scaling with sample points
        print("\n--- Laplace Sample Points Scaling ---")
        for n in [100, 500, 1000, 2000, 5000]:
            self.bench_laplace_transform(n=n)


def run_cpu_benchmarks() -> None:
    """Run CPU benchmarks."""
    bench = BenchTransforms(warmup=5, iterations=20, device="cpu")
    bench.run_all()
    print("\n")
    bench.run_scaling()


def run_cuda_benchmarks() -> None:
    """Run CUDA benchmarks if available."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")
        return

    bench = BenchTransforms(warmup=5, iterations=20, device="cuda")
    bench.run_all()
    print("\n")
    bench.run_scaling()


if __name__ == "__main__":
    print("Running CPU benchmarks...\n")
    run_cpu_benchmarks()

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Running CUDA benchmarks...\n")
        run_cuda_benchmarks()
