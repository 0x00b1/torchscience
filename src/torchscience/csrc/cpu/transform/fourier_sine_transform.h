#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of Discrete Sine Transform (DST).
 *
 * Implements DST types I-IV.
 *
 * @param input Input tensor (real)
 * @param n_param Signal length (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param type DST type (1, 2, 3, or 4)
 * @param norm Normalization mode (0=backward, 1=ortho)
 * @return DST of the input
 */
inline at::Tensor fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(input.numel() > 0, "fourier_sine_transform: input tensor must be non-empty");
    TORCH_CHECK(type >= 1 && type <= 4, "fourier_sine_transform: type must be 1, 2, 3, or 4");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "fourier_sine_transform: dim out of range");

    // Determine signal length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "fourier_sine_transform: n must be positive");

    // Handle truncation/padding
    at::Tensor x = input.contiguous();
    if (n != input_size) {
        if (n < input_size) {
            x = x.narrow(dim, 0, n);
        } else {
            // Zero-pad
            std::vector<int64_t> pad_shape(x.sizes().begin(), x.sizes().end());
            pad_shape[dim] = n;
            at::Tensor padded = at::zeros(pad_shape, x.options());
            padded.narrow(dim, 0, input_size).copy_(x);
            x = padded;
        }
    }

    at::Tensor result;
    double pi = M_PI;

    if (type == 2) {
        // DST-II: The most common "DST"
        // X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi * (2n+1) * (k+1) / (2N))
        //
        // Method: Use FFT by creating antisymmetric sequence
        // y[n] = x[n] for n=0,...,N-1
        // y[2N-1-n] = -x[n] for n=0,...,N-1
        // Then DST-II[k] = -imag(FFT[y][k+1] * exp(-i*pi*(k+1)/(2N)))

        at::Tensor x_flip_neg = -at::flip(x, {dim});
        at::Tensor y = at::cat({x, x_flip_neg}, dim);  // 2N points

        // Compute FFT
        at::Tensor fft_result = at::fft_fft(y, c10::nullopt, dim);

        // Extract points 1 to N (indices k+1 for k=0..N-1)
        fft_result = fft_result.narrow(dim, 1, n);

        // Phase shift: exp(-i * pi * (k+1) / (2N))
        std::vector<int64_t> k_shape(ndim, 1);
        k_shape[dim] = n;
        at::Tensor k = at::arange(1, n + 1, x.options()).view(k_shape);

        at::Tensor phase = k * (-pi / (2.0 * n));
        at::Tensor cos_phase = at::cos(phase);
        at::Tensor sin_phase = at::sin(phase);

        // result = -imag(fft_result * exp(i*phase)) = -(real * sin + imag * cos)
        result = -(at::real(fft_result) * sin_phase + at::imag(fft_result) * cos_phase);

        // Ortho normalization
        // scipy convention: last element scaled by sqrt(1/(4*n)), others by sqrt(1/(2*n))
        if (norm == 1) {
            double scale = std::sqrt(1.0 / (2.0 * n));
            result = result * scale;
            // Last element needs extra 1/sqrt(2) to get sqrt(1/(4*n))
            at::Tensor last = result.narrow(dim, n - 1, 1);
            last.div_(std::sqrt(2.0));
        }

    } else if (type == 3) {
        // DST-III: Inverse of DST-II (scipy convention)
        // Unnormalized: y[k] = (-1)^k * x[N-1] + 2 * sum_{n=0}^{N-2} x[n] * sin(pi * (n+1) * (2k+1) / (2N))
        // Orthonormal: y[k] = sum_m x[m] * w[m] * 2 * sin(pi * (2k+1) * (m+1) / (2N))
        //              where w[m] = sqrt(1/(2N)) for m < N-1, sqrt(1/(4N)) for m = N-1
        //
        // Direct O(N^2) implementation for correctness.

        // Move dim to last position for easier manipulation
        at::Tensor x_perm = x.movedim(dim, -1);
        std::vector<int64_t> batch_shape(x_perm.sizes().begin(), x_perm.sizes().end() - 1);

        // Flatten batch dimensions
        int64_t batch_size = 1;
        for (size_t i = 0; i < batch_shape.size(); i++) {
            batch_size *= batch_shape[i];
        }
        at::Tensor x_flat = x_perm.reshape({batch_size, n});

        at::Tensor result_flat;

        if (norm == 1) {
            // Ortho: y[k] = sum_m x[m] * w[m] * 2 * sin(pi * (2k+1) * (m+1) / (2N))
            // where w[m] = sqrt(1/(2N)) for m < N-1, sqrt(1/(4N)) for m = N-1
            // This is the transpose of DST-II ortho matrix

            // Build full sine matrix: sin_mat[k, m] = 2 * sin(pi*(2k+1)*(m+1)/(2N))
            at::Tensor k_range = at::arange(n, x.options()).unsqueeze(1);  // [N, 1]
            at::Tensor m_range = at::arange(n, x.options()).unsqueeze(0);  // [1, N]
            at::Tensor angles = pi * (2.0 * k_range + 1.0) * (m_range + 1.0) / (2.0 * n);
            at::Tensor sin_mat = 2.0 * at::sin(angles);  // [N, N]

            // Apply input weights: w[m] = sqrt(1/(2N)) for m < N-1, sqrt(1/(4N)) for m = N-1
            at::Tensor weights = at::full({n}, std::sqrt(1.0 / (2.0 * n)), x.options());
            weights.index_put_({n - 1}, std::sqrt(1.0 / (4.0 * n)));

            // Scale columns by weights
            at::Tensor sin_mat_weighted = sin_mat * weights;

            // Compute: result[b, k] = sum_m x_flat[b, m] * sin_mat_weighted[k, m]
            result_flat = at::matmul(x_flat, sin_mat_weighted.transpose(0, 1));
        } else {
            // Unnormalized: y[k] = (-1)^k * x[N-1] + 2 * sum_{n=0}^{N-2} x[n] * sin(pi*(n+1)*(2k+1)/(2N))
            at::Tensor k_range = at::arange(n, x.options()).unsqueeze(1);  // [N, 1]
            at::Tensor m_range = at::arange(n - 1, x.options()).unsqueeze(0);  // [1, N-1]
            at::Tensor angles = pi * (m_range + 1.0) * (2.0 * k_range + 1.0) / (2.0 * n);
            at::Tensor sin_mat = at::sin(angles);  // [N, N-1]

            // Extract x[0..N-2] and x[N-1]
            at::Tensor x_first = x_flat.narrow(1, 0, n - 1);  // [B, N-1]
            at::Tensor x_last = x_flat.narrow(1, n - 1, 1);   // [B, 1]

            // Compute: 2 * sum_{n=0}^{N-2} x[n] * sin_mat[k, n]
            at::Tensor sum_term = 2.0 * at::matmul(x_first, sin_mat.transpose(0, 1));  // [B, N]

            // Add (-1)^k * x[N-1]
            at::Tensor alt_signs = at::pow(-1.0, k_range).squeeze(1);  // [N]
            at::Tensor alt_term = x_last * alt_signs;  // [B, N]

            result_flat = sum_term + alt_term;
        }

        // Reshape back
        batch_shape.push_back(n);
        result = result_flat.reshape(batch_shape).movedim(-1, dim);

    } else if (type == 1) {
        // DST-I: Antisymmetric boundary conditions
        // Unnormalized: X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(n+1)*(k+1)/(N+1))
        // Ortho: y[k] = sqrt(2/(N+1)) * sum_{n=0}^{N-1} x[n] * sin(pi*(n+1)*(k+1)/(N+1))
        TORCH_CHECK(n >= 1, "DST-I requires n >= 1");

        // Move dim to last position for easier manipulation
        at::Tensor x_perm = x.movedim(dim, -1);
        std::vector<int64_t> batch_shape(x_perm.sizes().begin(), x_perm.sizes().end() - 1);

        // Flatten batch dimensions
        int64_t batch_size = 1;
        for (size_t i = 0; i < batch_shape.size(); i++) {
            batch_size *= batch_shape[i];
        }
        at::Tensor x_flat = x_perm.reshape({batch_size, n});

        // Build sine matrix: sin_mat[k, m] = sin(pi*(k+1)*(m+1)/(n+1))
        at::Tensor k_range = at::arange(1, n + 1, x.options()).unsqueeze(1);
        at::Tensor m_range = at::arange(1, n + 1, x.options()).unsqueeze(0);
        at::Tensor angles = pi * k_range * m_range / (n + 1);
        at::Tensor sin_mat = at::sin(angles);

        // Compute: result[b, k] = 2 * sum_m x_flat[b, m] * sin_mat[k, m]
        at::Tensor result_flat;
        if (norm == 1) {
            // Ortho: scale by sqrt(2/(n+1))
            double scale = std::sqrt(2.0 / (n + 1));
            result_flat = scale * at::matmul(x_flat, sin_mat.transpose(0, 1));
        } else {
            result_flat = 2.0 * at::matmul(x_flat, sin_mat.transpose(0, 1));
        }

        // Reshape back
        batch_shape.push_back(n);
        result = result_flat.reshape(batch_shape).movedim(-1, dim);

    } else {  // type == 4
        // DST-IV: Half-sample antisymmetric
        // X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi * (2k+1) * (2n+1) / (4N))

        // Move dim to last position for easier manipulation
        at::Tensor x_perm = x.movedim(dim, -1);
        std::vector<int64_t> batch_shape(x_perm.sizes().begin(), x_perm.sizes().end() - 1);

        // Flatten batch dimensions
        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - 1; i++) {
            batch_size *= x_perm.size(i);
        }
        at::Tensor x_flat = x_perm.reshape({batch_size, n});

        // Build sine matrix: sin_mat[k, m] = sin(pi*(2k+1)*(2m+1)/(4n))
        at::Tensor k_range = at::arange(n, x.options()).unsqueeze(1);
        at::Tensor m_range = at::arange(n, x.options()).unsqueeze(0);
        at::Tensor angles = pi * (2.0 * k_range + 1.0) * (2.0 * m_range + 1.0) / (4.0 * n);
        at::Tensor sin_mat = at::sin(angles);

        // Compute DST-IV: result[b, k] = 2 * sum_m x_flat[b, m] * sin_mat[k, m]
        at::Tensor result_flat = 2.0 * at::matmul(x_flat, sin_mat.transpose(0, 1));

        // Reshape back
        batch_shape.push_back(n);
        result = result_flat.reshape(batch_shape).movedim(-1, dim);

        if (norm == 1) {
            // Ortho normalization: scale by sqrt(1/(2N))
            result = result * std::sqrt(1.0 / (2.0 * n));
        }
    }

    return result;
}

/**
 * Backward pass for DST on CPU.
 *
 * For ortho norm, the DST matrix is orthogonal, so the adjoint equals the
 * inverse, and we can simply apply the swapped-type DST.
 *
 * For backward (unnormalized) norm, the transpose relationships are:
 * - DST-II^T @ g = DST-III(g) + (-1)^k g[N-1] offset applied
 * - DST-III^T @ g = DST-II(g) with last element halved
 * - DST-I^T = DST-I (symmetric matrix)
 * - DST-IV^T = DST-IV (symmetric matrix)
 */
inline at::Tensor fourier_sine_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    at::Tensor grad;

    if (norm == 1) {
        // Ortho norm: DST is orthogonal, so adjoint = inverse
        if (type == 2) {
            grad = fourier_sine_transform(grad_output, n, norm_dim, 3, norm);
        } else if (type == 3) {
            grad = fourier_sine_transform(grad_output, n, norm_dim, 2, norm);
        } else {
            grad = fourier_sine_transform(grad_output, n, norm_dim, type, norm);
        }
    } else {
        // Backward (unnormalized) norm: need exact transpose
        if (type == 2) {
            // DST-II matrix: S[k,n] = 2 * sin(pi*(2n+1)*(k+1)/(2N))
            // S^T[n,k] = 2 * sin(pi*(2n+1)*(k+1)/(2N))
            // This is DST-III with additional handling of the last element term
            grad = fourier_sine_transform(grad_output, n, norm_dim, 3, 0);
            // Add (-1)^n * g[N-1] to each element
            at::Tensor last = grad_output.narrow(norm_dim, n - 1, 1);
            std::vector<int64_t> alt_shape(ndim, 1);
            alt_shape[norm_dim] = n;
            at::Tensor n_idx = at::arange(n, grad_output.options()).view(alt_shape);
            at::Tensor alternating = at::pow(-1.0, n_idx);
            grad = grad + last * alternating;
        } else if (type == 3) {
            // DST-III^T @ g = DST-II(g) with last element halved
            grad = fourier_sine_transform(grad_output, n, norm_dim, 2, 0);
            at::Tensor last = grad.narrow(norm_dim, n - 1, 1);
            last.div_(2.0);
        } else if (type == 4) {
            // DST-IV is symmetric
            grad = fourier_sine_transform(grad_output, n, norm_dim, 4, 0);
        } else {
            // DST-I is symmetric
            grad = fourier_sine_transform(grad_output, n, norm_dim, 1, 0);
        }
    }

    // Adjust size if needed
    if (n > input_size) {
        grad = grad.narrow(norm_dim, 0, input_size);
    } else if (n < input_size) {
        std::vector<int64_t> pad_shape(grad.sizes().begin(), grad.sizes().end());
        pad_shape[norm_dim] = input_size;
        at::Tensor padded = at::zeros(pad_shape, grad.options());
        padded.narrow(norm_dim, 0, n).copy_(grad);
        grad = padded;
    }

    return grad;
}

/**
 * Double backward pass for DST on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> fourier_sine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    // DST is linear, so grad_grad_output = DST[grad_grad_input]
    at::Tensor grad_grad_output = fourier_sine_transform(
        grad_grad_input, n_param, dim, type, norm
    );

    // No second-order term
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "fourier_sine_transform",
        &torchscience::cpu::transform::fourier_sine_transform
    );

    module.impl(
        "fourier_sine_transform_backward",
        &torchscience::cpu::transform::fourier_sine_transform_backward
    );

    module.impl(
        "fourier_sine_transform_backward_backward",
        &torchscience::cpu::transform::fourier_sine_transform_backward_backward
    );
}
