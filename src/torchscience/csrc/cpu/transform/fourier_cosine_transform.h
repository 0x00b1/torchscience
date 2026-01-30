#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of Discrete Cosine Transform (DCT).
 *
 * Implements DCT types I-IV via FFT.
 *
 * @param input Input tensor (real)
 * @param n_param Signal length (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param type DCT type (1, 2, 3, or 4)
 * @param norm Normalization mode (0=backward, 1=ortho)
 * @return DCT of the input
 */
inline at::Tensor fourier_cosine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(input.numel() > 0, "fourier_cosine_transform: input tensor must be non-empty");
    TORCH_CHECK(type >= 1 && type <= 4, "fourier_cosine_transform: type must be 1, 2, 3, or 4");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "fourier_cosine_transform: dim out of range");

    // Determine signal length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "fourier_cosine_transform: n must be positive");

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

    if (type == 2) {
        // DCT-II: The most common "DCT"
        // X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n+1) / (2N))

        // Method: Create 2N-point symmetric sequence and take FFT
        // y[n] = x[n] for n=0,...,N-1
        // y[2N-1-n] = x[n] for n=0,...,N-1
        // Then DCT-II[k] = real(FFT[y][k] * exp(-i*pi*k/(2N)))

        at::Tensor x_flip = at::flip(x, {dim});
        at::Tensor y = at::cat({x, x_flip}, dim);  // 2N points

        // Compute FFT
        at::Tensor fft_result = at::fft_fft(y, c10::nullopt, dim);

        // Extract first N points and apply phase shift
        fft_result = fft_result.narrow(dim, 0, n);

        // Phase shift: exp(-i * pi * k / (2N))
        std::vector<int64_t> k_shape(ndim, 1);
        k_shape[dim] = n;
        at::Tensor k = at::arange(n, x.options()).view(k_shape);

        double pi = M_PI;
        at::Tensor phase = k * (-pi / (2.0 * n));
        at::Tensor cos_phase = at::cos(phase);
        at::Tensor sin_phase = at::sin(phase);

        // result = real(fft_result * exp(i*phase)) = real * cos - imag * sin
        result = at::real(fft_result) * cos_phase - at::imag(fft_result) * sin_phase;

        // Ortho normalization
        // scipy convention: first element scaled by sqrt(1/(4*n)), others by sqrt(1/(2*n))
        // This makes DCT-II orthonormal: DCT-III with ortho norm inverts DCT-II with ortho norm
        if (norm == 1) {
            // Scale all elements by sqrt(1/(2*n))
            double scale = std::sqrt(1.0 / (2.0 * n));
            result = result * scale;
            // First element needs extra 1/sqrt(2) to get sqrt(1/(4*n))
            at::Tensor first = result.narrow(dim, 0, 1);
            first.div_(std::sqrt(2.0));
        }

    } else if (type == 3) {
        // DCT-III: Inverse of DCT-II (scipy convention)
        // Unnormalized: y[k] = x[0] + 2 * sum_{n=1}^{N-1} x[n] * cos(pi * n * (2k+1) / (2N))
        // Orthonormal:  y[k] = x[0]*sqrt(1/N) + sum_{n=1}^{N-1} x[n]*sqrt(2/N) * cos(...)
        //
        // For ortho, we scale input such that when multiplied by the unnorm factors
        // (1 for n=0, 2 for n>0), we get the ortho factors (sqrt(1/N), sqrt(2/N))
        //
        // Method: Scale input, apply phase shift, use FFT

        at::Tensor x_scaled = x.clone();

        if (norm == 1) {
            // Ortho scaling: convert to input for unnormalized DCT-III
            // - First element: factor is sqrt(1/N) in ortho, 1 in unnorm => scale by sqrt(1/N)
            // - Others: factor is sqrt(2/N) in ortho, 2 in unnorm => scale by sqrt(2/N)/2 = sqrt(1/(2N))
            at::Tensor first = x_scaled.narrow(dim, 0, 1);
            first.mul_(std::sqrt(1.0 / n));
            if (n > 1) {
                at::Tensor rest = x_scaled.narrow(dim, 1, n - 1);
                rest.mul_(std::sqrt(1.0 / (2.0 * n)));
            }
        }

        // Apply factor of 2 for n > 0 (unnormalized DCT-III formula)
        if (n > 1) {
            at::Tensor rest = x_scaled.narrow(dim, 1, n - 1);
            rest.mul_(2.0);
        }

        // Phase shift: exp(i * pi * n / (2N))
        std::vector<int64_t> n_shape(ndim, 1);
        n_shape[dim] = n;
        at::Tensor n_idx = at::arange(n, x.options()).view(n_shape);

        double pi = M_PI;
        at::Tensor phase = n_idx * (pi / (2.0 * n));
        at::Tensor cos_phase = at::cos(phase);
        at::Tensor sin_phase = at::sin(phase);

        // Create complex input with phase
        at::Tensor x_complex = at::complex(x_scaled * cos_phase, x_scaled * sin_phase);

        // Pad to 2N with zeros
        std::vector<int64_t> pad_shape(x.sizes().begin(), x.sizes().end());
        pad_shape[dim] = 2 * n;
        at::Tensor y = at::zeros(pad_shape, x_complex.options());
        y.narrow(dim, 0, n).copy_(x_complex);

        // Compute IFFT
        at::Tensor ifft_result = at::fft_ifft(y, c10::nullopt, dim);

        // Extract first N points and take real part, scale by 2N
        result = at::real(ifft_result.narrow(dim, 0, n)) * (2.0 * n);

    } else if (type == 1) {
        // DCT-I: Symmetric boundary conditions
        // Unnormalized: X[k] = x[0] + (-1)^k * x[N-1] + 2 * sum_{n=1}^{N-2} x[n] * cos(pi*k*n/(N-1))
        // Ortho: y[k] = w[k] * sum_{n=0}^{N-1} x[n] * w[n] * cos(pi*k*n/(N-1)) * sqrt(2/(N-1))
        //        where w[0] = w[N-1] = 1/sqrt(2), w[other] = 1
        TORCH_CHECK(n >= 2, "DCT-I requires n >= 2");

        double pi = M_PI;

        if (norm == 0) {
            // FFT-based implementation for unnormalized case
            at::Tensor x_mid = x.narrow(dim, 1, n - 2);  // x[1:N-1]
            at::Tensor x_mid_flip = at::flip(x_mid, {dim});
            at::Tensor y = at::cat({x, x_mid_flip}, dim);

            at::Tensor fft_result = at::fft_fft(y, c10::nullopt, dim);
            result = at::real(fft_result.narrow(dim, 0, n));
        } else {
            // Direct O(N^2) implementation for ortho case
            // ortho[k] = w[k] * sqrt(2/(N-1)) * sum_{n=0}^{N-1} x[n] * w[n] * cos(pi*k*n/(N-1))

            // Move dim to last position for easier manipulation
            at::Tensor x_perm = x.movedim(dim, -1);
            std::vector<int64_t> batch_shape(x_perm.sizes().begin(), x_perm.sizes().end() - 1);

            // Flatten batch dimensions
            int64_t batch_size = 1;
            for (size_t i = 0; i < batch_shape.size(); i++) {
                batch_size *= batch_shape[i];
            }
            at::Tensor x_flat = x_perm.reshape({batch_size, n});

            // Build weighted cosine matrix: cos_mat[k, m] = w[k] * w[m] * cos(pi*k*m/(n-1)) * sqrt(2/(n-1))
            at::Tensor k_range = at::arange(n, x.options()).unsqueeze(1);
            at::Tensor m_range = at::arange(n, x.options()).unsqueeze(0);
            at::Tensor angles = pi * k_range * m_range / (n - 1);
            at::Tensor cos_mat = at::cos(angles);

            // Apply weights: w[0] = w[N-1] = 1/sqrt(2), w[other] = 1
            double w_end = 1.0 / std::sqrt(2.0);
            double scale = std::sqrt(2.0 / (n - 1));

            // Weight columns (for input weights w[n])
            cos_mat.index({at::indexing::Slice(), 0}) *= w_end;
            cos_mat.index({at::indexing::Slice(), n - 1}) *= w_end;

            // Weight rows (for output weights w[k])
            cos_mat.index({0, at::indexing::Slice()}) *= w_end;
            cos_mat.index({n - 1, at::indexing::Slice()}) *= w_end;

            // Apply global scale
            cos_mat *= scale;

            // Compute: result[b, k] = sum_m x_flat[b, m] * cos_mat[k, m]
            at::Tensor result_flat = at::matmul(x_flat, cos_mat.transpose(0, 1));

            // Reshape back
            batch_shape.push_back(n);
            result = result_flat.reshape(batch_shape).movedim(-1, dim);
        }

    } else {  // type == 4
        // DCT-IV: Half-sample symmetric
        // X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi * (2k+1) * (2n+1) / (4N))
        //
        // Direct O(N^2) implementation using matrix multiplication.
        // Note: An FFT-based O(N log N) algorithm exists but requires careful
        // handling of half-sample shifts and is deferred to future optimization.

        double pi = M_PI;

        // Move dim to last position for easier manipulation
        at::Tensor x_perm = x.movedim(dim, -1);  // [..., n]
        std::vector<int64_t> batch_shape(x_perm.sizes().begin(), x_perm.sizes().end() - 1);

        // Flatten batch dimensions
        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - 1; i++) {
            batch_size *= x_perm.size(i);
        }
        at::Tensor x_flat = x_perm.reshape({batch_size, n});  // [B, n]

        // Build cosine matrix: cos_mat[k, m] = cos(pi*(2k+1)*(2m+1)/(4n))
        at::Tensor k_range = at::arange(n, x.options()).unsqueeze(1);  // [n, 1]
        at::Tensor m_range = at::arange(n, x.options()).unsqueeze(0);  // [1, n]
        at::Tensor angles = pi * (2.0 * k_range + 1.0) * (2.0 * m_range + 1.0) / (4.0 * n);
        at::Tensor cos_mat = at::cos(angles);  // [n, n]

        // Compute DCT-IV: result[b, k] = 2 * sum_m x_flat[b, m] * cos_mat[k, m]
        // = 2 * x_flat @ cos_mat.T
        at::Tensor result_flat = 2.0 * at::matmul(x_flat, cos_mat.transpose(0, 1));  // [B, n]

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
 * Backward pass for DCT on CPU.
 *
 * For ortho norm, the DCT matrix is orthogonal, so the adjoint equals the
 * inverse, and we can simply apply the swapped-type DCT.
 *
 * For backward (unnormalized) norm, the transpose relationships are:
 * - DCT-II^T @ g = DCT-III(g) + g[0] (add first element to all outputs)
 * - DCT-III^T @ g = DCT-II(g) with first element halved
 * - DCT-I^T @ g = DCT-I with modified weights (complex, use direct method)
 * - DCT-IV^T = DCT-IV (symmetric matrix)
 */
inline at::Tensor fourier_cosine_transform_backward(
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
        // Ortho norm: DCT is orthogonal, so adjoint = inverse
        if (type == 2) {
            grad = fourier_cosine_transform(grad_output, n, norm_dim, 3, norm);
        } else if (type == 3) {
            grad = fourier_cosine_transform(grad_output, n, norm_dim, 2, norm);
        } else {
            grad = fourier_cosine_transform(grad_output, n, norm_dim, type, norm);
        }
    } else {
        // Backward (unnormalized) norm: need exact transpose
        if (type == 2) {
            // DCT-II^T @ g = DCT-III(g) + g[0]
            grad = fourier_cosine_transform(grad_output, n, norm_dim, 3, 0);
            at::Tensor first = grad_output.narrow(norm_dim, 0, 1);
            grad = grad + first;
        } else if (type == 3) {
            // DCT-III^T @ g = DCT-II(g) with first element halved
            grad = fourier_cosine_transform(grad_output, n, norm_dim, 2, 0);
            at::Tensor first = grad.narrow(norm_dim, 0, 1);
            first.div_(2.0);
        } else if (type == 4) {
            // DCT-IV is symmetric
            grad = fourier_cosine_transform(grad_output, n, norm_dim, 4, 0);
        } else {
            // DCT-I: Use direct O(N^2) computation for the transpose
            // C[k,n] = c_n * cos(pi*k*n/(N-1)) where c_0=c_{N-1}=1, c_other=2
            // C^T[n,k] = c_n * cos(pi*k*n/(N-1))
            // grad[n] = c_n * sum_k g[k] * cos(pi*k*n/(N-1))

            double pi = M_PI;
            at::Tensor g_perm = grad_output.movedim(norm_dim, -1);
            std::vector<int64_t> batch_shape(g_perm.sizes().begin(), g_perm.sizes().end() - 1);

            int64_t batch_size = 1;
            for (size_t i = 0; i < batch_shape.size(); i++) {
                batch_size *= batch_shape[i];
            }
            at::Tensor g_flat = g_perm.reshape({batch_size, n});

            // Build transpose matrix: C^T[n,k] = c_n * cos(pi*k*n/(N-1))
            at::Tensor n_range = at::arange(n, grad_output.options()).unsqueeze(1);
            at::Tensor k_range = at::arange(n, grad_output.options()).unsqueeze(0);
            at::Tensor angles = pi * k_range * n_range / (n - 1);
            at::Tensor cos_mat = at::cos(angles);  // [n, n]

            // Apply c_n weights to rows (n=0 and n=N-1 have weight 1, others have weight 2)
            cos_mat = cos_mat * 2.0;
            cos_mat.index({0, at::indexing::Slice()}) /= 2.0;
            cos_mat.index({n - 1, at::indexing::Slice()}) /= 2.0;

            at::Tensor grad_flat = at::matmul(g_flat, cos_mat.transpose(0, 1));
            batch_shape.push_back(n);
            grad = grad_flat.reshape(batch_shape).movedim(-1, norm_dim);
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
 * Double backward pass for DCT on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> fourier_cosine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    // DCT is linear, so grad_grad_output = DCT[grad_grad_input]
    at::Tensor grad_grad_output = fourier_cosine_transform(
        grad_grad_input, n_param, dim, type, norm
    );

    // No second-order term
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "fourier_cosine_transform",
        &torchscience::cpu::transform::fourier_cosine_transform
    );

    module.impl(
        "fourier_cosine_transform_backward",
        &torchscience::cpu::transform::fourier_cosine_transform_backward
    );

    module.impl(
        "fourier_cosine_transform_backward_backward",
        &torchscience::cpu::transform::fourier_cosine_transform_backward_backward
    );
}
