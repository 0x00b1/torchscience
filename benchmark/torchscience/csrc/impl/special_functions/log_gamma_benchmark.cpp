#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/log_gamma.h"

using namespace torchscience::impl::special_functions;

// Complex small positive real part - direct Lanczos
static void BM_LogGamma_Complex64_SmallPositive(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> real_dist(0.5f, 10.0f);
  std::uniform_real_distribution<float> imag_dist(-5.0f, 5.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex64_SmallPositive)->Range(64, 4096);

static void BM_LogGamma_Complex128_SmallPositive(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> real_dist(0.5, 10.0);
  std::uniform_real_distribution<double> imag_dist(-5.0, 5.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex128_SmallPositive)->Range(64, 4096);

// Complex negative real part - reflection formula path
static void BM_LogGamma_Complex64_Negative(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> real_dist(-10.0f, 0.0f);
  std::uniform_real_distribution<float> imag_dist(-5.0f, 5.0f);
  for (size_t i = 0; i < count; ++i) {
    // Avoid poles at non-positive integers
    float real = real_dist(gen);
    float imag = imag_dist(gen);
    if (std::abs(imag) < 0.1f && std::floor(real) == real) {
      imag = 0.5f;
    }
    data[i] = c10::complex<float>(real, imag);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex64_Negative)->Range(64, 4096);

static void BM_LogGamma_Complex128_Negative(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> real_dist(-10.0, 0.0);
  std::uniform_real_distribution<double> imag_dist(-5.0, 5.0);
  for (size_t i = 0; i < count; ++i) {
    // Avoid poles at non-positive integers
    double real = real_dist(gen);
    double imag = imag_dist(gen);
    if (std::abs(imag) < 0.1 && std::floor(real) == real) {
      imag = 0.5;
    }
    data[i] = c10::complex<double>(real, imag);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex128_Negative)->Range(64, 4096);

// Large positive values - tests overflow prevention
static void BM_LogGamma_Complex64_Large(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> real_dist(100.0f, 1000.0f);
  std::uniform_real_distribution<float> imag_dist(-10.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex64_Large)->Range(64, 4096);

static void BM_LogGamma_Complex128_Large(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> real_dist(100.0, 1000.0);
  std::uniform_real_distribution<double> imag_dist(-10.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex128_Large)->Range(64, 4096);

// Purely imaginary values
static void BM_LogGamma_Complex64_Imaginary(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> imag_dist(0.1f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(0.0f, imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex64_Imaginary)->Range(64, 4096);

static void BM_LogGamma_Complex128_Imaginary(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> imag_dist(0.1, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(0.0, imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(log_gamma_complex(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LogGamma_Complex128_Imaginary)->Range(64, 4096);

// Baseline: std::lgamma for real part
static void BM_StdLgamma_Float(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<float> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 100.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::lgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdLgamma_Float)->Range(64, 4096);

static void BM_StdLgamma_Double(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<double> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 100.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::lgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdLgamma_Double)->Range(64, 4096);

BENCHMARK_MAIN();
