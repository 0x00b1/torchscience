#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/is_nonpositive_integer.h"

using namespace torchscience::impl::special_functions;

// Non-integer values - should return false quickly
static void BM_IsNonpositiveInteger_Complex64_NonInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    // Ensure non-integer by adding 0.5
    data[i] = c10::complex<float>(dist(gen) + 0.5f, dist(gen) * 0.1f);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_NonInteger)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_NonInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    // Ensure non-integer by adding 0.5
    data[i] = c10::complex<double>(dist(gen) + 0.5, dist(gen) * 0.1);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_NonInteger)->Range(64, 4096);

// Positive integers - should return false
static void BM_IsNonpositiveInteger_Complex64_PositiveInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(static_cast<float>((i % 20) + 1), 0.0f);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_PositiveInteger)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_PositiveInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(static_cast<double>((i % 20) + 1), 0.0);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_PositiveInteger)->Range(64, 4096);

// Non-positive integers (poles) - should return true
static void BM_IsNonpositiveInteger_Complex64_Pole(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(-static_cast<float>(i % 20), 0.0f);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_Pole)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_Pole(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(-static_cast<double>(i % 20), 0.0);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_Pole)->Range(64, 4096);

// Large negative integers - tests relative tolerance path
static void BM_IsNonpositiveInteger_Complex64_LargeNegative(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(-static_cast<float>((i % 1000) + 100), 0.0f);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_LargeNegative)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_LargeNegative(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(-static_cast<double>((i % 1000) + 100), 0.0);
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_LargeNegative)->Range(64, 4096);

// Values with significant imaginary part - early exit path
static void BM_IsNonpositiveInteger_Complex64_LargeImag(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(dist(gen), 1.0f);  // Imaginary part above tolerance
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_LargeImag)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_LargeImag(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(dist(gen), 1.0);  // Imaginary part above tolerance
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_LargeImag)->Range(64, 4096);

// Near-integer values (within tolerance) - tests boundary conditions
static void BM_IsNonpositiveInteger_Complex64_NearInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> eps_dist(-1e-6f, 1e-6f);
  for (size_t i = 0; i < count; ++i) {
    float base = -static_cast<float>(i % 20);
    data[i] = c10::complex<float>(base + eps_dist(gen), eps_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex64_NearInteger)->Range(64, 4096);

static void BM_IsNonpositiveInteger_Complex128_NearInteger(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> eps_dist(-1e-13, 1e-13);
  for (size_t i = 0; i < count; ++i) {
    double base = -static_cast<double>(i % 20);
    data[i] = c10::complex<double>(base + eps_dist(gen), eps_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(is_nonpositive_integer(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IsNonpositiveInteger_Complex128_NearInteger)->Range(64, 4096);

BENCHMARK_MAIN();
