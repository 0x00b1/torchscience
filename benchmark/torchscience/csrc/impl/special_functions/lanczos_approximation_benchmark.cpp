#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/lanczos_approximation.h"

using namespace torchscience::impl::special_functions;

template <typename T>
std::vector<T> generate_random_data(size_t count, T min_val, T max_val, unsigned seed = 42) {
  std::vector<T> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(min_val, max_val);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  return data;
}

// Small positive values - typical gamma function input range
static void BM_LanczosSeries_Float_SmallPositive(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.0f);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Float_SmallPositive)->Range(64, 4096);

static void BM_LanczosSeries_Double_SmallPositive(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.0);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Double_SmallPositive)->Range(64, 4096);

// Larger positive values
static void BM_LanczosSeries_Float_LargePositive(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 10.0f, 100.0f);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Float_LargePositive)->Range(64, 4096);

static void BM_LanczosSeries_Double_LargePositive(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 10.0, 100.0);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Double_LargePositive)->Range(64, 4096);

// Values near zero (but positive, as used in gamma computation after shift)
static void BM_LanczosSeries_Float_NearZero(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.01f, 0.5f);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Float_NearZero)->Range(64, 4096);

static void BM_LanczosSeries_Double_NearZero(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.01, 0.5);
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Double_NearZero)->Range(64, 4096);

// Complex values - small
static void BM_LanczosSeries_Complex64_Small(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Complex64_Small)->Range(64, 4096);

static void BM_LanczosSeries_Complex128_Small(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Complex128_Small)->Range(64, 4096);

// Complex values - large imaginary part
static void BM_LanczosSeries_Complex64_LargeImag(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> real_dist(0.5f, 5.0f);
  std::uniform_real_distribution<float> imag_dist(10.0f, 50.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Complex64_LargeImag)->Range(64, 4096);

static void BM_LanczosSeries_Complex128_LargeImag(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> real_dist(0.5, 5.0);
  std::uniform_real_distribution<double> imag_dist(10.0, 50.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(lanczos_series(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_LanczosSeries_Complex128_LargeImag)->Range(64, 4096);

BENCHMARK_MAIN();
