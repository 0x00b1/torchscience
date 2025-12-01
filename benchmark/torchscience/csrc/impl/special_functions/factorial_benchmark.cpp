#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/factorial.h"

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

// Integer values - LUT path
static void BM_Factorial_Float_Integer(benchmark::State& state) {
  std::vector<float> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i % 35);  // 0! to 34! fit in float
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(factorial(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Float_Integer)->Range(64, 4096);

static void BM_Factorial_Double_Integer(benchmark::State& state) {
  std::vector<double> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<double>(i % 171);  // 0! to 170! fit in double
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(factorial(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Double_Integer)->Range(64, 4096);

// Non-integer values - gamma path
static void BM_Factorial_Float_NonInteger(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(factorial(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Float_NonInteger)->Range(64, 4096);

static void BM_Factorial_Double_NonInteger(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(factorial(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Double_NonInteger)->Range(64, 4096);

// Complex values
static void BM_Factorial_Complex64(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(factorial(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Complex64)->Range(64, 4096);

static void BM_Factorial_Complex128(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(factorial(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Factorial_Complex128)->Range(64, 4096);

// Baseline: std::tgamma(x + 1)
static void BM_StdTgamma_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::tgamma(x + 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdTgamma_Float)->Range(64, 4096);

static void BM_StdTgamma_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::tgamma(x + 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdTgamma_Double)->Range(64, 4096);

BENCHMARK_MAIN();
