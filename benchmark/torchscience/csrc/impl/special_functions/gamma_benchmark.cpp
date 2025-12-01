#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/gamma.h"

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
static void BM_Gamma_Float_Integer(benchmark::State& state) {
  std::vector<float> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>((i % 34) + 1);  // 1 to 35
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Float_Integer)->Range(64, 4096);

static void BM_Gamma_Double_Integer(benchmark::State& state) {
  std::vector<double> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<double>((i % 170) + 1);  // 1 to 171
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Double_Integer)->Range(64, 4096);

// Small positive non-integer - Lanczos path
static void BM_Gamma_Float_SmallPositive(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Float_SmallPositive)->Range(64, 4096);

static void BM_Gamma_Double_SmallPositive(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Double_SmallPositive)->Range(64, 4096);

// Negative values - reflection path
static void BM_Gamma_Float_Negative(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), -10.0f, -0.1f);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1f;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Float_Negative)->Range(64, 4096);

static void BM_Gamma_Double_Negative(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), -10.0, -0.1);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Double_Negative)->Range(64, 4096);

// Large negative values - tests range reduction in sin_pi
static void BM_Gamma_Float_LargeNegative(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), -100.0f, -50.0f);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1f;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Float_LargeNegative)->Range(64, 4096);

static void BM_Gamma_Double_LargeNegative(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), -100.0, -50.0);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Double_LargeNegative)->Range(64, 4096);

// Complex values
static void BM_Gamma_Complex64(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-5.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(gamma(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Complex64)->Range(64, 4096);

static void BM_Gamma_Complex128(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-5.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(gamma(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Gamma_Complex128)->Range(64, 4096);

// Baseline: std::tgamma
static void BM_StdTgamma_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::tgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdTgamma_Float)->Range(64, 4096);

static void BM_StdTgamma_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::tgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdTgamma_Double)->Range(64, 4096);

BENCHMARK_MAIN();
