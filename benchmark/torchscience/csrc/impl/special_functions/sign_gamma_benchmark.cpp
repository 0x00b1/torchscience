#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/sign_gamma.h"

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

// Positive values - always returns 1
static void BM_SignGamma_Float_Positive(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.1f, 100.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sign_gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignGamma_Float_Positive)->Range(64, 4096);

static void BM_SignGamma_Double_Positive(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.1, 100.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sign_gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignGamma_Double_Positive)->Range(64, 4096);

// Negative values - alternating sign computation
static void BM_SignGamma_Float_Negative(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), -100.0f, -0.1f);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1f;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sign_gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignGamma_Float_Negative)->Range(64, 4096);

static void BM_SignGamma_Double_Negative(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), -100.0, -0.1);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (std::floor(x) == x) x += 0.1;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sign_gamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignGamma_Double_Negative)->Range(64, 4096);

// signed_exp_lgamma - combines sign and lgamma
static void BM_SignedExpLgamma_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), -10.0f, 10.0f);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (x <= 0 && std::floor(x) == x) x += 0.1f;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(signed_exp_lgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignedExpLgamma_Float)->Range(64, 4096);

static void BM_SignedExpLgamma_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), -10.0, 10.0);
  // Avoid non-positive integers (poles)
  for (auto& x : data) {
    if (x <= 0 && std::floor(x) == x) x += 0.1;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(signed_exp_lgamma(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SignedExpLgamma_Double)->Range(64, 4096);

// gamma_ratio - ratio of two gamma functions
static void BM_GammaRatio_Float(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::pair<float, float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {dist(gen), dist(gen)};
  }
  for (auto _ : state) {
    for (const auto& [a, b] : data) {
      benchmark::DoNotOptimize(gamma_ratio(a, b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GammaRatio_Float)->Range(64, 4096);

static void BM_GammaRatio_Double(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::pair<double, double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {dist(gen), dist(gen)};
  }
  for (auto _ : state) {
    for (const auto& [a, b] : data) {
      benchmark::DoNotOptimize(gamma_ratio(a, b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GammaRatio_Double)->Range(64, 4096);

// gamma_ratio_4 - ratio of products of gamma functions
static void BM_GammaRatio4_Float(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::tuple<float, float, float, float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {dist(gen), dist(gen), dist(gen), dist(gen)};
  }
  for (auto _ : state) {
    for (const auto& [a1, a2, b1, b2] : data) {
      benchmark::DoNotOptimize(gamma_ratio_4(a1, a2, b1, b2));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GammaRatio4_Float)->Range(64, 4096);

static void BM_GammaRatio4_Double(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::tuple<double, double, double, double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {dist(gen), dist(gen), dist(gen), dist(gen)};
  }
  for (auto _ : state) {
    for (const auto& [a1, a2, b1, b2] : data) {
      benchmark::DoNotOptimize(gamma_ratio_4(a1, a2, b1, b2));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GammaRatio4_Double)->Range(64, 4096);

BENCHMARK_MAIN();
