#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/incomplete_beta.h"

using namespace torchscience::impl::special_functions;

// Helper to generate valid incomplete beta parameters
template <typename T>
struct BetaParams {
  T z, a, b;
};

template <typename T>
std::vector<BetaParams<T>> generate_beta_params(size_t count, T z_min, T z_max, T a_min, T a_max, T b_min, T b_max, unsigned seed = 42) {
  std::vector<BetaParams<T>> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> z_dist(z_min, z_max);
  std::uniform_real_distribution<T> a_dist(a_min, a_max);
  std::uniform_real_distribution<T> b_dist(b_min, b_max);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {z_dist(gen), a_dist(gen), b_dist(gen)};
  }
  return data;
}

// Small parameters - typical case
static void BM_IncompleteBeta_Float_Small(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.1f, 0.9f, 0.5f, 5.0f, 0.5f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_Small)->Range(64, 4096);

static void BM_IncompleteBeta_Double_Small(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.1, 0.9, 0.5, 5.0, 0.5, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_Small)->Range(64, 4096);

// Large parameters - more iterations needed
static void BM_IncompleteBeta_Float_Large(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.1f, 0.9f, 10.0f, 100.0f, 10.0f, 100.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_Large)->Range(64, 4096);

static void BM_IncompleteBeta_Double_Large(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.1, 0.9, 10.0, 100.0, 10.0, 100.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_Large)->Range(64, 4096);

// z near 0 - fast convergence
static void BM_IncompleteBeta_Float_ZNearZero(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.01f, 0.1f, 1.0f, 10.0f, 1.0f, 10.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_ZNearZero)->Range(64, 4096);

static void BM_IncompleteBeta_Double_ZNearZero(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.01, 0.1, 1.0, 10.0, 1.0, 10.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_ZNearZero)->Range(64, 4096);

// z near 1 - uses symmetry relation
static void BM_IncompleteBeta_Float_ZNearOne(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.9f, 0.99f, 1.0f, 10.0f, 1.0f, 10.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_ZNearOne)->Range(64, 4096);

static void BM_IncompleteBeta_Double_ZNearOne(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.9, 0.99, 1.0, 10.0, 1.0, 10.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_ZNearOne)->Range(64, 4096);

// Asymmetric parameters (a << b or a >> b)
static void BM_IncompleteBeta_Float_Asymmetric(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.1f, 0.9f, 0.5f, 2.0f, 10.0f, 50.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_Asymmetric)->Range(64, 4096);

static void BM_IncompleteBeta_Double_Asymmetric(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.1, 0.9, 0.5, 2.0, 10.0, 50.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_Asymmetric)->Range(64, 4096);

// Special case: a = 1 (exact formula)
static void BM_IncompleteBeta_Float_AOne(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.1f, 0.9f, 1.0f, 1.0f, 1.0f, 10.0f);
  for (auto& p : data) p.a = 1.0f;
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_AOne)->Range(64, 4096);

static void BM_IncompleteBeta_Double_AOne(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.1, 0.9, 1.0, 1.0, 1.0, 10.0);
  for (auto& p : data) p.a = 1.0;
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_AOne)->Range(64, 4096);

// Special case: b = 1 (exact formula)
static void BM_IncompleteBeta_Float_BOne(benchmark::State& state) {
  auto data = generate_beta_params<float>(state.range(0), 0.1f, 0.9f, 1.0f, 10.0f, 1.0f, 1.0f);
  for (auto& p : data) p.b = 1.0f;
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Float_BOne)->Range(64, 4096);

static void BM_IncompleteBeta_Double_BOne(benchmark::State& state) {
  auto data = generate_beta_params<double>(state.range(0), 0.1, 0.9, 1.0, 10.0, 1.0, 1.0);
  for (auto& p : data) p.b = 1.0;
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(regularized_incomplete_beta(p.z, p.a, p.b));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_IncompleteBeta_Double_BOne)->Range(64, 4096);

BENCHMARK_MAIN();
