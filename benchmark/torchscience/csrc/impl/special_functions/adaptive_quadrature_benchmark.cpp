#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/adaptive_quadrature.h"

using namespace torchscience::impl::special_functions;

// Helper to generate valid beta parameters
template <typename T>
struct BetaIntegralParams {
  T t_upper, a, b;
};

template <typename T>
std::vector<BetaIntegralParams<T>> generate_beta_integral_params(
    size_t count, T t_min, T t_max, T a_min, T a_max, T b_min, T b_max, unsigned seed = 42) {
  std::vector<BetaIntegralParams<T>> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> t_dist(t_min, t_max);
  std::uniform_real_distribution<T> a_dist(a_min, a_max);
  std::uniform_real_distribution<T> b_dist(b_min, b_max);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {t_dist(gen), a_dist(gen), b_dist(gen)};
  }
  return data;
}

// ============================================================================
// Gauss-Kronrod G7-K15 Single-Level Benchmarks
// ============================================================================

// Typical parameters - moderate singularity
static void BM_GaussKronrod15_Float_Typical(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.1f, 0.9f, 1.0f, 5.0f, 1.0f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod15Policy>(p.t_upper, p.a, p.b, 0.0f, 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod15_Float_Typical)->Range(64, 4096);

static void BM_GaussKronrod15_Double_Typical(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.1, 0.9, 1.0, 5.0, 1.0, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod15Policy>(p.t_upper, p.a, p.b, 0.0, 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod15_Double_Typical)->Range(64, 4096);

// Small parameters - strong singularity near t=0
static void BM_GaussKronrod15_Float_SmallParams(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.1f, 0.5f, 0.1f, 0.5f, 0.1f, 0.5f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod15Policy>(p.t_upper, p.a, p.b, 0.0f, 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod15_Float_SmallParams)->Range(64, 4096);

static void BM_GaussKronrod15_Double_SmallParams(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.1, 0.5, 0.1, 0.5, 0.1, 0.5);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod15Policy>(p.t_upper, p.a, p.b, 0.0, 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod15_Double_SmallParams)->Range(64, 4096);

// ============================================================================
// Gauss-Kronrod G15-K31 Single-Level Benchmarks (Higher Order)
// ============================================================================

static void BM_GaussKronrod31_Float_Typical(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.1f, 0.9f, 1.0f, 5.0f, 1.0f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod31Policy>(p.t_upper, p.a, p.b, 0.0f, 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod31_Float_Typical)->Range(64, 4096);

static void BM_GaussKronrod31_Double_Typical(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.1, 0.9, 1.0, 5.0, 1.0, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_beta_integrals<GaussKronrod31Policy>(p.t_upper, p.a, p.b, 0.0, 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrod31_Double_Typical)->Range(64, 4096);

// ============================================================================
// Upper Region Quadrature Benchmarks
// ============================================================================

// Helper struct for upper region params
template <typename T>
struct UpperRegionParams {
  T z, t_split, a, b;
};

template <typename T>
std::vector<UpperRegionParams<T>> generate_upper_region_params(
    size_t count, unsigned seed = 42) {
  std::vector<UpperRegionParams<T>> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> z_dist(T(0.5), T(0.95));
  std::uniform_real_distribution<T> split_frac_dist(T(0.3), T(0.7));
  std::uniform_real_distribution<T> ab_dist(T(0.5), T(5.0));
  for (size_t i = 0; i < count; ++i) {
    T z = z_dist(gen);
    T t_split = z * split_frac_dist(gen);
    data[i] = {z, t_split, ab_dist(gen), ab_dist(gen)};
  }
  return data;
}

static void BM_GaussKronrodUpperRegion_Float(benchmark::State& state) {
  auto data = generate_upper_region_params<float>(state.range(0));
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_upper_region<GaussKronrod15Policy>(p.z, p.t_split, p.a, p.b, 0.0f, 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrodUpperRegion_Float)->Range(64, 4096);

static void BM_GaussKronrodUpperRegion_Double(benchmark::State& state) {
  auto data = generate_upper_region_params<double>(state.range(0));
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_upper_region<GaussKronrod15Policy>(p.z, p.t_split, p.a, p.b, 0.0, 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrodUpperRegion_Double)->Range(64, 4096);

// ============================================================================
// Adaptive Quadrature Benchmarks
// ============================================================================

// Easy case - smooth integrand, fast convergence
static void BM_AdaptiveLogWeighted_Float_Easy(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.3f, 0.7f, 2.0f, 5.0f, 2.0f, 5.0f);
  float tol = adaptive_tolerance<float>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Float_Easy)->Range(64, 1024);

static void BM_AdaptiveLogWeighted_Double_Easy(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.3, 0.7, 2.0, 5.0, 2.0, 5.0);
  double tol = adaptive_tolerance<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Double_Easy)->Range(64, 1024);

// Difficult case - small parameters, strong singularity
static void BM_AdaptiveLogWeighted_Float_Difficult(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.1f, 0.5f, 0.1f, 0.5f, 0.1f, 0.5f);
  float tol = adaptive_tolerance_difficult<float>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, true));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Float_Difficult)->Range(64, 512);

static void BM_AdaptiveLogWeighted_Double_Difficult(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.1, 0.5, 0.1, 0.5, 0.1, 0.5);
  double tol = adaptive_tolerance_difficult<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, true));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Double_Difficult)->Range(64, 512);

// High-order vs low-order comparison
static void BM_AdaptiveLogWeighted_Double_LowOrder(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.2, 0.8, 1.0, 3.0, 1.0, 3.0);
  double tol = adaptive_tolerance<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Double_LowOrder)->Range(64, 1024);

static void BM_AdaptiveLogWeighted_Double_HighOrder(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.2, 0.8, 1.0, 3.0, 1.0, 3.0);
  double tol = adaptive_tolerance<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol, true));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveLogWeighted_Double_HighOrder)->Range(64, 1024);

// ============================================================================
// Adaptive Upper Region Benchmarks
// ============================================================================

static void BM_AdaptiveUpperRegion_Float(benchmark::State& state) {
  auto data = generate_upper_region_params<float>(state.range(0));
  float tol = adaptive_tolerance<float>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_upper_region_integrals(p.z, p.t_split, p.a, p.b, tol, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveUpperRegion_Float)->Range(64, 1024);

static void BM_AdaptiveUpperRegion_Double(benchmark::State& state) {
  auto data = generate_upper_region_params<double>(state.range(0));
  double tol = adaptive_tolerance<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_upper_region_integrals(p.z, p.t_split, p.a, p.b, tol, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveUpperRegion_Double)->Range(64, 1024);

// ============================================================================
// Doubly Log-Weighted Integral Benchmarks
// ============================================================================

static void BM_GaussKronrodDoublyWeighted_Float(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.2f, 0.8f, 1.0f, 3.0f, 1.0f, 3.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_doubly_weighted_integrals<GaussKronrod15Policy>(
              p.t_upper, p.a, p.b, 0.0f, 1.0f));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrodDoublyWeighted_Float)->Range(64, 4096);

static void BM_GaussKronrodDoublyWeighted_Double(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.2, 0.8, 1.0, 3.0, 1.0, 3.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          gauss_kronrod_doubly_weighted_integrals<GaussKronrod15Policy>(
              p.t_upper, p.a, p.b, 0.0, 1.0));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_GaussKronrodDoublyWeighted_Double)->Range(64, 4096);

static void BM_AdaptiveDoublyWeighted_Float(benchmark::State& state) {
  auto data = generate_beta_integral_params<float>(state.range(0), 0.2f, 0.8f, 1.0f, 3.0f, 1.0f, 3.0f);
  float tol = adaptive_tolerance<float>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_doubly_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveDoublyWeighted_Float)->Range(64, 512);

static void BM_AdaptiveDoublyWeighted_Double(benchmark::State& state) {
  auto data = generate_beta_integral_params<double>(state.range(0), 0.2, 0.8, 1.0, 3.0, 1.0, 3.0);
  double tol = adaptive_tolerance<double>();
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(
          adaptive_doubly_log_weighted_beta_integrals(p.t_upper, p.a, p.b, tol));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_AdaptiveDoublyWeighted_Double)->Range(64, 512);

// ============================================================================
// Kahan Accumulator Benchmark
// ============================================================================

static void BM_KahanAccumulator_Float(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<float> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(1e-6f, 1e-3f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    KahanAccumulator<float> acc;
    for (const auto& x : data) {
      acc.add(x);
    }
    benchmark::DoNotOptimize(acc.result());
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_KahanAccumulator_Float)->Range(64, 4096);

static void BM_KahanAccumulator_Double(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<double> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(1e-12, 1e-6);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    KahanAccumulator<double> acc;
    for (const auto& x : data) {
      acc.add(x);
    }
    benchmark::DoNotOptimize(acc.result());
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_KahanAccumulator_Double)->Range(64, 4096);

// Baseline: naive summation
static void BM_NaiveSummation_Float(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<float> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(1e-6f, 1e-3f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    float sum = 0.0f;
    for (const auto& x : data) {
      sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_NaiveSummation_Float)->Range(64, 4096);

static void BM_NaiveSummation_Double(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<double> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(1e-12, 1e-6);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& x : data) {
      sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_NaiveSummation_Double)->Range(64, 4096);

BENCHMARK_MAIN();
