#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/hypergeometric_2_f_1.h"

using namespace torchscience::impl::special_functions;

// Helper to generate valid 2F1 parameters
template <typename T>
struct Hyp2F1Params {
  T a, b, c, z;
};

template <typename T>
std::vector<Hyp2F1Params<T>> generate_2f1_params(
    size_t count, T z_min, T z_max, T a_min, T a_max, T b_min, T b_max, T c_min, T c_max, unsigned seed = 42) {
  std::vector<Hyp2F1Params<T>> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> z_dist(z_min, z_max);
  std::uniform_real_distribution<T> a_dist(a_min, a_max);
  std::uniform_real_distribution<T> b_dist(b_min, b_max);
  std::uniform_real_distribution<T> c_dist(c_min, c_max);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {a_dist(gen), b_dist(gen), c_dist(gen), z_dist(gen)};
  }
  return data;
}

// |z| < 0.5 - direct series path
static void BM_Hyp2F1_Float_DirectSeries(benchmark::State& state) {
  auto data = generate_2f1_params<float>(state.range(0), -0.4f, 0.4f, 0.5f, 3.0f, 0.5f, 3.0f, 1.0f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_DirectSeries)->Range(64, 4096);

static void BM_Hyp2F1_Double_DirectSeries(benchmark::State& state) {
  auto data = generate_2f1_params<double>(state.range(0), -0.4, 0.4, 0.5, 3.0, 0.5, 3.0, 1.0, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_DirectSeries)->Range(64, 4096);

// 0.5 <= |z| < 1, |1-z| < |z| - 1-z transformation path (DLMF 15.8.4)
static void BM_Hyp2F1_Float_OneMinusZ(benchmark::State& state) {
  auto data = generate_2f1_params<float>(state.range(0), 0.6f, 0.95f, 0.5f, 3.0f, 0.5f, 3.0f, 1.0f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_OneMinusZ)->Range(64, 4096);

static void BM_Hyp2F1_Double_OneMinusZ(benchmark::State& state) {
  auto data = generate_2f1_params<double>(state.range(0), 0.6, 0.95, 0.5, 3.0, 0.5, 3.0, 1.0, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_OneMinusZ)->Range(64, 4096);

// Negative z - tests 1/z transformation
static void BM_Hyp2F1_Float_NegativeZ(benchmark::State& state) {
  auto data = generate_2f1_params<float>(state.range(0), -0.9f, -0.1f, 0.5f, 3.0f, 0.5f, 3.0f, 1.0f, 5.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_NegativeZ)->Range(64, 4096);

static void BM_Hyp2F1_Double_NegativeZ(benchmark::State& state) {
  auto data = generate_2f1_params<double>(state.range(0), -0.9, -0.1, 0.5, 3.0, 0.5, 3.0, 1.0, 5.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_NegativeZ)->Range(64, 4096);

// Small parameters - fast convergence
static void BM_Hyp2F1_Float_SmallParams(benchmark::State& state) {
  auto data = generate_2f1_params<float>(state.range(0), 0.1f, 0.5f, 0.1f, 1.0f, 0.1f, 1.0f, 0.5f, 2.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_SmallParams)->Range(64, 4096);

static void BM_Hyp2F1_Double_SmallParams(benchmark::State& state) {
  auto data = generate_2f1_params<double>(state.range(0), 0.1, 0.5, 0.1, 1.0, 0.1, 1.0, 0.5, 2.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_SmallParams)->Range(64, 4096);

// Larger parameters - more terms needed
static void BM_Hyp2F1_Float_LargeParams(benchmark::State& state) {
  auto data = generate_2f1_params<float>(state.range(0), 0.1f, 0.5f, 5.0f, 15.0f, 5.0f, 15.0f, 10.0f, 30.0f);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_LargeParams)->Range(64, 4096);

static void BM_Hyp2F1_Double_LargeParams(benchmark::State& state) {
  auto data = generate_2f1_params<double>(state.range(0), 0.1, 0.5, 5.0, 15.0, 5.0, 15.0, 10.0, 30.0);
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_LargeParams)->Range(64, 4096);

// Complex parameters
static void BM_Hyp2F1_Complex64(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::tuple<c10::complex<float>, c10::complex<float>, c10::complex<float>, c10::complex<float>>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 3.0f);
  std::uniform_real_distribution<float> z_dist(0.1f, 0.4f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {
      c10::complex<float>(dist(gen), dist(gen) * 0.1f),
      c10::complex<float>(dist(gen), dist(gen) * 0.1f),
      c10::complex<float>(dist(gen) + 1.0f, dist(gen) * 0.1f),
      c10::complex<float>(z_dist(gen), z_dist(gen) * 0.1f)
    };
  }
  for (auto _ : state) {
    for (const auto& [a, b, c, z] : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(a, b, c, z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Complex64)->Range(64, 4096);

static void BM_Hyp2F1_Complex128(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<std::tuple<c10::complex<double>, c10::complex<double>, c10::complex<double>, c10::complex<double>>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.5, 3.0);
  std::uniform_real_distribution<double> z_dist(0.1, 0.4);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {
      c10::complex<double>(dist(gen), dist(gen) * 0.1),
      c10::complex<double>(dist(gen), dist(gen) * 0.1),
      c10::complex<double>(dist(gen) + 1.0, dist(gen) * 0.1),
      c10::complex<double>(z_dist(gen), z_dist(gen) * 0.1)
    };
  }
  for (auto _ : state) {
    for (const auto& [a, b, c, z] : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(a, b, c, z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Complex128)->Range(64, 4096);

// Special case: a or b is negative integer (terminates)
static void BM_Hyp2F1_Float_Terminating(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<Hyp2F1Params<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.1f, 0.8f);
  std::uniform_int_distribution<int> neg_int_dist(-10, -1);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {
      static_cast<float>(neg_int_dist(gen)),  // a is negative integer
      dist(gen) + 0.5f,
      dist(gen) + 1.0f,
      dist(gen)
    };
  }
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Float_Terminating)->Range(64, 4096);

static void BM_Hyp2F1_Double_Terminating(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<Hyp2F1Params<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.1, 0.8);
  std::uniform_int_distribution<int> neg_int_dist(-10, -1);
  for (size_t i = 0; i < count; ++i) {
    data[i] = {
      static_cast<double>(neg_int_dist(gen)),  // a is negative integer
      dist(gen) + 0.5,
      dist(gen) + 1.0,
      dist(gen)
    };
  }
  for (auto _ : state) {
    for (const auto& p : data) {
      benchmark::DoNotOptimize(hypergeometric_2_f_1(p.a, p.b, p.c, p.z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Hyp2F1_Double_Terminating)->Range(64, 4096);

BENCHMARK_MAIN();
