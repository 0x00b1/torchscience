#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/sin_pi.h"

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

// Small values - direct computation path
static void BM_SinPi_Float_Small(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), -100.0f, 100.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Float_Small)->Range(64, 4096);

static void BM_SinPi_Double_Small(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), -100.0, 100.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Double_Small)->Range(64, 4096);

// Integer values - exact 0 path
static void BM_SinPi_Float_Integer(benchmark::State& state) {
  std::vector<float> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(static_cast<int>(i) - static_cast<int>(data.size() / 2));
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Float_Integer)->Range(64, 4096);

static void BM_SinPi_Double_Integer(benchmark::State& state) {
  std::vector<double> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<double>(static_cast<int>(i) - static_cast<int>(data.size() / 2));
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Double_Integer)->Range(64, 4096);

// Half-integer values - exact +/-1 path
static void BM_SinPi_Float_HalfInteger(benchmark::State& state) {
  std::vector<float> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(static_cast<int>(i) - static_cast<int>(data.size() / 2)) + 0.5f;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Float_HalfInteger)->Range(64, 4096);

static void BM_SinPi_Double_HalfInteger(benchmark::State& state) {
  std::vector<double> data(state.range(0));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<double>(static_cast<int>(i) - static_cast<int>(data.size() / 2)) + 0.5;
  }
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Double_HalfInteger)->Range(64, 4096);

// Large values - range reduction path
static void BM_SinPi_Float_Large(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 1e10f, 1e15f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Float_Large)->Range(64, 4096);

static void BM_SinPi_Double_Large(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 1e10, 1e15);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(sin_pi(x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Double_Large)->Range(64, 4096);

// Complex values
static void BM_SinPi_Complex64(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(sin_pi(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Complex64)->Range(64, 4096);

static void BM_SinPi_Complex128(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(sin_pi(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Complex128)->Range(64, 4096);

// Complex with large real part - range reduction path
static void BM_SinPi_Complex64_LargeReal(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> real_dist(1e10f, 1e15f);
  std::uniform_real_distribution<float> imag_dist(-10.0f, 10.0f);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<float>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(sin_pi(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Complex64_LargeReal)->Range(64, 4096);

static void BM_SinPi_Complex128_LargeReal(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> real_dist(1e10, 1e15);
  std::uniform_real_distribution<double> imag_dist(-10.0, 10.0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = c10::complex<double>(real_dist(gen), imag_dist(gen));
  }
  for (auto _ : state) {
    for (const auto& z : data) {
      benchmark::DoNotOptimize(sin_pi(z));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_SinPi_Complex128_LargeReal)->Range(64, 4096);

// Baseline: std::sin(pi * x)
static void BM_StdSin_Pi_Float(benchmark::State& state) {
  const float pi = 3.14159265358979323846f;
  auto data = generate_random_data<float>(state.range(0), -100.0f, 100.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::sin(pi * x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdSin_Pi_Float)->Range(64, 4096);

static void BM_StdSin_Pi_Double(benchmark::State& state) {
  const double pi = 3.14159265358979323846;
  auto data = generate_random_data<double>(state.range(0), -100.0, 100.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::sin(pi * x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdSin_Pi_Double)->Range(64, 4096);

// Baseline for large values - shows precision loss
static void BM_StdSin_Pi_Float_Large(benchmark::State& state) {
  const float pi = 3.14159265358979323846f;
  auto data = generate_random_data<float>(state.range(0), 1e10f, 1e15f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::sin(pi * x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdSin_Pi_Float_Large)->Range(64, 4096);

static void BM_StdSin_Pi_Double_Large(benchmark::State& state) {
  const double pi = 3.14159265358979323846;
  auto data = generate_random_data<double>(state.range(0), 1e10, 1e15);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(std::sin(pi * x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_StdSin_Pi_Double_Large)->Range(64, 4096);

BENCHMARK_MAIN();
