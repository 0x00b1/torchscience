#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/chebyshev_polynomial_t.h"

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

// Integer degree, small values (0-7) - unrolled path
static void BM_ChebyshevT_Float_IntegerSmall(benchmark::State& state) {
  auto z_data = generate_random_data<float>(state.range(0), -1.0f, 1.0f);
  std::vector<float> v_data(state.range(0));
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = static_cast<float>(i % 8);
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Float_IntegerSmall)->Range(64, 4096);

static void BM_ChebyshevT_Double_IntegerSmall(benchmark::State& state) {
  auto z_data = generate_random_data<double>(state.range(0), -1.0, 1.0);
  std::vector<double> v_data(state.range(0));
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = static_cast<double>(i % 8);
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Double_IntegerSmall)->Range(64, 4096);

// Integer degree, larger values - recurrence path
static void BM_ChebyshevT_Float_IntegerLarge(benchmark::State& state) {
  auto z_data = generate_random_data<float>(state.range(0), -1.0f, 1.0f);
  std::vector<float> v_data(state.range(0));
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = static_cast<float>((i % 50) + 10);  // degrees 10-59
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Float_IntegerLarge)->Range(64, 4096);

static void BM_ChebyshevT_Double_IntegerLarge(benchmark::State& state) {
  auto z_data = generate_random_data<double>(state.range(0), -1.0, 1.0);
  std::vector<double> v_data(state.range(0));
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = static_cast<double>((i % 50) + 10);  // degrees 10-59
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Double_IntegerLarge)->Range(64, 4096);

// Non-integer degree, |z| <= 1 - analytic path
static void BM_ChebyshevT_Float_NonInteger(benchmark::State& state) {
  auto z_data = generate_random_data<float>(state.range(0), -0.99f, 0.99f);
  auto v_data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Float_NonInteger)->Range(64, 4096);

static void BM_ChebyshevT_Double_NonInteger(benchmark::State& state) {
  auto z_data = generate_random_data<double>(state.range(0), -0.99, 0.99);
  auto v_data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Double_NonInteger)->Range(64, 4096);

// Non-integer degree, z > 1 - hyperbolic path
static void BM_ChebyshevT_Float_Hyperbolic(benchmark::State& state) {
  auto z_data = generate_random_data<float>(state.range(0), 1.1f, 5.0f);
  auto v_data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Float_Hyperbolic)->Range(64, 4096);

static void BM_ChebyshevT_Double_Hyperbolic(benchmark::State& state) {
  auto z_data = generate_random_data<double>(state.range(0), 1.1, 5.0);
  auto v_data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Double_Hyperbolic)->Range(64, 4096);

// Complex values
static void BM_ChebyshevT_Complex64(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<float>> v_data(count);
  std::vector<c10::complex<float>> z_data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (size_t i = 0; i < count; ++i) {
    v_data[i] = c10::complex<float>(dist(gen), dist(gen));
    z_data[i] = c10::complex<float>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Complex64)->Range(64, 4096);

static void BM_ChebyshevT_Complex128(benchmark::State& state) {
  size_t count = state.range(0);
  std::vector<c10::complex<double>> v_data(count);
  std::vector<c10::complex<double>> z_data(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  for (size_t i = 0; i < count; ++i) {
    v_data[i] = c10::complex<double>(dist(gen), dist(gen));
    z_data[i] = c10::complex<double>(dist(gen), dist(gen));
  }
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t(v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Complex128)->Range(64, 4096);

// Backward pass
static void BM_ChebyshevT_Backward_Float(benchmark::State& state) {
  auto z_data = generate_random_data<float>(state.range(0), -0.99f, 0.99f);
  auto v_data = generate_random_data<float>(state.range(0), 0.5f, 10.5f);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t_backward(1.0f, v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Backward_Float)->Range(64, 4096);

static void BM_ChebyshevT_Backward_Double(benchmark::State& state) {
  auto z_data = generate_random_data<double>(state.range(0), -0.99, 0.99);
  auto v_data = generate_random_data<double>(state.range(0), 0.5, 10.5);
  for (auto _ : state) {
    for (size_t i = 0; i < z_data.size(); ++i) {
      benchmark::DoNotOptimize(chebyshev_polynomial_t_backward(1.0, v_data[i], z_data[i]));
    }
  }
  state.SetItemsProcessed(state.iterations() * z_data.size());
}
BENCHMARK(BM_ChebyshevT_Backward_Double)->Range(64, 4096);

BENCHMARK_MAIN();
