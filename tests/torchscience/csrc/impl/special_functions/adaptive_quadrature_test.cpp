#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/adaptive_quadrature.h"

using namespace torchscience::impl::special_functions;

class AdaptiveQuadratureTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Gauss-Legendre Constants Tests
// ============================================================================

TEST_F(AdaptiveQuadratureTest, GaussLegendre_NodesInRange) {
  // All nodes should be in [0, 1) (positive half)
  for (int i = 0; i < 16; ++i) {
    EXPECT_GE(gauss_legendre::kNodes[i], 0.0);
    EXPECT_LE(gauss_legendre::kNodes[i], 1.0);
  }
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_WeightsPositive) {
  // All weights should be positive
  for (int i = 0; i < 16; ++i) {
    EXPECT_GT(gauss_legendre::kWeights[i], 0.0);
  }
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_WeightsSum) {
  // Sum of all weights should equal 2 (since we're using half-interval values)
  double sum = 0.0;
  for (int i = 0; i < 16; ++i) {
    sum += 2.0 * gauss_legendre::kWeights[i];  // Both positive and negative
  }
  EXPECT_NEAR(sum, 2.0, 1e-14);
}

// ============================================================================
// Gauss-Kronrod K15 Constants Tests
// ============================================================================

TEST_F(AdaptiveQuadratureTest, GaussKronrod_K15_NodesInRange) {
  for (int i = 0; i < 8; ++i) {
    EXPECT_GE(gauss_kronrod::kK15Nodes[i], 0.0);
    EXPECT_LE(gauss_kronrod::kK15Nodes[i], 1.0);
  }
}

TEST_F(AdaptiveQuadratureTest, GaussKronrod_K15_WeightsPositive) {
  for (int i = 0; i < 8; ++i) {
    EXPECT_GT(gauss_kronrod::kK15Weights[i], 0.0);
  }
}

TEST_F(AdaptiveQuadratureTest, GaussKronrod_K15_G7WeightsPositive) {
  for (int i = 0; i < 4; ++i) {
    EXPECT_GT(gauss_kronrod::kG7Weights[i], 0.0);
  }
}

// ============================================================================
// Gauss-Kronrod K31 Constants Tests
// ============================================================================

TEST_F(AdaptiveQuadratureTest, GaussKronrod_K31_NodesInRange) {
  for (int i = 0; i < 16; ++i) {
    EXPECT_GE(gauss_kronrod::kK31Nodes[i], 0.0);
    EXPECT_LE(gauss_kronrod::kK31Nodes[i], 1.0);
  }
}

TEST_F(AdaptiveQuadratureTest, GaussKronrod_K31_WeightsPositive) {
  for (int i = 0; i < 16; ++i) {
    EXPECT_GT(gauss_kronrod::kK31Weights[i], 0.0);
  }
}

// ============================================================================
// Basic Integration Tests (using Gauss-Legendre)
// ============================================================================

// Simple test using hand-computed quadrature
template <typename Func>
double simple_gauss_legendre_32(Func f, double a, double b) {
  double c = (b - a) / 2.0;
  double d = (b + a) / 2.0;
  double sum = 0.0;

  for (int i = 0; i < 16; ++i) {
    double xi_pos = d + c * gauss_legendre::kNodes[i];
    double xi_neg = d - c * gauss_legendre::kNodes[i];
    sum += gauss_legendre::kWeights[i] * (f(xi_pos) + f(xi_neg));
  }

  return c * sum;
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_IntegratePolynomial) {
  // Integrate x^2 from 0 to 1, should be 1/3
  auto f = [](double x) { return x * x; };
  double result = simple_gauss_legendre_32(f, 0.0, 1.0);
  EXPECT_NEAR(result, 1.0 / 3.0, 1e-14);
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_IntegrateCubic) {
  // Integrate x^3 from 0 to 1, should be 1/4
  auto f = [](double x) { return x * x * x; };
  double result = simple_gauss_legendre_32(f, 0.0, 1.0);
  EXPECT_NEAR(result, 0.25, 1e-14);
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_IntegrateSine) {
  // Integrate sin(x) from 0 to pi, should be 2
  auto f = [](double x) { return std::sin(x); };
  double result = simple_gauss_legendre_32(f, 0.0, kPi);
  EXPECT_NEAR(result, 2.0, 1e-12);
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_IntegrateExponential) {
  // Integrate exp(x) from 0 to 1, should be e - 1
  auto f = [](double x) { return std::exp(x); };
  double result = simple_gauss_legendre_32(f, 0.0, 1.0);
  EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-14);
}

TEST_F(AdaptiveQuadratureTest, GaussLegendre_IntegrateGaussian) {
  // Integrate exp(-x^2) from -1 to 1
  auto f = [](double x) { return std::exp(-x * x); };
  double result = simple_gauss_legendre_32(f, -1.0, 1.0);
  // Expected value from numerical tables
  double expected = 1.4936482656248540507989348722637;
  EXPECT_NEAR(result, expected, 1e-12);
}

// ============================================================================
// K15 Rule Tests
// ============================================================================

// Simple K15 quadrature
template <typename Func>
double simple_k15(Func f, double a, double b) {
  double c = (b - a) / 2.0;
  double d = (b + a) / 2.0;
  double sum = 0.0;

  // Node 0 is at center (weight applies once)
  sum += gauss_kronrod::kK15Weights[0] * f(d);

  // Other nodes are symmetric
  for (int i = 1; i < 8; ++i) {
    double xi_pos = d + c * gauss_kronrod::kK15Nodes[i];
    double xi_neg = d - c * gauss_kronrod::kK15Nodes[i];
    sum += gauss_kronrod::kK15Weights[i] * (f(xi_pos) + f(xi_neg));
  }

  return c * sum;
}

TEST_F(AdaptiveQuadratureTest, K15_IntegratePolynomial) {
  auto f = [](double x) { return x * x; };
  double result = simple_k15(f, 0.0, 1.0);
  EXPECT_NEAR(result, 1.0 / 3.0, 1e-14);
}

TEST_F(AdaptiveQuadratureTest, K15_IntegrateSine) {
  auto f = [](double x) { return std::sin(x); };
  double result = simple_k15(f, 0.0, kPi);
  EXPECT_NEAR(result, 2.0, 1e-10);
}

// ============================================================================
// K31 Rule Tests
// ============================================================================

template <typename Func>
double simple_k31(Func f, double a, double b) {
  double c = (b - a) / 2.0;
  double d = (b + a) / 2.0;
  double sum = 0.0;

  // Node 0 is at center
  sum += gauss_kronrod::kK31Weights[0] * f(d);

  // Other nodes are symmetric
  for (int i = 1; i < 16; ++i) {
    double xi_pos = d + c * gauss_kronrod::kK31Nodes[i];
    double xi_neg = d - c * gauss_kronrod::kK31Nodes[i];
    sum += gauss_kronrod::kK31Weights[i] * (f(xi_pos) + f(xi_neg));
  }

  return c * sum;
}

TEST_F(AdaptiveQuadratureTest, K31_IntegratePolynomial) {
  auto f = [](double x) { return x * x; };
  double result = simple_k31(f, 0.0, 1.0);
  EXPECT_NEAR(result, 1.0 / 3.0, 1e-14);
}

TEST_F(AdaptiveQuadratureTest, K31_IntegrateSine) {
  auto f = [](double x) { return std::sin(x); };
  double result = simple_k31(f, 0.0, kPi);
  EXPECT_NEAR(result, 2.0, 1e-12);
}

TEST_F(AdaptiveQuadratureTest, K31_IntegrateExponential) {
  auto f = [](double x) { return std::exp(x); };
  double result = simple_k31(f, 0.0, 1.0);
  EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-14);
}

// ============================================================================
// Comparison Tests (K15 vs K31 for smooth functions)
// ============================================================================

TEST_F(AdaptiveQuadratureTest, K15vsK31_SmoothFunction) {
  auto f = [](double x) { return std::exp(-x * x); };
  double k15_result = simple_k15(f, -1.0, 1.0);
  double k31_result = simple_k31(f, -1.0, 1.0);

  // Both should give similar results for smooth functions
  EXPECT_NEAR(k15_result, k31_result, 1e-10);
}

// ============================================================================
// Accuracy Order Tests
// ============================================================================

TEST_F(AdaptiveQuadratureTest, GaussLegendre_HighDegreePolynomial) {
  // 32-point Gauss-Legendre is exact for polynomials up to degree 63
  // Test x^30
  auto f = [](double x) { return std::pow(x, 30); };
  double result = simple_gauss_legendre_32(f, 0.0, 1.0);
  double expected = 1.0 / 31.0;
  EXPECT_NEAR(result, expected, 1e-13);
}

// ============================================================================
// Node Ordering Tests
// ============================================================================

TEST_F(AdaptiveQuadratureTest, GaussLegendre_NodesIncreasing) {
  for (int i = 1; i < 16; ++i) {
    EXPECT_GT(gauss_legendre::kNodes[i], gauss_legendre::kNodes[i - 1]);
  }
}

TEST_F(AdaptiveQuadratureTest, K15_NodesIncreasing) {
  for (int i = 1; i < 8; ++i) {
    EXPECT_GT(gauss_kronrod::kK15Nodes[i], gauss_kronrod::kK15Nodes[i - 1]);
  }
}

TEST_F(AdaptiveQuadratureTest, K31_NodesIncreasing) {
  for (int i = 1; i < 16; ++i) {
    EXPECT_GT(gauss_kronrod::kK31Nodes[i], gauss_kronrod::kK31Nodes[i - 1]);
  }
}

// ============================================================================
// Float vs Double Consistency
// ============================================================================

TEST_F(AdaptiveQuadratureTest, FloatDoubleConsistency) {
  auto f_double = [](double x) { return std::sin(x); };
  auto f_float = [](float x) { return std::sin(x); };

  double result_double = simple_k15(f_double, 0.0, kPi);
  float result_float = static_cast<float>(simple_k15(
      [&](double x) { return static_cast<double>(f_float(static_cast<float>(x))); },
      0.0, kPi));

  EXPECT_NEAR(result_float, result_double, 1e-5);
}
