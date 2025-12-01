function(add_benchmark NAME SOURCE)
  add_executable(${NAME} ${SOURCE})

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/torchscience/csrc
    ${TORCH_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${NAME}
    PRIVATE
    benchmark::benchmark
    ${TORCH_LIBRARIES}
  )

  set_target_properties(
    ${NAME}
    PROPERTIES
    CXX_STANDARD 17
  )
endfunction()

add_benchmark(
  cos_pi_benchmark
  torchscience/csrc/impl/special_functions/cos_pi_benchmark.cpp
)

add_benchmark(
  sin_pi_benchmark
  torchscience/csrc/impl/special_functions/sin_pi_benchmark.cpp
)

add_benchmark(
  factorial_benchmark
  torchscience/csrc/impl/special_functions/factorial_benchmark.cpp
)

add_benchmark(
  digamma_benchmark
  torchscience/csrc/impl/special_functions/digamma_benchmark.cpp
)

add_benchmark(
  trigamma_benchmark
  torchscience/csrc/impl/special_functions/trigamma_benchmark.cpp
)

add_benchmark(
  tetragamma_benchmark
  torchscience/csrc/impl/special_functions/tetragamma_benchmark.cpp
)

add_benchmark(
  pentagamma_benchmark
  torchscience/csrc/impl/special_functions/pentagamma_benchmark.cpp
)

add_benchmark(
  gamma_benchmark
  torchscience/csrc/impl/special_functions/gamma_benchmark.cpp
)

add_benchmark(
  log_gamma_benchmark
  torchscience/csrc/impl/special_functions/log_gamma_benchmark.cpp
)

add_benchmark(
  sign_gamma_benchmark
  torchscience/csrc/impl/special_functions/sign_gamma_benchmark.cpp
)

add_benchmark(
  chebyshev_polynomial_t_benchmark
  torchscience/csrc/impl/special_functions/chebyshev_polynomial_t_benchmark.cpp
)

add_benchmark(
  incomplete_beta_benchmark
  torchscience/csrc/impl/special_functions/incomplete_beta_benchmark.cpp
)

add_benchmark(
  hypergeometric_2_f_1_benchmark
  torchscience/csrc/impl/special_functions/hypergeometric_2_f_1_benchmark.cpp
)

add_benchmark(
  lanczos_approximation_benchmark
  torchscience/csrc/impl/special_functions/lanczos_approximation_benchmark.cpp
)

add_benchmark(
  is_nonpositive_integer_benchmark
  torchscience/csrc/impl/special_functions/is_nonpositive_integer_benchmark.cpp
)

add_benchmark(
  adaptive_quadrature_benchmark
  torchscience/csrc/impl/special_functions/adaptive_quadrature_benchmark.cpp
)
