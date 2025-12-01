function(add_special_function_test NAME SOURCE)
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
    GTest::gtest_main
    ${TORCH_LIBRARIES}
  )

  set_target_properties(
    ${NAME}
    PROPERTIES
    CXX_STANDARD 17
  )

  add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_special_function_test(
  sin_pi_test
  torchscience/csrc/impl/special_functions/sin_pi_test.cpp
)

add_special_function_test(
  cos_pi_test
  torchscience/csrc/impl/special_functions/cos_pi_test.cpp
)

add_special_function_test(
  is_nonpositive_integer_test
  torchscience/csrc/impl/special_functions/is_nonpositive_integer_test.cpp
)

add_special_function_test(
  lanczos_approximation_test
  torchscience/csrc/impl/special_functions/lanczos_approximation_test.cpp
)

add_special_function_test(
  factorial_test
  torchscience/csrc/impl/special_functions/factorial_test.cpp
)

add_special_function_test(
  gamma_test
  torchscience/csrc/impl/special_functions/gamma_test.cpp
)

add_special_function_test(
  log_gamma_test
  torchscience/csrc/impl/special_functions/log_gamma_test.cpp
)

add_special_function_test(
  sign_gamma_test
  torchscience/csrc/impl/special_functions/sign_gamma_test.cpp
)

add_special_function_test(
  digamma_test
  torchscience/csrc/impl/special_functions/digamma_test.cpp
)

add_special_function_test(
  trigamma_test
  torchscience/csrc/impl/special_functions/trigamma_test.cpp
)

add_special_function_test(
  tetragamma_test
  torchscience/csrc/impl/special_functions/tetragamma_test.cpp
)

add_special_function_test(
  pentagamma_test
  torchscience/csrc/impl/special_functions/pentagamma_test.cpp
)

add_special_function_test(
  chebyshev_polynomial_t_test
  torchscience/csrc/impl/special_functions/chebyshev_polynomial_t_test.cpp
)

add_special_function_test(
  adaptive_quadrature_test
  torchscience/csrc/impl/special_functions/adaptive_quadrature_test.cpp
)

add_special_function_test(
  hypergeometric_2_f_1_test
  torchscience/csrc/impl/special_functions/hypergeometric_2_f_1_test.cpp
)

add_special_function_test(
  incomplete_beta_test
  torchscience/csrc/impl/special_functions/incomplete_beta_test.cpp
)
