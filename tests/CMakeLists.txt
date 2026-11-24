# SPDX-License-Identifier: GPL-3.0-or-later
# Each test_<name>.c is a standalone executable registered with CTest.
function(taw_test name)
  add_executable(test_${name} test_${name}.c)
  target_link_libraries(test_${name} PRIVATE tadamw_internal)
  target_include_directories(test_${name} PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

taw_test(version)
taw_test(crc32c)
taw_test(rc)
taw_test(aw)
taw_test(awseek)
taw_test(p1)
taw_test(quant)
taw_test(fit_opt) # shared Newton-Schulz polynomial, bit-exactness oracle
taw_test(gpu)
set_tests_properties(gpu PROPERTIES SKIP_RETURN_CODE 77)

# Dynamic .awseek container tests.
taw_test(dynfit)

taw_test(fitgolden)

# Knob-off archive bytes for every optimizer profile, from the foundation.
add_test(NAME cli_fit_golden
         COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
                 -DDOCS=${PROJECT_SOURCE_DIR}/docs
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_fit_golden_work
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_fit_golden.cmake)

# NAMO arm: pins both W1 paths and the level-5 (s1 > 1) untie mirror.
taw_test(namo)

# Seek path calls taw_model_fit_source directly; the stream golden does not.
add_test(NAME cli_fit_namo_seek
  COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
          "-DSOURCES=${PROJECT_SOURCE_DIR}/docs/format.md|${PROJECT_SOURCE_DIR}/docs/design.md"
          -DWORK=${CMAKE_CURRENT_BINARY_DIR}/fit_namo_work
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_fit_namo_seek.cmake)
set_tests_properties(cli_fit_namo_seek PROPERTIES ENVIRONMENT
  "TAW_HYBRID=namod;TAW_MUON_LR0=0.3;TAW_MUON_LR1=0.015;TAW_VEC_LR0=0.003;TAW_VEC_LR1=0.00015;TAW_NAMO_WD=0.01;TAW_NAMO_BETA=0.999;TAW_NAMO_CLAMP=0.9")
# PR #7 LR-override arm.
taw_test(tuned_lr)
# C3 gate arm: gate math, knob-off equality, mean/shrink, layer/scope goldens.
taw_test(c3_fit)

# EMA/SWA arm and its seek-path CLI test.
taw_test(ema)
add_test(NAME cli_fit_ema_seek
         COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
                 "-DSOURCES=${PROJECT_SOURCE_DIR}/docs/format.md|${PROJECT_SOURCE_DIR}/docs/design.md"
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/fit_ema_work
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_fit_ema_seek.cmake)
set_tests_properties(cli_fit_ema_seek PROPERTIES ENVIRONMENT "TAW_FIT_AVG=ema999")

add_test(NAME cli_version COMMAND taw_cli --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^taw [0-9]+\\.[0-9]+\\.[0-9]+")

add_test(NAME cli_roundtrip
         COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
                 "-DSOURCES=${PROJECT_SOURCE_DIR}/docs/format.md|${PROJECT_SOURCE_DIR}/docs/design.md"
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_work
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_roundtrip.cmake)

# C3 gate through the CLI's stream and awseek fit paths, plus the GPU guard.
add_test(NAME cli_fit_c3
         COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
                 "-DSOURCES=${PROJECT_SOURCE_DIR}/docs/format.md|${PROJECT_SOURCE_DIR}/docs/design.md"
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_fit_c3_work
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_fit_c3.cmake)
set_tests_properties(cli_fit_c3 PROPERTIES
  ENVIRONMENT "TAW_EPS_MODE=c3;TAW_EPS_GATE_MODE=shrink")

# taw-nx (built only with OpenSSL and LZ4): synthetic keys and NSP, no real
# dumps or keys needed.
if(TARGET taw_nx)
  add_executable(test_nx test_nx.c)
  target_include_directories(test_nx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                             ${PROJECT_SOURCE_DIR}/tools/hactool)
  target_compile_definitions(test_nx PRIVATE _DEFAULT_SOURCE _FILE_OFFSET_BITS=64)
  set_target_properties(test_nx PROPERTIES C_EXTENSIONS ON)
  target_link_libraries(test_nx PRIVATE hactool_lib)
  add_test(NAME nx_crypto
           COMMAND ${CMAKE_COMMAND} -E env sh -c
                   "rm -rf '${CMAKE_CURRENT_BINARY_DIR}/nx_work' && mkdir -p '${CMAKE_CURRENT_BINARY_DIR}/nx_work' && '$<TARGET_FILE:test_nx>' '${CMAKE_CURRENT_BINARY_DIR}/nx_work'")
  add_test(NAME cli_nx
           COMMAND ${CMAKE_COMMAND} -DTAWNX=$<TARGET_FILE:taw_nx> -DTAW=$<TARGET_FILE:taw_cli>
                   -DWORK=${CMAKE_CURRENT_BINARY_DIR}/nx_work
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_nx.cmake)
  set_tests_properties(cli_nx PROPERTIES DEPENDS nx_crypto)
endif()
