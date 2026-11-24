# SPDX-License-Identifier: GPL-3.0-or-later
# C3 gate through the CLI (both stream and awseek fit paths). Env arms are set
# by the CTest ENVIRONMENT property, so this exercises the real knob plumbing.
# cmake -DTAW=<taw> -DSOURCES=<f1|f2> -DWORK=<dir> -P cli_fit_c3.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

set(INPUT "${WORK}/input.bin")
file(WRITE "${INPUT}" "")
string(REPLACE "|" ";" files "${SOURCES}")
foreach(rep 1 2 3)
  foreach(f IN LISTS files)
    file(READ "${f}" chunk)
    file(APPEND "${INPUT}" "${chunk}")
  endforeach()
endforeach()

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed (${rc}): ${ARGN}\n${err}")
  endif()
endfunction()

function(same a b)
  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${a}" "${b}"
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "files differ: ${a} ${b}")
  endif()
endfunction()

# Stream fit with the gate enabled.
run("${TAW}" c -l 1 -p mix "${INPUT}" "${WORK}/c3.aw")
run("${TAW}" d "${WORK}/c3.aw" "${WORK}/c3.out")
same("${INPUT}" "${WORK}/c3.out")
execute_process(COMMAND "${TAW}" i "${WORK}/c3.aw" OUTPUT_VARIABLE info)
if(NOT info MATCHES "profile: +1 \\(F32_STRICT_MIX\\)")
  message(FATAL_ERROR "gate arm did not produce the fitted mixer:\n${info}")
endif()

# awseek calls taw_model_fit_source directly; the gate must reach that path.
run("${TAW}" c -l 1 -p mix -B 4096 "${INPUT}" "${WORK}/c3.awseek")
run("${TAW}" d "${WORK}/c3.awseek" "${WORK}/c3seek.out")
same("${INPUT}" "${WORK}/c3seek.out")

# GPU guard: require the guard's OWN diagnostic, not just any failure —
# a GPU-less CI would otherwise pass vacuously on "no Vulkan device".
execute_process(COMMAND "${TAW}" c -l 1 -p mix -g on "${INPUT}" "${WORK}/no.aw"
                RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0 AND err MATCHES "TAW_EPS_MODE=c3 is CPU-only")
  # guard fired as intended
elseif(rc EQUAL 0)
  message(FATAL_ERROR "TAW_EPS_MODE=c3 with -g on was not rejected")
else()
  message(FATAL_ERROR "guard did not fire (rc=${rc}): ${err}")
endif()

# c2/nc9/junk are errors, not implicit gate enables.
foreach(bad c2 nc9 nonsense)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env "TAW_EPS_MODE=${bad}"
                  "${TAW}" c -l 1 -p mix "${INPUT}" "${WORK}/bad_${bad}.aw"
                  RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR "TAW_EPS_MODE=${bad} was accepted (rc=${rc}): ${err}")
  endif()
  if(NOT err MATCHES "TAW_EPS_MODE=${bad} is invalid")
    message(FATAL_ERROR "unexpected diagnostic for ${bad}: ${err}")
  endif()
endforeach()

# Same knob with GPU off must still succeed (proves the failure is the guard).
# run() FATAL_ERRORs unless the encode/decode return codes are exactly 0.
run("${TAW}" c -l 1 -p mix -g off "${INPUT}" "${WORK}/cpu_ok.aw")
run("${TAW}" d "${WORK}/cpu_ok.aw" "${WORK}/cpu_ok.out")
same("${INPUT}" "${WORK}/cpu_ok.out")
