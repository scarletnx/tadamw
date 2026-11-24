# SPDX-License-Identifier: GPL-3.0-or-later
# Fit-arm seek coverage: taw_model_fit_source is reached through the .awseek
# path, which cli_roundtrip's stream golden does not exercise. The NAMO env
# is supplied by the CTest ENVIRONMENT property on cli_fit_namo_seek.
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
set(INPUT "${WORK}/input.bin")
file(WRITE "${INPUT}" "")
foreach(rep 1 2 3)
  string(REPLACE "|" ";" files "${SOURCES}")
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

run("${TAW}" c -l 1 -p mix -B 4096 "${INPUT}" "${WORK}/namo.awseek")
execute_process(COMMAND "${TAW}" i "${WORK}/namo.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "kind: +awseek" OR NOT info MATCHES "profile: +1 \\(F32_STRICT_MIX\\)")
  message(FATAL_ERROR "unexpected info output:\n${info}")
endif()
run("${TAW}" d "${WORK}/namo.awseek" "${WORK}/back.bin")
same("${INPUT}" "${WORK}/back.bin")

file(SIZE "${INPUT}" size)
run("${TAW}" x "${WORK}/namo.awseek" 4000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 4000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "NAMO fit-arm seek range mismatch")
endif()
