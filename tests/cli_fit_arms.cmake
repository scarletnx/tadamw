# SPDX-License-Identifier: GPL-3.0-or-later
# CLI seek-path check for a fitted profile:
# cmake -DTAW=<taw> -DSOURCES=<f1|f2> -DWORK=<dir> -DPRED=mix|gate
#       -P cli_fit_arms.cmake
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

# fit_all is exercised through taw_model_fit_source, not just taw_compress.
run("${TAW}" c -l 1 -p "${PRED}" -B 4096 "${INPUT}" "${WORK}/in.awseek")
run("${TAW}" d "${WORK}/in.awseek" "${WORK}/out.bin")
same("${INPUT}" "${WORK}/out.bin")

execute_process(COMMAND "${TAW}" i "${WORK}/in.awseek" OUTPUT_VARIABLE info)
if(PRED STREQUAL "cal")
  set(want "profile: +6 \\(CALIBRATED_MIX\\)")
elseif(PRED STREQUAL "gate")
  set(want "profile: +11 \\(ENTROPY_GATE\\)")
elseif(PRED STREQUAL "hnet")
  set(want "profile: +14 \\(HNET_BOUNDARY\\)")
else()
  set(want "profile: +1 \\(F32_STRICT_MIX\\)")
endif()
if(NOT info MATCHES "${want}")
  message(FATAL_ERROR "unexpected profile for ${PRED}:\n${info}")
endif()

# Range extraction that straddles a block boundary.
file(SIZE "${INPUT}" size)
if(size LESS 9000)
  message(FATAL_ERROR "INPUT too small for the range test")
endif()
run("${TAW}" x "${WORK}/in.awseek" 4000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 4000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "${PRED} range extract mismatch")
endif()
