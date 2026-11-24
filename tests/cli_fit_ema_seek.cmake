# SPDX-License-Identifier: GPL-3.0-or-later
# CLI seek-path check for an EMA-fitted profile: exercises taw_model_fit_source
# (not just taw_compress). ENVIRONMENT sets TAW_FIT_AVG=ema999.
# cmake -DTAW=<taw> -DSOURCES=<f1|f2> -DWORK=<dir> -P cli_fit_ema_seek.cmake
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

# Fitted mixer, seekable variant with small blocks.
run("${TAW}" c -l 1 -p mix -B 4096 "${INPUT}" "${WORK}/ema.awseek")
run("${TAW}" d "${WORK}/ema.awseek" "${WORK}/out.bin")
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files
                "${INPUT}" "${WORK}/out.bin" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "EMA seek archive did not decode back to the input")
endif()

execute_process(COMMAND "${TAW}" i "${WORK}/ema.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "kind: +awseek" OR
   NOT info MATCHES "profile: +1 \\(F32_STRICT_MIX\\)" OR
   NOT info MATCHES "block_size: +4096")
  message(FATAL_ERROR "unexpected EMA seek archive info:\n${info}")
endif()

# Range extraction straddling a block boundary.
file(SIZE "${INPUT}" size)
if(size LESS 9000)
  message(FATAL_ERROR "INPUT too small for the range test")
endif()
run("${TAW}" x "${WORK}/ema.awseek" 4000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 4000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "EMA seek range extract mismatch")
endif()
