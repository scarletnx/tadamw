# SPDX-License-Identifier: GPL-3.0-or-later
# X6 seek-path fit golden for profile 14: cmake -DTAW=<taw> -DWORK=<dir>
# -P cli_fit_hnet_seek.cmake. ENVIRONMENT pins the knob-off arms. The source
# is generated below, not passed in; no -DSOURCES argument exists.
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

# Fixed synthetic source: a golden must not change when the docs it happens
# to be built from are edited. A deterministic counter keeps the text varied
# (a purely repeated line stores almost without fitting). ~70 KiB, enough
# for the range extract below.
set(INPUT "${WORK}/input.bin")
set(payload "")
foreach(i RANGE 0 699)
  math(EXPR k "(${i} * 1103515245 + 12345) % 2147483647")
  string(APPEND payload "line ${i} seed ${k}: the quick brown fox jumps over the lazy dog; H-Net marks a boundary.\n")
endforeach()
file(WRITE "${INPUT}" "${payload}")

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

run("${TAW}" c -l 1 -p hnet -B 16384 "${INPUT}" "${WORK}/hnet.awseek")
file(SIZE "${WORK}/hnet.awseek" arc_n)
if(NOT arc_n EQUAL 4535)
  message(FATAL_ERROR "hnet seek golden: arc_n ${arc_n}, expected 4535")
endif()

execute_process(COMMAND "${TAW}" i "${WORK}/hnet.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "profile: +14 \\(HNET_BOUNDARY\\)")
  message(FATAL_ERROR "unexpected profile in info:\n${info}")
endif()
if(NOT info MATCHES "model_size: +([0-9]+)")
  message(FATAL_ERROR "no model_size in info:\n${info}")
endif()
set(model_n "${CMAKE_MATCH_1}")
if(NOT model_n EQUAL 240)
  message(FATAL_ERROR "hnet seek golden: model_size ${model_n}, expected 240")
endif()

run("${TAW}" d "${WORK}/hnet.awseek" "${WORK}/out.bin")
same("${INPUT}" "${WORK}/out.bin")
run("${TAW}" x "${WORK}/hnet.awseek" 15000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 15000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "hnet seek range extract mismatch")
endif()
