# SPDX-License-Identifier: GPL-3.0-or-later
# CLI roundtrip: cmake -DTAW=<taw> -DSOURCES=<f1|f2|...> -DWORK=<dir> -P cli_roundtrip.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

# Input: the given files concatenated three times (tens of KiB of text).
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

# Stream variant.
run("${TAW}" c -l 3 "${INPUT}" "${WORK}/in.aw")
run("${TAW}" d "${WORK}/in.aw" "${WORK}/out.aw.bin")
same("${INPUT}" "${WORK}/out.aw.bin")

# Seekable variant, selected by extension, small blocks.
run("${TAW}" c -B 4096 "${INPUT}" "${WORK}/in.awseek")
run("${TAW}" d "${WORK}/in.awseek" "${WORK}/out.seek.bin")
same("${INPUT}" "${WORK}/out.seek.bin")

execute_process(COMMAND "${TAW}" i "${WORK}/in.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "kind: +awseek" OR NOT info MATCHES "block_size: +4096")
  message(FATAL_ERROR "unexpected info output:\n${info}")
endif()

# Range extraction straddling a block boundary.
file(SIZE "${INPUT}" size)
if(size LESS 9000)
  message(FATAL_ERROR "INPUT too small for the range test")
endif()
run("${TAW}" x "${WORK}/in.awseek" 4000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 4000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "range extract mismatch")
endif()

# Out-of-range extraction must fail.
execute_process(COMMAND "${TAW}" x "${WORK}/in.awseek" ${size} 1 "${WORK}/bad.bin"
                RESULT_VARIABLE rc ERROR_QUIET)
if(rc EQUAL 0)
  message(FATAL_ERROR "out-of-range extract succeeded")
endif()

# Profile 1 (the fitted mixer), forced, both weight formats.
foreach(w f16 q8)
  run("${TAW}" c -l 1 -p mix -w ${w} "${INPUT}" "${WORK}/in.${w}.aw")
  run("${TAW}" d "${WORK}/in.${w}.aw" "${WORK}/out.${w}.bin")
  same("${INPUT}" "${WORK}/out.${w}.bin")
  execute_process(COMMAND "${TAW}" i "${WORK}/in.${w}.aw" OUTPUT_VARIABLE info)
  if(NOT info MATCHES "profile: +1 \\(F32_STRICT_MIX\\)" OR NOT info MATCHES "\\(${w}\\)")
    message(FATAL_ERROR "unexpected info output:\n${info}")
  endif()
endforeach()

# Profile 1, seekable, with a range that straddles blocks.
run("${TAW}" c -l 1 -p mix -B 16384 "${INPUT}" "${WORK}/mix.awseek")
run("${TAW}" x "${WORK}/mix.awseek" 15000 5000 "${WORK}/mix_range.bin")
file(READ "${INPUT}" want OFFSET 15000 LIMIT 5000 HEX)
file(READ "${WORK}/mix_range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "profile 1 range extract mismatch")
endif()

# Profile 2 (the chunk stage), stream and seekable.
run("${TAW}" c -l 2 -p chunk "${INPUT}" "${WORK}/in.chunk.aw")
run("${TAW}" d "${WORK}/in.chunk.aw" "${WORK}/out.chunk.bin")
same("${INPUT}" "${WORK}/out.chunk.bin")
execute_process(COMMAND "${TAW}" i "${WORK}/in.chunk.aw" OUTPUT_VARIABLE info)
if(NOT info MATCHES "profile: +2 \\(F32_STRICT_CHUNK\\)")
  message(FATAL_ERROR "unexpected info output:\n${info}")
endif()
run("${TAW}" c -l 1 -p chunk -j 2 -B 16384 "${INPUT}" "${WORK}/chunk.awseek")
run("${TAW}" x "${WORK}/chunk.awseek" 15000 5000 "${WORK}/chunk_range.bin")
file(READ "${INPUT}" want OFFSET 15000 LIMIT 5000 HEX)
file(READ "${WORK}/chunk_range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "profile 2 range extract mismatch")
endif()

# Bad option values are usage errors.
execute_process(COMMAND "${TAW}" c -p bogus "${INPUT}" "${WORK}/bad.aw"
                RESULT_VARIABLE rc ERROR_QUIET)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "bad -p accepted (${rc})")
endif()
