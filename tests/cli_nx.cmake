# SPDX-License-Identifier: GPL-3.0-or-later
# taw-nx on the synthetic NSP from test_nx:
# cmake -DTAWNX=<taw-nx> -DTAW=<taw> -DWORK=<dir> -P cli_nx.cmake
function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed (${rc}): ${ARGN}\n${out}${err}")
  endif()
  set(last_err "${err}" PARENT_SCOPE)
  set(last_out "${out}" PARENT_SCOPE)
endfunction()

set(K "${WORK}/prod.keys")
run("${TAWNX}" enc -l 1 -j 2 -B 1 -k "${K}" "${WORK}/test.nsp" "${WORK}/test.awseek")
if(NOT last_err MATCHES "NCAs: 1 decrypted")
  message(FATAL_ERROR "the NCA was not stored plain:\n${last_err}")
endif()
if(EXISTS "${WORK}/test.awseek.plain")
  message(FATAL_ERROR "the temporary plain image was left behind")
endif()

# An ordinary .awseek: the dump and /.meta, readable with taw.
run("${TAW}" i "${WORK}/test.awseek")
if(NOT last_out MATCHES "files: +2" OR NOT last_out MATCHES "/test.nsp" OR
   NOT last_out MATCHES "/.meta")
  message(FATAL_ERROR "unexpected info:\n${last_out}")
endif()
run("${TAW}" x "${WORK}/test.awseek" /.meta "${WORK}/meta.txt")
file(READ "${WORK}/meta.txt" meta)
if(NOT meta MATCHES "^tadamw-nx 1\nkind nsp\n" OR NOT meta MATCHES "\nnca 56 396288\n")
  message(FATAL_ERROR "unexpected /.meta:\n${meta}")
endif()

run("${TAWNX}" dec -k "${K}" "${WORK}/test.awseek" "${WORK}/back.nsp")
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/test.nsp" "${WORK}/back.nsp"
                RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "decoded NSP differs from the original")
endif()

# The plain form compresses; the encrypted one would not.
file(SIZE "${WORK}/test.nsp" in_size)
file(SIZE "${WORK}/test.awseek" arc_size)
math(EXPR limit "${in_size} / 2")
if(NOT arc_size LESS limit)
  message(FATAL_ERROR "archive not smaller than half the NSP (${arc_size} of ${in_size})")
endif()

# Wrong keys: decoding must fail on the SHA-256 and leave no output.
file(WRITE "${WORK}/wrong.keys"
  "header_key = 00112233445566778899aabbccddeeff0f1e2d3c4b5a69788796a5b4c3d2e1f0\n"
  "key_area_key_application_00 = 00112233445566778899aabbccddeeff\n")
execute_process(COMMAND "${TAWNX}" dec -q -k "${WORK}/wrong.keys" "${WORK}/test.awseek" "${WORK}/bad.nsp"
                RESULT_VARIABLE rc ERROR_QUIET)
if(rc EQUAL 0 OR EXISTS "${WORK}/bad.nsp")
  message(FATAL_ERROR "decoding with wrong keys succeeded or left output")
endif()

# A dump is not an archive.
execute_process(COMMAND "${TAWNX}" dec -q -k "${K}" "${WORK}/test.nsp" "${WORK}/bad2.nsp"
                RESULT_VARIABLE rc ERROR_QUIET OUTPUT_QUIET)
if(rc EQUAL 0 OR EXISTS "${WORK}/bad2.nsp")
  message(FATAL_ERROR "an NSP was accepted as a taw-nx archive")
endif()

# The same dump with opt-in dynamic blocks: taw i/x and taw-nx dec still work.
run("${TAWNX}" enc -l 1 -j 2 -B 1 -D -T 65536 -k "${K}" "${WORK}/test.nsp" "${WORK}/test.dyn.awseek")
execute_process(COMMAND "${TAW}" i "${WORK}/test.dyn.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "dynamic: +yes" OR NOT info MATCHES "files: +2" OR
   NOT info MATCHES "/.meta")
  message(FATAL_ERROR "unexpected dynamic info:\n${info}")
endif()
run("${TAW}" x "${WORK}/test.dyn.awseek" /.meta "${WORK}/dyn_meta.txt")
file(READ "${WORK}/dyn_meta.txt" dyn_meta)
if(NOT dyn_meta STREQUAL meta)
  message(FATAL_ERROR "dynamic /.meta differs from the fixed-block one")
endif()
run("${TAW}" x "${WORK}/test.dyn.awseek" /test.nsp "${WORK}/dyn_dump.bin")
run("${TAW}" x "${WORK}/test.dyn.awseek" 100 500 "${WORK}/dyn_range.bin")
file(READ "${WORK}/dyn_dump.bin" want OFFSET 100 LIMIT 500 HEX)
file(READ "${WORK}/dyn_range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "dynamic named/range extract mismatch")
endif()
run("${TAWNX}" dec -k "${K}" "${WORK}/test.dyn.awseek" "${WORK}/back.dyn.nsp")
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/test.nsp" "${WORK}/back.dyn.nsp"
                 RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "dynamic-block decoded NSP differs from the original")
endif()
