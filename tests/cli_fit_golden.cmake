# SPDX-License-Identifier: GPL-3.0-or-later
# Knob-off archive bytes and the seven optimizer profiles. Hashes were
# recorded from unmodified origin/main @ 1aa3dd7, built with both GCC and
# Clang; both produced the same bytes. Any change to the shipped fitting
# arithmetic (including the Newton-Schulz extraction) breaks these.
#
# cmake -DTAW=<taw> -DDOCS=<docs dir> -DWORK=<dir> -P cli_fit_golden.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

# profile tag | environment override (empty = shipped defaults)
set(PROFILES
  "default|"
  "muon|TAW_HYBRID=muon"
  "blend|TAW_HYBRID=blend"
  "alt|TAW_HYBRID=alt"
  "route|TAW_HYBRID=route"
  "malt|TAW_MALT=malt"
  "malter|TAW_MALT=malter")

# SHA256 of the archive from baseline main, level 3, -g off -j 1.
set(GOLDEN_format_default   5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_muon      5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_blend     5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_alt       5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_route     5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_malt      5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_format_malter    5e100e1616c204ccffb9ce7cc37221f12c5a41bb045cf5d78db89b5c461aa29a)
set(GOLDEN_design_default   aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)
set(GOLDEN_design_muon      aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)
set(GOLDEN_design_blend     aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)
set(GOLDEN_design_alt       aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)
set(GOLDEN_design_route     aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)
set(GOLDEN_design_malt      d4c096f60af6272458bcfd69f84d4fb97835c0705819f24948c9cc7ecfa4daf3)
set(GOLDEN_design_malter    aa9bcef65f01fce145dda8ebc3eed4a83a4a079b27cb75b3aff07787d9956370)

foreach(corpus format design)
  foreach(entry IN LISTS PROFILES)
    string(REPLACE "|" ";" kv "${entry}")
    list(GET kv 0 tag)
    list(GET kv 1 knob)
    set(out "${WORK}/${corpus}-${tag}.aw")
    if(knob STREQUAL "")
      execute_process(COMMAND "${TAW}" c -l 3 -g off -j 1
                      "${DOCS}/${corpus}.md" "${out}"
                      RESULT_VARIABLE rc ERROR_VARIABLE err)
    else()
      execute_process(COMMAND "${CMAKE_COMMAND}" -E env "${knob}"
                      "${TAW}" c -l 3 -g off -j 1
                      "${DOCS}/${corpus}.md" "${out}"
                      RESULT_VARIABLE rc ERROR_VARIABLE err)
    endif()
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "encode ${corpus}-${tag} failed (${rc}): ${err}")
    endif()
    file(SHA256 "${out}" sha)
    if(NOT sha STREQUAL "${GOLDEN_${corpus}_${tag}}")
      message(FATAL_ERROR
        "archive changed for ${corpus}-${tag}: got ${sha}, want ${GOLDEN_${corpus}_${tag}}")
    endif()
  endforeach()
endforeach()
