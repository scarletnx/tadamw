# SPDX-License-Identifier: GPL-3.0-or-later
# Archive-byte golden for the default fit, the seven optimizer profiles and
# the five FARO arms (routing + ERC, including TAW_FARO_STAT=lin). The corpus
# is docs/format.md and docs/design.md, encoded at level 3 with -g off -j 1.
# The archives are byte-identical across GCC and Clang builds.
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
  "malter|TAW_MALT=malter"
  "faro_muon|TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8"
  "faro_probe|TAW_FARO=1"
  "faro_lin|TAW_FARO=1 TAW_FARO_STAT=lin"
  "faro_adam|TAW_FARO=1 TAW_FARO_ROUTE=adam TAW_FARO_ALLOC=none"
  "faro_layer|TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_BLOCKS=layer")

# SHA256 of level 3, -g off -j 1 archives over the in-repo docs corpus.
set(GOLDEN_format_default   32d545057073a7e87206ed7a83b424cb32d92834834613b275cf15e73488070b)
set(GOLDEN_format_muon      71e377995506080136d987f717fadaaa0ef7d58bf1bf5a89ea350d0bb9301f43)
set(GOLDEN_format_blend     056f127452ee328cb4160f1955a4e25420df751402b879f54cdcdfbbeba902ca)
set(GOLDEN_format_alt       abc2b3a9b8db318ebb3ad413a14689ceabc7f639d7e842ec66aad90abb285cab)
set(GOLDEN_format_route     71e377995506080136d987f717fadaaa0ef7d58bf1bf5a89ea350d0bb9301f43)
set(GOLDEN_format_malt      1277143326f69233ad9ec314566b6c2dc4d48596931e73b7a4809b7257a60a74)
set(GOLDEN_format_malter    3949911775e602f101bce0ad58afbc9b8e80f44e88afc8812c2e1db4066d5600)
set(GOLDEN_format_faro_muon   e41cba51290d62912ff50ed519a2d249037c1e643f9f73e1102ddc01e806e46e)
set(GOLDEN_format_faro_probe  122386f27943f2bf6c50493a2e67addfdbcd0c632b939495d0a8a0b006016c54)
set(GOLDEN_format_faro_lin    122386f27943f2bf6c50493a2e67addfdbcd0c632b939495d0a8a0b006016c54)
set(GOLDEN_format_faro_adam   32d545057073a7e87206ed7a83b424cb32d92834834613b275cf15e73488070b)
set(GOLDEN_format_faro_layer  0ac5b74d93e6d87b7d79c2ea393f220dbbae54f3c265907c08c6ce0fdb4479bf)
set(GOLDEN_design_default   d7c2d0673330527b44584053d0cb96a5ece8ad100a04d3e6ef97004740597539)
set(GOLDEN_design_muon      9711710b3307c906b805816d35249d2dda5562c22847b5c69ecb9d8d630d9ff1)
set(GOLDEN_design_blend     001336e8949474c033d913b23cbd1b1a092190a488b2d03c084e925a2f426795)
set(GOLDEN_design_alt       1e25ed0859fae3cee1cb43dd09e783b88bbbe6547c027d19b0c12b716570f74e)
set(GOLDEN_design_route     9711710b3307c906b805816d35249d2dda5562c22847b5c69ecb9d8d630d9ff1)
set(GOLDEN_design_malt      c161a1f46e93f3da755f9a286cd517449d7e9f60c8c85dc7ed5b0dfb7021c7b6)
set(GOLDEN_design_malter    3402130e4d3d83a0b6b5acb18eb3d906955f1f7905ef471e14ddb66474c5f88f)
set(GOLDEN_design_faro_muon  e22bd04a89b0fdd86aedff66479907f669afb2935727a57d248316f289ca3d75)
set(GOLDEN_design_faro_probe 43efc436dbdfb0a6e772e9caead0729baa23e57632f74288e37a7531a95eb8c5)
set(GOLDEN_design_faro_lin   43efc436dbdfb0a6e772e9caead0729baa23e57632f74288e37a7531a95eb8c5)
set(GOLDEN_design_faro_adam  d7c2d0673330527b44584053d0cb96a5ece8ad100a04d3e6ef97004740597539)
set(GOLDEN_design_faro_layer a694b9adfaa31dfb3adf9737bf4d2269bf444185852d78325a9b13926f00f859)

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
      # FARO arms carry several VAR=VALUE pairs; -E env needs them separate.
      separate_arguments(env NATIVE_COMMAND "${knob}")
      execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${env}
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
