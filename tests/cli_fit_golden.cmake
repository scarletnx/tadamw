# SPDX-License-Identifier: GPL-3.0-or-later
# Knob-off archive bytes and the seven optimizer profiles. The corpus here is
# the in-repo docs/format.md and docs/design.md; both gained central registry
# sections during foundation coalescing, so these hashes were re-measured on
# the coalesced tree (GCC 16.2.1 and Clang 23.1.5 produce identical bytes).
# Fitter byte-identity to origin/main is pinned separately by golden_corpus.sh
# on the fixed external corpora (text1m/libc1m); a fitter regression breaks
# that suite and/or these hashes.
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
set(GOLDEN_format_default   0bf2ba79efa2efdd797c74f2dfda9bcbd5117a03bc70d4983fd1b8e574ca5296)
set(GOLDEN_format_muon   24640c08e700bfe46058c36dcb5e97d53ced5a2f983e9909b215a3b724eed350)
set(GOLDEN_format_blend   4414e5c2f89780bdffd037731eedc9bd54cad92a58abbdf457702f1751006ab7)
set(GOLDEN_format_alt   ff18b42d95f490a136c4ba27341d56cb437b3756d0e17d233fb482477c590074)
set(GOLDEN_format_route   24640c08e700bfe46058c36dcb5e97d53ced5a2f983e9909b215a3b724eed350)
set(GOLDEN_format_malt   b978f4a95c92307b7a824d9e0fc31c45daffeff612fa9738968a981f056546ab)
set(GOLDEN_format_malter   c1e417958c809b12a44ed5e6b5482743bb1080e6612d085166c4b6d5d402e50c)
set(GOLDEN_design_default   928ace1d47483efad126788faf128a80e9a9964d4fff095c96efee9d9169dab6)
set(GOLDEN_design_muon   27dadbda5b4b7387bbe5e8109204ebe0ddd279b6dd626613b47e4b44b6597392)
set(GOLDEN_design_blend   c8f039da2fb5b5be0840833bfae40f112e421368e06275c0fc68af565120c980)
set(GOLDEN_design_alt   dc58dcaa6daa2cb47dfde36880b7c384fcf5dd1f514d348f05397533de6380f5)
set(GOLDEN_design_route   27dadbda5b4b7387bbe5e8109204ebe0ddd279b6dd626613b47e4b44b6597392)
set(GOLDEN_design_malt   7ba19da6356be2da5ef1343b66fb90086c8cf3e8bdfd2d929c2d1b83df33bb52)
set(GOLDEN_design_malter   0241f4f5461367a7477069ab763df518a8083c649671aa1c564b5205216cb8bc)

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
