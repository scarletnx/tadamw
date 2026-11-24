# SPDX-License-Identifier: GPL-3.0-or-later
# Archive-byte golden for the default fit and the seven optimizer profiles.
# The corpus is docs/format.md and docs/design.md, encoded at level 3 with
# -g off -j 1. The archives are byte-identical across GCC and Clang builds.
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

# SHA256 of level 3, -g off -j 1 archives over the in-repo docs corpus.
set(GOLDEN_format_default   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_muon   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_blend   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_alt   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_route   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_malt   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_malter   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_design_default   289e2f71e942b04f9ba556aa75a96afae5e00cccad5f8477bc509661ff742073)
set(GOLDEN_design_muon   a1df4af4bb99303073b4cf6d06ad91979953cbd0d9cb7f365ef1685ab0e07e01)
set(GOLDEN_design_blend   25d6b3c3b3c4828e6916db4ca9f10dde62ea4a5dc256a82d73aea6b36015a037)
set(GOLDEN_design_alt   b64bdea4dfa8065dee7a899890d39b6ac264999211a6d280946beb1930c179c5)
set(GOLDEN_design_route   a1df4af4bb99303073b4cf6d06ad91979953cbd0d9cb7f365ef1685ab0e07e01)
set(GOLDEN_design_malt   ef95da2c26630ffb2c68a6e422f61aa7f5891a376c2eec172066252329b87301)
set(GOLDEN_design_malter   5eb31d55afe466c58aabf361be16714650be698954af0b4dec9f0c0c5d9a91a2)

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
