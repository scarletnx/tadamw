# SPDX-License-Identifier: GPL-3.0-or-later
# Archive-byte golden for the default fit, the seven optimizer profiles and
# the five FARO arms (routing + ERC, including TAW_FARO_STAT=lin). The corpus
# is docs/format.md and docs/design.md, encoded at level 3 with -g off -j 1.
# The archives are byte-identical across GCC and Clang builds, and the FARO
# arms match upstream main c89ca25 byte for byte on the same corpus.
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
set(GOLDEN_format_default   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_muon      8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_blend     8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_alt       8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_route     8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_malt      8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_malter    8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_faro_muon   e25789f644b7b0d139db7e45d488b20dfcda9a8b260bad4be03bcad0b8061c2e)
set(GOLDEN_format_faro_probe  9b2312139957d149533e209ab6dd152669b4242646513a599fa66f535faa0c9b)
set(GOLDEN_format_faro_lin    9b2312139957d149533e209ab6dd152669b4242646513a599fa66f535faa0c9b)
set(GOLDEN_format_faro_adam   8bd87b13f1e72d8104b1a3e2914f0c92f865223dc9b6a99e86aaf1269b29d952)
set(GOLDEN_format_faro_layer  252fce62987325e2ead2ddd5794e31b6f8f486eb6b5c33aa6152412e97d99678)
set(GOLDEN_design_default   73d7554fe74bf6301de9d4da588ae2d515bb129295ed4a24b878bce416694695)
set(GOLDEN_design_muon      35936d4ae7a7a1c3e3a2fc70eee3cd3450ae7f7349f23c5b3ce5529742247d45)
set(GOLDEN_design_blend     418cbad3171ff31b0414b20ed003a12020b9bdfff50a6cb3d3105860e3c95a08)
set(GOLDEN_design_alt       a0cf55223988004654c5ab4a188724880d29db3acaaa3107b0f928277edcf79c)
set(GOLDEN_design_route     35936d4ae7a7a1c3e3a2fc70eee3cd3450ae7f7349f23c5b3ce5529742247d45)
set(GOLDEN_design_malt      7770af709c9e1d6cb4eeedd51da2c64c12e9d981ebf88476a0adeeb7d93342f9)
set(GOLDEN_design_malter    c51e36df9ac60ce1aad6c2a04f719c624c67ed5de0e4061cccfb53f3e4137cdc)
set(GOLDEN_design_faro_muon  de920cf5140628b60cdfa47218464b68b185453eb707a976ae310b0229cda638)
set(GOLDEN_design_faro_probe 829edb2dfdc3729ceba86498c876484391ae9d0011e23b28e966afe4f99b8a71)
set(GOLDEN_design_faro_lin   829edb2dfdc3729ceba86498c876484391ae9d0011e23b28e966afe4f99b8a71)
set(GOLDEN_design_faro_adam  73d7554fe74bf6301de9d4da588ae2d515bb129295ed4a24b878bce416694695)
set(GOLDEN_design_faro_layer 1f6e578733925bb03b72f64046435022089bdc183fdbfa7d990aae83ed84d16e)

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
