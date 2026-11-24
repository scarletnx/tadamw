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
set(GOLDEN_format_default   3c91b931401d70848939b157a9f1600723768987944cfce58b9870d64c461ad9)
set(GOLDEN_format_muon      e6edcb519ff54c388695f81ea0c6c09d6033cb25e127e32972887153f0d3fbcf)
set(GOLDEN_format_blend     132076b1f0c777aadb8f8eafdd43acbf117d7399b20f4a4f1e58961597abe5f8)
set(GOLDEN_format_alt       e6065d9be31ee5da7605203e21ccc0172e9f68a549ddfd95d6ebca77730b196c)
set(GOLDEN_format_route     e6edcb519ff54c388695f81ea0c6c09d6033cb25e127e32972887153f0d3fbcf)
set(GOLDEN_format_malt      2ae22d1103f639b409dce96c15bf2264610ba01c5020cf8fc36c51fd9e209412)
set(GOLDEN_format_malter    0ae38b2ac8423a30dadb7b656fea1550b7e88ce62dc9c8676ca10ab0b55bf7d5)
set(GOLDEN_format_faro_muon   b3e75d2d727b9d95fc00e72890e6c34a0619454342bda747d8332a1008254c78)
set(GOLDEN_format_faro_probe  c7c0a844511ca512bfbc7bdbb9332c8521e566cdc98255b07518720d4cc32e6f)
set(GOLDEN_format_faro_lin    c7c0a844511ca512bfbc7bdbb9332c8521e566cdc98255b07518720d4cc32e6f)
set(GOLDEN_format_faro_adam   3c91b931401d70848939b157a9f1600723768987944cfce58b9870d64c461ad9)
set(GOLDEN_format_faro_layer  8b1941727c0a0b7f65bb5a207adfd4783a875fa489ae75954beba56f410dbf24)
set(GOLDEN_design_default   55182debc3b3d0ceb086e6eefc8d7aa5fad5d6bc519716d3e824b7782da6b679)
set(GOLDEN_design_muon      97d2eaa83b033a359baf3ad35a626af9ac52b824ab51b449186dfd64c21f9419)
set(GOLDEN_design_blend     a8e73d3977ee9b51a9fc67ce5bb9ae6456d897cfd1f7dc259e96cc0e05a97906)
set(GOLDEN_design_alt       e4089b31e660f5f4ddb8bb59c97bddbf581456c159df769cefb499f179c8fa58)
set(GOLDEN_design_route     97d2eaa83b033a359baf3ad35a626af9ac52b824ab51b449186dfd64c21f9419)
set(GOLDEN_design_malt      6156ef2691be3b3dcab632fc5486a92851ba884312b2be09a40f3ba778196fac)
set(GOLDEN_design_malter    6a4dda186f3669249e7d0d9823104f713bca741584c05144febac1c54684999d)
set(GOLDEN_design_faro_muon  ca4583aa6ed1499255b3d7326aababe8df4510dbcc253fb6eece29a45bf52902)
set(GOLDEN_design_faro_probe 0e9d922c49554b66199e88d596fc1df6f876deac9be9423de116c7597ecd27b9)
set(GOLDEN_design_faro_lin   0e9d922c49554b66199e88d596fc1df6f876deac9be9423de116c7597ecd27b9)
set(GOLDEN_design_faro_adam  55182debc3b3d0ceb086e6eefc8d7aa5fad5d6bc519716d3e824b7782da6b679)
set(GOLDEN_design_faro_layer c2a5067a369397acb247fc6cd33dbb6fa0feb203c1f81ffaaeb610642458a513)

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
