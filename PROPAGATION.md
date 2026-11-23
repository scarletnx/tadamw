# PROPAGATION.md — canonical fitgolden header adoption check

Canonical: this worktree `integration/golden-canonical` @ `d245fc6`,
`tests/fitgolden.h` git blob **`99f3b1820bf11d5762d90d6eb0915b6a6734ef15`**
(SHA-256 `b6474e1578c396c7c4d149a6b653b0a577f4290e2170bd437ea9eeb0b5e7c5e…`).
Comparison is by `git hash-object` of each staged worktree's
`tests/fitgolden.h`, read-only. No sibling worktree was modified; no push.

If a decoder from main reads this archive, then the archive must still be
main's — and if the golden harness is the thing proving it, then every arm
must include the *same* header. Three header variants are still in flight.

## Per-PR result

| PR | worktree / SHA checked | header blob | status | exact required action |
|---|---|---|---|---|
| #5 FARO | `exp-faro` `908e613` | `4e8059e2…` | **STALE (pre-F1)** | Adopt the `d245fc6` header verbatim: add the portable `TAW_TEST_UNUSED` macro and its seven helper annotations. No local modification to preserve. |
| #6 NAMO | `exp-namo` `588223f` | `4e8059e2…` | **STALE (pre-F1)** | Adopt the `d245fc6` header. Also finish the in-flight `fit_opt` drop: `tests/CMakeLists.txt:18` still says `taw_test(fit_opt)` but `tests/test_fit_opt.c` is gone — remove that line or the configure/build fails; #5 owns `fit_opt`. |
| #7 LR-baseline | `exp-lr-baseline` `fb81c4d` | `4e8059e2…` | **STALE (pre-F1)** | Adopt the `d245fc6` header. No local modification to preserve. |
| #8 C3 gate | `exp-eps-gate` `3608f64` | `99f3b182…` | **CANONICAL** | None — already adopted. |
| #10 calibration | `exp-calibration` `8906f3b` | `4e8059e2…` | **STALE (pre-F1)** | Adopt the `d245fc6` header. No local modification to preserve. |
| #11 entropygate | `exp-entropygate` `d6a7fca` | `4e8059e2…` | **STALE (pre-F1)** | Adopt the `d245fc6` header. No local modification to preserve. |
| #13 hnet-feature | `exp-hnet-feature` `532ed22` | `3fb9351a…` | **STALE (private variant)** | Replace with the `d245fc6` header. The private "Local integration note" at `tests/fitgolden.h:16–20` and the seven `static void TAW_TEST_UNUSED …` placements (`:58`, `:81`, `:100`, `:125`, `:133`, `:146`, `:176`) are a stale workaround, not a legitimate modification. |
| #15 hnet-blocks | `exp-hnet-blocks` `faae264` (verified at tip `7edf9ba`, which includes the in-flight R-4 error-code fix; header blob identical at both commits) | `3fb9351a…` | **STALE (private variant)** | Replace with the `d245fc6` header. Same private note at `tests/fitgolden.h:16–20` and same seven attribute placements; no legitimate local change. The error-code fix touched `src/header.c`, `tests/test_aw.c`, `RESULTS.md` only. |
| #19 ema | `exp-ema` `9be5faa` | `99f3b182…` | **CANONICAL** | None — already adopted. |
| #21 final-matrix | `exp-final-matrix` `3d6d11e` | n/a (no `tests/fitgolden.h`) | **N/A** | None — docs/scripts only; no test helper to propagate. |

## Legitimate local modifications

**None found.** The five stale pre-F1 headers (`4e8059e2…`) are the exact
`14b1c0f` canonical blob—byte-identical to each other—with only the F1 patch
missing. The two private variants (`3fb9351a…`, #13 and #15) are the same
`14b1c0f` blob plus `TAW_TEST_UNUSED` and a local note; their own comment
states "The only change is the unused attribute". They differ from canonical
only in the comment wording and in attribute *placement*
(`static void TAW_TEST_UNUSED f` vs canonical `static TAW_TEST_UNUSED void f`),
not in any helper, table, alias, scrub logic, or assertion. There is nothing
to merge forward; replace the file.

## Other checks

- **Blanket `-Wno-unused-function`:** removed from all ten staged
  `tests/CMakeLists.txt` files, including #13 (previously on `test_fitgolden`
  and `test_hnet_fit`). No action remains there.
- **Assertion bypass:** no env-var assertion skip in any staged test.
- **Aliases / layout:** where headers are canonical or the stale canonical
  blob, `fgold`, the six-row `fitgolden_base[]`, `FITGOLDEN_BASE_N`,
  `base_golden` and `fgold_env_scrub()` are all present and unchanged.
- **Canonical branch itself** still compiles `test_fitgolden.c` under
  `-Werror` (rc=0), full `ctest` green, golden corpus 12/12; no code changed
  in this pass.

## Closing conditional

If any of the five pre-F1 branches merges before adopting `d245fc6`'s header,
then a `-DTADAMW_WERROR=ON` build of its arm fails on `fit_probe` again. If
#13 or #15 merges with the private variant, then the one-helper invariant is
broken again even though the build passes. If #6 merges with the dangling
`taw_test(fit_opt)` line, then CMake fails on a missing source. Until all
eight actions above are applied, the propagation is **NOT GREEN**.
