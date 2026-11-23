# RECONCILE.md — fitgolden reconciliation for the staged merges

Scope: local worktrees only, read-only. Snapshots checked: foundation
`foundation/golden-harness` @ `d08f019`; #7 `0fe5ff5`, #8 `af5fd00`,
#10 `0ed49b2`, #11 `eff6ac4`, #13 `12de443`, #15 `2c075a1`, #5 `09311ce`,
#6 `e40b8c0`, #19 `16d91c4`. Nothing fetched, no corpora touched, no push.

If a decoder from main reads this archive, then the archive must still be
main's — and if a *test* is supposed to prove that, there must be exactly one
copy of the test. Today there are eight.

## 1. Inventory of the collision

| Worktree | `tests/fitgolden.h` | `tests/test_fitgolden.c` | `taw_test()` added | extra arm targets / CLI tests |
|---|---|---|---|---|
| foundation `d08f019` | **yes** (`TAW_TESTS_FITGOLDEN_H`) | **yes** (4-row baseline, own `main`) | `fitgolden` | — |
| #7 lr-baseline | yes (same guard) | yes (4-row baseline + tuned) | `fitgolden` | none — tuned table is in the same file |
| #8 eps-gate | yes (same guard, `static inline` helper, `TAW_GOLDEN_PRINT`) | **no** | `c3_fit` | `cli_fit_c3` |
| #10 calibration | yes (same guard, `fit_check(..., bool cal)`) | yes (4-row baseline) | `fitgolden`, `cal_fit` | — |
| #11 entropygate | **no** (helper + `fgold` + tables inlined in the `.c`) | yes (inline helper, base + gate tables) | `fitgolden` | `cli_fit_mix_seek`, `cli_fit_gate_seek` (foreach) |
| #13 hnet-feature | yes (same guard, `fgold` has `pred` field) | yes (4-row baseline with `pred`) | `hnet`, `fitgolden`, `hnet_fit` | `cli_fit_hnet_seek` |
| #15 hnet-blocks | **no** (own `dfgold` + own `static fit_check`) | **no** | `dynfit` | — |
| #5 faro | yes (**different guard** `TAW_TEST_FITGOLDEN_H`) | yes (4-row baseline) | `fitgolden`, `faro_fit` | `cli_fit_faro_seek` |
| #6 namo | yes (same guard, embeds `base_golden[]` in header) | yes (uses header's table) | `fitgolden`, `namo` | — |
| #19 ema | yes (same guard, adds `fit_probe`/`fit_roundtrip`, embeds 6-row `fitgolden_base[]`) | yes (uses header's table) | `fitgolden`, `ema` | — |

Counts that matter:

- **`tests/fitgolden.h`: 8 copies** (foundation, #7, #8, #10, #13, #5, #6, #19).
- **`tests/test_fitgolden.c`: 8 copies** (foundation, #7, #10, #11, #13, #5, #6, #19).
- **`taw_test(fitgolden)`: 8 copies** of `add_executable(test_fitgolden ...)` and
  `add_test(NAME fitgolden ...)`. The second one is a hard CMake error
  ("cannot create target … another target with the same name already exists"),
  not a soft duplicate-test warning.
- All 4 baseline tables are the same council/e values; #19 adds two L3 rows,
  #11/#13 add a `pred` field, #6/#19 put the table in the header, #8 changes
  `fit_check` to `static inline` and adds a `TAW_GOLDEN_PRINT` early-out that
  **skips all assertions** when set.
- Different `fgold` layouts: foundation `uint64_t model_size`; most PRs
  `size_t model_size`; #11/#13 insert `taw_predictor pred` as the **third**
  field. Copying a 6-field initializer into the 5-field survivor silently
  shifts every value — this is the highest-risk mechanical error.
- Different include guards: faro's `TAW_TEST_FITGOLDEN_H` vs the other seven
  `TAW_TESTS_FITGOLDEN_H`. A TU including both gets duplicate `fgold` and
  duplicate `fit_check` definitions.

**CTest target-name audit (all 10 worktrees):** the *only* duplicated
`taw_test()` name is `fitgolden` (8×). `c3_fit`, `cal_fit`, `hnet`, `hnet_fit`,
`dynfit`, `faro_fit`, `namo`, `ema` are each unique. CLI test names are unique
too: `cli_fit_c3`, `cli_fit_mix_seek`, `cli_fit_gate_seek`, `cli_fit_hnet_seek`,
`cli_fit_faro_seek`. The `add_test(NAME ${name})` inside `taw_test()` is a
function template, not a second collision. No existing 12 test names are
reused by any PR.

## 2. Ruling — exactly one survivor

**Survivor helper: `tests/fitgolden.h` from `d08f019`** (foundation). Delete the
other seven copies. Required minimal extensions, in this order:

1. Keep the guard `TAW_TESTS_FITGOLDEN_H` (reject faro's variant) and keep
   `fgold` **bit-for-bit as it is**: `{int level; taw_weights w; size_t arc_n;
   uint64_t model_size; uint32_t model_crc;}`. Field order must never change;
   every existing 5-field initializer is positional.
2. Add an options struct and an extended entry point; do **not** add `pred` to
   `fgold`:
   ```c
   typedef struct { taw_predictor pred; uint8_t profile; } fgold_opts;
   static void fit_check_ex(const char *arm, const fgold *g, size_t ng,
                            const fgold_opts *o);
   static void fit_check(const char *arm, const fgold *g, size_t ng)
   {   /* wrapper: .pred = TAW_PREDICTOR_MIX, .profile = 0 (don't check) */ }
   ```
   #10's `bool cal` becomes `.profile = 6`; #11's gate becomes
   `.pred = TAW_PREDICTOR_GATE, .profile = 11`; #13's hnet becomes
   `.pred = TAW_PREDICTOR_HNET, .profile = 14`. The 6-field tables in #11/#13
   are rewritten to 5 fields plus an opts argument — never copied verbatim.
3. Adopt #19's `fit_source()`, `fit_probe()`, `fit_roundtrip()` so the MALTER
   A/B and any future "measure, don't assert" arm never reimplements the fit.
   `fit_check_ex()` is then a thin loop over `fit_probe()` + comparison +
   `fit_roundtrip()`.
4. Adopt the *idea* of #8's `TAW_GOLDEN_PRINT`, but as a separate
   `fit_check_print()` that only prints. **Reject** #8's current behaviour
   where setting the env var makes `fit_check` skip every assertion — a CI
   machine with that variable set would report a false green.
5. Put the one canonical baseline table in the header as `static const fgold
   fitgolden_base[]` + `FITGOLDEN_BASE_N` (the #6/#19 pattern, generalized).
   `static const` gives internal linkage, so arm translation units may include
   it without duplicate symbols. **No other file may define a baseline table.**
   Extend it from 4 to 6 rows with #19's L3 cells; I re-measured both on main
   during this reconciliation and they match exactly: `L3/F16 arc=261701
   model=568 crc=0x1388A8EA`, `L3/Q8 arc=263030 model=1888 crc=0x1A6E3799`.
6. Add a single `fgold_env_scrub()` helper that unsets **every** `TAW_*`
   variable via `environ` (with `_POSIX_C_SOURCE 200809L` at the top of the
   header, which every test includes first). This replaces the incomplete
   hand-written knob lists in #6/#7/#8/#10/#19 — e.g. #7's "shipped" arm does
   not unset `TAW_FIT_AVG`, so #19's default-on build would silently invalidate
   it. Each arm calls `fgold_env_scrub()` first, then sets only its own knobs.
7. Reject #15's local `static fit_check` name: rename to `dyn_fit_check()`.
   Its `dfgold` struct (index CRC, block count, first/last raw_len) stays
   local to `test_dynfit.c` — it is a container golden, not a fitter golden.

**Survivor test: `tests/test_fitgolden.c` from `d08f019`.** Delete the other
seven copies. It shrinks to a single `main()` that calls
`fit_check("base", fitgolden_base, FITGOLDEN_BASE_N)` and returns
`CHECK_DONE()`. It owns the sole `taw_test(fitgolden)` registration.

## 3. Where the per-PR arm tables go

One CTest target per PR (or per distinct mechanism), all with unique names.
Every arm file includes `fitgolden.h`, calls `fgold_env_scrub()`, and contains
**only** its own tables — no baseline table, no helper copy, no `pred` field
in `fgold`, no duplicate `main`.

| PR | Arm tables to move | Surviving file / target |
|---|---|---|
| #7 | `tuned[]` (4 rows, LR/WD knobs); drop the duplicate `shipped[]` | new `tests/test_tuned_lr.c`, `taw_test(tuned_lr)` |
| #8 | `SHRINK`, `SHRINK_PLAIN`, `MEAN`, `MEAN_PLAIN`, `LAYERS1`, `SCOPE_COLD`, `BETA95`, `GATE_P2`, `gate_direct()`, distinctness CHECKs; drop `BASELINE[]` | `tests/test_c3_fit.c`, `taw_test(c3_fit)` (keep name; include `fit_gate.h` from `src/`) |
| #10 | `cal`, `ident`, `calonly`, `apmonly`; drop `base[]`; convert `bool cal` → `.profile = 6` | `tests/test_cal_fit.c`, `taw_test(cal_fit)` (keep) |
| #11 | `gate[]` (rewrite 6-field → 5-field + `fgold_opts`); drop inlined helper and `base[]` | new `tests/test_gate_fit.c`, `taw_test(gate_fit)` (new target — today it hides inside `fitgolden`) |
| #13 | `hnet[]` (rewrite with opts); drop duplicate `base[]` | `tests/test_hnet_fit.c`, `taw_test(hnet_fit)` (keep); also keep `taw_test(hnet)` (profile unit test, distinct) |
| #15 | `goldens[]` (`dfgold`) and `dyn_fit_check()` | `tests/test_dynfit.c`, `taw_test(dynfit)` (keep) |
| #5 | `erc`, `erc_muon`, `malter`, `malter_erc`; drop `base[]` | `tests/test_faro_fit.c`, `taw_test(faro_fit)` (keep) |
| #6 | `namo`, `namod`, `namo_gate`, `namod_row`, `namo_nest`; drop `base_golden[]` from header | `tests/test_namo.c`, `taw_test(namo)` (keep) |
| #19 | `ema999_gold`, `swa_gold`, `ema9_gold`, `late_gold`, `malt_ab()`; drop `fitgolden_base[]` from header | `tests/test_ema.c`, `taw_test(ema)` (keep) |

Final CTest inventory after all nine merges: the original 12 + `fitgolden` +
9 arm targets (`tuned_lr`, `c3_fit`, `cal_fit`, `gate_fit`, `hnet_fit`,
`dynfit`, `faro_fit`, `namo`, `ema`) + `hnet` + 5 CLI-fit tests
(`cli_fit_c3`, `cli_fit_mix_seek`, `cli_fit_gate_seek`, `cli_fit_hnet_seek`,
`cli_fit_faro_seek`) = **28 tests** (gpu skipped on this host). No name repeats.
The CLI scripts stay as five distinct files (`cli_fit_c3.cmake`,
`cli_fit_arms.cmake`, `cli_fit_hnet_seek.cmake`, plus #5's reuse of
`cli_roundtrip.cmake`); their `WORK` dirs are already distinct.

## 4. Per-merge conflict handling

`tests/CMakeLists.txt` is modified by all ten worktrees and will conflict at
every batch. Resolve in COMPAT.md batch order by **keeping the foundation
block** (single `taw_test(fitgolden)`) and appending only the current PR's
arm line(s) from the table above. Concretely:

- Batch 2 (#5): delete its `fitgolden.h`, `test_fitgolden.c`, and its
  `taw_test(fitgolden)` line; keep `faro_fit` + `cli_fit_faro_seek`.
- Batch 3 (#8): its `fitgolden.h` goes; `test_c3_fit.c` switches to the
  survivor header and loses the print early-out; keep `c3_fit` + `cli_fit_c3`.
- Batch 4 (#6): header and `test_fitgolden.c` go; `base_golden[]` goes; keep
  `namo` and add `fgold_env_scrub()`.
- Batch 5 (#19): header and `test_fitgolden.c` go; `fit_probe`/
  `fit_roundtrip` move up into the survivor header; keep `ema`.
- Batch 6 (#15): no fitgolden file conflict; rename its local helper; keep
  `dynfit`.
- Batch 7 (#11): its inlined helper and baseline go; gate table moves to
  `test_gate_fit.c`; replace `taw_test(fitgolden)` with `taw_test(gate_fit)`;
  keep both `cli_fit_*_seek` names.
- Batch 8 (#13): header and `test_fitgolden.c` go; rewrite tables with opts;
  keep `hnet`, `hnet_fit`, `cli_fit_hnet_seek`.
- Batch 9 (#10): header and `test_fitgolden.c` go; convert `bool cal`; keep
  `cal_fit`.
- #7 was batch 1: its `fitgolden.h`/`test_fitgolden.c` go; the tuned table
  moves to `test_tuned_lr.c`; add that target.

Acceptance check after each batch: `cmake --build build -j4 && ctest` green
with no `add_executable ... same name` error, `golden_corpus.sh` still prints
`golden corpus set OK (12 pairs)`, and `grep -c taw_test\(fitgolden\)` on
`tests/CMakeLists.txt` is exactly **1**.

## 5. The one-line COMPAT.md fix

`COMPAT.md` says `ctest` is `12/12`; with `d08f019` it is `13/13`. The one-line
substitution is **`12/12` → `13/13`**, applied at all three occurrences
(the batch rule, batch 0's green condition, and the §4 build-command comment).
Applied in this commit. Longer-term the count will change again with every
merged arm (28 after all nine); the durable wording would be "all CTest tests
green, gpu skipped", but that is a rewording, not the requested one-line fix.

## 6. Compatibility keeper's conditional

If two `taw_test(fitgolden)` registrations survive a batch, then the build
fails before any archive is written — a loud failure, acceptable. If two
*baseline tables* survive, then a maintainer can update one and leave the
other pinning stale bytes — a silent failure, not acceptable. Hence the
single header-owned `fitgolden_base[]` and the single surviving
`test_fitgolden.c`. If an arm test keeps its own copy of the baseline, it is
not an arm test; it is a fork of the invariant.
