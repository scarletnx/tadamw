# FINAL.md — Compatibility Keeper's final-pass verification

Base of record: canonical `integration/golden-canonical` @ `14b1c0f` in this
worktree. Final staged branches checked read-only at the requested SHAs:
#5 `6db34ea`, #6 `d8506fc`, #7 `fb81c4d`, #8 `a7457ec`, #10 `8906f3b`,
#11 `0a91327`, #13 `92638a7`, #15 `af4e989`, #19 `d3368de`, #21 `fb0d51f`.
All ten heads matched; no fetch, no corpus change, no sibling edit, no push.

If a decoder from main reads this archive, then the archive must still be
main's — and if a golden test is the only thing standing between a merge and a
silent default change, then that test must compile under the strictest build
and be registered exactly once. Two staged deviations and one duplicate target
break that condition.

## Verdict: **NOT CLEAN — three fixes required** (none is a format regression)

### Checked and clean

- **One fitgolden helper per worktree.** Eight of the nine fitgolden-carrying
  PRs have `tests/fitgolden.h` byte-identical to canonical (SHA `4dd99584…`);
  #15's copy is canonical **plus** the unused attribute only (see F1). No PR
  carries a second helper, a private baseline table, or a private
  `fit_check()` definition.
- **One `test_fitgolden.c` and one `taw_test(fitgolden)`** per worktree where
  present (#6 and #21 legitimately have neither; #6 consumes `base_golden`
  from the canonical header). Counts: 1 for #5/#7/#8/#10/#11/#13/#15/#19 and
  the canonical branch, 0 for #6/#21. No worktree has two.
- **No assertion bypass.** The only `TAW_GOLDEN_PRINT` occurrences in any
  final branch are the canonical header's *prohibition comment*. No env var,
  `#if`, or CTest property turns a golden assertion into a no-op.
- **No incompatible `fgold` layout.** Every `fgold` table in every final arm
  is the canonical five-field positional form; predictor/profile arms use
  `fit_check_ex()` with `fgold_opts` (#10 `.profile = 6`, #11 `.profile = 11`,
  #13 `.profile = 14`). #15's `dfgold` is a separate container struct and its
  helper was renamed `dyn_fit_check`, as specified.
- **Alias compatibility holds.** NAMO's `base_golden` is a macro alias, not a
  second table; EMA's `fit_probe`/`fit_roundtrip`/`fitgolden_base`/
  `FITGOLDEN_BASE_N` names are present in the canonical header.
- **COMPAT.md counts are consistent right now:** the canonical branch says
  `13/13`, and no builder registers an extra target on main. (Forward-looking
  caveat in F3.)
- **#21** has no test changes, restored the upstream `.gitignore` patterns,
  and keeps its report out of the way of the other `RESULTS.md` writers.

### F1 — BLOCKER: canonical header fails `-Werror`, and both staged workarounds fork or blanket-suppress

Measured on this worktree:

```
cc -c -std=c2x -O2 -Wall -Wextra -Wpedantic -Werror \
   -Iinclude -Ibuild/include -Isrc -Itests tests/test_fitgolden.c
-> error: 'fit_probe' defined but not used [-Werror=unused-function]   (rc=1)

same compile against #15's fitgolden.h -> rc=0
```

Root cause: the canonical header ships seven `static` helpers; an arm that
uses only `fit_check()` leaves `fit_probe()` unused. Both PRs patched around
it:

- #15 adds `TAW_TEST_UNUSED` (`__attribute__((unused))` with a portable empty
  fallback) to all seven helpers — targeted, portable, and it works — but it
  lives in a **private copy of the canonical header**, which reintroduces the
  duplicate-helper problem the whole harness was built to remove.
- #13 adds `target_compile_options(test_fitgolden PRIVATE -Wno-unused-function)`
  and the same for `test_hnet_fit` — a blanket suppression that would hide
  genuinely dead code in those test files. Not acceptable.

**Exact fix (canonical-header patch required):**
1. In this worktree's `tests/fitgolden.h`, define once (before the helper
   definitions):
   ```c
   #if defined(__GNUC__) || defined(__clang__)
   #define TAW_TEST_UNUSED __attribute__((unused))
   #else
   #define TAW_TEST_UNUSED
   #endif
   ```
   and annotate all seven static helpers (`fgold_env_scrub`, `fit_source`,
   `fgold_fit`, `fit_probe`, `fit_roundtrip`, `fit_check_ex`, `fit_check`),
   exactly as #15's header does.
2. Verify `cc ... -Werror ... tests/test_fitgolden.c` now returns 0 and
   `ctest` is still green.
3. Re-copy that patched header to **all nine** fitgolden-carrying PR branches
   in one coordinated commit (otherwise eight copies diverge again).
4. Delete #15's private header variant; drop #13's two
   `-Wno-unused-function` lines from `tests/CMakeLists.txt`.
5. Keep the prohibition comment — an env var must never turn goldens into
   no-ops.

### F2 — BLOCKER: duplicate `taw_test(fit_opt)` between #5 and #6

Both final branches add the identical `tests/test_fit_opt.c` (blob
`2115d6b7…`, the shared Newton–Schulz bit-exactness oracle) **and** the same
`taw_test(fit_opt)` line. Merging batches 2 and 4 without resolution produces
`add_executable(test_fit_opt …): another target with the same name already
exists`, plus a duplicate `add_test(NAME fit_opt …`.

**Exact fix:** keep one owner. Let #5 land `taw_test(fit_opt)` and
`test_fit_opt.c` in batch 2. At #6's batch, drop its `taw_test(fit_opt)` line
and its copy of the file (the blobs are identical, so this is a clean removal,
not a semantic loss). #6's own `taw_test(namo)` and `cli_fit_namo_seek` stay.
No other cross-branch target or CLI test name repeats; the nine CLI fit names
(`cli_fit_golden`, `cli_fit_faro_seek`, `cli_faro_gpu_guard`,
`cli_fit_namo_seek`, `cli_fit_c3`, `cli_fit_mix_seek`, `cli_fit_gate_seek`,
`cli_fit_hnet_seek`, `cli_fit_ema_seek`) are all distinct.

### F3 — ADVISORY: COMPAT.md's count expires at the first arm merge

`13/13` is correct for the canonical branch alone. After all nine batches the
suite is **33 tests** (12 original + `fitgolden` + 9 arm targets + `fit_opt` +
`hnet` + 9 CLI fit tests), gpu skipped. Either update the count at each batch
or replace the three numeric occurrences with the durable wording "all CTest
tests green, gpu skipped". This is not a merge blocker on main today.

## Closing conditional

If F1 is not applied, then a `-DTADAMW_WERROR=ON` build of every arm that uses
only part of the canonical header fails, and maintainers will keep forking the
header to work around it. If F2 is not applied, then batches 2 and 4 cannot
both be green. If F3 is ignored, then the golden corpus still passes but the
document lies about the test count. Until all three are closed: **NOT GREEN**.
