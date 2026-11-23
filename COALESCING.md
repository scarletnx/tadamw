# COALESCING.md — one foundation branch from four integration branches

Worktree `coal-foundation`, branch `coal/coal-foundation`, base `origin/main`
`1aa3dd7`. Local only; nothing pushed, no PR branch touched.

If a decoder from main reads this archive, then the archive must still be
main's — and if four foundations are merged into one, then each unique change
must appear exactly once or the decoder's next read depends on merge order.

## Source branches (merged in this exact order)

| # | branch | tip | unique contribution |
|---|---|---|---|
| 1 | `integration/opt-plumbing` | `8b93197` | shared `fit_opt.h/.c` ABI + `opt_hooks` + one gate owner, `src/fit.c` refactor, `test_fit_opt`, `cli_fit_golden`, `STRUCTURE.md` |
| 2 | `integration/golden-canonical` | `11ffaa7` | canonical `tests/fitgolden.h` (blob `99f3b182…`), `tests/golden_corpus.sh`, `COMPAT.md`, `PROPAGATION.md` |
| 3 | `integration/predictor-enum` | `a8af462` | central allocation registry in `docs/format.md`, predictor enums `6/11/14` with value `3` void, sparse CLI table in `tools/taw.c` |
| 4 | `integration/knob-final` | `2c6fe47` | `KNOBS.md`, `docs/design.md` knob registry + historical appendix |

Merge graph: `6e14c28` (1) → `1de9a6e` (2) → `dfbc901` (3) → `bc4406f` (4).
The final tip is the commit adding this document (the `cli_fit_golden`
re-baseline is included there); `git log -1 --format=%H` gives its exact hash.

## Conflict resolution

- `tests/CMakeLists.txt` — auto-merged: opt-plumbing's `taw_test(fit_opt)`
  plus golden-canonical's single `taw_test(fitgolden)` both survive, once each.
- `INTEGRATION.md`, `RECONCILE.md`, `FINAL.md` — add/add conflicts. Resolved by
  **coalescing, not choosing**: each source document is preserved verbatim once,
  under a labelled `## Part I/II/III` heading with its source branch and SHA.
  No section was dropped; no section was duplicated.
- `docs/format.md` — only `integration/predictor-enum` changed this file, so
  there was no textual conflict. There is exactly **one** central allocation
  registry (`docs/format.md:19`, "Allocation registry (central ledger)") and
  one set of predictor enums (`include/tadamw/tadamw.h:88–94`). The doc corpus
  used by `cli_fit_golden.cmake` changed as a result; see below.
- `docs/design.md` — only `integration/knob-final` changed this file; its knob
  registry and historical appendix are intact.

## Required adaptation: `cli_fit_golden.cmake`

The opt-plumbing test hashes were measured with `docs/format.md` and
`docs/design.md` as their corpora. Both documents gained registry sections
during coalescing, so the fixture bytes changed — not the fitter. All 14
`GOLDEN_*` constants were re-measured on the coalesced tree; GCC 16.2.1 and
Clang 23.1.5 produced **identical** hashes, and the comment now records why.
Fitter byte-identity to `1aa3dd7` is independently pinned by
`tests/golden_corpus.sh` on the fixed external corpora: 12/12 pairs pass with
both the GCC and the Clang binary.

## Verification (all green)

- GCC 16.2.1, `-DCMAKE_BUILD_TYPE=Release -DTADAMW_GPU=OFF -DTADAMW_WERROR=ON`:
  build clean, `ctest` **15/15** (gpu skipped).
- Clang 23.1.5, same flags: build clean, `ctest` **15/15** (gpu skipped).
- `golden_corpus.sh` with each compiler's `taw`: `golden corpus set OK (12 pairs)`.
- `git rev-parse HEAD:tests/fitgolden.h` == `99f3b1820bf11d5762d90d6eb0915b6a6734ef15`.
- `grep -c taw_test(fitgolden) tests/CMakeLists.txt` == 1;
  `grep -c taw_test(fit_opt) tests/CMakeLists.txt` == 1.
- No conflict markers remain.

## What downstream coalesced branches should depend on

Depend on this branch's tip (or on `bc4406f` plus the final tip commit), never
on the four source branches individually:

- `coal/coal-fitting` — for the shared `fit_opt.h/.c` ABI and `opt_hooks`; the
  single gate owner (`step_shape`/`fit_gate_step`) lives here.
- `coal/coal-profiles` — for the central predictor/profile allocation
  (`6/11/14`, value 3 void) and the `docs/format.md` registry; do not add a new
  predictor value by appending at 3.
- `coal/coal-container-docs` — for the knob registry (`KNOBS.md`,
  `docs/design.md`) and the durable COMPAT green condition.
- Every staged PR arm — for the canonical `tests/fitgolden.h` blob and the
  `TAW_TEST_UNUSED` annotations; arms must include this header, not a private
  copy (see `PROPAGATION.md`).

Re-baselining rule: if a downstream branch changes a doc used as a
`cli_fit_golden` corpus, it must re-measure those 14 hashes in the same commit
and re-run the GCC/Clang cross-compiler comparison; it must not quietly edit
the constants.
