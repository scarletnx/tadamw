# INTEGRATION.md — canonical fitgolden harness, implemented

Branch `integration/golden-canonical` (from `foundation/golden-harness` @
`bd32f43`). Local only; no fetch, no corpora touched, no push.

If a decoder from main reads this archive, then the archive must still be
main's — and if a golden test is meant to prove it, there must be exactly one
copy of the test and one copy of the baseline table. This branch implements
that single copy.

## 1. What is now in the tree

| File | Status |
|---|---|
| `tests/fitgolden.h` | **the only** copy. Fixed `fgold` layout, `fgold_opts`/`fit_check_ex`, `fit_probe`/`fit_roundtrip`/`fit_source`, `fitgolden_base[]` (6 rows) + `FITGOLDEN_BASE_N`, `fgold_env_scrub()`, `base_golden` alias. No `TAW_GOLDEN_PRINT`. |
| `tests/test_fitgolden.c` | **the only** copy. One `main()`: `fgold_env_scrub(); fit_check("base", fitgolden_base, FITGOLDEN_BASE_N);`. |
| `tests/CMakeLists.txt` | **one** `taw_test(fitgolden)` (line 21, unchanged from `d08f019`). No other fitgolden target. |
| `tests/golden_corpus.sh` | unchanged; still the 12 external SHA/size pairs from COMPAT.md. |

Public API of the canonical header (signatures are the contract for staged
arms):

```c
typedef struct { int level; taw_weights w; size_t arc_n;
                 uint64_t model_size; uint32_t model_crc; } fgold;
typedef struct { taw_predictor pred; uint8_t profile; } fgold_opts;
static void fgold_env_scrub(void);                 /* unset every TAW_* */
static void fit_probe(int, taw_weights, fgold *, void **, size_t *);
static void fit_roundtrip(const void *, size_t);
static void fit_check_ex(const char *, const fgold *, size_t,
                         const fgold_opts *);
static void fit_check(const char *, const fgold *, size_t); /* MIX wrapper */
static const fgold fitgolden_base[];               /* 6 rows */
#define FITGOLDEN_BASE_N ...
#define base_golden fitgolden_base                 /* #6 alias */
```

The six baseline rows are the four council/e cells plus #19's two L3 cells:
`L3/F16 261701/568/0x1388A8EA`, `L3/Q8 263030/1888/0x1A6E3799` (re-measured
on main during round 2, both matched).

**Alias compatibility verified by compile check in this branch:** the staged
`test_namo.c` (uses `base_golden`), `test_ema.c` (uses `fit_probe`,
`fit_roundtrip`, `fitgolden_base`, `FITGOLDEN_BASE_N`), `test_faro_fit.c` and
`test_lr`'s table all compile unchanged with `-fsyntax-only` against this
header. Their private copies can simply be deleted at merge; only the env
scrub call needs adding to each `main()` where it is not already present.

## 2. Unique per-PR arm files and targets

Each arm file includes **this** header, contains **only** its own arm tables,
calls `fgold_env_scrub()` first, and has its own unique CTest target. None
defines a baseline table, a helper, or `main()` for `fitgolden`.

| PR | Arm file | CTest target | Tables / code to move from the PR | CMake line to add at that batch |
|---|---|---|---|---|
| #5 faro | `tests/test_faro_fit.c` | `faro_fit` | `erc`, `erc_muon`, `malter`, `malter_erc`; drop `base[]` | `taw_test(faro_fit)` |
| #6 namo | `tests/test_namo.c` | `namo` | `namo`, `namod`, `namo_gate`, `namod_row`, `namo_nest`; drop header's `base_golden[]` (alias only) | `taw_test(namo)` |
| #7 lr-baseline | `tests/test_tuned_lr.c` (**new**) | `tuned_lr` | `tuned[]` (LR/WD); drop duplicate `shipped[]` and the private `fitgolden.h` | `taw_test(tuned_lr)` |
| #8 eps-gate | `tests/test_c3_fit.c` | `c3_fit` | `SHRINK`, `SHRINK_PLAIN`, `MEAN`, `MEAN_PLAIN`, `LAYERS1`, `SCOPE_COLD`, `BETA95`, `GATE_P2`, `gate_direct()`, distinctness CHECKs; drop `BASELINE[]` and the `TAW_GOLDEN_PRINT` early-out | `taw_test(c3_fit)` |
| #10 calibration | `tests/test_cal_fit.c` | `cal_fit` | `cal`, `ident`, `calonly`, `apmonly`; drop `base[]`; replace `fit_check(..., bool cal)` with `fit_check_ex(..., &(fgold_opts){.profile = 6})` | `taw_test(cal_fit)` |
| #11 entropygate | `tests/test_gate_fit.c` (**new**) | `gate_fit` | `gate[]` rewritten 6-field → 5-field + `&(fgold_opts){.pred = TAW_PREDICTOR_GATE, .profile = 11}`; drop inlined helper and `base[]` | `taw_test(gate_fit)` |
| #13 hnet-feature | `tests/test_hnet_fit.c` | `hnet_fit` | `hnet[]` rewritten with `&(fgold_opts){.pred = TAW_PREDICTOR_HNET, .profile = 14}`; drop `base[]` | `taw_test(hnet_fit)` (+ keep distinct `taw_test(hnet)` profile unit test) |
| #15 hnet-blocks | `tests/test_dynfit.c` | `dynfit` | `dfgold` table; rename local `static fit_check` → `dyn_fit_check` (container golden, not a fitter golden) | `taw_test(dynfit)` |
| #19 ema | `tests/test_ema.c` | `ema` | `ema999_gold`, `swa_gold`, `ema9_gold`, `late_gold`, `malt_ab()`; drop header's `fitgolden_base[]` | `taw_test(ema)` |

CLI fit scripts stay distinct and are registered under the already-unique
names `cli_fit_c3`, `cli_fit_mix_seek`, `cli_fit_gate_seek`,
`cli_fit_hnet_seek`, `cli_fit_faro_seek`.

**Why the arm files are specified here but not registered on main:** their
knobs and predictor enum values do not exist in `1aa3dd7`. A registered arm on
main would either fail to compile or (if made conditional) skip silently —
both are false greens, which is precisely the failure mode this harness
exists to prevent. The files are created/registered at the batch that merges
the corresponding PR, using the exact names above. The final inventory after
all nine batches is 28 CTest names, none repeated.

## 3. Migration rule for each staged PR

At merge time, for every PR in the table:

1. Delete its `tests/fitgolden.h` and `tests/test_fitgolden.c` (if present).
2. Delete its `taw_test(fitgolden)` line; keep only the arm target line.
3. Replace arm tables' `pred` field with `fgold_opts` passed to
   `fit_check_ex()`; for #10 replace `bool cal` with `.profile = 6`.
4. Call `fgold_env_scrub()` at the start of each arm `main()` **before**
   setting arm knobs; this supersedes the incomplete per-PR knob lists.
5. Never reintroduce an env var that suppresses assertions.

`tests/CMakeLists.txt` conflicts in every batch; resolve by keeping the
foundation `taw_test(fitgolden)` block and appending only that batch's arm
line. Acceptance after each batch: build clean, `ctest` green, and
`grep -c 'taw_test(fitgolden)' tests/CMakeLists.txt` == 1.

## 4. Verification performed on this branch

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTADAMW_GPU=OFF` +
  `cmake --build build -j4` — clean.
- `./build/tests/test_fitgolden` — pass with all six baseline rows.
- Four staged arm tests compile unchanged against the canonical header
  (NAMO, EMA, FARO, LR-baseline), confirming the aliases.
- `ctest` — full suite green (gpu skipped). Count is still **13/13**: no new
  target was registered on main, so COMPAT.md's `13/13` needs no further edit
  in this round.
- `tests/golden_corpus.sh` — expected to remain `golden corpus set OK (12
  pairs)`; the encoder was not touched.
