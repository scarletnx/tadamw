# RECONCILE.md — staged PR fixes vs the knob docs

> "The clipboard and the code must agree, or the clipboard is a work of fiction."

Scope: read-only inspection of the four staged local branches at the exact
SHAs named in the request, through the shared git object database from the
`struct-c` worktree. No sibling worktree file was modified; nothing was pushed.

Documents reconciled:

- `KNOBS.md` @ `b080612` (the ruling) on `struct/c`
- `docs/design.md` §"Environment knobs (encoder-only)" @ `694f862` on
  `foundation/knob-docs` (the published knob doc)

Method: for each branch, enumerate every `getenv`/`fit_env_*` read in tracked
C sources, plus env-like tokens in scripts/docs; compare name, default, value
vocabulary and byte-identity claim against the two documents above.

## Verdict summary

| PR | SHA | Verdict | Headline drift |
|----|-----|---------|----------------|
| #6 NAMO | `e40b8c0` | **DRIFT** | Doc claims a knob (`TAW_NAMO_NEST`) that does not exist; doc denies a gate alias that does exist; `TAW_MUON_NEST` default still conditional |
| #8 EPS gate | `af5fd00` | **DRIFT** | `TAW_GATE_MODE` not renamed; `TAW_EPS_GRAN` removed but still documented; GPU guard + `TAW_GOLDEN_PRINT` undocumented; new README cross-reference wrong |
| #19 EMA | `16d91c4` | **MATCH** (minor) | Default is `off`, `auto` gone; numeric aliases `0..7`/`ema09` undocumented |
| #21 final matrix | `0b9b8d8` | **DRIFT** | Stale "PR #19 flipped default to late" comment; old/rejected knob names survive in scripts, `analyze.py` and `RESULTS.md` |

No branch changes `--help` or the CLI predictor tables. The "pending
allocation" note for `-p gate`/`-p hnet` in `docs/design.md` remains correct.

---

## 1. PR #6 NAMO — `e40b8c0`

### 1.1 Names/defaults actually in code

| name | default | code behaviour | doc says | verdict |
|------|---------|----------------|----------|---------|
| `TAW_HYBRID=namo\|namod` (`namo-d` alias still accepted) | `adam` | `fit.c:771-774` | `namo\|namod`, default `adam` | ✓ (alias undocumented — minor) |
| `TAW_NAMO_BETA` | `0.99` | `fit.c:806` | `0.99` | ✓ |
| `TAW_NAMO_SCALE` | `1.0` | `fit.c:807` | `1.0` | ✓ |
| `TAW_NAMO_CLAMP` | `0.1` | `fit.c:808` | `0.1` | ✓ |
| `TAW_NAMO_WD` | `1e-3` (inherits `o.wd`) | `fit.c:829-830` | `1e-3` | ✓ |
| `TAW_NAMO_D_AXIS=col\|row` | `col` | `fit.c:818` | `col` | ✓ |
| `TAW_NAMO_GATE` | `none`; `c3` enables the **shared** shrink gate, W1-only, inert outside NAMO | `fit.c:811-813`, `src/fit_opt.h` | "there is no NAMO-specific gate setting" | ✗ **false** |
| `TAW_NAMO_GATE_BETA` / `_P` | `0.9` / `1.0` | `fit.c:813-814` | not documented at all | ✗ **undocumented** |
| `TAW_NAMO_NEST` | — | **does not exist** | documented, default `0` | ✗ **phantom** |
| `TAW_MUON_NEST` | `1`, but **`0` when NAMO mode active** | `fit.c:817` | "keeps its shipped default of `1`" | ✗ **false** |

`TAW_MALT`/`TAW_MALTER` still override `TAW_HYBRID` (parsed later) — the
doc's precedence note is correct. The fix's real achievement is structural:
`src/fit_opt.h` now owns one Newton–Schulz loop and one
`shrink_begin`/`shrink_at` gate; the private `c3_gate` is gone. That is an
implementation fix, not a knob-surface fix.

### 1.2 Drift to report

1. **Phantom `TAW_NAMO_NEST`.** My ruling invented a rename that the branch
   did not implement. The branch keeps the conditional default on the shared
   `TAW_MUON_NEST` instead.
2. **Gate alias exists but is denied in docs.** The ruling said "delete
   `TAW_NAMO_GATE*`"; the fix deliberately kept them as aliases into the
   shared gate. The docs must describe the alias, not deny it.
3. **`TAW_MUON_NEST` conditional default** — explicitly forbidden by the
   ruling ("never a hidden branch on another knob's value") but still present.
   Since the branch is staged, document the actual behaviour and record the
   unmet convention as a follow-up; do not silently claim it was fixed.

### 1.3 Exact doc edits

In `docs/design.md` @ `694f862`, replace the NAMO table row:

```diff
-| `TAW_NAMO_NEST` | `0` | Nesterov momentum for the NAMO W1 update |
+| `TAW_NAMO_GATE=none\|c3` | `none` | NAMO-only alias into the shared sign-shrink gate; inert outside NAMO modes |
+| `TAW_NAMO_GATE_BETA` / `TAW_NAMO_GATE_P` | `0.9` / `1.0` | sign-EMA decay / gate exponent for that alias |
```

and replace the paragraph under that table:

```diff
-`TAW_MALT`/`TAW_MALTER` take precedence over `TAW_HYBRID`; leave them unset
-when selecting a NAMO mode. Sign gating has one owner, `TAW_EPS_MODE=c3`
-below — there is no NAMO-specific gate setting. The shared `TAW_MUON_NEST`
-keeps its shipped default of `1`; NAMO has its own `TAW_NAMO_NEST`.
+`TAW_MALT`/`TAW_MALTER` take precedence over `TAW_HYBRID`; leave them unset
+when selecting a NAMO mode. Gate *arithmetic* has one owner
+(`shrink_begin`/`shrink_at` in `src/fit_opt.h`); `TAW_NAMO_GATE=c3` is the
+NAMO-only alias into it, while `TAW_EPS_MODE=c3` is the AdamW-side owner.
+`TAW_MUON_NEST` keeps its shipped default of `1` outside NAMO and is forced
+to `0` in NAMO modes (the paper uses un-extrapolated momentum); this is a
+known deviation from the one-knob-one-meaning convention, not a new knob.
```

In `KNOBS.md` @ `b080612` (ruling), replace lines 258-259:

```diff
-| `TAW_NAMO_GATE`, `TAW_NAMO_GATE_BETA`, `TAW_NAMO_GATE_P`, `sig1`, `c3_gate` | #6 | **delete** — C3 owned by #8 | cross-PR feature duplication |
-| `TAW_MUON_NEST` conditional default under NAMO | #6 | add `TAW_NAMO_NEST` (default `0`) and leave `TAW_MUON_NEST=1` alone | shared knob must not change meaning by another knob's value |
+| private `c3_gate` implementation | #6 | **done** in `e40b8c0`: shared `shrink_begin`/`shrink_at` in `src/fit_opt.h` | one gate implementation |
+| `TAW_NAMO_GATE`, `_BETA`, `_P` | #6 | **retained** as NAMO-only alias into the shared gate; document, do not deny | staged fix chose aliasing over deletion |
+| `TAW_MUON_NEST` conditional default under NAMO | #6 | **still open**: document `0`-under-NAMO now; follow up with `TAW_NAMO_NEST` if the convention is enforced | staged fix did not add `TAW_NAMO_NEST` |
```

and line 167's clause "Remove `TAW_NAMO_GATE*` (C3 belongs to #8).
Do not change `TAW_MUON_NEST`'s default conditionally." → "Gate arithmetic is
shared (`fit_opt.h`); `TAW_NAMO_GATE*` is a documented NAMO alias. Conditional
`TAW_MUON_NEST` remains an open convention violation."

---

## 2. PR #8 EPS gate — `af5fd00`

### 2.1 Names/defaults actually in code

| name | default | code behaviour | doc says | verdict |
|------|---------|----------------|----------|---------|
| `TAW_EPS_MODE` | unset → off | `eps_requested()` (lines 509-522): `none`/`off`/`0` off; **any other token enables the gate**; `c2`/`nc9` additionally print "dropped" | `none\|c3` | ~ (see below) |
| `TAW_EPS_SIGN_BETA` | `0.9` | `fit.c:523` | `0.9` | ✓ |
| `TAW_EPS_GATE_P` | `1.0` | `fit.c:524` | `1.0` | ✓ |
| `TAW_EPS_GATE_MODE` | — | **does not exist** | `mean\|shrink`, default `mean` | ✗ **phantom** |
| `TAW_GATE_MODE` | unset → mean-preserving | `fit.c:525-526` (still the old name) | not documented | ✗ **leftover old name** |
| `TAW_EPS_LAYERS=1\|2\|12` | `2` | `fit.c:527-532` | `2` | ✓ |
| `TAW_EPS_SCOPE=all\|cold\|warm` | `all` | `fit.c:533-537` | `all` | ✓ |
| `TAW_EPS_GRAN=row\|set` | — | **removed** with C2 | documented, default `row` | ✗ **phantom** |
| `TAW_EPS_DEBUG` | unset | `fit.c:538` | unset | ✓ |

Removed as ruled: `TAW_EPS_ALPHA`, `TAW_EPS_C`, `TAW_EPS_FLOOR`,
`TAW_FORCE_SEL1/2`, and C2/NC9 code paths — none appear in the docs, good.
`adam_rows_adv` now wraps one shared `adam_rows` implementation and the gate
math lives in `src/fit_gate.h`; both are structural, not knobs.

New constraints introduced by the fix but absent from the doc:

- **GPU guard** (`fit.c:1511-1514`): any requested gate with `-g on|auto`
  returns `TAW_E_ARG`; CPU-only. Documented in the PR's own README/RESULTS but
  not in `docs/design.md`.
- **`TAW_GOLDEN_PRINT`** (`tests/fitgolden.h:47`, test-only): reprints golden
  vectors. Undocumented in `docs/design.md`; violates the `_DEBUG` diagnostic
  naming convention.

Behaviour quirk: `TAW_EPS_MODE=c2` or `nc9` prints a "dropped" warning but
`eps_requested()` still treats it as ON, so the gate is applied. The doc says
only `c3` is supported — a doc/behavior mismatch worth flagging.

The fix also **added** a README section that itself documents the unrenamed
`TAW_GATE_MODE` and points to `docs/design.md ("C3 sign-consistency gate")`,
a section title that does not exist on `foundation/knob-docs`.

### 2.2 Exact doc edits

In `docs/design.md` @ `694f862`:

```diff
-| `TAW_EPS_MODE=none\|c3` | `none` | sign-EMA shrink gate on AdamW vector updates |
+| `TAW_EPS_MODE=none\|c3` | `none` | sign-EMA shrink gate on AdamW vector updates; any non-`none`/`off`/`0` token currently enables it, `c2`/`nc9` print a "dropped" warning |
@@
-| `TAW_EPS_GATE_MODE=mean\|shrink` | `mean` | mean-preserving or shrink-only gate |
+| `TAW_GATE_MODE=mean\|shrink` | unset → `mean` | mean-preserving or shrink-only gate (**not yet renamed to `TAW_EPS_GATE_MODE`**) |
@@
-| `TAW_EPS_GRAN=row\|set` | `row` | per-weight or per-set granularity |
```
(delete the `TAW_EPS_GRAN` row entirely)

After the paragraph ending "...every update is the shipped one.", add:

```diff
+With `TAW_EPS_MODE=c3` and `-g on` or `-g auto`, fitting is rejected with
+`TAW_E_ARG` (the gate is implemented in the CPU fitter only). The test suite
+also reads `TAW_GOLDEN_PRINT` (any value) to reprint golden vectors; it is a
+test-only diagnostic, not a product knob.
```

In PR #8's `README.md` at `af5fd00` (currently a drift source):

```diff
-| `TAW_GATE_MODE` | `mean` (default), `shrink` | normalize by the row mean, or shrink directly |
+| `TAW_GATE_MODE` | `mean` (default), `shrink` | normalize by the row mean, or shrink directly (pending rename to `TAW_EPS_GATE_MODE`) |
@@
-See `docs/design.md` ("C3 sign-consistency gate") for the method and measurements.
+See `docs/design.md` ("Environment knobs (encoder-only)" → "Epsilon / sign gate") for the method and measurements.
```

In `KNOBS.md`, line 256 row stays as the **target** rename, but add a status
note: "not applied in staged fix `af5fd00` — code and PR README still use
`TAW_GATE_MODE`"; delete line 122 (`TAW_EPS_GRAN`) from the registry or mark
it "removed in `af5fd00`"; line 412's doc line drops `TAW_EPS_GRAN=row|set`.

**Code change still required for the doc to become true:** rename
`TAW_GATE_MODE` → `TAW_EPS_GATE_MODE` in `src/fit.c:525`, PR README, PR
RESULTS examples (lines 209, 389) and the #21 harness strings; then flip the
two diff lines above back to the canonical name. Also change
`eps_requested()` to test `c3` explicitly if `c2`/`nc9` must be rejected
rather than merely warned about.

---

## 3. PR #19 EMA — `16d91c4`

### 3.1 Verification

| check | result |
|-------|--------|
| `TAW_FIT_AVG_DEFAULT` | `TAW_AVG_OFF` (`fit.c:38`) ✓ |
| unset `TAW_FIT_AVG` | returns `TAW_AVG_DEFAULT` = `off`; shipped fitter byte-identical ✓ |
| `auto` alias | removed (`fit.c:59` now `7`/`late` only) ✓ |
| `late` | still EMA(0.999) for `epochs >= 6` (levels 3+), opt-in ✓ |
| GPU warm-start nets | initialised `TAW_AVG_OFF` ✓ |
| `bench/control.sh` | no `TAW_FIT_LRSCALE`; uses `TAW_LR0`/`TAW_LR1` ✓ |
| old names in C sources | none ✗→✓ |
| new production knobs | none beyond `TAW_FIT_AVG` ✓ |

The foundation doc's `TAW_FIT_AVG` line is **accurate**. The only gap is
value-vocabulary: the code still accepts numeric aliases `0..7` and `ema09`,
and `RESULTS_EMA.md:32` says "(also `0..7`)" while the published doc lists
only canonical names.

Harness-only env vars introduced by the fix: `TAW_BENCH_DATA` and `TAW`
(binary path) in `bench/{bench,control}.sh`, documented in `control.sh`'s
usage comment. `TAW` is a very generic name; recommend `TAW_BENCH_BIN` in a
follow-up, but no product-doc line is required.

### 3.2 Exact doc edit (minor, canonical vocabulary)

In `docs/design.md` @ `694f862`:

```diff
-`TAW_FIT_AVG=off|late|swa|ema9|ema95|ema99|ema999|swa_all` (default `off`)
+`TAW_FIT_AVG=off|late|swa|ema9|ema95|ema99|ema999|swa_all` (default `off`; numeric aliases `0..7` and `ema09` are also accepted)
```

`KNOBS.md` line 414 needs the same parenthetical. No other KNOBS.md edit for
#19.

---

## 4. PR #21 final matrix — `0b9b8d8`

### 4.1 Verification

| check | result |
|-------|--------|
| `.gitignore` | original four build/archive patterns restored ✓ |
| private absolute paths in scripts | none; all paths via `BENCH_*` ✓ |
| `TAW_FIT_AVG=off` | pinned via `FITPIN` in every cell; `env -i` scrubs inherited state ✓ |
| `BENCH_*` harness knobs | documented in `RESULTS.md:344` ✓ |
| unmerged-branch provenance | stated in `RESULTS.md:31` ("LOCAL APPENDIX") ✓ |
| stale default claim in comment | `bench_one.sh:13-14` still says PR #19 flipped the default to `TAW_AVG_LATE` ✗ |
| old/rejected knob names | `TAW_SIGN_BETA`, `TAW_GATE_P`, `TAW_GATE_MODE`, `TAW_FARO_ROUTE` still in scripts and report ✗ |

Note: `RESULTS.md:13` and `bench_config.sh:20` still default `BENCH_ROOT` to
`/tmp/opencode/tadamw-bench`. That is a shared temp path, not a private home,
and it is overridable — acceptable, but worth a "read-only corpus root"
footnote (already present at `RESULTS.md:13`).

### 4.2 Exact doc edits

In `bench_one.sh` at `0b9b8d8`, replace lines 13-14:

```diff
-# PR #19 flipped the fitter's default to TAW_AVG_LATE for L3+, which changes
-# shipped L5/L9 bytes. Pin the historic behaviour for every single cell.
+# PR #19 now defaults TAW_FIT_AVG=off. Keep this pin so the recorded cells
+# remain byte-comparable to the pre-fix matrix and survive a future re-flip
+# of the compile-time default.
```

In `bench_one.sh:19`, `analyze.py:129` and `RESULTS.md:43` (the C3+ELR config
string), the names `TAW_SIGN_BETA`, `TAW_GATE_P`, `TAW_GATE_MODE` are
**pre-rename historical strings** used with an unmerged binary. Do **not**
silently rewrite them in the runner — that would stop reproducing the
archived bytes. Instead append a footnote in `RESULTS.md` §"Configurations":

```diff
+The `c3elr` row deliberately uses the pre-rename env names
+(`TAW_SIGN_BETA`, `TAW_GATE_P`, `TAW_GATE_MODE`) because its binary is an
+unmerged branch built before PR #8's rename. Current equivalents are
+`TAW_EPS_SIGN_BETA`, `TAW_EPS_GATE_P`, `TAW_EPS_GATE_MODE`; the latter is
+still pending in the staged code (`af5fd00`).
```

Likewise for the FARO row (`bench_one.sh:45`, `analyze.py:124`,
`RESULTS.md:38`): add a footnote that `TAW_FARO_ROUTE=muon` belongs to the
rejected routing lane and is retained only to reproduce the historical
measurements; the supported surface is ERC-only (`TAW_FARO`, `_ALLOC`,
`_BLOCKS`, `_CAP`, `_GAMMA`, `_KAPPA`, `_FLOOR`, `_CEIL`, `_DEBUG`).

`TAW_ELR*`, `TAW_ADAGO*`, `TAW_SEED` remain unmerged-branch knobs; they are
already labelled as a local appendix, so no further doc change is needed
beyond keeping `analyze.py`'s generated template in sync with `RESULTS.md`.

---

## 5. Cross-cutting summary of drift classes

| class | items |
|-------|-------|
| **Phantom knobs** (documented, not in code) | `TAW_NAMO_NEST` (#6); `TAW_EPS_GATE_MODE` (#8); `TAW_EPS_GRAN` (#8) |
| **Leftover old names** (in code/docs) | `TAW_GATE_MODE` (#8 code + README + RESULTS); `TAW_SIGN_BETA`, `TAW_GATE_P`, `TAW_GATE_MODE`, `TAW_FARO_ROUTE` (#21 harness/report, historical) |
| **Undocumented new knobs** | `TAW_GOLDEN_PRINT` (#8, test-only); `TAW_BENCH_DATA`, `TAW` (#19 harness); `BENCH_*` (#21 harness — documented in its RESULTS) |
| **False/undocumented semantics** | `TAW_MUON_NEST` conditional default (#6); NAMO gate alias (#6); GPU guard + `c2`/`nc9` warn-but-enable (#8); stale #19-default comment (#21) |
| **Undocumented aliases** | `namo-d` (#6); `0..7`, `ema09` (#19) |
| **Correct as written** | FARO ERC names/defaults; `TAW_LR1=flat` (unchanged by these fixes); `TAW_FIT_AVG=off`; `TAW_EPS_SIGN_BETA`/`TAW_EPS_GATE_P`; `-p gate`/`-p hnet` pending-allocation note; byte-identity policy paragraph |

## 6. Order of operations for the doc owner

1. Apply §2 edits to `docs/design.md` and PR #8's README (largest drift).
2. Apply §1 edits (NAMO alias + remove phantom `TAW_NAMO_NEST`).
3. Apply §3 one-line alias note and §4 stale-comment/footnote edits.
4. Update `KNOBS.md` rows listed in §§1.3, 2.2, 3.2 so the ruling stops
   asserting the pre-fix state.
5. Land the `TAW_GATE_MODE` → `TAW_EPS_GATE_MODE` code rename in #8 before
   removing the "pending rename" hedge from the docs.
6. Re-run this reconciliation; it should then be all MATCH.
