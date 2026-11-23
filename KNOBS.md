# KNOBS.md — the Knob Marshal's registry

> "No undocumented knob enters this repo on my watch."

Scope: the env/CLI/build surface on `origin/main` @ `1aa3dd7` and every knob
introduced by PRs **#5, #6, #7, #8, #11, #13, #19, #21**. PR branches were read
read-only through `myfork`; no corpus, sibling worktree, or upstream ref was
touched, and nothing was pushed. Snapshot SHAs reviewed:

| PR | branch | HEAD reviewed | base |
|----|--------|---------------|------|
| #5 FARO | `exp/faro` | `b94d9fd` | rebased onto `1aa3dd7` |
| #6 NAMO | `exp/namo` | `41b47bf` | rebased onto `1aa3dd7` |
| #7 LR baseline | `exp/lr-baseline` | `96a7919` | `1aa3dd7` |
| #8 C2/C3 gate | `exp/eps-gate` | `9487c26` | `1aa3dd7` |
| #11 entropy gate | `exp/entropygate` | `f13f614` | `1aa3dd7` |
| #13 H-Net boundary | `exp/hnet-feature` | `1897fec` | `1aa3dd7` |
| #19 gated EMA/SWA | `exp/ema` | `d671eb2` | `1aa3dd7` |
| #21 final matrix | `exp/final-matrix` | `d9f0ba7` | `1aa3dd7` |

Documentation notation: **help** = `taw --help`; **README** = `README.md`
usage/options; **design** = `docs/design.md` / `docs/format.md`; **RESULTS** =
a `RESULTS*.md` research report (not user documentation). A knob whose only
documentation is a RESULTS report is **undocumented** for registry purposes.

---

## 1. Registry

### 1.A CLI flags (public surface)

| Name | Kind | Source | Default | Documented? | Conflicts? | Effect on default archives |
|------|------|--------|---------|-------------|------------|----------------------------|
| `taw c/d/x/i`, `-h`, `--help`, `--version` | subcommands | main | — | help ✓ README ✓ | none | none |
| `-l 1..9` | encoder flag | main | `5` | help ✓ README ✓ | none | selects fitting effort; default unchanged |
| `-s` | encoder flag | main | off (`.aw`) | help ✓ README ✓ | none | container variant only |
| `-B N` | encoder flag | main | `1 MiB` | help ✓ README ✓ | none | seek block size |
| `-p auto\|order0\|mix` | encoder flag | main | `auto` | help ✓ README ✓ | none | `auto`/`mix` = byte-identical default |
| `-w auto\|f16\|…\|iq2` | encoder flag | main | `auto` | help ✓ README ✓ | none | encoder-only |
| `-j N` | encoder/decoder flag | main | `1` | help ✓ README ✓ | none | threads; decoder output pinning tested |
| `-g off\|auto\|on` | encoder flag | main | `off` | help ✓ README ✓ | none | opt-in GPU |
| `-p gate` | encoder flag + enum `TAW_PREDICTOR_GATE` | **#11** | `auto` (opt-in) | help ✓ (branch), design/format ✓, **README ✗** (audit required it) | profile id 11 reserved by other unmerged branches (audit/b) | none while `auto`/`mix` unchanged ✓ |
| `-p hnet` | encoder flag + enum `TAW_PREDICTOR_HNET` | **#13** | `auto` (opt-in) | help ✓ README ✓ design/format ✓ | profile id 14 also claimed by sibling branches | none while `auto`/`mix` unchanged ✓ |

There is **no CLI flag** for any optimizer/averaging/epsilon knob in any PR.
#21 adds no CLI flag.

### 1.B Build-time knobs (CMake → `<tadamw/config.h>`)

| Name | Source | Default | Documented? | Effect on bytes |
|------|--------|---------|-------------|-----------------|
| `TADAMW_ENCODER`, `TADAMW_DECODER`, `TADAMW_TOOLS`, `TADAMW_TESTS`, `TADAMW_INSTALL`, `BUILD_SHARED_LIBS` | main | ON/ON/ON/ON/ON/OFF | README build table ✓ | none (capability only) |
| `TADAMW_GPU` | main | auto (Vulkan + glslang) | README ✓ | none (capability only) |
| `TADAMW_WERROR`, `TADAMW_SANITIZE` | main | OFF | README (text) ✓ | none (diagnostic builds) |
| `TAW_HAVE_ENCODER`, `TAW_HAVE_DECODER`, `TAW_HAVE_GPU` | `cmake/config.h.in` | derived | `tadamw.h` comment ✓ | none — read-only capability macros |
| `TAW_FIT_AVG_DEFAULT` | **#19** (compile-time macro in `fit.c`, *not* in config.h) | **`TAW_AVG_LATE`** | RESULTS_EMA only | **CHANGES bytes at L3–L9** ✗ |

`TADAMW_NX` is auto-detected (OpenSSL 3 + LZ4), not a CMake option, despite
the README table wording. No PR other than #19 adds a build-time knob.

### 1.C Environment knobs already on main (baseline)

All are `getenv`/`fit_env_*` in `src/fit.c` unless noted. `unset = shipped
behaviour` is the implicit contract; none appears in `--help` or `README.md`.
`design.md` documents only `TAW_FIT_DEBUG` and `TAW_GPU_DEVICE`. `RESULTS.md`
documents the MALT/Muon subset only.

| Name | Default | Value vocabulary | Documented? | Default-archive effect |
|------|---------|------------------|-------------|------------------------|
| `TAW_HYBRID` | `adam` | `adam\|muon\|blend\|alt\|route` | RESULTS (partial) | none |
| `TAW_MALT` | `off` | `off\|0\|1\|on\|malt\|malter` | RESULTS ✓ | none |
| `TAW_MALTER` | `0` | `0\|1` | RESULTS ✓ | none |
| `TAW_LR0` / `TAW_LR1` | `0.02` / `lr0/20` | float | ✗ | none |
| `TAW_VEC_LR0` / `TAW_VEC_LR1` | `lr0` / `lr1` | float | RESULTS ✓ (winner values) | none |
| `TAW_MUON_LR0` / `TAW_MUON_LR1` | `0.01` / `lr0/20` | float | RESULTS ✓ (winner values) | none |
| `TAW_ADAM1_LR0` / `TAW_ADAM1_LR1` | `vec_lr0` / `vec_lr1` | float | ✗ | none |
| `TAW_MALT_LR0` / `TAW_MALT_LR1` | `0.055` / `0.0011` | float | RESULTS ✓ | none |
| `TAW_B1` / `TAW_B2` | `0.9` / `0.999` | float | ✗ | none |
| `TAW_MUON_MOM` / `TAW_MUON_GAIN` | `0.95` / `0.2` | float | RESULTS (MOM only) | none |
| `TAW_MUON_NS` / `TAW_MUON_NEST` | `5` / `1` | uint | RESULTS ✓ | none |
| `TAW_MALT_B2` / `TAW_MALT_EPS` / `TAW_MALT_GRAFT` | `0.999` / `1e-8` / `o` | float/float/`o\|m\|d` | RESULTS ✓ | none |
| `TAW_WD` / `TAW_MUON_WD` / `TAW_ADAM1_WD` | `1e-5` / (`1e-5` in adam mode, else `1e-3`) / `vecwd` | float | ✗ | none |
| `TAW_ALPHA` | `0.5` | float | ✗ | none |
| `TAW_WARM` | `muon` | `muon\|adam` | ✗ | none |
| `TAW_ALT` / `TAW_ALT_FIRST` | `step` / `muon` | `step\|half`, `muon\|adam` | ✗ | none |
| `TAW_ROUTE_Q` | `0.5` | float | ✗ | none |
| `TAW_FIT_DEBUG` | unset | `1` (print), `a` (all candidates) | design ✓ | none (diagnostic) |
| `TAW_MALT_DEBUG` | unset | any | RESULTS ✓ | none (diagnostic) |
| `TAW_GPU_DEVICE` (`fit_gpu.c`) | unset | integer device index | design ✓ | none (capability selection) |

**Finding:** 30+ tuning knobs already ship with no user-facing documentation.
This is the root cause of the PR drift: contributors follow the existing
"RESULTS.md is the doc" pattern. Fix the pattern, not just the new knobs.

### 1.D Environment knobs introduced by the PRs

| Name | PR | Default | Value vocabulary | Documented? | Conflicts with existing name? | Effect on default archives |
|------|----|---------|------------------|-------------|-------------------------------|----------------------------|
| `TAW_FARO` | #5 | `0` (off) | `0\|1\|on\|yes\|true` | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_ROUTE` | #5 | `ret` | `ret\|muon\|adam` | RESULTS only | no — but PR's own routing lane is a documented failure | none (off) ✓ |
| `TAW_FARO_ALLOC` | #5 | `erc` | `erc\|none` | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_STAT` | #5 | `probe` | `probe\|lin` | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_BLOCKS` | #5 | `set` | `set\|layer` | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_EXPLORE` | #5 | `20` | uint | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_GAMMA` / `_KAPPA` / `_CAP` / `_FLOOR` / `_CEIL` | #5 | `0.9` / `3.0` / `1.4` / `0.5` / `2.0` | float | RESULTS only | no | none (off) ✓ |
| `TAW_FARO_DEBUG` | #5 | `0` | uint | RESULTS only | no | none (diagnostic) ✓ |
| `TAW_HYBRID=namo\|namod\|namo-d` | #6 | `adam` | **new values** on existing knob | RESULTS only | **value extension**; `TAW_MALT` is parsed *after* and silently overrides it | none (opt-in) ✓ |
| `TAW_NAMO_BETA` / `_SCALE` / `_CLAMP` | #6 | `0.99` / `1.0` / `0.1` | float (`CLAMP=0` = off) | RESULTS only | no | none (opt-in) ✓ |
| `TAW_NAMO_WD` | #6 | `o.wd` (`1e-3` for NAMO) | float | RESULTS only | **duplicates** `TAW_MUON_WD`/`TAW_WD` | none (opt-in) ✓ |
| `TAW_NAMO_D_AXIS` | #6 | `col` | `col\|row` | RESULTS only | no | none (opt-in) ✓ |
| `TAW_NAMO_GATE` | #6 | `none` | `none\|c3` | RESULTS only | **collides conceptually with #8's `TAW_EPS_MODE=c3`** | none (opt-in) ✓ |
| `TAW_NAMO_GATE_BETA` / `TAW_NAMO_GATE_P` | #6 | `0.9` / `1.0` | float | RESULTS only | `_P` near-duplicates #8's `TAW_GATE_P` | none (opt-in) ✓ |
| `TAW_MUON_NEST` (default changed to `0` under NAMO) | #6 | `1` → **`0` when NAMO** | existing knob | RESULTS only | **semantic overload** of a shared knob | none (NAMO off) ✓ |
| `TAW_LR1=flat` | #7 | numeric `lr0/20` | **new sentinel** on existing numeric knob | RESULTS/sweep only | **value collision**: `flat` is not a float; needs special-case parse | none (opt-in) ✓ |
| `TAW_EPS_MODE` | #8 | unset → `none` | `none\|c2\|c3\|nc9` | RESULTS only | no | none (opt-in) ✓ |
| `TAW_EPS_ALPHA` / `_C` / `_FLOOR` | #8 | `0.5` / `1e-3` / `1e-8` | float | RESULTS only | no | none (opt-in) ✓ |
| `TAW_SIGN_BETA` | #8 | `0.9` | float | RESULTS only | **unprefixed**; should be `TAW_EPS_SIGN_BETA` | none (opt-in) ✓ |
| `TAW_GATE_P` | #8 | `1.0` | float | RESULTS only | **unprefixed** + near-duplicate of #6 `TAW_NAMO_GATE_P` | none (opt-in) ✓ |
| `TAW_GATE_MODE` | #8 | unset → mean-preserving | `shrink` | RESULTS only | **unprefixed**; ambiguous next to `TAW_NAMO_GATE` | none (opt-in) ✓ |
| `TAW_EPS_LAYERS` | #8 | `2` | bitstring `1\|2\|12` | RESULTS only | no | none (opt-in) ✓ |
| `TAW_EPS_SCOPE` | #8 | `all` | `all\|cold\|warm` | RESULTS only | no | none (opt-in) ✓ |
| `TAW_EPS_GRAN` | #8 | `row` | `row\|set` | RESULTS only | no | none (opt-in) ✓ |
| `TAW_EPS_DEBUG` | #8 | unset | any | RESULTS only | no | none (diagnostic) ✓ |
| `TAW_FORCE_SEL1` / `TAW_FORCE_SEL2` | #8 | unset | int | RESULTS only | **wrong layer**: diagnostic shape filter inside `enumerate()` (core candidate selection) | none unless set; should not ship in production ✓ |
| `TAW_FIT_AVG` | #19 | **unset → `late`** | `off\|swa\|ema9\|ema99\|ema999\|swa_all\|ema95\|late\|auto`, `0..7` | RESULTS_EMA only | no exact name, but **sentinel `auto` duplicates `late`** | **CHANGES bytes at L3–L9** ✗ |
| `TAW_FIT_AVG_DEFAULT` (compile-time) | #19 | **`TAW_AVG_LATE`** | compile-time int | RESULTS_EMA only | `TAW_`-prefixed macro is not in `config.h` and not documented as build-time | **root cause of the L3+ default change** ✗ |
| *(none)* | #11 | — | — | CLI enum `TAW_PREDICTOR_GATE` / profile 11 | — | ✓ |
| *(none)* | #13 | — | — | CLI enum `TAW_PREDICTOR_HNET` / profile 14 | — | ✓ |
| *(none)* | #21 | — | — | scripts only | — | ✓ (no code knobs) |

### 1.E #21 script-only knob references (documentation without implementation)

`exp/final-matrix`'s `bench_one.sh` / `analyze.py` reference knobs from
**unmerged** branches: `TAW_ELR`, `TAW_ELR_C/SCOPE/BETA/CLAMP/SCHED`,
`TAW_ADAGO`, `TAW_ADAGO_LR/EPS`, `TAW_SEED`. These are not knobs of #21 and do
not exist on main. They must not be copied into user docs; #21's repro section
has to state the unmerged-branch provenance (audit/d finding) or drop the
runner scripts. #21 also **regresses `.gitignore`** (deletes `/build*/`,
`/compile_commands.json`, `*.aw`, `*.awseek`) and hardcodes sibling-worktree
paths — restore the four lines and parameterise paths before merge.

---

## 2. Default policy ruling

### 2.1 The rule

1. **Byte-identity is the default.** With no env override and no explicit
   `-p`/`-w`/`-g`/API selection, compression must reproduce the shipped
   archives bit-for-bit for every level and every golden corpus. Any knob that
   can change fitted weights must default to the shipped path.
2. **Opt-in is explicit.** A knob may change bytes only when a user sets an
   env var, a CLI flag, or an API field. "Gated by epoch count" is not opt-in —
   it is a silent default change.
3. **`auto`/`mix` are sacred.** New predictor profiles may change bytes only
   under their own `-p` value. `auto`, `order0`, `mix`, `-w auto` stay exactly
   as on main (this is why #17/#12 were rejected by the other audits).
4. **Build knobs never change bytes** for the same input + params.
5. **Diagnostic knobs do not ship in production paths** (`enumerate()`,
   candidate scoring, decoder hot loops) unless test-only-gated.

### 2.2 Per-PR ruling

| PR | Ruling |
|----|--------|
| #5 FARO | **Condition: drop the routing lane.** `TAW_FARO=0` already preserves bytes ✓. `TAW_FARO_ROUTE/STAT/EXPLORE` implement a lane the PR itself calls unvalidated (audit verdict DRAFT); drop them or re-label as a negative-result doc. Retain only the ERC lane knobs, then rename the feature "FARO-ERC". |
| #6 NAMO | **Condition: fix precedence + drop duplicated C3.** Default `TAW_HYBRID=adam` is byte-identical ✓. `TAW_MALT` is parsed after `namo|namod` and silently overrides them — document precedence or unify into one `TAW_HYBRID` vocabulary. Remove `TAW_NAMO_GATE*` (C3 belongs to #8). Do not change `TAW_MUON_NEST`'s default conditionally. |
| #7 `flat` | **Approve with formalisation.** Default stays numeric ✓. Accept `flat` as an explicit sentinel, but document it; preferably move it to `TAW_LR_SCHEDULE=flat\|decay` so `TAW_LR1` stays a float. |
| #8 EPS | **Condition: narrow.** `TAW_EPS_MODE` unset → `none` is byte-identical ✓. Remove `TAW_FORCE_SEL1/2` from `enumerate()`. Drop or clearly mark rejected C2/NC9 paths. Keep one owner for C3 (this PR, not #6). |
| #11 `-p gate` | **Approve as explicit opt-in** — `auto`/`mix` unchanged ✓. Add the missing README line. Keep profile 11 gated behind its own CLI value. |
| #13 `-p hnet` | **Approve as explicit opt-in** — default unchanged ✓. Keep profile 14 behind `-p hnet`; keep the `TAW_GPU_OFF` guard (GPU path hardcodes `P1_F=26`). |
| #19 EMA | **REJECT the default. Must default OFF.** See 2.3. |
| #21 scripts | **No runtime default impact**, but block on `.gitignore` restore + path parameterisation; no new user-facing knobs. |

### 2.3 Fixing PR #19's "default ON for levels 3+"

Root cause: `src/fit.c`

```c
#ifndef TAW_FIT_AVG_DEFAULT
#define TAW_FIT_AVG_DEFAULT TAW_AVG_LATE
#endif
static int parse_avg_mode(const char *s) {
    if (!s || !*s) return TAW_FIT_AVG_DEFAULT;   /* silent default */
    ...
}
static int resolve_avg_mode(int mode, int epochs) {
    if (mode == TAW_AVG_LATE)
        return epochs >= 6 ? TAW_AVG_EMA999 : TAW_AVG_OFF;  /* L3+ */
    return mode;
}
```

Because `-l 3..9` yields ≥6 epochs, an unset `TAW_FIT_AVG` folds EMA(0.999)
into the fitted model and changes the archive bytes of every level-3+ encode —
including MALTER/Muon runs, on which the PR never benchmarked (audit/d §19).
The in-code comment "With no environment overrides this is the shipped AdamW
fitter, bit for bit" is therefore false on that branch.

Required fix, in order:

1. `#define TAW_FIT_AVG_DEFAULT TAW_AVG_OFF` (or delete the macro and have
   `parse_avg_mode(NULL|"")` return `TAW_AVG_OFF`). The macro must be the
   canonical, single default source.
2. Keep the whole `enum`, `resolve_avg_mode()`, and the `late` value: it becomes
   an **explicit opt-in** (`TAW_FIT_AVG=late` reproduces the PR's levels-3+
   behaviour on demand). Do not delete the experiment.
3. Make `parse_avg_mode()` reject unknown strings to `TAW_AVG_OFF`, not to a
   compile-time default; remove the `auto` alias (it collides with `-p auto`
   vocabulary) and pick one name for the late mode (`late`).
4. Correct the stale "bit for bit" comment.
5. Rewrite `bench/control.sh`: `TAW_FIT_LRSCALE=$s` references a knob that no
   longer exists — use `TAW_LR0`/`TAW_LR1`.
6. Re-run or relabel the bench TSVs (they are byte-identical to the pre-rebase
   branch) and add an A/B on the merged MALTER champion before anyone may
   sign off on a default change.
7. Add tests: (a) `TAW_FIT_AVG=off` reproduces the four golden archives;
   (b) `TAW_FIT_AVG=late` is deterministic; (c) the averaged model decodes.
8. Document `TAW_FIT_AVG` in `docs/design.md`'s env section and add it to this
   registry. If the maintainers later *do* want a changed default, that is a
   separate, explicitly-documented decision with README/format sign-off.

---

## 3. Naming conventions to enforce

### 3.1 Rules

- **Prefix:** every env knob is `TAW_<FAMILY>_<PARAM>`. Feature-local params
  carry the feature family prefix (`TAW_FARO_*`, `TAW_NAMO_*`, `TAW_EPS_*`).
  Bare `TAW_<GENERIC>` names for a feature-local parameter are rejected.
- **Boolean vocabulary:** canonical `off|on`; accept `0|1` as aliases. Pick one
  and use it everywhere — today FARO accepts `1/on/yes/true`, MALT accepts
  `1/on/malt`, NAMO accepts `col/row`, which is incoherent.
- **Value vocabulary:** lowercase single tokens, documented in one place, with
  one canonical name per value and a short alias list. **No token may mean two
  different things across PRs** (`c3` must belong to exactly one owner). Avoid
  magic sentinels inside numeric knobs (`TAW_LR1=flat`); prefer an explicit
  mode knob (`TAW_LR_SCHEDULE`).
- **Env vs CLI parity:** a setting that changes archive bytes and is meant for
  users gets a CLI flag + public API enum and full docs. A tuning-only setting
  stays env-only, encoder-only, and is documented in `docs/design.md` — **not**
  in `--help`. CLI token names must match the public enum name and the
  `profile_name()` string.
- **Defaults live in exactly one place** (code default or documented
  compile-time macro), never a hidden branch on epoch count.
- **Diagnostics** use a `_DEBUG` suffix and are gated/stderr-only;
  they never alter candidate enumeration.

### 3.2 Renames required for colliders

| Current | PR | Rename / action | Reason |
|---------|----|-----------------|--------|
| `TAW_SIGN_BETA` | #8 | `TAW_EPS_SIGN_BETA` | missing family prefix |
| `TAW_GATE_P` | #8 | `TAW_EPS_GATE_P` | missing prefix; near-duplicate of `TAW_NAMO_GATE_P` |
| `TAW_GATE_MODE` | #8 | `TAW_EPS_GATE_MODE` | ambiguous next to `TAW_NAMO_GATE` |
| `TAW_FORCE_SEL1/2` | #8 | **delete** (or `TAW_TEST_FORCE_SEL1/2` behind a test-only build flag) | diagnostic filter in core `enumerate()` |
| `TAW_NAMO_GATE`, `TAW_NAMO_GATE_BETA`, `TAW_NAMO_GATE_P`, `sig1`, `c3_gate` | #6 | **delete** — C3 owned by #8 | cross-PR feature duplication |
| `TAW_MUON_NEST` conditional default under NAMO | #6 | add `TAW_NAMO_NEST` (default `0`) and leave `TAW_MUON_NEST=1` alone | shared knob must not change meaning by another knob's value |
| `TAW_HYBRID=namod` + alias `namo-d` | #6 | one canonical token `namod` (drop `namo-d` or document as alias) | value vocabulary hygiene |
| `TAW_FIT_AVG=auto` | #19 | drop alias; use `late` only | collides with `-p auto` vocabulary |
| `TAW_FIT_AVG_DEFAULT` | #19 | keep name but set `TAW_AVG_OFF`; if it is ever user-settable, move to `cmake/config.h.in` as `TAW_HAVE_*`-style documented build macro | currently a `TAW_`-prefixed macro invisible to `config.h` |
| `TAW_FARO_ROUTE/STAT/EXPLORE` | #5 | drop with the rejected routing lane | ships a self-declared failure |

### 3.3 Consolidation targets (accuracy/effort knobs duplicated across optimizers)

1. **LR endpoints — 5 families for 2 concepts.**
   `TAW_LR0/1`, `TAW_VEC_LR0/1`, `TAW_MUON_LR0/1`, `TAW_ADAM1_LR0/1`,
   `TAW_MALT_LR0/1` (+ #6 `TAW_NAMO_WD`). Keep `TAW_LR0/LR1` as the universal
   default; allow per-component overrides only where a measured reason exists
   (Muon and MALT need them today). Document the inheritance chain in one
   table instead of re-deriving it in each PR.
2. **Weight decay — 4 names.**
   `TAW_WD`, `TAW_MUON_WD`, `TAW_ADAM1_WD`, `TAW_NAMO_WD` → `TAW_WD` plus
   explicit `_WD` overrides only for components that differ.
3. **Second-moment / decay knobs.**
   `TAW_B1/B2`, `TAW_MALT_B2`, `TAW_NAMO_BETA`, `TAW_NAMO_GATE_BETA` → a
   shared `TAW_BETA1/BETA2` with documented per-optimizer overrides; `TAW_EPS`
   damping currently hard-coded `1e-6` while `TAW_MALT_EPS` is `1e-8` — pick
   one vocabulary for "epsilon".
4. **C3 sign gate — two owners.** #6 `TAW_NAMO_GATE*` and #8
   `TAW_EPS_MODE=c3` + `TAW_SIGN_BETA`/`TAW_GATE_P`/`TAW_GATE_MODE`. One owner
   (recommend #8's `TAW_EPS_*`); #6's comparison moves to #8's RESULTS.
5. **NS hyperparameters.** `TAW_MUON_NS` / `TAW_MUON_NEST` are shared by Muon,
   MALT and NAMO — factor the Newton–Schulz polynomial into one helper (audit
   findings for #5/#6) and keep these knobs single-owned by that helper.
6. **Feature profiles.** `-p gate` (11) and `-p hnet` (14) are correctly
   explicit and need no consolidation, but profile IDs must be reserved in one
   table (`docs/format.md`) to stop sibling branches reusing 11/14.

---

## 4. Minimal public-API / CLI change policy

### 4.1 When a new predictor enum + CLI flag is justified

Introduce a new `taw_predictor` enumerator, a CLI `-p <name>` token, and a new
profile **only if** the feature changes the archive format — i.e. fitted
parameters must be stored in the archive and the decoder must understand them.

- #11 qualifies: profile 11 stores `gate_a/gate_b` in a 4-byte tail → needs
  `TAW_PREDICTOR_GATE`, profile 11 enum in `src/format.h`, format §2.x,
  `profile_name()`, CLI table entry, `--help`, README, loader validation, golden
  vectors, and a CLI round-trip test.
- #13 qualifies: profile 14 stores a 27-wide first layer → needs
  `TAW_PREDICTOR_HNET`, profile 14, format §2.14, loader width rejection,
  README/design/format updates, goldens, CLI round-trip. Its `TAW_GPU_OFF`
  guard must stay until `fit_gpu.c` stops hardcoding `P1_F=26`.

Everything else — optimizers, epsilon/C3 gates, FARO allocation, EMA/SWA,
LR schedule, MALT — is **encoder-side fitting**. It must remain an env-only
knob with **no** new public enum, CLI flag, or `taw.h` change. If a user-facing
selection is genuinely needed for such a feature, add a `taw_params` field
(and the matching CLI flag) rather than inventing a predictor profile; that
still requires the full doc/test checklist below.

### 4.2 What `taw.h` and `config.h` must gain

- `include/tadamw/tadamw.h`: **append** the new `TAW_PREDICTOR_*` enumerator
  (never renumber existing values — ABI/source compatibility), and document
  its opt-in nature inline. No new field for env-only knobs.
- `cmake/config.h.in` / generated `<tadamw/config.h>`: only build-time
  *capability* macros (today `TAW_HAVE_ENCODER/DECODER/GPU`). Add a
  `TAW_HAVE_<FEATURE>` macro **only** if gate/hnet become conditionally
  compiled (e.g. `TADAMW_GATE=ON`). The PRs as written need no config.h change.
  `TAW_FIT_AVG_DEFAULT` is not a config.h macro and must not masquerade as one.
- `tools/taw.c`: `predictors[]` table, `profile_name()` case, and the
  `--help` line must all change together, plus README usage and
  `docs/format.md`. A change to one without the others is a merge blocker.
- Required doc/test checklist for any new profile: format spec section, loader
  validation, golden vectors, scalar/AVX2 parity, `auto`/`mix` byte-identity
  proof, CLI round-trip test.

### 4.3 Env-vs-CLI parity rule

| Setting class | Env knob | CLI flag | Public API | Docs |
|---------------|----------|----------|------------|------|
| Archive format / predictor selection | no | yes (`-p`) | enum in `taw.h` | help, README, format.md |
| Encoder tuning (optimizer, gates, averaging) | yes | no | none | design.md env section + this registry |
| Build capability | CMake option | no | `TAW_HAVE_*` in config.h | README build table |
| Diagnostics | `*_DEBUG` | no | none | design.md (stderr-only) |

---

## 5. Required help-text / README lines for each retained knob

### 5.1 `--help` (exact lines to add; both predictors together)

```
      -p auto|order0|mix|gate|hnet  predictor (default auto: the fitted mixer
                       if it beats order0; gate = experimental entropy-adaptive
                       profile 11; hnet = integer H-Net boundary input,
                       profile 14)
```

### 5.2 README additions

```sh
taw c -p gate in out.aw                # experimental profile 11: entropy-adaptive confidence gate (opt-in)
taw c -p hnet in out.aw                # profile 14: + integer boundary input (opt-in)
```

Plus one line under "Status": *"Profiles 11 (`ENTROPY_GATE`) and 14
(`HNET_BOUNDARY`) are opt-in via `-p gate` / `-p hnet`; the default `auto`/`mix`
path is unchanged."*

### 5.3 New `docs/design.md` section: "Environment knobs (encoder-only)"

Mandatory preamble (one line, applies to every row below):

> All `TAW_*` environment variables are encoder-only tuning knobs. Unset means
> shipped behaviour and byte-identical archives; they are not CLI options and
> do not change the archive format. Values are lowercase; booleans accept
> `off|on` and `0|1`.

One required line per retained knob (baseline knobs first, then PR knobs that
survive review):

```
TAW_HYBRID=adam|muon|blend|alt|route        optimizer family (default adam; TAW_MALT/TAW_MALTER take precedence when set).
TAW_MALT=off|on|malter                     MALT/MALTER on W1 (default off).
TAW_MALTER=0|1                             promote TAW_MALT=on to MALTER (default 0).
TAW_LR0 / TAW_LR1=<float>                  AdamW LR endpoints (default 0.02 / LR0/20); TAW_LR1=flat holds LR constant (PR #7).
TAW_VEC_LR0 / TAW_VEC_LR1=<float>          W2 AdamW LR overrides (default TAW_LR0 / TAW_LR1).
TAW_MUON_LR0 / TAW_MUON_LR1=<float>        Muon W1 LR endpoints (default 0.01 / 0.0005).
TAW_ADAM1_LR0 / TAW_ADAM1_LR1=<float>      Adam-half W1 LR overrides (default TAW_VEC_LR*).
TAW_MALT_LR0 / TAW_MALT_LR1=<float>        MALT/MALTER W1 LR endpoints (default 0.055 / 0.0011).
TAW_B1 / TAW_B2=<float>                    AdamW betas (default 0.9 / 0.999).
TAW_MUON_MOM / TAW_MUON_GAIN=<float>       Muon momentum / update gain (default 0.95 / 0.2).
TAW_MUON_NS / TAW_MUON_NEST=<uint>         Newton-Schulz steps / Nesterov (default 5 / 1).
TAW_MALT_B2 / TAW_MALT_EPS=<float>         MALT preconditioner EMA decay / damping (default 0.999 / 1e-8).
TAW_MALT_GRAFT=o|m|d                       MALT magnitude graft source (default o).
TAW_WD / TAW_MUON_WD / TAW_ADAM1_WD=<float> weight decay per component (default 1e-5 / TAW_WD in adam mode else 1e-3 / TAW_WD).
TAW_ALPHA=<float>                          blend mixing factor (default 0.5).
TAW_WARM=muon|adam                         warm-start optimizer (default muon).
TAW_ALT=step|half, TAW_ALT_FIRST=muon|adam alternate-step schedule controls (defaults step / muon).
TAW_ROUTE_Q=<float>                        routing quantile for HYBRID=route (default 0.5).
TAW_FIT_DEBUG=1|a                          print fitted candidates / fit all (default unset).
TAW_MALT_DEBUG=1                           print final l/r/nu statistics (default unset).
TAW_GPU_DEVICE=<int>                       Vulkan device index (default unset).
TAW_FARO=off|on                            FARO-ERC allocation lane (PR #5; default off; drop the ROUTE/STAT/EXPLORE lane).
TAW_FARO_ALLOC=erc|none                    enable ERC reallocation (default erc).
TAW_FARO_BLOCKS=set|layer                  risk-block granularity (default set).
TAW_FARO_CAP / TAW_FARO_GAMMA / TAW_FARO_KAPPA / TAW_FARO_FLOOR / TAW_FARO_CEIL=<float>  ERC tunables (defaults 1.4 / 0.9 / 3.0 / 0.5 / 2.0).
TAW_HYBRID=namo|namod                      NAMO / NAMO-D W1 optimizer (PR #6; opt-in; one owner for C3).
TAW_NAMO_BETA / TAW_NAMO_SCALE / TAW_NAMO_CLAMP / TAW_NAMO_WD=<float>  NAMO tunables (defaults 0.99 / 1.0 / 0.1 / 1e-3).
TAW_NAMO_D_AXIS=col|row                    NAMO-D diagonal axis (default col).
TAW_LR_SCHEDULE=decay|flat                 LR schedule (PR #7; default decay; replaces the TAW_LR1=flat sentinel if adopted).
TAW_EPS_MODE=none|c2|c3|nc9                AdamW epsilon/sign-gate mode (PR #8; default none; only C3 retained).
TAW_EPS_SIGN_BETA / TAW_EPS_GATE_P=<float>  C3 sign-EMA decay / exponent (defaults 0.9 / 1.0; renamed from TAW_SIGN_BETA / TAW_GATE_P).
TAW_EPS_GATE_MODE=mean|shrink              C3 gate shape (default mean; renamed from TAW_GATE_MODE).
TAW_EPS_LAYERS=1|2|12, TAW_EPS_SCOPE=all|cold|warm, TAW_EPS_GRAN=row|set   C3 scope controls (defaults 2 / all / row).
TAW_EPS_DEBUG=1                            print per-layer gating statistics (default unset).
TAW_FIT_AVG=off|late|swa|ema9|ema95|ema99|ema999|swa_all  EMA/SWA fold-in (PR #19; default off; late = EMA999 for >=6 epochs, opt-in only).
```

Each line above is a **required** line; a PR that adds or changes a knob without
adding/updating its line is a merge blocker under the Knob Marshal rule. No
line is required for #11/#13 beyond §5.1–5.2 (they add no env knob), and none
for #21 (scripts only; its private paths and unmerged-branch env references
must be removed instead).
