# INTEGRATION.md — opt plumbing, exact API and per-PR deletion list

**Branch:** `integration/opt-plumbing` (local only, no push).
**Base:** `foundation/ns-refactor` @ `e682872` + `RECONCILE.md` @ `51671a9`.
**Inputs read read-only:** `exp-namo` @ `e40b8c0`, `exp-faro` @ `09311ce`,
`exp-eps-gate` @ `af5fd00`, `exp-ema` @ `16d91c4`.

This commits the plumbing half of `RECONCILE.md`: one header, one gate
implementation, one hook ABI with functional risk accounting, and no dead
NS helper. No PR-specific behaviour is enabled — every hook is NULL on the
shipped paths, so archives stay byte-identical.

---

## 1. What changed here

| File | Change |
|---|---|
| `src/fit_opt.h` | **only** optimizer header (guard `TAW_FIT_OPT_H` — the same one NAMO uses). Added `struct tnet;` forward decl, `step_shape`, full `opt_hooks`, `fit_gate_step()` decl. Removed `ns_score()` decl. |
| `src/fit_opt.c` | Added `fit_gate_step()` verbatim from staged #8 (`fit.c:544-550`). `adam_rows` now honours `shape.begin/gate` and writes `risk[bid]`; `muon_rows`/`malt_rows` write `risk[bid]`. Ungated `w -=` expressions are unchanged source. Removed `ns_score()` definition. |
| `src/fit_internal.h` | `typedef struct tnet { ... } tnet;` — named the struct so hooks can take an opaque pointer. No layout change. |
| `src/fit.c` | stale comment only (`ns_score` removed from the wrapper list). |
| `tests/test_fit_opt.c` | New `hooks_abi()`: #8's direct gate numbers, a real `step_shape` begin/gate pair versus a verbatim `adam_rows_gated` oracle, and exact FARO risk-bank values. |

---

## 2. The exact API (as implemented)

```c
/* src/fit_opt.h — one header, one guard, no fit_gate.h */
struct tnet;                         /* definition stays in fit_internal.h */

/* the single sign-consistency gate, adopted from #8 */
float fit_gate_step(float *sig, float gk, float beta, float p);

typedef struct {
    void  *ctx;                                              /* gate state */
    void  (*begin)(void *ctx, const float *g, size_t len, float scale);
    float (*gate) (void *ctx, size_t k, float gk);
    float (*eps)  (void *ctx, size_t k, float vh, float dflt); /* C2: NULL */
} step_shape;

typedef struct {
    void      *ctx;     /* feature-private state */
    step_shape shape;   /* the ONLY gate / coordinate-eps slot */
    float     *risk;    /* FARO bank [nb]; NULL = no accounting */
    unsigned   nb;      /* bank length */

    void (*pre_step )(void *ctx, struct tnet *, optcfg *, uint64_t step,
                      bool tied);
    void (*pre_set  )(void *ctx, struct tnet *, unsigned layer, unsigned set,
                      opt_plan *);
    void (*post_set )(void *ctx, struct tnet *, unsigned layer, unsigned set,
                      size_t off, size_t len, uint64_t step);
    void (*post_step)(void *ctx, struct tnet *, uint64_t step);
    void (*finalize) (void *ctx, struct tnet *);
} opt_hooks;
```

Wrapper signatures are unchanged from the foundation:

```c
void adam_rows(optcfg *, float *w, float *m, float *v, const float *g,
    size_t len, float scale, float lr, float wd, float mix,
    const opt_hooks *, unsigned bid);
void muon_rows(const optcfg *, float *w, float *m, const float *g,
    size_t rows, size_t cols, float scale, float lr, float wd, float mix,
    const opt_hooks *, unsigned bid);
void malt_rows(const optcfg *, float *w, float *m, float *l, float *r,
    float *nu, const float *g, size_t rows, size_t cols, float scale,
    float lr, float wd, float mix, bool adaptive,
    const opt_hooks *, unsigned bid);
```

### 2.1 Gate semantics (why there is only one)

`fit_gate_step` is #8's exact arithmetic:
`s = beta*s + (1-beta)*sign(gk); return p==2 ? |s|*|s| : |s|`.

`step_shape.begin` is the **first pass** (advance the sign EMA for every
weight, cache the multiplier, apply mean-preserving normalization with the
`gmean < 1e-12f → 1` guard); `step_shape.gate` is the **second pass**. This
two-pass shape is required by #8's `mean_pres` and is also exactly NAMO's
`shrink_begin`/`shrink_at` shape, so both PRs fit without a second
implementation.

`adam_rows` applies the multiplier to `upd` **before** `lr`
(`upd *= gate; w -= mix*lr*upd`) — #8's deliberate order (`af5fd00`
`fit.c:594-596`: moving it shifts bpc by ~0.02%). When `shape.begin` or
`shape.gate` is NULL it takes the untouched main expression, so the shipped
path is bit-for-bit.

### 2.2 Risk semantics (FARO)

Each wrapper adds `Δ²` to `h->risk[bid]` only when the bank is non-NULL:
`adam_rows` uses `mix*lr*(mh/(fsqrt(vh)+eps))` (**weight decay excluded**,
FARO's exact expression), `muon_rows` uses `factor*x[at]`, `malt_rows` uses
`fac*x[at]`. A generic `w_after - w_before` hook would change rounding and
is deliberately not used.

### 2.3 `ns_score()` — removed, with rationale

The foundation had `ns_score()` for FARO's `TAW_FARO_STAT=lin` routing score.
Staged FARO @ `09311ce` **removed the whole routing lane** (`fit.c:720-726`:
"the routing lane was prototyped here and REMOVED ... no Newton–Schulz
polynomial is duplicated here"). There is no caller left. Removed from
`fit_opt.h`, `fit_opt.c` and the stale `fit.c` comment. If FARO ever restores
`lin`, re-add it as a thin `ns_ortho()` caller — never a fourth copy.

---

## 3. NAMO port patch description (`e40b8c0` → integration)

### 3.1 Delete from NAMO's diff

| # | What | Where in `e40b8c0` | Why |
|---|---|---|---|
| 1 | `src/fit_opt.h` (the whole file) | new file, 74 lines | Foundation's file wins; same `TAW_FIT_OPT_H` guard, byte-identical `ns_ortho` body. Keep `#include "fit_opt.h"` at `fit.c:19`. |
| 2 | `static float fsqrt()` | `fit.c:28-40` | Redefines `fit_opt.h:29`. |
| 3 | `enum { MUON_MAXDIM = 32, MUON_MAXEL = ... }` | `fit.c:450` | `fit_opt.h:46-48` already defines `NS_MAX*` + `MUON_*` aliases. |
| 4 | `typedef enum { ... HYB_NAMO, HYB_NAMOD } hyb_mode` | `fit.c:456-462` | Move the two new values into `fit_opt.h:102`. |
| 5 | `typedef struct { ... shrink_cfg shrink; ... } optcfg` | `fit.c:467-478` | Move `n1t, n2t, nb2, nscale, nclamp, drow` to `fit_opt.h:117`; replace `shrink_cfg shrink` with the shared gate config in `optcfg` (`bool on; float beta, p; bool mean_pres;`). |
| 6 | `adam_rows` / `muon_rows` / `malt_rows` | `fit.c:508`, `:523`, `:653` | Now in `fit_opt.c`; leaving them is a duplicate-symbol link error. Their NS loops are already replaced — do not reintroduce them. |
| 7 | `shrink_cfg`, `shrink_begin`, `shrink_at` | NAMO `fit_opt.h:49-72` | Deleted with the header; use `step_shape` + `fit_gate_step`. |
| 8 | `o.shrink = {...}` initializer | `fit.c:795-802` | Replace with `o.gate = {...}`; keep the `is_namo &&` leak guard. |
| 9 | `src/CMakeLists.txt` change (if any) | — | Foundation already compiles `fit_opt.c`. |

### 3.2 Port `namo_rows` verbatim

Move the body from `exp-namo/src/fit.c:564-634` into `fit_opt.c` unchanged
except for the interface:

```c
void namo_rows(const optcfg *o, float *w, float *m, const float *g,
    float *vs, float *vc, size_t set, size_t rows, size_t cols, float scale,
    float lr, float wd, bool diag, const opt_hooks *h, unsigned bid);
```

* drop the `float *sig` parameter — the gate comes from `h->shape`;
* `shrink_begin(sig, g, rows*cols, scale, &o->shrink)` →
  `h->shape.begin(h->shape.ctx, g, rows*cols, scale)`;
* `upd *= shrink_at(sig, i, &o->shrink)` →
  `upd *= h->shape.gate(h->shape.ctx, i, g[i]*scale)`;
* keep `step = lr * o->nscale` and `w -= step * (diag ? d[...] : alpha) * upd`
  exactly as staged;
* the `ns_ortho(x,a,b,y,mr,mc,o->ns_steps)` call is already there — keep it.

Add the declaration to `fit_opt.h`. Keep NAMO's `v2s/v2c/sig` state, the untie
mirror (`fit.c:427-436`), all `TAW_NAMO_*` env parsing, the `is_namo` warm-up
guard, `n1t/n2t` updates and the dispatch branch (`fit.c:951-958`).

### 3.3 Why no new re-baseline

`tests/test_namo.c:26` already says the CRCs were re-baselined after the X3
shared-NS/shared-gate change, and its gate order now matches #8's (gate
applied to `upd` before the outer scale — NAMO `fit.c:617-622` versus #8
`fit.c:590-597`). These published values must stay **unchanged**:

| arm | L1/F16 | L1/Q8 | L5/F16 | L5/Q8 |
|---|---|---|---|---|
| `namo` | `0xBDD1E363` | `0x8479754C` | `0x8D502120` | `0xB6AD329D` |
| `namod` | `0x7E870997` | `0x4C6F6736` | `0xA331EA04` | `0x5744DC67` |
| `namo+c3` | `0xE08B6854` | `0xB4BA0916` | `0xCFD4F585` | `0xFFC26241` |

Any change after rebase means a hook was left on, the gate was applied twice,
or a local `fsqrt` survived — not a measurement update.

---

## 4. Per-PR deletion list

### #6 NAMO (`e40b8c0`)
Everything in §3.1, plus: switch `tests/test_namo.c` to the shared gate
(include `fit_opt.h`, not its deleted header); resolve the duplicated
`tests/fitgolden.h` (all tables agree). No `fit_opt.c` CMake change needed.

### #5 FARO (`09311ce`)
Delete local `fsqrt` (`:27`), `MUON_MAX*` (`:435`), `hyb_mode` (`:437-445`),
`optcfg` (`:452-463`), and all three local wrappers (`:491/:511/:589`,
including their two inline NS loops at `:548/:657`). Keep `faro_ctx` and the
ERC math (`:718+`) verbatim; replace positional `risk` with `h->risk` and use
the already-present `bid` argument. `faro_ns_score` is already gone — do not
resurrect it. Feature CRCs (`erc_muon` L5/Q8 `0x54A5EB53`) must not move.

### #8 C3 (`af5fd00`)
Delete local `fsqrt` (`:28`), `MUON_MAX*` (`:436`), `hyb_mode` (`:438-446`),
`optcfg` (`:453-464`), `adam_rows_gated` (`:564`), thin `adam_rows` (`:604`),
`muon_rows` (`:613`) and `malt_rows` (`:686`, two NS copies) — foundation's
`adam_rows` + `step_shape` replaces them. **Delete `src/fit_gate.h`** (the
single header rule); update `tests/test_c3_fit.c` to include `fit_opt.h`.
Delete `fit_gate_step`'s definition in `fit.c:544-550` — it now lives in
`fit_opt.c`. Keep `eps_cfg` only for layers/scope/debug, and its feature CRCs
(e.g. `SHRINK` L5/Q8 `0x84D8C2B4`). Preserve gate-before-`lr` ordering.

### #19 EMA (`16d91c4`)
Delete local `fsqrt` (`:108`), `MUON_MAX*` (`:529`), `hyb_mode` (`:531-539`),
`optcfg` (`:546-557`), and all three local wrappers (`:582/:597/:670`, two
more NS loops). Keep `wa/last1/last2` in `tnet` and `ema_pow`; convert the two
inline fold blocks to `post_set`, the end-of-run copy to `finalize`, and drop
`avg_mode` from `tnet_init`/`tnet_train`. GPU nets get `h = NULL`.

---

## 5. Verification performed

* GCC 16 and Clang 23, `-Werror`, `-ffp-contract=off -fno-fast-math`, Release:
  both build clean.
* `ctest`: **14/14 on both compilers** (the new `hooks_abi()` runs inside
  `test_fit_opt`).
* `cli_fit_golden`: all 2 corpora × 7 optimizer profiles byte-identical to
  unmodified `1aa3dd7`; GCC and Clang agree byte-for-byte.
* `test_fit_opt` still compares `ns_ortho` and all wrappers against verbatim
  main loops, raw float bytes, both transpose cases, `H=32`, `ns=1/5`.
* New `hooks_abi()` checks #8's direct gate values (0.1 / 0.19 / 0.01), a
  real `step_shape` pair versus a verbatim `adam_rows_gated` oracle, and the
  exact FARO risk sum.

## 6. Remaining work (deliberately not faked)

1. **Dispatch is still the old `if/else` ladder** in `tnet_train`
   (`fit.c:621-657`). `opt_plan`, `pre_step`, `pre_set`, `post_step` and
   `finalize` are defined but have no call sites. `plan_w1()`/`apply_w1()`
   must be added before FARO/NAMO/EMA can use them.
2. **No PR-specific hook implementation is enabled** — `fit.c` passes
   `&hooks` with `{0}`. That is intentional: enabling gate/EMA/FARO is each
   PR's rebase work.
3. **`namo_rows` is not implemented here.** It is PR #6 code; §3.2 is the
   exact port description. No NAMO behaviour was smuggled into the foundation.
4. **`step_shape.eps` is an unused reserved slot** — staged #8 dropped C2.
5. **GPU/GLSL untouched**; no hook is passed there.
