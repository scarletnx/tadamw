# RECONCILE.md — Foundation vs. the staged PR fixes

**Same persona, same verdict style:** this is a merge-order design review, not
a rematch. I read every file side by side; nothing below is inferred from a
PR title.

**Refs read (all local worktrees, read-only):**

| What | Ref | Worktree |
|---|---|---|
| Foundation (mine) | `foundation/ns-refactor` @ `e682872` | `struct-a` |
| Design memo | `struct/a` @ `10d3fd4` | `struct-a` |
| #6 NAMO | `e40b8c0` | `exp-namo` |
| #5 FARO | `09311ce` | `exp-faro` |
| #8 C3 gate | `af5fd00` | `exp-eps-gate` |
| #19 EMA | `16d91c4` | `exp-ema` |

---

## 0. Verdict in one paragraph

**Foundation's `src/fit_opt.h` wins.** NAMO's own `src/fit_opt.h` is a strict
subset: its `ns_ortho()` body is *byte-identical* to mine (verified by text
comparison, not eyeballing), and its only extra symbols are the shrink gate,
which belongs to #8's single-gate owner anyway. Every PR still carries its own
`fsqrt`, `MUON_MAXDIM/MUON_MAXEL`, `hyb_mode`, `optcfg` and local
`adam_rows/muon_rows/malt_rows` in `fit.c` — those are hard compile/link
collisions with the foundation and are the bulk of the deletion checklists.
The gate concept should be **one** `fit_gate_step()` (#8) reached through
**one** `step_shape` slot in foundation's `opt_hooks`; NAMO's `shrink_*` and
#8's `eps_cfg` gate fields are configuration aliases into it. FARO's risk bank
and EMA's averaging attach to the same `opt_hooks`, not to optimizer branches.
NAMO's published numbers do **not** need a new re-baseline: `e40b8c0` already
re-baselined them after its shared-NS/shared-gate change (§3).

---

## 1. Exact collisions (file / symbol, with line numbers)

### 1.1 Physical file collisions

| File | Foundation | NAMO `e40b8c0` | Others | Resolution |
|---|---|---|---|---|
| `src/fit_opt.h` | 173 lines: guard `TAW_FIT_OPT_H`, `fsqrt`, NS helpers, `hyb_mode`, `optcfg`, `opt_plan`, `opt_hooks`, 4 wrapper decls | **new file**, 74 lines: **same guard**, `NS_MAXDIM/EL`, `ns_ortho`, `shrink_cfg`, `shrink_begin`, `shrink_at` | absent in #5/#8/#19 | **Delete NAMO's copy.** Foundation's is a superset; same guard means a naive file merge silently drops one side. |
| `src/fit_opt.c` | new, defines `adam_rows`, `muon_rows`, `malt_rows`, `ns_score` | absent (header-only) | absent | Keep foundation's; add `namo_rows` here on #6 rebase. |
| `tests/fitgolden.h` | absent | new, 73 lines with `base_golden[]` | **also new in #5, #8, #19** — all four copies carry the *same* `base_golden` values (`0x197291E6u`, `0x47F69FFBu`, `0xC9DED846u`, `0xB0F10114u`) | One copy wins at integration; the tables agree, so no numeric risk. Foundation's `cli_fit_golden.cmake` is a separate, complementary check. |

### 1.2 Symbol collisions inside `fit.c`

Every PR was branched from `1aa3dd7`, so all of them still own the original
definitions. Line numbers are from the staged refs:

| Symbol | Foundation | NAMO | FARO | #8 | #19 | Collision kind |
|---|---|---|---|---|---|---|
| `fsqrt()` | `static inline` in `fit_opt.h:29` | `static` `fit.c:28` | `fit.c:27` | `fit.c:28` | `fit.c:108` | **Redefinition** in every PR once `#include "fit_opt.h"` is present. Delete all four locals. |
| `NS_MAXDIM/NS_MAXEL` | `enum` in `fit_opt.h:46` (+`MUON_*` aliases at :48) | `#define` in its `fit_opt.h:13-14` | `enum MUON_MAX*` `fit.c:435` | `fit.c:436` | `fit.c:529` | Macro-vs-enum clash if both headers merge; duplicate enum constants if PR locals stay. Use foundation's enums + aliases. |
| `ns_ortho()` | `fit_opt.h:70` | NAMO `fit_opt.h:16` — **body byte-identical (verified)** | deleted with `faro_ns_score` | still inline in `fit.c:636,741` (2 copies) | still inline in `fit.c:600,674` (2 copies) | Duplicate definition only in NAMO's file; elsewhere it is *duplication to delete*, not a name clash. |
| `hyb_mode` | `fit_opt.h:102` (7 values) | `fit.c:456` (**+HYB_NAMO, HYB_NAMOD**) | `fit.c:437` | `fit.c:438` | `fit.c:531` | Duplicate typedef. Foundation's must gain the two NAMO values; PR locals deleted. |
| `optcfg` | `fit_opt.h:117` | `fit.c:467` (+`n1t,n2t,nb2,nscale,nclamp,shrink,drow`) | `fit.c:452` | `fit.c:453` | `fit.c:546` | Duplicate typedef; foundation's must gain NAMO fields + one gate-config field. |
| `adam_rows()` | `fit_opt.c:16`, `(…, mix, const opt_hooks *h, unsigned bid)` | `fit.c:508` (old 10-arg) | `fit.c:491` (**11-arg `float *risk, unsigned bid`**) | `fit.c:604` + `adam_rows_gated` at :564 | `fit.c:582` | **Duplicate external symbol** at link time; FARO's `(risk,bid)` maps 1:1 onto foundation's `(h,bid)`. |
| `muon_rows()` | `fit_opt.c:33` | `fit.c:523` | `fit.c:511` | `fit.c:613` | `fit.c:597` | Same. |
| `malt_rows()` | `fit_opt.c:79` | `fit.c:653` | `fit.c:589` | `fit.c:686` | `fit.c:670` | Same. |
| `namo_rows()` | **absent** (reserved `W1_NAMO/NAMOD` in `opt_plan`, `fit_opt.h:136`) | `fit.c:564` (static, already calls `ns_ortho`) | absent | absent | absent | Not a collision — the one wrapper that must be *added* to foundation. |
| Gate math | none | `shrink_cfg/shrink_begin/shrink_at` in its `fit_opt.h:49-72` | none | `fit_gate_step()` `fit.c:544`, declared in `src/fit_gate.h` (`TAW_FIT_GATE_H`) | none | **Two gate owners** with identical math (§4). |
| `ns_score()` | `fit_opt.c:180` | absent | **deleted** at `09311ce` (routing lane removed; comment `fit.c:724-726` says shared refactor is upstream's) | absent | absent | Dead code in foundation after #5's staged fix — delete or keep only as an explicit test helper (§6.4). |
| Hook plumbing | `opt_hooks { void *ctx; }` `fit_opt.h:149` | none | `risk` threaded as raw params | `eps_cfg` + `sig` threaded through `adam_rows_gated` | `avg_mode` through `tnet_init/tnet_train` signatures | All four must attach to the one `opt_hooks` (§5). |

### 1.3 Confirmed non-collision

The NS polynomial bodies in foundation `fit_opt.h:70-98` and NAMO
`fit_opt.h:16-44` are **byte-for-byte identical** (coefficients, `float s`
accumulators, `i/j/k` nesting, `mr*mc` final loop). NAMO already replaced the
inline loops in its `muon_rows` (branch diff `@@ -523`) and `malt_rows`
(`@@ -628`) with `ns_ortho()`; there is no numeric difference to arbitrate.

---

## 2. Which `fit_opt.h` wins, and why

1. **Superset.** Foundation has the ABI (`hyb_mode`, `optcfg`, `opt_plan`,
   `opt_hooks`), the `fsqrt` single source, the transpose helpers, four
   wrapper declarations and a compiled `fit_opt.c`. NAMO's file has one
   algorithm and a gate subset. Choosing NAMO's file would delete the hook
   ABI that #5/#8/#19 need.
2. **Numerics are preserved.** The shared NS body is identical (§1.3), so
   adopting foundation's header cannot move an orthogonalization result by a
   single bit.
3. **Build integration.** Foundation registers `fit_opt.c` in
   `src/CMakeLists.txt` and has `tests/test_fit_opt.c` pinning both transpose
   cases and `H=32`. NAMO's header-only variant has no equivalent test.
4. **The gate is not a header concern once owned.** NAMO's `shrink_*` helpers
   are configuration + two-pass state; they belong behind `step_shape` (§4),
   not in the single-algorithm header.

**Action:** delete `src/fit_opt.h` from NAMO's diff entirely; keep its
`#include "fit_opt.h"` in `fit.c` (branch line 19), which then resolves to the
foundation file.

---

## 3. Merging NAMO's `namo_rows` without moving its published numbers

### 3.1 The NS part is already merged

NAMO's staged `namo_rows` (`fit.c:564`) already calls `ns_ortho()`. Its
normalization is `x[i] *= 1.0f / norm` after `fsqrt(ss)` and the `1e-12f`
guard — identical to foundation's `muon_rows`/`malt_rows`. Moving the body
verbatim into foundation `fit_opt.c` therefore changes **nothing**.

### 3.2 The gate order is already the unified order

The old concern in `STRUCTURE.md` §5 R4 was that NAMO multiplied
`(mix*lr*upd)*g` while #8 did `mix*lr*(upd*g)`. That is **obsolete for the
staged refs**:

* NAMO `fit.c:617-622`: `float upd = x[at] + wd*w[i]; if (sig) upd *=
  shrink_at(...); w[i] -= step * (diag ? d[...] : alpha) * upd;` — gate
  applied to `upd` *before* the outer scale.
* #8 `fit.c:590-597`: `float upd = ratio + wd*w[k]; upd *= fac;
  w[k] -= mix*lr*upd;` — same order, and the comment at :594-596 says moving
  the gate after `lr` shifts bpc by ~0.02%, so it is deliberate.

Both gate implementations also compute the same state:
`s = beta*s + (1-beta)*sign(g*scale)`, `g = |s|^p` with `p==2 → a*a`. NAMO's
`shrink_begin` and #8's `fit_gate_step` (`fit.c:544-550`) are line-for-line
the same rule.

### 3.3 The numbers are already re-baselined

`tests/test_namo.c:26` says: *"Re-baselined after the X3 shared-NS /
shared-gate refactor."* Its published CRCs are:

| arm | L1/F16 | L1/Q8 | L5/F16 | L5/Q8 |
|---|---|---|---|---|
| `namo` | `0xBDD1E363` | `0x8479754C` | `0x8D502120` | `0xB6AD329D` |
| `namod` | `0x7E870997` | `0x4C6F6736` | `0xA331EA04` | `0x5744DC67` |
| `namo+c3` | `0xE08B6854` | `0xB4BA0916` | `0xCFD4F585` | `0xFFC26241` |

**Therefore: no second re-baseline.** Keep these tables verbatim in the
rebase; if the rebased build changes any of them, the merge is wrong (a hook
was left on, a gate applied twice, or `fsqrt` got duplicated), not "a new
measurement".

### 3.4 Concrete port of `namo_rows`

Move the body from `exp-namo/src/fit.c:564-634` to `fit_opt.c` unchanged
except for the interface:

```c
void namo_rows(const optcfg *o, float *w, float *m, const float *g,
    float *vs, float *vc, size_t set, size_t rows, size_t cols, float scale,
    float lr, float wd, bool diag, const opt_hooks *h, unsigned bid);
```

* drop the `float *sig` parameter — the gate comes from `h->shape`;
* replace `shrink_begin(sig, g, rows*cols, scale, &o->shrink)` with
  `h->shape.begin(h->shape.ctx, g, rows*cols, scale)` (no-op when the hook is
  absent);
* replace `upd *= shrink_at(sig, i, &o->shrink)` with
  `upd *= h->shape.gate(h->shape.ctx, i, g[i]*scale)`;
* keep `step = lr * o->nscale` and the `w -= step * (diag ? d[...] : alpha) *
  upd` expression exactly as staged;
* add the declaration to `fit_opt.h` next to the other wrappers.

---

## 4. One gate owner: #8's `fit_gate_step` behind foundation's `step_shape`

There are currently two implementations of the same rule:

* NAMO `fit_opt.h:49-72` — `shrink_cfg` + `shrink_begin` + `shrink_at`;
* #8 `fit_gate.h` + `fit.c:544-550` — `float fit_gate_step(float *sig, float
  gk, float beta, float p)`, already exposed to `tests/test_c3_fit.c`
  (`gate_direct()` checks 0.1 / 0.19 / 0.01).

**#8 wins the math** (it is the exported symbol with direct unit tests). The
foundation owns the *plumbing*:

```c
typedef struct {
	void  *ctx;
	void  (*begin)(void *ctx, const float *g, size_t len, float scale);
	float (*gate) (void *ctx, size_t k, float gk);
	float (*eps)  (void *ctx, size_t k, float vh, float dflt); /* NULL: C2 dropped */
} step_shape;
```

* `begin` advances the sign EMA **once per weight** by calling
  `fit_gate_step(sig+k, g[k]*scale, beta, p)`, stores the multiplier in a
  per-set bank, and — for `mean_pres` — divides by the row mean with the
  `gmean < 1e-12f → 1` guard (#8 `fit.c:586-588`). This two-pass shape is
  required by #8's mean-preserving mode and is also exactly NAMO's
  `shrink_begin`/`shrink_at` shape.
* `gate` returns the stored multiplier. `gk` is kept in the signature so a
  shrink-only implementation can stay stateless if desired.
* Wrappers (`adam_rows`, `muon_rows`, `malt_rows`, `namo_rows`) call
  `h->shape.begin/gate` only. They never contain a sign EMA.
* `optcfg` gains one config block used by both PRs:

```c
struct { bool on; float beta, p; bool mean_pres; } gate;
```

  with `#8` setting `beta=TAW_EPS_SIGN_BETA` (0.9), `p=TAW_EPS_GATE_P` (1.0),
  `mean_pres = (TAW_GATE_MODE != "shrink")`; and NAMO setting
  `beta=TAW_NAMO_GATE_BETA` (0.9), `p=TAW_NAMO_GATE_P` (1.0),
  `mean_pres=false`, `on = is_namo && TAW_NAMO_GATE=="c3"` (NAMO's P6-1 leak
  guard at `fit.c:795-802` must be preserved).
* #8 keeps `eps_cfg` only for the *dispatch* concerns (`layers`, `scope`,
  `debug`) and drops its C2/C3/nc9 mode enum and gate fields; `muon_rows` and
  `malt_rows` in #8 remain untouched by the gate (its own comment at
  `fit.c:494-495`).
* Delete NAMO's `shrink_cfg`, `shrink_begin`, `shrink_at` with its header.
  `t->sig` (NAMO `fit_internal.h` diff, `[n1]`) becomes the shared gate's
  sign bank; allocate `[n1+n2]` as #8 does and mirror both layers at untie.
  NAMO only reads W1 entries, so the extra W2 mirror cannot change its
  numbers — verify with the §3.3 CRCs.

---

## 5. How #8 / #6 / #5 / #19 attach to `opt_hooks`

Foundation currently has only `void *ctx`. Extend it — once — to:

```c
typedef struct {
	void      *ctx;       /* keep: feature-private state */
	step_shape shape;     /* §4: the single C3 / NAMO_GATE gate */
	float     *risk;      /* FARO bank [nb]; NULL = no accounting */
	unsigned   nb;        /* bank length (debug/bounds) */
	void (*pre_step )(void *, tnet *, optcfg *, uint64_t step, bool tied);
	void (*pre_set  )(void *, tnet *, unsigned layer, unsigned set,
	                  opt_plan *);
	void (*post_set )(void *, tnet *, unsigned layer, unsigned set,
	                  size_t off, size_t len, uint64_t step);
	void (*post_step)(void *, tnet *, uint64_t step);
	void (*finalize) (void *, tnet *);
} opt_hooks;
```

**#8 (gate):** fill only `shape` + `ctx` (the `eps_cfg` layers/scope logic
decides *which* set gets a non-NULL shape, in the dispatch, not in the
wrapper). Ungated sets see `shape.begin == NULL` and get the verbatim main
expression.

**#6 (NAMO):** fill `shape` from `optcfg.gate` for W1, plus the NAMO branch in
`plan_w1`. No new hook slot.

**#5 (FARO):** set `h.risk = fr.usum` when `fr.frozen && fr.alloc`; use
`pre_step` for `faro_allocate`, `pre_set` for `fb`/`mu[fb]` LR scaling and
forced geometry, `post_step` for `faro_measure`. The staged FARO passes
`risk, bid` positionally (`fit.c:1129-1169`); those become `h->risk` and the
already-present `bid` argument — a mechanical rename, **no** new signature
change. Keep the exact expression `float upd = mix*lr*(mh/(fsqrt(vh)+eps));
risk[bid] += upd*upd;` (wd excluded, `fit.c:502-505`) — a generic
`w_after - w_before` hook would change rounding and is not acceptable.

**#19 (EMA):** one `post_set` (fold the touched `off/len` using
`ema_pow(decay, step - last[set])`) and one `finalize` (`wa → w`, SWA mean).
Drop `avg_mode` from `tnet_init`/`tnet_train`; wire it in `fit_one()`.
GPU nets get `h = NULL` (staged EMA already keeps last iterate for GPU).

**GPU boundary:** `fit_gpu.c/.comp` stay untouched; no hook is ever passed
there.

---

## 6. What each staged PR must delete to rebase onto `foundation/ns-refactor`

### 6.1 NAMO (`e40b8c0`) — the full checklist

**Delete:**
1. `src/fit_opt.h` (whole file) — foundation's wins (§2).
2. `fit.c:28-40` local `static float fsqrt`.
3. `fit.c:450` `enum { MUON_MAXDIM = 32, MUON_MAXEL = ... }`.
4. `fit.c:456-462` `typedef enum { ... HYB_NAMO, HYB_NAMOD } hyb_mode` —
   instead add those two values to `fit_opt.h:102`.
5. `fit.c:467-478` `typedef struct optcfg` — instead add `n1t, n2t, nb2,
   nscale, nclamp, drow` and the shared `gate` config (§4) to
   `fit_opt.h:117`.
6. `fit.c:508` `adam_rows`, `:523` `muon_rows`, `:653` `malt_rows` (all three
   are now in `fit_opt.c`; leaving them is a duplicate-symbol link error).
   Their NS loops are already gone at this ref — do not reintroduce them.
7. `shrink_cfg` / `shrink_begin` / `shrink_at` (deleted with the header) and
   the `o.shrink` initializer at `fit.c:795-802` → replaced by `o.gate` plus
   `h->shape` wiring.

**Add / keep:**
8. Move `namo_rows` (`fit.c:564-634`) to `fit_opt.c` with the §3.4 signature;
   add its declaration to `fit_opt.h`.
9. Keep `t->v2s/v2c/sig` state, the untie mirror (`fit.c:427-436`), all
   `TAW_NAMO_*` env parsing, the `is_namo` warm-up guard, `n1t/n2t` updates
   and the dispatch branch (`fit.c:951-958`).
10. Keep `tests/test_namo.c` and its CRCs unchanged (§3.3); resolve the
    duplicated `tests/fitgolden.h` with the other PRs.
11. `src/CMakeLists.txt`: no new change — foundation already compiles
    `fit_opt.c`.

### 6.2 FARO (`09311ce`)

Delete its local `fsqrt` (`:27`), `MUON_MAX*` enum (`:435`), `hyb_mode`
(`:437-445`), `optcfg` (`:452-463`), and all three local wrappers
(`:491/:511/:589`, including their two inline NS loops at `:548/:657`).
Keep `faro_ctx` and the ERC math (`:718+`) verbatim; replace the positional
`risk` argument with `h->risk` (§5). `faro_ns_score` is already gone — do not
resurrect it. The feature CRCs in `tests/test_faro_fit.c` (e.g.
`erc_muon` L5/Q8 `0x54A5EB53`) must stay unchanged.

### 6.3 #8 C3 (`af5fd00`)

Delete its local `fsqrt` (`:28`), `MUON_MAX*` (`:436`), `hyb_mode`
(`:438-446`), `optcfg` (`:453-464`), `adam_rows_gated` (`:564`) and the thin
`adam_rows` (`:604`) — foundation's `adam_rows` plus `step_shape` replaces
both. Delete `muon_rows` (`:613`) and `malt_rows` (`:686`) entirely (two NS
copies) along with their `MUON_MAXDIM` scratch. **Keep** `src/fit_gate.h` and
`fit_gate_step` (`:544-550`) as the single gate math, keep `eps_cfg` as
layers/scope/debug config, and keep its feature CRCs (e.g. `SHRINK` L5/Q8
`0x84D8C2B4`). Preserve the gate-before-`lr` rounding order at `:594-597`.

### 6.4 #19 EMA (`16d91c4`)

Delete its local `fsqrt` (`:108`), `MUON_MAX*` (`:529`), `hyb_mode`
(`:531-539`), `optcfg` (`:546-557`) and all three local wrappers
(`:582/:597/:670`, two more NS loops). Keep `wa/last1/last2` in `tnet` and
`ema_pow`, convert the two inline fold blocks to `post_set`, and drop
`avg_mode` from both signatures.

### 6.5 Foundation clean-up found during reconciliation

`ns_score()` in `fit_opt.c:180` was written for FARO's `TAW_FARO_STAT=lin`
score, but staged #5 **removed the routing lane** (`fit.c:720-726`). It now has
no caller. Either delete it from `fit_opt.h`/`fit_opt.c` or keep it with a
comment that it is a test-only helper for `tests/test_fit_opt.c`; do not leave
an unexplained dead export. Nothing else in the foundation is invalidated.

---

## 7. Verification at each step

1. **Bit-identity of the shared primitive** — `tests/test_fit_opt.c` already
   compares `ns_ortho` and all wrappers against verbatim main loops, raw
   float bytes, both transpose cases, `H=32`, `ns=1/5`. Re-run after every
   rebase (`ctest -R fit_opt`).
2. **Common baseline** — all four PRs' `fitgolden.h` agree on `base_golden`
   (`0x197291E6u / 0x47F69FFBu / 0xC9DED846u / 0xB0F10114u`). After each
   rebase, knob-off must reproduce this table *and* foundation's
   `cli_fit_golden.cmake` archive hashes.
3. **Feature CRCs unchanged** — NAMO §3.3, #8 `SHRINK`/`MEAN`/`GATE_P2`,
   FARO `erc_muon`/`malter_erc`, EMA mode tables. Any change is a merge bug,
   not a measurement update.
4. **Gate equivalence** — after unifying, run NAMO's `namo+c3` golden with
   both `shrink_at` (old) and `step_shape.gate` (new) in one build and
   compare raw bytes; the two must be identical, not merely within 1 ulp.
5. **Build matrix** — GCC + Clang, `-Werror`, `-ffp-contract=off
   -fno-fast-math` unchanged; GPU tests still skip/pass; no GLSL edits.

---

## Appendix — the side-by-side NS check I actually ran

I extracted the `ns_ortho()` function bodies from
`struct-a@e682872:src/fit_opt.h` (lines 70-98) and
`exp-namo@e40b8c0:src/fit_opt.h` (lines 16-44) and compared them as strings:

```
NS bodies byte-identical: True
foundation constants: enum { NS_MAXDIM = 32, NS_MAXEL = NS_MAXDIM * NS_MAXDIM };
namo constants:       #define NS_MAXDIM 32 / #define NS_MAXEL (NS_MAXDIM * NS_MAXDIM)
```

That is the whole arbitration: same algorithm, different packaging, and one
ABI that can carry all four features.
