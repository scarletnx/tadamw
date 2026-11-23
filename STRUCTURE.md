# STRUCTURE.md — Design review: the fitting-side optimizer layer

**Baseline reviewed:** `origin/main` @ `1aa3dd7` (merged #3 MALTER, #4 Muon).
**PRs reviewed (read-only):** #5 `exp/faro`, #6 `exp/namo`, #7 `exp/lr-baseline`,
#8 `exp/eps-gate`, #19 `exp/ema`.
**Reviewer:** The Refactor Architect. **Verdict:** approve the experiments,
reject the shape. Ship a small dispatcher first, then land the features as
plugins behind it.

> "Here is the shape this codebase should have, and here is the migration path."

---

## 0. What is on the table (and why I wince)

`origin/main` already has a clean, if slightly crowded, CPU fitter:

* `src/fit_internal.h` owns the data: `tnet` is one flat allocation of
  `[n1+n2]` weights, moments and gradients (`w, m, v, g`), a second AdamW
  moment pair for W1 (`m1, v1`), the MALT preconditioners (`malt_l, malt_r,
  malt_nu`), and the lazy per-set bookkeeping (`touched1/2, list1/2, cnt1,
  route1, tied`). Everything is allocated unconditionally so the shipped
  fitter is untouched. `tnet_untie()` mirrors set 0 into every set.
* `src/fit.c` owns the policy: `fit_env_float/uint/str` read knobs,
  `hyb_mode` + `optcfg` describe the chosen optimizer, `tnet_train()` parses
  the env, computes the per-step LR schedule, walks the lazy touched sets,
  and dispatches to `adam_rows`, `muon_rows` or `malt_rows`.
* `src/fit_gpu.c` / `fit_gpu.comp` are a **separate world**: the GPU path has
  its own GLSL `adam()` and only reads `t->w/m/v`, `row1/row2`, `n1/n2`, `s1/s2`
  and `s.hidden` (`layout_of`, `pad_rows`). It never calls a CPU optimizer.
  This is a hard boundary: nothing below moves the GLSL.

Now the five PRs. Four of them contain the *same 25-line Newton–Schulz loop*:

| Where | Function | Site | Coefficients |
|---|---|---|---|
| main | `muon_rows` | `src/fit.c:526–550` | `-4.7750f, 2.0315f, 3.4445f` |
| main | `malt_rows` | `src/fit.c:631–655` | identical |
| #6 NAMO | `namo_rows` | branch `src/fit.c:668–691` | identical |
| #5 FARO | `faro_ns_score` | branch `src/fit.c:862–885` | identical |

They are textually identical down to the `float s` accumulators, the
`i/j/k` loop nesting, and the transpose map
`at = tr ? (i % cols) * mc + i / cols : i`. (FARO's last loop uses `n`
instead of `mr*mc`; `n == mr*mc`, so even that is the same arithmetic.)
Four copies of an approximation whose coefficients are load-bearing is four
ways to break bit-exactness in a future edit. I winced.

Two concepts are also duplicated:

* **"Gate"** — #8 has `eps_cfg` + `adam_rows_adv()` with a two-pass sign-EMA
  (`src/fit.c:523–640`), local `sig` state, `TAW_SIGN_BETA/TAW_GATE_P`.
  #6 has `static inline c3_gate()` (`src/fit.c:507–515`), `t->sig1`,
  `TAW_NAMO_GATE_BETA/TAW_NAMO_GATE_P`. Two implementations of one idea, with
  *different multiplication order* (§5, risk R4).
* **"After the update, do something to the touched set"** — #19 EMA folds
  `t->wa` into each touched row in **two** places (W1 and W2 loops), #5 FARO
  measures risk and scales LRs in **every** dispatch branch. Both are
  cross-cutting concerns currently expressed as per-branch code.

And #7 is a six-line `TAW_LR1=flat` convenience in the LR endpoint block.
It is innocent; keep it innocent.

---

## 1. The target shape

### 1.1 File layout

Two new internal modules, everything else stays where upstream put it:

```
src/fit_opt.h     NEW  optcfg, hyb_mode, opt_plan, opt_hooks, wrapper decls,
                       static-inline NS primitive + transpose helpers
src/fit_opt.c     NEW  adam_rows / muon_rows / malt_rows / namo_rows /
                       ns_score — exactly one NS loop, in ns_ortho()
src/fit.c             tnet lifecycle, env parsing, LR schedule, dispatch,
                       FARO and EMA as hook implementations (unchanged style)
src/fit_internal.h    tnet gains ONE new field: float *sig (gate state);
                      NAMO's v2s/v2c and EMA's wa/last1/last2 stay where the
                      PRs put them; struct is named (see below)
src/fit_gpu.c/.comp   UNTOUCHED; hooks are NULL on the GPU path
```

`fit_opt.h` needs `tnet` only as an opaque pointer for hook callbacks, so
change `typedef struct { ... } tnet;` in `fit_internal.h` to
`typedef struct tnet { ... } tnet;` and `typedef struct tnet tnet;` in
`fit_opt.h`. No layout change, no other edits.

`CMakeLists.txt:45` already sets `-ffp-contract=off -fno-fast-math` globally;
`fit_opt.c` inherits it. Do not add per-target flags.

### 1.2 The one polynomial

One function, one copy, in `src/fit_opt.h` as `static inline` (header-only so
the compiler sees the same TU as the callers — zero chance of ISA/flag drift):

```c
#define NS_MAXDIM 32
#define NS_MAXEL  (NS_MAXDIM * NS_MAXDIM)

/* x: mr x mc, already transposed to short-first AND already normalized to
 * unit Frobenius norm. a,b,y: [NS_MAXEL] caller scratch. This is the entire
 * orthogonalization; the three coefficient literals and the i/j/k order are
 * ABI, not style. */
static inline void ns_ortho(float *x, float *a, float *b, float *y,
                            size_t mr, size_t mc, unsigned steps);

/* shared geometry: at = tr ? (i%cols)*mc + i/cols : i, tr = rows>cols */
static inline void ns_geom(size_t rows, size_t cols, bool *tr,
                           size_t *mr, size_t *mc);
static inline size_t ns_at(size_t i, size_t cols, bool tr, size_t mc);
```

The body is Appendix A — verbatim from `muon_rows`, with `o->ns_steps`
renamed `steps`. Each wrapper owns everything *around* it: how `x` is built,
the `fsqrt(ss)` + reciprocal normalization, and what is done with the result.

### 1.3 Per-optimizer wrappers (`src/fit_opt.c`)

All four keep the current FP expressions character-for-character. The only
ABI change is a uniform trailing pair `(const opt_hooks *h, unsigned bid)`
so that cross-cutting concerns have somewhere to land; a wrapper never
mentions FARO, EMA or NAMO by name.

```c
void adam_rows(optcfg *o, float *w, float *m, float *v, const float *g,
    size_t len, float scale, float lr, float wd, float mix,
    const opt_hooks *h, unsigned bid);

void muon_rows(const optcfg *o, float *w, float *m, const float *g,
    size_t rows, size_t cols, float scale, float lr, float wd, float mix,
    const opt_hooks *h, unsigned bid);

void malt_rows(const optcfg *o, float *w, float *m, float *l, float *r,
    float *nu, const float *g, size_t rows, size_t cols, float scale,
    float lr, float wd, float mix, bool adaptive,
    const opt_hooks *h, unsigned bid);

void namo_rows(const optcfg *o, float *w, float *m, const float *g,
    float *vs, float *vc, size_t set, size_t rows, size_t cols, float scale,
    float lr, float wd, bool diag,     /* diag == NAMO-D */
    const opt_hooks *h, unsigned bid);

/* FARO's linearized spectral score; calls ns_ortho, does not duplicate it. */
float ns_score(const optcfg *o, const float *g, unsigned rows, unsigned cols);
```

Each orthogonalized wrapper becomes:

1. build `x` (momentum / Nesterov / preconditioner / NAMO second moments),
2. `ss = Σ x²`, `norm = fsqrt(ss)`, early-out `!(norm > 1e-12f)` as today,
3. `recip = 1.0f/norm; x[i] *= recip;` (exact current order),
4. **`ns_ortho(x, a, b, y, mr, mc, o->ns_steps);`**
5. wrapper-specific post-processing (graft, L/R map-back, adaptive alpha,
   NAMO scale, `factor`); apply gate if `h` has one;
6. `w -= ...` **verbatim**, then, if `h->risk`, add `Δ²` to `h->risk[bid]` using
   the wrapper's own applied-update expression (see §3.2).

### 1.4 Dispatch: `hyb_mode` → `optcfg` → `opt_plan` → one `apply_w1`

`tnet_train()` still owns env parsing and the schedule — that part of main is
good design and must not move. What changes is the per-set branch jungle at
`fit.c:858–901`. Replace it with a *decision* and an *execution* step:

```c
typedef enum { W1_ADAM, W1_MUON, W1_MALT, W1_MALTER, W1_NAMO, W1_NAMOD } w1_kind;

typedef struct {
    w1_kind  kind;         /* geometry for this set/step */
    bool     adam_half;    /* HYB_BLEND: also run AdamW on m1/v1 */
    bool     adam_primary; /* HYB_ADAM uses m/v, NOT m1/v1 (load-bearing!) */
    float    scale;        /* 1/nb */
    float    lr_geo, wd_geo, mix_geo;    /* Muon/MALT/NAMO half */
    float    lr_adam, wd_adam, mix_adam; /* Adam half / pure Adam path */
    unsigned bid;          /* FARO risk block, 0 when bank is off */
} opt_plan;

static opt_plan plan_w1(const optcfg *o, const tnet *t, unsigned set,
    uint64_t step, bool tied, bool warm_adam, bool alt_muon,
    bool route_muon, const opt_hooks *h);
static void apply_w1(const optcfg *o, tnet *t, unsigned set,
    const opt_plan *p, const opt_hooks *h);
static void apply_w2(const optcfg *o, tnet *t, unsigned set,
    float scale, const opt_hooks *h);
```

`plan_w1()` is a pure translation of main's current `if/else` ladder; it must
preserve these exact cases (I am listing them so nobody "tidies" them):

| mode / condition | plan |
|---|---|
| `HYB_ADAM` | `W1_ADAM`, `adam_primary=true`, `lr_adam=o.lrvec`, `wd_adam=o.vecwd`, `mix=1`. Ignores `warm_adam` (as main does). |
| `HYB_MALT`/`HYB_MALTER`, not `(tied && warm_adam)` | `W1_MALT`/`W1_MALTER`, `lr_geo=o.lr`, `wd_geo=o.wd` |
| `HYB_BLEND` | `adam_half=true` on `m1/v1` with `lr1/wd1/(1-alpha)` **plus** `W1_MUON` on `m` with `o.lr/o.wd/alpha` |
| Muon (MUON, ALT-on, ROUTE-hot), not tied-warm | `W1_MUON`, `mix=1` |
| otherwise (ALT-off, ROUTE-cold, tied-warm) | `W1_ADAM` on `m1/v1`, `lr1/wd1` |
| W2, every mode | `adam_rows` on `m/v`, `o.lrvec/o.vecwd` |

`apply_w1()` is one `switch (p->kind)` calling the four wrappers; it no longer
contains any optimizer arithmetic. `h->pre_set()` runs *before* `plan_w1`
(so FARO can force Muon/Adam and multiply `lr_geo/lr_adam`), `h->post_set()`
runs once at the end of the set body (so EMA can fold). `h->pre_step()` runs
where main updates `b1t/b2t/malt_b*t`; `h->post_step()` where `g` is cleared.

This is how `optcfg` selects the polynomial path: `hyb_mode` → `plan_w1` →
`switch (kind)` → wrapper → `ns_ortho`. One selector, one polynomial.

### 1.5 `tnet` / `fit_internal.h`

Keep the existing "allocate unconditionally, mirror at untie, free in
`tnet_free`" convention. Add exactly one new common field, `float *sig`
(gate sign EMA, `[n1+n2]`), mirrored for **both** layers in `tnet_untie()` —
this preserves #8's behaviour and is harmless for NAMO (which only reads W1
entries). NAMO's `v2s/v2c` and EMA's `wa/last1/last2` stay as their PRs have
them. Do **not** change the order of the `rng_unif()` calls in `tnet_init()`
or the order of fields before `t->route1`: `fit_gpu.c` indexes by offsets
computed from the existing fields and appending is safe, inserting is not.

---

## 2. One owner for "gate"

**Decision: PR #8's gate becomes the single implementation.** It is the richer
and better-specified one (C2, C3, NC9, mean-preserving vs shrink, layer/scope
selection, debug stats); #6's `c3_gate` is a strict subset. Extract it into
`src/fit_opt.h` as a `step-shaping` sub-object of `opt_hooks`:

```c
typedef struct {
    void  *ctx;
    /* one pass over g: update sign EMA, fill mul[] and (mean-preserving) norm */
    void  (*begin)(void *ctx, const float *g, size_t len, float scale);
    /* per-weight multiplier; implements |s|^p, shrink or mean-preserving */
    float (*gate)(void *ctx, size_t k, float gk);
    /* coordinatewise epsilon (C2); returns o->eps when unused */
    float (*eps) (void *ctx, size_t k, float vh, float dflt);
} step_shape;
```

* `gate_begin()` / `gate_at()` are the *only* place that computes
  `s = beta*s + (1-beta)*sign(gk)`, `|s|^p`, the `gmean < 1e-12f → 1` guard,
  and the mean-preserving division. This two-pass shape is required: shrinking
  alone is one pass, mean-preserving needs the row mean first (#8's
  `adam_rows_adv` already does exactly this).
* #8 keeps `eps_cfg` (C2/NC9, layers, scope, debug) as its *configuration*, but
  **deletes `adam_rows_adv()`**: `adam_rows()` gains the `step_shape` hook and
  applies `eps_at`/`gate_at` inline in its existing output loop. With
  `TAW_EPS_MODE` unset all hook pointers are NULL and the verbatim main
  expression runs — byte-identical.
* #6's `TAW_NAMO_GATE=c3` becomes a *configuration alias* into the same gate:
  `mode = SHRINK`, `beta = TAW_NAMO_GATE_BETA` (0.9), `p = TAW_NAMO_GATE_P`
  (1.0), `mean_pres = false`. **Delete `c3_gate()`, delete `t->sig1`** (use
  the shared `t->sig`), delete `o.ngbeta/o.ngp/o.gate_c3`. The `v2s/v2c` fields
  remain NAMO's alone.
* No optimizer branch ever gets a gate concept of its own; the shared
  wrappers call `h->shape.gate()` uniformly.

One numeric caveat: #8 applies the multiplier as `mix*lr*(upd*gate)` while #6
applies `(mix*lr*upd)*gate`. Unification picks **#8's order** (it owns the
concept). NAMO_GATE trajectories may move by ~1 ulp; re-baseline those experimental
numbers, but note that *main's profiles* do not move because the hook is off.

---

## 3. EMA and FARO: hooks, not wrappers

### 3.1 The hook object

```c
typedef struct {
    step_shape shape;                       /* §2, NULL when off */
    float *risk; unsigned nb;               /* §3.2, NULL when off */
    void  *ctx;                             /* FARO / EMA private state */
    void (*pre_step )(void *, tnet *, optcfg *, uint64_t step, bool tied);
    void (*pre_set  )(void *, tnet *, unsigned layer, unsigned set, opt_plan *);
    void (*post_set )(void *, tnet *, unsigned layer, unsigned set,
                      size_t off, size_t len, uint64_t step);
    void (*post_step)(void *, tnet *, uint64_t step);
    void (*finalize) (void *, tnet *);       /* EMA/SWA: copy wa -> w */
} opt_hooks;
```

`tnet_train()` takes one extra parameter, `const opt_hooks *h` (EMA's
`avg_mode` stops being a second parameter to `tnet_init()` / `tnet_train()`;
`fit_job.avg_mode` resolves to hook wiring in `fit_one()`). GPU nets get
`h = NULL`, which is exactly PR #19's `TAW_AVG_OFF` behaviour — no GLSL change.

### 3.2 EMA — one `post_set`, one `finalize`

PR #19's two inline fold blocks (`branch fit.c:1025–1055`) become one function:

```c
static void avg_post_set(void *ctx, tnet *t, unsigned layer, unsigned set,
                         size_t off, size_t len, uint64_t step);
static void avg_finalize(void *ctx, tnet *t);   /* SWA mean or EMA copy */
```

`avg_post_set` needs only `(layer, set, off, len, step)` — exactly what the
hook supplies — and touches only the rows that were just updated, preserving
the lazy-state design and `ema_pow()`'s `decay^(steps since last fold)`
arithmetic. Epoch-boundary SWA snapshots and the final `wa → w` copy are
`pre_step`/`finalize`. No optimizer branch learns about averaging.

### 3.3 FARO — one risk sink + two lifecycle hooks

FARO has three needs. All three are served without editing any branch:

1. **Risk accounting.** A generic `(w_before, w_after)` post-hook is *not*
   acceptable: it includes decoupled weight decay and changes rounding
   (`w_after - w_before` ≠ `upd`). Instead the wrappers share one uniform
   sink field. Each wrapper adds its own applied-update square at the exact
   point PR #5 already does:

   ```c
   /* adam_rows, verbatim shipped update first */
   w[k] -= mix * lr * (mh / (fsqrt(vh) + o->eps) + wd * w[k]);
   if (h && h->risk) {
       float upd = mix * lr * (mh / (fsqrt(vh) + o->eps)); /* wd excluded */
       h->risk[bid] += upd * upd;
   }
   ```
   (`muon_rows`: `upd = factor*x[at]`; `malt_rows`: `upd = fac*x[at]`;
   `namo_rows`: `upd = step*scale*x[at]`.) This is one generic line per
   wrapper, no FARO names. FARO just sets `h.risk = fr.usum` when
   `fr.frozen && fr.alloc`, and gets `bid` from `plan_w1`/`apply_w1`.

2. **Pre-set / pre-step policy.** `faro_pre_set` computes `fb`, multiplies
   `plan->lr_geo/lr_adam` by `fr.mu[fb]`, sets `plan->bid = fb`, and — only
   in `HYB_MUON` — forces `W1_MUON`/`W1_ADAM` during exploration / frozen
   routing. `faro_pre_step` freezes and calls `faro_allocate()`. `faro_post_step`
   calls `faro_measure()`. All currently live scattered through
   `branch fit.c:1258–1416`; they collapse into one hook implementation.

3. **The fourth NS copy.** `faro_ns_score()` (branch `fit.c:841–900`) becomes a
   thin wrapper that fills `x`, normalizes, calls `ns_ortho`, then keeps its
   `double dot/xn` accumulation. `faro_cf_probe()` keeps calling
   `muon_rows()` / `adam_rows()` on shadow copies, but passes a zeroed
   `opt_hooks` so counterfactual probes never pollute the risk bank.

`faro_init/free/freeze/allocate/measure/report`, `faro_ctx` and the ERC math
are untouched — that is the actual experiment and it stays intact.

---

## 4. Migration order

**Stage 0 — land the foundation refactor on `1aa3dd7` before any feature PR.**
At that point there are only two NS copies and no hooks, so the extraction is
a small, reviewable diff: add `fit_opt.[ch]`, move `optcfg/hyb_mode`, add
`ns_ortho`, convert `muon_rows/malt_rows`, add `opt_plan` + `plan_w1/apply_w1`,
add the empty `opt_hooks`, add the golden-vector test (§5). Then rebase every
feature onto it.

Feature PR order: **#7 → #8 → #6 → #5 → #19.**

| Step | PR | Why here | What it must delete / change after rebase |
|---|---|---|---|
| 1 | **#7 LR overrides** | 6 lines, touches only the LR endpoint block (`fit.c:730–740`), no shared surface. Merge first so it cannot conflict with the refactor's `tnet_train` rewrite. | Nothing. Keep `TAW_LR1=flat` verbatim; do **not** fold it into a new schedule helper yet. |
| 2 | **#8 C3 gate** | Establishes the single gate owner before NAMO needs it. | Delete `adam_rows_adv()`; implement `step_shape` instead. Keep `eps_cfg`, C2/NC9, layer/scope/debug, `positive_median`, `in_scope`, `TAW_FORCE_SEL*`. Replace local `sig` with `t->sig` + mirroring at untie. |
| 3 | **#6 NAMO** | Now consumes the gate and the shared NS; no private copies. | Delete `c3_gate()`, `t->sig1`, `o.ngbeta/o.ngp/o.gate_c3`, `adam_rows`'s `sig` arg (use hook), the `sig` arg on `muon_rows`, and the entire NS loop in `namo_rows` (call `ns_ortho`). Keep `HYB_NAMO/NAMOD`, `v2s/v2c`, NAMO LR/WD/env, `n1t/n2t` updates. Alias `TAW_NAMO_GATE=c3` to the #8 gate in shrink mode. |
| 4 | **#5 FARO** | Most invasive; last of the optimizer features so it rebases over a stable dispatch + NS + gate. | Delete `risk, bid` params (use `h->risk` + `plan.bid`), delete `faro_ns_score`'s NS loop (`ns_score`), delete every branch-local `lr_w1/lr_a1/lrvec_b *= fr.mu[...]` and `risk` call, delete `fr.risk` assignment style in favour of `h.risk`. Keep all `faro_*` algorithm functions and `faro_ctx`. Pass zeroed hooks in `faro_cf_probe`. |
| 5 | **#19 EMA** | Pure additive consumer of `post_set`/`finalize`; no optimizer arithmetic of its own, so it rebases cleanly last. | Delete the two inline EMA fold blocks and the epoch/SWA code from `tnet_train`; implement `avg_post_set`/`avg_finalize`. Delete `avg_mode` from `tnet_init`/`tnet_train` signatures; wire from `fit_job.avg_mode` in `fit_one()`. Keep `t->wa/last1/last2`, `ema_pow`, mode parser, `resolve_avg_mode`, GPU nets at last iterate. |

If the foundation cannot land as its own PR, fold it into the #7 commit and
keep #7's 6-line change as the first hunk — but do not let any feature PR be
the first to touch `fit.c`, because then its private NS loop becomes the
"canonical" one by accident.

---

## 5. Bit-exactness: risks and verification

### Risks

* **R1 — FP contraction / TU drift.** Moving the loop into another
  translation unit could change rounding if flags differ. Mitigation:
  `static inline` in a header included only by `fit.c`; global
  `-ffp-contract=off -fno-fast-math` stays; never replace `fsqrt` with libm
  or `-mfma` paths. `src/fit_gpu.comp` is a different ISA and is out of scope.
* **R2 — the `w -=` expressions.** Do not "simplify"  
  `w[k] -= mix*lr*(mh/(fsqrt(vh)+eps) + wd*w[k])` into two products. The risk
  and gate lines are *additive*, adjacent, and only active with a hook.
* **R3 — transpose / normalization order.** `tr = rows > cols`, `NS_AT`,
  `fsqrt(ss)`, `!(norm > 1e-12f)`, `recip = 1.0f/norm` then `x[i] *= recip`
  must match main literally. Test both `H <= 26` (no transpose) and `H=32`
  (`tr = true`, since `P1_F = 26`).
* **R4 — gate multiplication order.** #8 uses `mix*lr*(upd*g)`, #6 uses
  `(mix*lr*upd)*g`; unify on #8. Main's profiles are unaffected (hook off),
  but NAMO_GATE experimental numbers must be re-baselined.
* **R5 — risk excludes wd.** A `(w_before, w_after)` diff hook changes FARO's
  ERC solve. Use the wrapper-emitted `Δ²` slot (§3.2).
* **R6 — lazy state / untie.** New arrays must be allocated without consuming
  RNG, mirrored in `tnet_untie()`, cleared with `touched1/2`, and freed in
  `tnet_free()`. Gate `sig` is mirrored for both layers; EMA's `last1/last2`
  use the *global* step. Forgetting a mirror changes weights only when that
  knob is on, but forgetting to free is a leak in every profile.
* **R7 — GPU boundary.** `fit_all()` must keep the exact candidate split
  (`taw_gpu_fits`) and pass `NULL` hooks to GPU nets; no new field may be
  inserted before `t->route1` in `tnet`.

### Verification gates (run at every step, not just at the end)

1. **Record goldens from unmodified `1aa3dd7` first.** New
   `tests/test_fit_opt.c` builds a deterministic `tnet` (fixed seed, fixed
   sample slice from `tests/testdata.h`) and FNV/CRC32c-hashes the **raw float
   bytes** of `w, m, v, m1, v1, malt_l, malt_r, malt_nu` after a fixed number
   of steps. Cover `adam` (default), `muon`, `blend`, `alt`, `route`, `malt`,
   `malter`, `muon` with `ns=1/5`, `nesterov=0/1`, and one non-default graft.
   Archive-level CRC alone is not enough — quantization can hide a 1-ulp drift.
2. **`ns_ortho` equivalence test.** Fill `x` with a deterministic pattern for
   shapes `(H,P1_F) = (8,26), (26,26), (32,26), (16,26)` and `ns_steps=1,5`;
   compare bit patterns against the old inlined loop recorded in step 1.
3. **Knob-off byte identity.** For each feature PR, with all its knobs
   unset/default, `taw c` on ≥3 fixed inputs × ≥2 levels and `cmp` the
   archives against the baseline binary: #8 (`TAW_EPS_MODE` unset), #6
   (`TAW_HYBRID` unset, `TAW_NAMO_GATE` unset), #5 (`TAW_FARO=0`), #19
   (`TAW_FIT_AVG=off`), #7 (`TAW_LR1` unset). This is the same claim each PR
   body already makes; make it executable.
4. **Feature-on equivalence.** Per-PR golden trajectories: #8 gate values and
   per-set `w`; #6 NAMO/NAMO-D `w` + `v2s/v2c`; #5 per-step `usum`, final
   `mu[]`, `geo[]`; #19 each `avg_mode`'s final `w`. Bitwise compare against
   the pre-refactor branch binary on the same inputs.
5. **Build/ISA matrix.** GCC and Clang, `-Werror`, sanitizer build
   (`-DTADAMW_SANITIZE=ON`), x86 SSE `fsqrt` and the generic `#else` `fsqrt`
   path, and `ctest` (including `gpu`, skipped without Vulkan). NS goldens are
   recorded per ISA because the fallback `fsqrt` is an approximation.
6. **Dispatch invariant test.** Assert `plan_w1` reproduces the table in §1.4
   for a matrix of `(mode, tied, warm_adam, alt_muon, route1[set])`, including
   that `HYB_ADAM` still uses `m/v` and every other Adam half uses `m1/v1`.

---

## Appendix A — the only Newton–Schulz loop that should exist

```c
static inline void ns_ortho(float *x, float *a, float *b, float *y,
                            size_t mr, size_t mc, unsigned steps)
{
	for (unsigned it = 0; it < steps; it++) {
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mc; k++)
					s += x[i * mc + k] * x[j * mc + k];
				a[i * mr + j] = s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mr; k++)
					s += a[i * mr + k] * a[k * mr + j];
				b[i * mr + j] = -4.7750f * a[i * mr + j] + 2.0315f * s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t k = 0; k < mc; k++) {
				float s = 0.0f;
				for (size_t j = 0; j < mr; j++)
					s += b[i * mr + j] * x[j * mc + k];
				y[i * mc + k] = s;
			}
		for (size_t i = 0; i < mr * mc; i++)
			x[i] = 3.4445f * x[i] + y[i];
	}
}
```

## Appendix B — one-line summary of the new call path

```
tnet_train (fit.c)
  parse env -> hyb_mode + optcfg               [unchanged]
  per step:  schedule LRs, b1t/b2t, malt_b*t   [unchanged]
             h->pre_step(...)                  [FARO freeze/allocate]
  per set:   h->pre_set(..., &plan)            [FARO LR/route]
             plan = plan_w1(...)               [main's ladder, pure]
             apply_w1 -> {adam,muon,malt,namo}_rows -> ns_ortho
                        -> h->shape.gate/eps -> h->risk[bid]
             apply_w2 -> adam_rows
             h->post_set(...)                  [EMA fold]
  per step:  h->post_step(...), clear g        [FARO measure]
  end:       h->finalize(...)                  [EMA/SWA wa -> w]
  GPU nets:  h = NULL                          [GLSL adam, untouched]
```
