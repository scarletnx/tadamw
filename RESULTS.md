# MALT / MALTER on the current leader (`exp/malt`, base `d41bd57`)

Encoder-only implementation of **MALT: Lightweight Curvature-Aware Muon via
Diagonal Preconditioning** (arXiv:2608.05088, Wu, Dong, Sun, Ma) and its
noise-adaptive variant **MALTER**. Format, decoder, quantization and golden
vectors are untouched. No push. Corpora in `/tmp/opencode/tadamw-bench` were
read-only; all archives live in `bench/arcs/` in this worktree.

## TL;DR / verdict

**Verified win over the tuned-Muon leader.** Best configuration:

```
TAW_MALT=malter \
TAW_MALT_LR0=0.28 TAW_MALT_LR1=0.0056 \
TAW_MALT_B2=0.9 TAW_MALT_EPS=1e-8 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015
```

| corpus / level | shipped AdamW | tuned AdamW | tuned Muon (base) | best plain MALT | **best MALTER** |
|---|---:|---:|---:|---:|---:|
| text1m L1 | 181557 | 179913 | 179678 | 179635 | **179405** |
| text1m L5 | 179166 | 178886 | 177816 | 177804 | **177138** |
| libc1m L1 | 379197 | 373887 | 371373 | 371743 | **370898** |
| libc1m L5 | 375291 | 371894 | 366997 | 366329 | **366504** |
| **sum L1+L5** | 1,115,211 | 1,104,580 | 1,095,864 | 1,095,511 | **1,093,945** |

MALTER beats tuned Muon by **−1,919 B (−0.18 %)** overall and is
**Pareto-better on all four cells** (−273 / −678 / −475 / −493 B). It beats
tuned AdamW by −10,635 B (−0.96 %) and shipped AdamW by −21,266 B (−1.91 %).
Plain MALT is also slightly better in sum (−353 B) but *not* Pareto: it loses
370 B on libc1m L1. MALTER is the clear winner, matching the paper's claim
that its noise-adaptive step-size adds a complementary gain over MALT.

Winner reproduces byte-identically on a rerun; four decode checks returned
`ok=1`; knob-off paths reproduce all four baseline archives exactly;
`ctest` = **12/12**.

## Paper summary

Muon orthogonalizes the momentum matrix (Newton–Schulz) to flatten its
singular spectrum, but it ignores the *curvature* anisotropy of the loss: with
an ill-conditioned Hessian, a fixed-magnitude update is too aggressive along
sharp directions and too timid along flat ones. MALT addresses this by
preconditioning **before** orthogonalization with two lightweight diagonal
factors built from row/column squared-gradient-norm EMAs (a diagonal surrogate
for the Shampoo/Kronecker curvature factors), then mapping the orthogonalized
direction back and grafting the update magnitude. MALTER adds a scalar
Adam/NAMO-style signal-to-noise step-size in the preconditioned space.
The paper proves stochastic non-convex convergence for MALT and reports
GPT-2 Small/Medium/Large pretraining wins over Muon at nearly identical memory
and wall-clock.

## Exact equations (paper Algorithms 1–2, §3.3, §3.5)

For a matrix parameter `W ∈ R^{m×n}`, gradient `G_t`, momentum decay `μ1`,
preconditioner EMA `μ2`, damping `ε`, learning rate `η`:

```
M_t  = μ1 M_{t-1} + (1-μ1) G_t
l_t  = μ2 l_{t-1} + (1-μ2) [ ||G_t[1,:]||², ..., ||G_t[m,:]||² ]
r_t  = μ2 r_{t-1} + (1-μ2) [ ||G_t[:,1]||², ..., ||G_t[:,n]||² ]
L_t  = Diag((l_t + ε)^(-1/8)),   R_t = Diag((r_t + ε)^(-1/8))
M~_t = L_t M_t R_t
O_t  = Orth(M~_t)                     # NS polar factor, singular values -> 1
D_t  = L_t O_t R_t                    # map back to parameter space
MALT:    W_t = W_{t-1} - η · (||O_t||_F / (||D_t||_F + ε)) · D_t

MALTER adds:
ν_t  = μ2 ν_{t-1} + (1-μ2) ||L_t G_t R_t||_F²
α_t  = [ sqrt(1-μ2^t) / (1-μ1^t + ε) ] · ||M~_t||_F / (sqrt(ν_t) + ε)
W_t  = W_{t-1} - η · α_t · (||O_t||_F / (||D_t||_F + ε)) · D_t
```

`Orth` is the same 5-step Newton–Schulz polynomial used by this codebase's
Muon (`A = X Xᵀ`, `B = −4.7750 A + 2.0315 A²`, `X = 3.4445 X + B X`), with the
preconditioned momentum first Frobenius-normalized. `(l+ε)^(-1/8)` is computed
as `1/sqrt(sqrt(sqrt(l+ε)))` — no libm dependency.

Two deliberate adaptations to the existing fitter (documented in the code):

* **Momentum convention.** The codebase uses `M = μ1 M + G` (no `(1-μ1)`),
  which Muon's normalize-then-orthogonalize path is invariant to. MALTER's
  `α` is *not* invariant to it, so `||M~||` is multiplied by `(1-μ1)` before
  forming `α`, recovering the paper's signal-to-noise ratio. Without this,
  `α` is ~20× too large and training is unstable.
* **Nesterov.** We keep the codebase's Nesterov extrapolation of the
  momentum before preconditioning, so MALT/MALTER differ from the tuned Muon
  only in the two-sided diagonal preconditioning and (for MALTER) the step
  rescale.

W1 is one 16×26 matrix; W2 (8×17 vectors) stays on AdamW exactly as in the
base. Per-set `l`/`r` vectors and the scalar `ν` are copied at untying like
the other W1 state; the extra state is `O(m+n)` per matrix.

## Environment knobs (all unset = shipped behaviour)

| variable | default | meaning |
|---|---|---|
| `TAW_MALT` | `off` | `off`/`0`: shipped AdamW; `1`/`on`/`malt`: MALT on W1; `malter`: adaptive MALTER. Overrides `TAW_HYBRID`. |
| `TAW_MALTER` | `0` | `1` promotes `TAW_MALT=malt` to MALTER. |
| `TAW_MALT_LR0` / `TAW_MALT_LR1` | `0.055` / `0.0011` | linear W1 LR endpoints (Muon-tuned defaults). |
| `TAW_MALT_B2` | `0.999` | preconditioner EMA decay `μ2` (also MALTER's `ν` decay). |
| `TAW_MALT_EPS` | `1e-8` | damping `ε` in `(l+ε)^(-1/8)`, graft and MALTER denominator. |
| `TAW_MALT_GRAFT` | `o` | graft source: `o` = `||O||_F` (paper), `m` = `||M~||_F`, `d` = none (`||D||_F`). |
| `TAW_MUON_MOM` / `TAW_MUON_NS` / `TAW_MUON_NEST` | `0.95` / `5` / `1` | shared with Muon: momentum `μ1`, NS steps, Nesterov. |
| `TAW_VEC_LR0` / `TAW_VEC_LR1` | `TAW_LR0` / `TAW_LR1` | W2 AdamW LR (winner uses `0.003` / `0.00015`). |
| `TAW_MALT_DEBUG` | unset | print final `l`/`r`/`ν` statistics to stderr. |

## Sweep table

Full sensitivity grid (all four cells, `-p mix`, bytes; W2 fixed to
`0.003→0.00015` unless noted). Baseline controls: Muon = 1,095,864 B;
tuned AdamW = 1,104,580 B; shipped = 1,115,211 B.

| W1 mode | LR0 → LR1 | b2 | eps | W2 | sum bytes | vs Muon |
|---|---|---:|---:|---|---:|---:|
| MALTER | 0.28 → 0.0056 | 0.90 | 1e-8 | 0.003→1.5e-4 | **1,093,945** | **−1,919** |
| MALTER | 0.28 → 0.0070 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,003 | −1,861 |
| MALTER | 0.26 → 0.0052 | 0.90 | 1e-8 | 0.005→2.5e-4 | 1,094,044 | −1,820 |
| MALTER | 0.25 → 0.0050 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,284 | −1,580 |
| MALTER | 0.40 → 0.0080 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,526 | −1,338 |
| MALTER | 0.35 → 0.0070 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,536 | −1,328 |
| MALTER | 0.30 → 0.0060 | 0.90 | 1e-4 | 0.003→1.5e-4 | 1,094,605 | −1,259 |
| MALTER | 0.30 → 0.0030 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,087 | −777 |
| MALT | 0.055 → 0.0011 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,095,511 | −353 |
| MALT | 0.065 → 0.00065 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,600 | −264 |
| MALT | 0.055 → 0.0011 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,958 | +94 |
| MALTER | 0.25 → 0.0050 | 0.90 | 1e-8 | 0.055→5.5e-4, b1=.95 | 1,100,016 | +4,152 |

Ablations (earlier proxy sweep with default W2, text1m+libc1m L1 sum;
Muon proxy 551,051): graft `o` = 550,647; graft `m` = 552,544; graft `d`
= 599,926 (diverges). With MALTER's winner: `eps` 1e-9 → 1,095,313,
1e-7 → 1,095,480, 1e-6 → 1,094,358 (all worse than 1e-8);
`b2` 0.85 → 1,094,576, 0.95 → 1,095,790 (both worse than 0.90);
final-LR ratio 1/30 → 1,094,374, 1/60 → 1,094,343, 1/75 → 1,094,214.
W2 tuned-Adam settings (b1=.95, wd=1e-3) are much worse for MALTER than
the leader's small W2 LR.

## Verification

* `bench.sh` winner run: 179405 / 177138 / 370898 / 366504, all four
  `decode_ok=1` (byte-exact vs the shared corpora).
* Knob-off: `TAW_MALT=off TAW_MALTER=0` reproduces all four baseline archives
  byte-for-byte (`cmp`), including with the env vars explicitly set to their
  off values.
* Determinism: rerunning the winner archive is byte-identical.
* `ctest --test-dir build --output-on-failure`: **12/12 passed**.
* Clean GCC 16 `-Werror`-compatible build; no libm added. A separate
  `-DTADAMW_WERROR=ON -DTADAMW_SANITIZE=ON` build compiles cleanly and passes
  11/12; the lone `gpu` failure is LeakSanitizer reporting leaks inside
  `libdbus-1.so` from the Vulkan runtime (no MALT code on that path).

## Caveats

* **Scale transfer is weak.** The paper's GPT-2 matrices are large; our W1
  is a single 16×26 matrix. With `l ~ 1e-6…1e-3`, `(l+ε)^(-1/8)` only varies
  by ~1.3–2.2×, so the diagonal preconditioner is much milder here than in
  the paper's regime. The gain is correspondingly small (~0.18 % over Muon).
* **Benchmark-specific tuning.** The sweep is over this one fitter/corpus
  pair; several near-best MALTER configurations differ by only a few hundred
  bytes and the top configuration may not generalize to other shapes. Treat
  the numbers as an existence proof, not a universal hyperparameter recipe.
* **LR does not transfer from the paper.** MALTER's `α` is below 1 in this
  noisy, lazily-updated regime, so it needs ~5–10× the W1 LR of MALT/Muon
  (0.25–0.4 vs 0.055). Using paper-like LRs with MALTER is much worse.
* **Not Pareto for plain MALT.** Its sum win comes from libc1m L5; it loses
  on libc1m L1. Only MALTER improves every cell.
* **Single W1 set.** `sel1=NONE` on the winning shape, so per-set routing and
  the multi-set aspects of the hybrid code are not exercised by MALT here.
* **CPU/encoder only.** The Vulkan trainer is unchanged; MALT is not
  implemented there. Memory overhead is `O(m+n)` extra floats per matrix,
  plus one scalar for MALTER; compute is one NS pass plus row/column sums,
  i.e. essentially Muon's cost.

---

# FARO on the fitted mixer (branch `exp/faro`, rebased onto `origin/main` 1aa3dd7)

The section above is upstream's MALT/MALTER report from merged PR #3, preserved
verbatim by the rebase. This section is the unique FARO delta (PR #5); it is an
independent opt-in and coexists with upstream's Muon and MALTER code — no
duplicate Muon implementation was introduced.

FARO reference point after the rebase is the tuned-Muon recipe (total
**1,095,864 B**), i.e. the same A/B baseline used in the FARO experiments;
the new main's MALTER leader (1,093,945 B) is a different opt-in arm and is
not used as the FARO control. Encoder-only changes in `src/fit.c`; format,
decoder, quantization, golden vectors and the GPU trainer are untouched.
Corpora in `/tmp/opencode/tadamw-bench` were read only; all archives live in
`bench/arcs/` here. Pushed to `myfork exp/faro` only; upstream untouched.

## TL;DR / verdict

**The allocation lane works; the routing lane does not.** With each block's
geometry forced to the tuned recipe (W1 = Muon, W2 = AdamW) and only the
cross-block equal-risk-contribution (ERC) budget reallocation enabled, four
archive sizes improve together:

```
TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 \
TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8
```

| corpus / level | shipped | tuned AdamW | tuned Muon (FARO-Adaptive) | **Muon + FARO-ERC c=1.8** | Δ vs tuned Muon |
|---|---:|---:|---:|---:|---:|
| text1m L1 | 181557 | 179913 | 179678 | **179502** | −176 |
| text1m L5 | 179166 | 178886 | 177816 | **177738** | −78 |
| libc1m L1 | 379197 | 373887 | 371373 | **371208** | −165 |
| libc1m L5 | 375291 | 371894 | 366997 | **366890** | −107 |
| **sum L1+L5** | 1,115,211 | 1,104,580 | 1,095,864 | **1,095,338** | **−526 (−0.048 %)** |

That is a **4/4 cell win** on the deterministic benchmark, with no change to
model size (608 B) or decode path. The paper's claim that FARO-ERC improves a
near-optimal hand-designed recipe therefore transfers to this fitter, at a much
smaller scale (0.048 % vs −0.0136 nats). The gain is modest but systematic
across corpora and levels; the cap sweep has a flat optimum at 1.5–1.8.

`FARO-Adaptive` (routing forced, ERC off) is **byte-identical** to the tuned
Muon control (all four archives verified with `cmp`, including against sibling
`exp-muon-adam`'s `muon_055_lowfin_*`). With `TAW_FARO` unset the whole build is
bit-identical to branch base.

## Paper summary (arXiv:2610.40858, Wu et al., 2026)

FARO makes the *parameter block* the unit of optimizer decision and separates
two choices:

1. **Routing (return side).** Partition parameters into blocks (B = 117 in the
   paper). For each block b and candidate geometry φ ∈ {coordinate,
   row_column, spectral} — Adam-style, row/column-rescaled matrix, orthogonalized
   matrix — apply the update to a *held-out counterfactual probe* and record the
   realized return ρ_b(φ) = L_probe(θ) − L_probe(θ − η u_b(φ)). Route to the
   largest return and **freeze once the ranking settles**, which bounds the
   search and removes routing variance.
2. **Allocation (risk side).** After the freeze, each block carries a reference
   budget β̄_b; the update is θ_b ← θ_b − α β_b d_b(φ_b). Keep the total fixed,
   Σβ_b = Σβ̄_b =: Bud (a constraint the paper contrasts with scalar
   "Kelly"/volatility controllers that only move along the simplex diagonal).
   Risk is the scale-free update-to-weight ratio uwr_b = ‖Δ_b‖/(‖θ_b‖+ε),
   smoothed as r_b ← γ r_b + (1−γ) uwr_b; allocation at step t reads r(t−1)
   only. Winsorize against the cross-block median, r̃_b = clip(r_b, m/κ, κm),
   then solve the capped, budget-neutral system
   Λ(λ) = Σ_b β̄_b·clip(λ/r̃_b, c⁻¹, c) = Bud by bisection and apply
   μ_b = clip(λ⋆/r̃_b, c⁻¹, c), β_b = clip(β̄_b μ_b, β_min, β_max).

Defaults: γ = 0.9, κ = 3.0, c = 1.4, ERC clamp [0.5, …]. Reported results:
FARO-Adaptive reaches 3.2780 nats, matching the hand-designed NorMuonAndAdam
(3.2794) without a hand-set split; FARO-ERC then improves at 6/6 seeds by
−0.0136 nats (paired p ≈ 0.031, CI excludes zero), cheaper than the measured
0.004–0.01 nat noise floor. The gain comes from *trimming high-risk blocks*
(58.7 % of multipliers < 1, mean 0.98), not from expanding safe ones; a larger
cap deepens it and saturates, lowering the floor deepens it more than raising
the ceiling. Under a 60 % corrupted stream it still wins 5/6 seeds. System
cost is reported as nil (O(B) solve, no parameters/optimizer state).

## Mapping to `tadamw`, and what is exact vs simplified

The fitted mixer has W1 (one H×26 matrix per selector set) and W2 (one vector
row per selector set, 17 params each). Natural blocks:

| block mode | blocks on the winning shape (`H=16, s1=NONE, s2=BITPOS`) |
|---|---|
| `set` (default) | 9: the single W1 matrix + 8 W2 set-rows |
| `layer` | 2: all W1 params vs all W2 params |

Candidate geometries: **W1: spectral (Muon) vs coordinate (AdamW)**; W2 is
fixed to coordinate/AdamW (a vector "spectral" arm is not implemented). The
frozen reference budget β̄_b is that block's LR endpoint (Muon LR for W1,
vector LR for W2), so ERC moves update mass between W1 and W2 at fixed total.

**Implemented exactly:** Algorithm 1 of the paper — median winsorization,
capped multiplier, 48-step bisection on Λ(λ), previous-step-only risk, EMA
γ, clamp [floor, ceil], μ_b multiplying the block's LR. Neutrality was checked
in debug mode (a diagnostic prints if Σβ̄μ drifts by >1e-4 relative; it never
fired). Risk Δ uses only the geometry-induced update, excluding decoupled WD.

**Simplified / approximated (this is not a full FARO reproduction):**

* The paper's routing probe is a disjoint held-out shard and evaluates each
  candidate's update over the training trajectory. Here a deterministic slice
  of 16 strided samples per batch is used, with shadow copies of the block's
  weights and of both optimizer states; one step is evaluated per exploration
  batch. No extra training data is withheld, so the probe is *in-sample*.
* `TAW_FARO_STAT=probe` (default) computes the actual one-step **counterfactual
  loss decrease in bits** for both candidates via the fitter's real cost table
  (`faro_probe_loss`), i.e. Eq. (1) with a 16-sample probe.
  `TAW_FARO_STAT=lin` is a cheaper first-order variant:
  ‖g‖₁/√N (coordinate) vs ⟨g, NS(g)⟩/‖NS(g)‖_F (spectral), both per unit
  update (Appendix G's linearized return). Full multi-step counterfactual
  evaluations over all blocks would multiply fitting cost many-fold and were
  not attempted.
* Routes freeze after un-tie plus `TAW_FARO_EXPLORE` batches (default 20);
  during exploration the applied W1 geometry alternates by step parity so
  both candidates mature their momentum state. With `TAW_FARO_ROUTE=muon|adam`
  exploration is skipped and the route is imposed.
* B = 9 (or 2), not 117. With `s1=NONE` there is only one W1 set, so
  per-selector-family routing degenerates, as already observed for
  `TAW_HYBRID=route` on the base branch.

**Knob-off is byte-identical:** `adam_rows`/`muon_rows` keep their original
weight-update expressions verbatim; extra parameters are a NULL risk pointer
and a block id, and all FARO state is allocated/freed only when `TAW_FARO` is
on. Verified with `cmp` against the base branch's archives.

## Environment knobs (all unset = shipped behaviour)

| variable | default | meaning |
|---|---|---|
| `TAW_FARO` | `0` | `1`/`on`/`yes`/`true` enables FARO; `0` is bit-exact base |
| `TAW_FARO_ROUTE` | `ret` | W1 geometry: `ret` (measured return), `muon`, `adam` |
| `TAW_FARO_STAT` | `probe` | `probe` (one-step counterfactual bits) or `lin` (linearized) |
| `TAW_FARO_EXPLORE` | `20` | post-untie batches used to measure returns (freeze point) |
| `TAW_FARO_ALLOC` | `erc` | `none` disables reallocation (μ ≡ 1) |
| `TAW_FARO_BLOCKS` | `set` | `set` (one block per W1/W2 set) or `layer` (W1 vs W2) |
| `TAW_FARO_CAP` | `1.4` | allocation cap c (μ ∈ [1/c, c]) |
| `TAW_FARO_GAMMA` | `0.9` | risk-EWMA decay γ |
| `TAW_FARO_KAPPA` | `3.0` | winsor bound κ |
| `TAW_FARO_FLOOR` / `TAW_FARO_CEIL` | `0.5` / `2.0` | final clamp on μ |
| `TAW_FARO_DEBUG` | `0` | print freeze decision, returns, multiplier range, risks |

## Sweep (archive bytes, lower is better; all `-p mix`, both corpora)

All rows below use the tuned Muon recipe unless marked. `Δ` is vs the tuned
Muon control sum 1,095,864 (or vs tuned AdamW 1,104,580 for the last two rows).
"cells" counts archive cells improved out of four. Every archive was decoded
and `cmp`-checked (`decode_ok=1` for all 100 new `bench/results.tsv` rows).

| config | text L1 | text L5 | libc L1 | libc L5 | sum | Δ | cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned Muon = FARO-Adaptive | 179678 | 177816 | 371373 | 366997 | 1,095,864 | 0 | — |
| **ERC c=1.8, set blocks (winner)** | **179502** | **177738** | **371208** | **366890** | **1,095,338** | **−526** | **4/4** |
| ERC c=1.5 | 179473 | 177884 | 371111 | 366890 | 1,095,358 | −506 | 3/4 |
| ERC c=1.8, κ=1.5 | 179606 | 177756 | 371145 | 366835 | 1,095,342 | −522 | 4/4 |
| ERC c=1.8, γ=0.5 | 179560 | 177823 | 371244 | 366790 | 1,095,417 | −447 | 3/4 |
| ERC c=1.6 | 179601 | 177785 | 371150 | 366940 | 1,095,476 | −388 | 4/4 |
| ERC c=2.2 | 179448 | 177874 | 371151 | 367126 | 1,095,599 | −265 | 2/4 |
| ERC c=2.6 | 179502 | 177879 | 371151 | 367126 | 1,095,658 | −206 | 2/4 |
| ERC c=4.0 | 179501 | 177879 | 371151 | 367126 | 1,095,657 | −207 | 2/4 |
| ERC c=1.8, layer blocks | 179694 | 177919 | 371301 | 366746 | 1,095,660 | −204 | 2/4 |
| ERC c=2.0 | 179500 | 177844 | 371203 | 367126 | 1,095,673 | −191 | 2/4 |
| ERC c=1.4 (paper default) | 179542 | 177774 | 371285 | 367159 | 1,095,760 | −104 | 3/4 |
| routing `ret` (probe), alloc=none | 179993 | 177983 | 372260 | 368099 | 1,098,335 | +2471 | 0/4 |
| routing `ret` (probe) + ERC c=1.8 | 179924 | 177873 | 372119 | 367809 | 1,097,725 | +1861 | 0/4 |
| routing `ret`, `STAT=lin` + ERC c=1.8 | 179924 | 177926 | 372119 | 367809 | 1,097,778 | +1914 | 0/4 |
| tuned AdamW + ERC c=1.8 (W2 only¹) | 179898 | 178678 | 373875 | 371261 | 1,103,712 | −868² | 4/4 |
| tuned AdamW + ERC c=1.8 (both) | 179965 | 178589 | 374023 | 371460 | 1,104,037 | −543² | 4/4 |

¹ `TAW_HYBRID=adam` routes W1 through the shipped AdamW path; an early version
of the hook scaled only W2 there. Both variants are shown for transparency;
the final code scales both (last row).
² Δ against tuned AdamW (1,104,580).

Other observations:

* **Cap response.** The gain appears at 1.4, peaks around 1.5–1.8 and then
  partly reverses for 2.0–4.0, unlike the paper's monotone-then-flat curve;
  with only 9 blocks and a 608-byte model the discrete quantized weights make
  larger tilts lose their benefit. The paper's default 1.4 is not optimal here.
* **floor/κ are inert at c=1.8.** `TAW_FARO_FLOOR=0.25` and `TAW_FARO_KAPPA=10`
  reproduced the winner byte-for-byte (the downward clamp cannot bind at
  c=1.8, and κ=3 already covers the observed risk spread). Tightening κ to 1.5
  kept 4/4 wins. γ=0.5 (faster risk response) was slightly worse.
* **Freeze point / granularity.** With routes forced, `TAW_FARO_EXPLORE` ∈ {5,
  20, 40} is byte-identical (there is nothing to measure). Per-set blocks beat
  the 2-block `layer` partition: risk is more informative cross-sectionally
  with 9 blocks, consistent with the paper's 117-block premise.
* **Allocation is not a global LR change.** FARO-Adaptive (ERC off) is exactly
  the tuned Muon control; the whole gain is the cross-block reallocation at
  fixed total. ERC also transfers to tuned AdamW (4/4 cells), so the effect is
  not specific to the Muon geometry.

## Routing lane: a negative result

With `route=ret`, both statistics chose **coordinate/AdamW for W1**:
the one-step probe returned 0.477 bits (coordinate) vs 0.336 (spectral) on
the sampled batches; the linearized statistic also preferred coordinate.
Consequently W1 loses orthogonalization and the archives get ~2.5 KB worse
than the tuned recipe, with ERC recovering only ~0.6 KB of that. This is the
opposite of the paper's "recovers the human prior" result. Likely reasons:
(a) the probe is in-sample and one-step, so it rewards the geometry with the
best immediate first-order alignment, whereas Muon's advantage is in
long-horizon conditioning over 800–1270 steps; (b) the probe is 16 samples
from the current batch rather than a disjoint held-out shard; (c) this model
has one W1 matrix, so there is no cross-block routing structure to exploit.
A faithful reproduction would need a held-out probe and multi-step/full-horizon
returns; that was deliberately not attempted because it would multiply fitting
cost many times over. **The routing approximation is therefore not validated
here, and FARO-Adaptive is implemented as a forced-route reference arm.**

## Verification

* `ctest --test-dir build --output-on-failure`: **12/12** with no overrides and
  **12/12** with the winning FARO environment.
* Knob-off control archives `cmp`-identical to `exp-muon-adam`'s
  `muon_055_lowfin_*` and to the shipped controls 181557/179166/379197/375291.
* All 100 new benchmark rows have `decode_ok=1`; the format/decoder were never
  modified, so byte-identical decode is expected and observed.
* Neutrality check (debug mode): no budget drift reported in any run.

## Reproduce the winner

```sh
cd /home/andrew/.local/share/opencode/worktree/8500c0/exp-faro
cmake --build build -j1
NAME=faro_repro \
TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 \
TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8 \
  ./bench.sh
# expect 179502 / 177738 / 371208 / 366890, sum 1,095,338
```

## Rebase verification (new main `1aa3dd7`, PR #3 MALTER + PR #4 Muon)

After rebasing onto upstream main, a pristine build of `1aa3dd7` (from
`git archive`) and the rebased FARO build were compared on 12 identical
encodes with `cmp`:

| env | archives | result |
|---|---|---|
| no overrides (shipped) | 4 | byte-identical to new main (181557/179166/379197/375291) |
| tuned Muon (`TAW_HYBRID=muon`, tuned LRs) | 4 | byte-identical to new main (179678/177816/371373/366997) |
| MALTER winner (`TAW_MALT=malter`, tuned LRs) | 4 | byte-identical to new main (179405/177138/370898/366504) |

So upstream's Muon and MALTER implementations are fully preserved, with no
duplicate Muon code: the only `fit.c` delta vs `origin/main` is the optional
FARO state/routing/ERC logic plus optional risk hooks on the three W1 update
routines (NULL/no-op when `TAW_FARO` is off). ctest: **12/12** with no
overrides and **12/12** with the winning FARO env. The FARO winner reproduced
exactly after the rebase: **179502 / 177738 / 371208 / 366890**
(sum 1,095,338, −526 B vs tuned Muon), all four decodes verified byte-exact.

## Caveats

* Archive bytes are deterministic for a fixed build, so "4/4 cells" is four
  corpus/level cells of one deterministic fitter, **not** a 6-seed statistical
  test like the paper's. The margin (0.05 %) is small in absolute terms.
* Only levels 1 and 5, `-p mix`, two 1 MiB corpora, and the current winning
  shape were benchmarked; transfers to other shapes/L9 are untested.
* The 9-block partition puts the entire Muon matrix in one block, so ERC's
  "risk parity" here mostly means trading W1 LR against eight W2 LR rows.
* Routing uses in-sample, one-step counterfactuals and only W1 has a second
  geometry candidate; see the negative result above.
* Timing was not used as a metric (shared 12-core box); the implementation adds
  O(B) arithmetic per step (B ≤ 9 here) and a few hundred bytes of state, but
  the probe path adds ~16 forward evaluations per W1 block per exploration
  batch and is **not** free when `route=ret` is used.
