# FINAL.md — last knob-surface pass

> Clipboard out, pen uncapped. "Clean" is a word I only write when every name
> on the clipboard is a name in the code — and vice versa. **This is not clean.**

Scope: read-only audit of the ten final staged branches at the exact SHAs
named in the request, compared against canonical `integration/knob-final`
@ `b1dadf9` (`docs/design.md` env-knob section, `README.md`, `KNOBS.md`,
`RECONCILE.md`, `INTEGRATION.md`). No sibling worktree was modified; no push.

## Verdict per PR

| PR | SHA | drift names in live knob surface | undocumented live knobs | default byte-identical | GPU guard | verdict |
|----|-----|----------------------------------|-------------------------|------------------------|-----------|---------|
| #5 FARO | `6db34ea` | none in knobs; **1 stale code comment** | none | ✓ | FARO→CPU ✓ documented | **DRIFT (comment)** |
| #6 NAMO | `d8506fc` | none | none | ✓ | shared C3 guard via #8 ✓ | MATCH |
| #7 flat LR | `fb81c4d` | none | none | ✓ | n/a | MATCH |
| #8 EPS gate | `a7457ec` | none in knobs; **1 stale code comment** | `TAW_DATA` (harness only) | ✓ | C3→`TAW_E_ARG` ✓ documented | **DRIFT (comment)** |
| #10 calibrated | `8906f3b` | none | profile-6 predictor CLI missing from canonical registry | ✓ | CPU-only, documented | **DRIFT (canonical doc)** |
| #11 gate | `0a91327` | none | none | ✓ | n/a | MATCH |
| #13 hnet | `92638a7` | none | none | ✓ | n/a | MATCH |
| #15 dynamic blocks | `af4e989` | none | **`-D`/`-T`/`-C` + 4 API fields absent from canonical registry** | ✓ | n/a | **DRIFT (canonical registry)** |
| #19 EMA | `d3368de` | none | harness `TAW`/`TAW_BENCH_DATA` only | ✓ (`TAW_AVG_OFF`) | GPU nets not averaged ✓ | MATCH |
| #21 matrix | `fb0d51f` | old tokens only in appendix ✓ | `BENCH_*` documented in its own RESULTS | ✓ pin intact | n/a | MATCH |

What is genuinely clean: every product knob name/default in the final code
matches the live tables — `TAW_FARO*` (no route knob), `TAW_NAMO_NEST` (no
`TAW_NAMO_GATE`), `TAW_MUON_NEST` back to `1`, `TAW_EPS_GATE_MODE`, no
`TAW_EPS_GRAN`, invalid `TAW_EPS_MODE` now returns `TAW_E_ARG`,
`TAW_FIT_AVG` default `off` with `auto` gone, `TAW_CALIBRATED` default `0`.
#21's pre-rename tokens are confined to its appendix with warnings and are
covered by the canonical "Historical names" appendix. Both GPU guards are
documented with their correct asymmetry.

## Exact drift — required fixes

**D1 — stale drift-name comment (blocking the "no live drift names" claim).**
`src/fit_opt.c:15` in **both** #5 `6db34ea` and #8 `a7457ec` reads:

```c
/* The single sign-consistency gate, verbatim from staged #8 (af5fd00). Both
 * #8's TAW_EPS_MODE=c3 and #6's TAW_NAMO_GATE=c3 must call this, not copy it. */
```

`TAW_NAMO_GATE` no longer exists in final #6 (`d8506fc` uses the shared
`TAW_EPS_MODE=c3`). Fix:

```diff
- * #8's TAW_EPS_MODE=c3 and #6's TAW_NAMO_GATE=c3 must call this, not copy it. */
+ * #8's TAW_EPS_MODE=c3 and #6's shared NAMO gate must call this, not copy it. */
```

**D2 — canonical calibrated-mix section is stale vs final #10.**
`docs/design.md` (canonical) says "there is no predictor enum or `-p` token
yet" and files profile 6 under "pending allocation". Final `8906f3b` adds
`TAW_PREDICTOR_CALIBRATED=6` and `-p cal` (`tools/taw.c:35,153`,
`taw.h`), and its format registry allocates value 6 under ruling R-4. Fix
the canonical text to: "Profile 6 is allocated under R-4 as
`TAW_PREDICTOR_CALIBRATED` (value 6, `-p cal`); `auto`/`mix` still fit profile
1 only." Update the "Predictor choices pending format allocation" heading
to cover only the remaining cases.

**D3 — canonical predictor-allocation note is stale vs final #11/#13.**
Final #11/#13 also implement R-4: predictor values 11 (`TAW_PREDICTOR_GATE`)
and 14 (`TAW_PREDICTOR_HNET`) are allocated, with `-p gate`/`-p hnet` in their
own `--help` and token tables. Canonical `docs/design.md` still says "ids are
therefore **pending allocation** ... neither flag is accepted yet". Fix to
"allocated under R-4; the `-p` tokens ship with the respective PRs" while
keeping the "default `auto`/`mix` byte-identical" guarantee.

**D4 — #15's CLI/API knobs missing from the canonical registry.**
`af4e989` adds `-D` (dynamic blocks), `-T N` (target bytes), `-C N` (change
threshold), a `-B` help line, public `taw_params` fields `dyn_blocks`,
`chunk_target`, `change_threshold`, and `taw_info.has_dyn_blocks`. They are
documented in #15's own `--help`/`docs/format.md` §7.2 but are **absent from
canonical `KNOBS.md` §1.A, `docs/design.md` and `README.md`**. Add to
`KNOBS.md` §1.A:

```
| `-D` | encoder flag | #15 | off | help ✓ (PR branch) | none | default fixed blocks, byte-identical |
| `-T N` | encoder flag | #15 | `-B` value | help ✓ (PR branch) | none | dynamic-block target |
| `-C N` | encoder flag | #15 | `0` (ratio) | help ✓ (PR branch) | none | change spike threshold 0..256 |
```

and a short canonical design-doc line: "`.awseek` dynamic blocks are opt-in
via `-D` (`-T`/`-C` tune them); the default archive is byte-identical to
fixed blocks and `taw i` reports `dynamic:`."

**D5 — #21 appendix wording stale after R-4.**
`fb0d51f` `analyze.py:470-474` calls `TAW_PREDICTOR_GATE=11` and
`TAW_PREDICTOR_HNET=14` "pending predictor slots". They are allocated in the
final PRs. Fix the generated line to "predictor ids allocated under R-4
(recorded here as historical binary tags)", keeping the surrounding
"canonical names are `TAW_EPS_SIGN_BETA` / `TAW_EPS_GATE_P` /
`TAW_EPS_GATE_MODE`; FARO routing has no equivalent" text, which is correct.

**Minor (not blockers) — harness-only env vars.** `TAW_DATA` (#8
`sweep_eps.sh:8`), `TAW` and `TAW_BENCH_DATA` (#19 bench scripts) and `BENCH_*`
(#21) are configurable path variables in committed harnesses, not product
knobs; add one "harness env" note to `KNOBS.md` §1.E if the strict registry is
to cover them. This does not affect archive bytes or the public API.

## Ruling

Five exact drift items (D1-D5), one of them (D1) a live source comment that
still names a dead knob. The historical appendix usage for #21, the
byte-identical defaults and both GPU guards all pass. Fix D1-D5, then this
pass can be rerun as a clean one.
