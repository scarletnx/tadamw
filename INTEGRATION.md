# INTEGRATION.md — final knob-doc integration

Branch `integration/knob-final`, based on `foundation/knob-docs` @ `4f5cb9b`
(which contains `KNOBS.md` @ `b080612`, the docs patch @ `694f862`, and
`RECONCILE.md` @ `4f5cb9b`). Local only; nothing pushed.

## 1. Doc changes applied to `docs/design.md`

| item | change |
|------|--------|
| `TAW_EPS_GATE_MODE` | kept as the canonical documented name (already in the "Epsilon / sign gate" table); the old `TAW_GATE_MODE` appears only in the historical appendix |
| `TAW_EPS_GRAN` | **removed** from the live table — the knob was deleted with C2 in staged #8 |
| GPU guards | FARO: `TAW_FARO=on` + `-g on\|auto` silently forces CPU (`TAW_GPU_OFF`). C3: `TAW_EPS_MODE=c3` + `-g on\|auto` is rejected with `TAW_E_ARG`. Both documented, with the asymmetry called out |
| NAMO | live section now documents `TAW_NAMO_NEST` (default `0`) as the NAMO-only switch; the drift text about a NAMO gate alias and a conditional `TAW_MUON_NEST` was removed |
| `TAW_FIT_AVG` | numeric aliases `0..7` and `ema09` documented as retained |
| `TAW_CALIBRATED` | new "Calibrated mixer (profile 6, pending allocation)" subsection: default `0`/off, `1`/`on`/`cal` enable, CPU-only, test-only `TAW_CAL_PARAMS`/`TAW_CAL_NO_APM`, byte-identical default |
| historical appendix | new "Historical names (provenance only)" subsection listing `TAW_SIGN_BETA`, `TAW_GATE_P`, `TAW_GATE_MODE`, `TAW_FARO_ROUTE`, `TAW_FIT_LRSCALE` with current equivalents/status |

## 2. Verification (no live phantom references)

`grep -E 'TAW_NAMO_GATE|TAW_GATE_MODE|TAW_EPS_GRAN|TAW_SIGN_BETA|TAW_GATE_P|TAW_FARO_ROUTE|TAW_FIT_LRSCALE' docs/design.md`
returns matches only at lines 377-383 — i.e. **only inside the historical
appendix**. `README.md` has none. Live knob tables use only canonical names:
`TAW_EPS_GATE_MODE`, `TAW_NAMO_NEST`, `TAW_CALIBRATED`, plus the existing
FARO/EMA entries.

Two clarifications on the verification wording:

- `TAW_NAMO_NEST` is documented live as the **target** name. It does not yet
  exist in staged #6 (`e40b8c0`), which still uses a conditional default on
  `TAW_MUON_NEST`. This is an intentional "docs lead, code follows" state;
  the required code fix is listed in §3. The drift names (`TAW_NAMO_GATE*` and
  conditional `TAW_MUON_NEST`) are absent from the live doc.
- `KNOBS.md` and `RECONCILE.md` intentionally retain old tokens — they are a
  registry and an audit record, not live user docs. Their references are
  provenance and must not be "cleaned up" into silence.
- `TAW_FIT_LRSCALE` is not present in the staged #21 tree; it was removed from
  #19's pre-fix `bench/control.sh` and is cited in that PR's results. It is
  listed in the appendix because the integration brief names it, with its
  provenance stated explicitly in the appendix text.

No `--help`, CLI token tables, predictor enums or source files were changed:
profile ids 6/11/14 remain pending allocation in `docs/format.md` §2 (only 0
and 1 are implemented).

## 3. Code-side renames still required

| PR | staged SHA | required code/doc work | why the doc is currently ahead of code |
|----|-----------|------------------------|----------------------------------------|
| **#6 NAMO** | `e40b8c0` | In `src/fit.c`: add `TAW_NAMO_NEST` (default `0`) and use it at line 817 in place of `fit_env_uint("TAW_MUON_NEST", is_namo ? 0 : 1)`; delete `TAW_NAMO_GATE`, `TAW_NAMO_GATE_BETA`, `TAW_NAMO_GATE_P` (lines 811-814) and the NAMO `shrink` field; update `RESULTS.md` knob table (lines 342-344), tests and bench strings | Live doc promises a NAMO-only NEST switch and no private gate; staged code has neither |
| **#8 EPS gate** | `af5fd00` | Rename `TAW_GATE_MODE` → `TAW_EPS_GATE_MODE` in `src/fit.c:525`, the README table added by the fix, `RESULTS.md` examples (lines 209, 389) and any tests; optionally make `eps_requested()` accept only `c3` (today `c2`/`nc9` warn but still enable); rename or document test-only `TAW_GOLDEN_PRINT` | Live doc uses the canonical renamed name; staged code/PR README still use the old one |
| **#21 final matrix** | `0b9b8d8` | Annotate historical env strings in `bench_one.sh:19,45`, `analyze.py:124,129` and `RESULTS.md:38,43` with a "pre-rename historical binary" footnote (do not silently rewrite them — that would break reproduction); fix the stale `bench_one.sh:13-14` comment claiming #19 defaults to `TAW_AVG_LATE`; keep `FITPIN='TAW_FIT_AVG=off'` | The archived matrix legitimately records pre-rename tokens, but must not read as current API guidance |

Existing `RECONCILE.md` §§1.3, 2.2, 3.2, 4.2 contain the verbatim replacement
text for the #6, #8 and #21 documentation edits; this file records the
integration state, not a duplicate patch.

## 4. Follow-ups (non-blocking)

- `TAW_GOLDEN_PRINT` (#8, `tests/fitgolden.h:47`) is a test-only diagnostic
  that violates the `_DEBUG` diagnostic convention; rename to
  `TAW_TEST_GOLDEN_PRINT` or add it to the diagnostic doc line.
- Decide the `namo-d` alias in #6 (currently accepted at `fit.c:773`): keep and
  document, or drop to one canonical token `namod`.
- `TAW` (binary path) in #19's harness scripts is a very generic env name;
  prefer `TAW_BENCH_BIN` if the harness is kept long-term.
