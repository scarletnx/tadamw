# COMPAT.md — the Compatibility Keeper's invariant suite and merge staging

Base of record: `origin/main` @ `1aa3dd746c73c2e60939dc38355eb09df6e907d6`
(PR #3 MALTER + PR #4 Muon merged). All PR heads were read read-only from
`myfork` and match the SHAs in `audit/a..d/REVIEW.md`; no branch, corpus, or
upstream ref was touched. Golden hashes below were produced on this base with
GCC 16.2.1 / CMake 4.5.0, `-DCMAKE_BUILD_TYPE=Release -DTADAMW_GPU=OFF`
(the CMake `-ffp-contract=off -fno-fast-math` flags apply).

If a decoder from main reads an archive produced by a merged PR with no new
knob set, then it MUST produce exactly the bytes that main would have produced,
or the merge has broken the contract. That single conditional governs this
document. A format extension is only legitimate when it is unreachable without
an explicit, centrally allocated predictor/flag choice.

---

## 1. Invariants every PR must satisfy before merge

These are gates, not aspirations. "Knob-off" means the scrubbed environment of
§4, with **no** new profile flag and **no** new optimizer knob.

### I1 — Knob-off byte-identity to main on the fixed corpus set

- Every archive of the §4 golden corpus set must match the recorded SHA-256
  **and** byte length exactly. `auto` and `-p mix` must both remain profile-1
  archives with the main bytes.
- No PR may change the default value of an existing knob; no PR may add a knob
  whose default is "on" for any shipped level (1..9). Consequences:
  - **#19** currently sets `TAW_FIT_AVG_DEFAULT = TAW_AVG_LATE`, which is EMA999
    for epochs ≥ 6 (levels 3+). If that default survives, every level-3..9
    archive changes bytes and I1 fails at L5. It must default to `off`, or be
    an explicitly signed-off default-change batch with rebaselined goldens.
  - **#10** currently computes `profiles = 3u` for `TAW_PREDICTOR_AUTO`, i.e.
    profile 6 candidates participate in plain `auto` fitting ("AUTO fitting
    evaluates profile 1 and profile 6 candidates together" in its RESULTS.md).
    A main decoder rejects a profile-6 archive with `TAW_E_UNSUPPORTED`. If
    that survives, the shipped default emits unreadable-by-main archives.
    Auto (and `-p mix`) must enumerate profile 1 only; p6 must require
    `-p cal`.
  - #5/#6/#7/#8/#15 are default-off/opt-in as written (verified in their
    diffs) and satisfy I1 structurally; #11/#13 keep `auto`/`mix` p1-only.
- A knob-off `cmp` in a PR's RESULTS.md is not evidence; the §4 hashes are.

### I2 — Profile-1 golden CRCs unchanged

`tests/test_p1.c:golden()` must keep these 9 `(quant, coded_len, CRC32C)` values
on **every** available ISA (scalar, AVX2, AVX512 if present):

| quant | len | CRC |
|---|---|---|
| F16 (1) | 59885 | `0x74309CD4` |
| Q8 (4) | 48112 | `0x899F4057` |
| Q4 (5) | 47020 | `0xA78FE21D` |
| Q2 (6) | 23304 | `0x2D02777A` |
| FP8 (7) | 43808 | `0xF5A4AC7B` |
| NVFP4 (8) | 44641 | `0xD47885B8` |
| IQ8 (9) | 62559 | `0x2AC391B8` |
| IQ4 (10) | 45922 | `0x84FA96FD` |
| IQ2 (11) | 48256 | `0xAF5179FA` |

Also required: `archive_parity()` (all ISAs emit identical whole archives),
`ctx_parity()` (scalar vs AVX2 contexts across unit resets), and
`archive_model_damage()` (`E_CHECKSUM` for a raw flip, `E_FORMAT` for a
consistent but invalid model). A PR that changes any of these needs a new
profile ID, not a golden update.

### I3 — New-profile goldens for F16/Q8 and every ISA path

Only when a profile PR is merged, and only for the profile it owns:

| PR | profile | predictor | F16 (len / CRC) | Q8 (len / CRC) | ISA coverage in branch |
|---|---|---|---|---|---|
| #10 | 6 `CALIBRATED_MIX` | `cal` | 43914 / `0xD593C4D1` | 39189 / `0xEBF4AF55` | all available ISAs (scalar p6 path) |
| #11 | 11 `ENTROPY_GATE` | `gate` | 59403 / `0xE70A25C1` | 47616 / `0x4310D2DB` | all available ISAs (gate in AVX2 loop) |
| #13 | 14 `HNET_BOUNDARY` | `hnet` | 57761 / `0xEE2926BD` | 48573 / `0x482211FB` | all available ISAs (f=27, no fused loop) |

Plus identity vectors in the same test binary: p6 `(temp=256,bias=0,w=(2,1,1))`
and p11 `(a=0,b=0)` MUST reproduce profile-1 bits and bytes. Profile 14 has no
identity vector (different width) but must reproduce profile-1 bytes with
`auto`/`mix`. If a merge fix changes any arithmetic, regenerate these goldens
in the same commit and record the reason; a silent golden edit is a format
break.

### I4 — Decoder rejection of unknown profile/quant pairs

- `taw_header_parse()` → `taw_pred_supported()` must return `TAW_E_UNSUPPORTED`
  for any profile outside {0,1} on main, and for {0,1} with a quant outside its
  allow-list (profile 0 only `QUANT_NONE`; profile 1 only the nine quant values
  in `p1_quant_supported`). Golden test for this already exists in
  `tests/test_p1.c:validation()` (size mismatch, BF16/F32, bad shape, non-finite
  F16, Q8 `-128`/non-finite scale).
- **Cross-profile loader matrix** (new test for any profile merge): load each
  merged profile's golden model bytes under every merged profile ID and every
  quant; exactly the owning pair returns `TAW_OK`, all others return
  `TAW_E_FORMAT` or `TAW_E_UNSUPPORTED`. This is the test that catches the
  `p1_net_load`/`net_load(head_len)` refactoring collision between #10 and
  #13 and the gate-tail parsing in #11.
- **Central allocation test** (new test for any profile merge): assert a
  bijection between `taw_predictor` enum values, `predictors[]` CLI names,
  `profile_name()` cases and arithmetic profile IDs. Today #10, #11 and #13
  each add `TAW_PREDICTOR_CALIBRATED` / `_GATE` / `_HNET` at **enum value 3**
  and each appends `cal`/`gate`/`hnet` at `predictors[3]`; merging two of them
  without reallocation produces a wrong CLI→profile mapping even if the C
  conflict is resolved. Profile IDs 6/11/14 are numerically distinct; quant
  IDs are untouched. Document `docs/format.md` section numbers centrally too
  (all three add §2.3/§5.2 and mark each other's IDs "reserved, refuse").

### I5 — awseek seek / named-file correctness if touched (#15, and any profile PR adding awseek CLI coverage)

- The §4 fixed-block awseek archive (`-B 4096`) must keep its hash; `taw x`
  range extraction must match the source bytes (hash in §4); `taw d` with
  `-j 1` and `-j 8` must both round-trip.
- Named-file extraction via `taw-nx`/`cli_nx.cmake` must still list and extract
  `/.meta` and `/test.nsp`; wrong keys must still fail. #15 changes the index
  entry from 16 to 20 bytes under flag bit 1 and adds a `dynamic:` line to
  `taw i` — scripts that parse info output must be checked.
- For dynamic-block archives: index CRC checked **before** entry parsing;
  entry size 20 only with `TAW_FLAG_DYN_BLOCKS`; `Σ raw_len == total`;
  `raw_len != 0`; `raw_len ≤ 2^26`; `coded_len ≤ raw_len`; offsets in range;
  unknown flag bits (4..255) rejected by `taw_header_parse()`. Audit-c notes
  only the sum-breaking corruption case is tested today; add `raw_len == 0`,
  `coded_len > raw_len` and out-of-range `coded_offset` cases before merge.

### I6 — Thread-count independence

- Encoding the same input with `-j 1`, `-j 3`, `-j 8` must produce identical
  archive bytes (the §4 `libc1m_L1_mix_j8.aw` cell pins one arm). Decoding with
  `-j 1`/`-j 8` must produce identical output bytes.
- Applies with equal force to #15's two-pass `choose_blocks()` and to #19's
  EMA step counters (`last1`/`last2`) at un-tie. Add a combined arm: L5,
  `-j 1` vs `-j 8`, `TAW_FIT_AVG=ema999`, and (if #15 merged) `-D -T 16384`.

### I7 — Gate/profile ownership (derived from I1/I3)

- Exactly one owner for each mechanism: C3 sign gating belongs to #8, not #6;
  APM-blend profiling belongs to one of #10 (p6) / #12 (p7, DRAFT, out of
  scope), not both. If #12 is ever revived, #10's ownership must be re-decided
  before either archive's bytes are considered stable.
- The NS polynomial has one implementation. #5's dormant `faro_ns_score()` and
  #6's live `namo_rows()` copy must be factored into the upstream helper, or
  the copies will silently diverge after any Muon/MALTER tuning change.
- No diagnostic knob in candidate selection: #8's `TAW_FORCE_SEL1/2` in
  `enumerate()` must be removed or test-build-only, otherwise the §4 hashes
  can change with a debug env var.

---

## 2. Interaction matrix — pairs that can change each other's bytes/defaults

"Detect" names the test that must be added/kept green in the batch where the
second PR lands. All pairs are read from the diffs; the audit reports supply
the known issues (audit-a #5–#8, audit-b #10/#11, audit-c #13/#15, audit-d
#19/#21).

| Pair | Shared surface | Failure mode if merged together | Detect |
|---|---|---|---|
| **#10 × #12** (given example; #12 is DRAFT/out of scope) | Learned APM blend: p6 `w0..w2` vs p7 blend; both find `w0=0` | Duplicate mechanisms, duplicate predictor enum value 3, duplicate `docs/format.md` §2.3/§5.2 and conflicting ID reservations. Whichever lands last silently redefines the other's `-p` name. | Central allocation test (I4) + "one APM-blend owner" decision recorded before either merge; hold #12 until #10 settled. |
| **#8 × #6** (given example) | C3 sign gate: #8 `adam_rows_adv` on all AdamW `(m,v)`; #6 `sig1` passed to AdamW W1, Muon W1, blend W1 | Plain `HYB_ADAM` W1 gets two independent gating passes; W2 only #8, Muon W1 only #6, MALT untouched. Bytes differ from either single-gate PR; no default-off test sees it. | Known-vector test with `TAW_EPS_MODE=c3` + `TAW_NAMO_GATE=c3` on fixed W1/W2, compared to each single gate and knob-off; remove #6's gate (single owner). |
| **#13 × #8** (and × #5) | W1 width: #13 `t->f = 27`; #8 `float gate[P1_HMAX*P1_F+1]` = 833 floats | With `-p hnet` + C3, row1 = H·27 = up to 864 floats → **stack buffer overflow**; #5's FARO scratch is likewise sized from `P1_F`. | ASan build, `TAW_EPS_MODE=c3` + `-p hnet` (and `TAW_FARO=1` + `-p hnet`) encode; assert no overflow + deterministic bytes. |
| **#13 × #5/#6/#7/#19** | `t->f` generalization in `tnet_init/step/untie/malt_rows/muon_rows/serialize` | Audit-c verified f=26 behaviour by inspection; a missed site changes RNG draw order or pointer offsets for *all* upstream profiles, not just p14. | §4 golden hashes (all cells) + tuned Muon/MALTER controls + `test_p1.c` goldens after the rebase that lands #13. |
| **#13 × #15** | CLI awseek/range tests, header/model layout | #15 adds `dynamic:` to `taw i` and 20-byte index entries; #13 adds `-p hnet` roundtrip assertions. A seekable p14 archive's index/model CRC could be mis-sized. | `-p hnet -s -B 4096 -D -T 16384` roundtrip + range extract + `taw i` profile line + index hash. |
| **#15 × #10/#11/#13** | Index math vs new head/tail bytes | Head/tail extensions live in the model section (CRC-covered), so index math should be unaffected — but a loader that mis-parses head length shifts the model bytes and therefore every block offset. | Cross-profile loader matrix (I4) run on seekable archives; decode + range + named-file for each profile with fixed and dynamic index. |
| **#19 × everything** | Default-on EMA for levels 3+ | With `TAW_FIT_AVG_DEFAULT=late`, *every* L3..9 golden cell changes bytes, including cells claimed byte-identical by #5/#6/#7/#8/#10/#11/#13/#15. Also evaluates calibration/gate fitting on a different final net than the serialized one. | §4 scrub sets `TAW_FIT_AVG=off`; separately assert `TAW_FIT_AVG=off` reproduces main L5 hash and `ema999` is deterministic; A/B on MALTER path. Preferred fix: default `off`. |
| **#19 × #6/#8** | EMA fold order vs C3/NAMO per-step state | Sign-EMA update, weight update, EMA fold must have one pinned order; un-tie copies (#6 `v2s/v2c/sig1`, #19 `wa/last1/last2`) must all be mirrored. | Known-vector test with both knobs on; L5 encode -j1/-j8 with un-tie forced; compare bytes. |
| **#19 × #7** | LR schedule (`flat`) + EMA | A flat LR changes the trajectory that EMA averages; audit-d says levels 3–4 and 6–9 are unmeasured. | `TAW_LR1=flat` with `TAW_FIT_AVG=off` and `=ema999`, two runs each, byte compare. |
| **#7 × #8** | Tuned AdamW config + `enumerate()` | #8's headline (C3 −4,838 B, `w0=0`) depends on #7's tuned LR/WD; #8's `TAW_FORCE_SEL1/2` can change the candidate set underneath #7's sweep numbers. | Pin exact `TAW_LR0/LR1/B1/WD` in the C3 repro; env scrub unsets `TAW_FORCE_SEL*`; assert force knobs have no effect unless explicitly set. |
| **#5 × #6** | NS duplicate + risk hook | #5 appends risk/bid hooks to `muon_rows`/`malt_rows`/`adam_rows` but not to #6's new `namo_rows`; FARO-ERC becomes a silent no-op under NAMO. Three NS copies (upstream, faro, namo) can diverge. | Enable `TAW_FARO=1` + `TAW_HYBRID=namod`, assert hook fires (counter) and archive changes; NS-coefficient equality test across all copies. |
| **#5 × #19** | W1 LR/budget multiplier + EMA fold | `lr_w1 = o.lr * fr.mu[fb]` changes W1 steps; #19 decays by `decay^(steps since last fold)`. Wrong order changes L5 bytes. | Known-vector W1 test with both on; §4 hashes with FARO off; deterministic repeat with FARO on. |
| **#15 × #21** | `.gitignore` and generated archives | #21 replaces `/build*/`, `/compile_commands.json`, `*.aw`, `*.awseek` with four unrelated patterns; #15's generated `.awseek` files become committable. #21 also hardcodes sibling-worktree binary paths and 9 unmerged branches. | `git check-ignore build/x.aw build/x.awseek build/compile_commands.json` after merge; restore the four lines before #21 lands. |
| **#21 × #5/#6/#7/#8/#13/#15** | `RESULTS.md` add/add conflict | Each PR appends a section to `RESULTS.md`; #21 adds 361 lines there. Textual conflict plus duplicated reports. | Follow #19's pattern: rename to `RESULTS_MATRIX.md`; `git merge-tree --write-tree` must be clean in each batch. |

Out-of-scope but noted because they constrain the plan: #9/#12/#16/#18 are
DRAFT and #14/#17/#20 are CLOSE per audits b/c/d. None of them should be
pulled into a batch to "resolve" an interaction here.

---

## 3. Staged merge plan

Rule for every batch: after the merge, `ctest` is 12/12 (gpu skipped on this
host), §4 hashes are unchanged, and a decoder built from the previous batch
can read every default archive produced by the new batch. Each batch is one
revertible commit (or one PR merge) so rollback is `git revert -m 1 <merge>`.

| Batch | Content (with required fixes) | Why this position | Green condition / rollback |
|---|---|---|---|
| **0 — infrastructure, no PR** | Land the §4 hash script + hash table, the central predictor/profile allocator test (I4), the knob-off golden test, and the cross-profile loader matrix skeleton. | Gives every later batch an objective pass/fail; no archive behaviour changes. | ctest 12/12 + hashes match. Rollback: delete new test files. |
| **1 — docs/hygiene, no format** | **#7** (delete stale v1 TSV/logs, consolidate v2 into RESULTS.md/bench/results.tsv, make `sweep.py` portable or drop it, test `TAW_LR1=flat`) and **#21** (rename report to `RESULTS_MATRIX.md`, restore `.gitignore`, parameterise private paths, mark unmerged arms). | #7's tuned-AdamW config is the reference for #8; #21 is a decision record, not a mechanism. No profile/format state. | Hashes unchanged; `git check-ignore` passes. Rollback: revert both commits; no state to unwind. |
| **2 — FARO-ERC** | **#5**: strip the routing lane (`faro_cf_probe/explore/ns_score`, `TAW_FARO_ROUTE/STAT/EXPLORE`), keep ERC only, fix the "no duplicate Muon" doc claim, add neutrality/risk-hook/knob-off tests. | Opt-in, default-off, no format. Landing before #6/#8 means only one NS duplicate is removed at a time. | Hashes unchanged; FARO-on deterministic vs MALTER control. Rollback: revert; no archive state. |
| **3 — C3 gate owner** | **#8**: ship C3 shrink-only, remove C2/NC9 dead paths or label them reproduction-only, factor `adam_rows` body out of `adam_rows_adv`, remove `TAW_FORCE_SEL1/2`, add known-vector + per-corpus text1m-L1 guard. | Establishes single C3 ownership before #6 is allowed anywhere near a gate. | Hashes unchanged; gate known-vector test green. Rollback: revert. |
| **4 — NAMO** | **#6**: remove `TAW_NAMO_GATE*`/`sig1`/C3 comparison, factor NS into the shared helper, add update-rule known vectors + "does not beat MALTER (80 B)" framing. | After batch 3 the gate conflict is gone; only the NS duplicate and new optimizer state remain. | Hashes unchanged; NAMO/NAMOD opt-in bytes deterministic. Rollback: revert. |
| **5 — EMA knob** | **#19**: default `TAW_FIT_AVG=off`, fix stale comment, fix `bench/control.sh` (`TAW_FIT_LRSCALE` removed), re-label/re-run TSVs, add knob-off golden + `late` determinism test, A/B on MALTER. | Default-off keeps all hashes green. If the maintainer insists on `late` as default, this becomes a separate *default-change* batch with rebaselined goldens and an explicit decoder-compatibility note (bytes change, format does not). | Hashes unchanged with `off`; L5 `ema999` deterministic. Rollback: revert; archives made with `late` still decode on main. |
| **6 — container feature** | **#15**: rebase, three new corruption tests, mixed-media structural number, equal-target table re-anchored to MALTER; note `taw i` output addition. | Opt-in flag bit 1; fixed-block index stays 16 bytes, so all §4 hashes (including awseek) stay green. | Hashes unchanged; `-D` arm golden recorded at merge; j=1/3/8 parity. Rollback: revert; any dynamic-block archive must be re-encoded with fixed blocks (main decoder rejects flag bit 1). |
| **7 — first arithmetic profile** | **#11**: allocate predictor value 3 → `gate`, keep `auto`/`mix` p1-only, add third corpus or second seed, UBSan/ASan check of AVX2 padding mask + match-active `st24` golden, README line. | Best-executed profile PR; smallest delta (4-byte tail), clean AVX2 integration, strong tests. Establishes the profile-allocation convention. | §4 hashes unchanged; p11 F16/Q8 goldens green on all ISAs. Rollback: revert; `-p gate` archives unreadable by post-rollback main. |
| **8 — W1-width profile** | **#13**: fix §2.14.2 max and sampling comment, add GPU `P1_F` hardcode guard, re-benchmark vs MALTER, allocate predictor value 4 → `hnet`, ASan cross-check with every optimizer knob from batches 2–5. | Depends on the allocator from batch 7 and on the W1-width audit; after it, all W1 scratch buffers have been re-audited once. | §4 hashes unchanged; p14 goldens green; optimizer knobs + `-p hnet` do not overflow. Rollback: revert; `-p hnet` archives unreadable. |
| **9 — conditional calibration** | **#10** only after #12 is closed/held: allocate predictor value 5 → `cal`, restrict `auto` to profile 1 (p6 only via `-p cal`), matched-quant + third-corpus evidence, fused-AVX2 throughput numbers or documented cost, README line. | Its APM-blend ownership is not stable while #12 exists, and its current auto semantics violate I1. | §4 hashes unchanged; p6 goldens green. Rollback: revert; `-p cal` archives unreadable. |
| **Closed / held** | #12 (DRAFT), #9/#16/#18 (DRAFT), #14/#17/#20 (CLOSE) — keep out of all batches. | Negative or unresolved evidence; see §5. | n/a |

**General rollback notes.** (a) A revert of a format PR must be accompanied by
a note that archives produced with its explicit predictor are no longer
readable — do not redistribute them. (b) A revert of #15 must be accompanied by
re-encoding of any dynamic-block archive. (c) A revert of #19 is byte-safe for
decoding but changes L3+ encode bytes back. (d) Never roll back a batch by
dropping the golden hash script; keep the hash ledger so the next attempt is
measured against the same corpus.

---

## 4. The single golden corpus set and command sequence after every merge

**Corpora (read-only, never modified).** `D=/tmp/opencode/tadamw-bench`:
`text1m` (sha256 `2364f861d5e7f418095fad55dfc1a2d7c53eccfe30c1f2d1578b3b1fe7422129`),
`libc1m` (`49bfd4c2dc804f7a35a87093b794a1d9c9d73d6f7e0c4626eec6d201b67c508d`).
Seekable large/mixed input for the optional `-D` arm:
`D/corpora2/mixed1m` (`45b01b975f9823f6f475b827868e240aa079de009a82a87ecd489f21a673d642`).
Do not substitute regenerated corpora: they change every hash.

**Build (fixed).**
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTADAMW_GPU=OFF
cmake --build build -j4
(cd build && ctest --output-on-failure)     # expect 12/12, gpu skipped
```

**Env scrub.** Each invocation is wrapped so no experiment knob leaks in:
```
scrub() { env -i PATH=/usr/bin:/bin HOME=/tmp LC_ALL=C TZ=UTC \
                TAW_FIT_AVG=off "$@"; }
```
`env -i` clears all `TAW_*` (FARO/NAMO/LR1/EPS/GATE/CAL/HYBRID/MALT/MUON/
FORCE_SEL*/GPU...). `TAW_FIT_AVG=off` is explicit belt-and-braces until #19's
default is fixed. Work dir: `build/compat/` (gitignored by `/build*/`).

**Commands.**
```
D=/tmp/opencode/tadamw-bench; T=./build/tools/taw; O=build/compat; mkdir -p "$O"
for c in text1m libc1m; do
  for l in 1 5; do
    for p in mix auto; do
      scrub "$T" c -l "$l" -p "$p"         "$D/$c" "$O/${c}_L${l}_${p}.aw"
      scrub "$T" d "$O/${c}_L${l}_${p}.aw" "$O/${c}_L${l}_${p}.dec"
      cmp "$O/${c}_L${l}_${p}.dec" "$D/$c"
    done
  done
done
scrub "$T" c -l 1 -p mix -w q8  "$D/text1m" "$O/text1m_L1_mix_q8.aw"
scrub "$T" c -l 1 -p mix -j 8   "$D/libc1m" "$O/libc1m_L1_mix_j8.aw"
scrub "$T" c -l 1 -p mix -B 4096 "$D/text1m" "$O/text1m_L1_mix_B4k.awseek"
scrub "$T" x "$O/text1m_L1_mix_B4k.awseek" 4000 5000 "$O/range.bin"
scrub "$T" d -j 8 "$O/text1m_L1_mix_B4k.awseek" "$O/seek_j8.dec"
cmp "$O/seek_j8.dec" "$D/text1m"
(cd "$O" && sha256sum *.aw *.awseek *.bin | sort -k2)
```
If #15 is merged, add the opt-in arm and record its hash at that merge:
```
scrub "$T" c -l 1 -p mix -s -B 4096 -D -T 16384 \
     "$D/corpora2/mixed1m" "$O/mixed1m_L1_dyn.awseek"
scrub "$T" d -j 8 "$O/mixed1m_L1_dyn.awseek" "$O/mixed1m_dyn.dec"
cmp "$O/mixed1m_dyn.dec" "$D/corpora2/mixed1m"
```
If a profile PR is merged, add its explicit predictor arm (`-p cal|gate|hnet`)
and require the §I3 CRC/`taw i` profile line; do **not** add it to `auto`.

**Hashes to compare (main, this base).** Any mismatch on a default cell is a
merge blocker; the `auto` cells are the ones that catch #10/#19.

| archive | bytes | sha256 |
|---|---|---|
| `text1m_L1_mix.aw` | 181557 | `085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d` |
| `text1m_L1_auto.aw` | 181557 | `085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d` |
| `text1m_L1_mix_q8.aw` | 181557 | `085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d` |
| `text1m_L5_mix.aw` | 179166 | `ca621be0b340b7e04fa65d4a781d9a25766aa6312be7d0be174963b36eae187a` |
| `text1m_L5_auto.aw` | 179166 | `ca621be0b340b7e04fa65d4a781d9a25766aa6312be7d0be174963b36eae187a` |
| `libc1m_L1_mix.aw` | 379197 | `6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b` |
| `libc1m_L1_auto.aw` | 379197 | `6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b` |
| `libc1m_L1_mix_j8.aw` | 379197 | `6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b` |
| `libc1m_L5_mix.aw` | 375291 | `89863914c8deff6b43522f58f711757abda94d5d6eb44f6c369905c628faccf2` |
| `libc1m_L5_auto.aw` | 375291 | `89863914c8deff6b43522f58f711757abda94d5d6eb44f6c369905c628faccf2` |
| `text1m_L1_mix_B4k.awseek` | 334373 | `82f38fe571e5206630b61942defdc8169da48666fceedb9d9717f4288aaa98df` |
| `range.bin` (offset 4000, len 5000) | 5000 | `e20185c652f152d0dc36b0f6a6903ea09576cd892461571c42bcb913887a21f0` |

All golden archives must report `profile: 1 (F32_STRICT_MIX)` in `taw i`.

---

## 5. Ruling

The Compatibility Keeper rules on the ten REVIEW-tier PRs as follows. If a
decoder from main reads this archive, then the archive must still be main's.

**Safe to merge as-is (compat-neutral; hygiene fixes still required by the
audits):** none, strictly. The two closest are:

- **#7 LR-baseline** — code delta is six default-off lines and no format/API
  change; the blocker is stale v1 artifacts and hardcoded `/tmp/opencode` in
  `sweep.py`, not bytes. Merge after pruning (batch 1).
- **#21 final-matrix** — the numbers verify and the format is untouched; but
  its `.gitignore` regression un-ignores every `.aw`/`.awseek` and its scripts
  point at unmerged sibling worktrees. Merge after rename + `.gitignore`
  restore (batch 1).

**Need fixes first (none may land in their current form):**

- **#19 ema** — default is *on* for levels 3+, so L5 default bytes change;
  flip to `off`, fix the stale comment and broken `bench/control.sh`, add tests.
- **#10 calibration** — `auto` currently enumerates profile 6, so the default
  emits archives a main decoder rejects; gate p6 behind `-p cal`, resolve APM
  ownership with #12, re-run at matched quant.
- **#6 namo** — third NS copy plus a C3 gate that wraps upstream Muon/Adam W1;
  factor NS, remove the gate, add known-vector tests.
- **#8 eps-gate** — ships its own rejected C2/NC9, duplicates `adam_rows`, and
  puts `TAW_FORCE_SEL1/2` in candidate selection; narrow to C3, factor, remove.
- **#5 faro** — ships a self-declared-failing routing lane with a dormant NS
  copy; ship ERC-only and prove the risk hook on MALTER.
- **#15 hnet-blocks** — correctness looks sound but three security-relevant
  index-corruption cases are untested and the value story is overstated as a
  compression win; add tests and re-anchor the table to MALTER.
- **#11 entropygate** — technically the cleanest, but adds a permanent profile
  and hot-loop branch for 0.04–0.30 % on two corpora; needs a third corpus,
  the match-active golden, and a centrally allocated predictor value.
- **#13 hnet-feature** — f=27 generalization is clean for f=26, but the
  benchmark is 15 kB behind MALTER, docs have two inaccuracies, and the GPU
  `P1_F` hardcode needs a guard; fix, re-benchmark, allocate predictor value 4.

**Close (among the in-scope ten):** none of the ten is a close candidate on
compatibility grounds alone — every defect above is fixable. The CLOSE rulings
belong to the adjacent tiers (#14, #17, #20), which are outside this suite and
are not staged. #12 is DRAFT and must be held, not pulled in to "pair" with
#10.

Final conditional: If any PR in batches 2–9 is merged without its listed fix,
then either the §4 hashes change, a main decoder rejects a default archive, or
a knob that is never tested silently rots. In each of those cases the merge is
not green and must be reverted, regardless of the reported byte savings.
