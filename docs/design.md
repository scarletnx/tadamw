# tar_aw design

## Goal

A general-purpose compressor in the spirit of zstd/xz, with a different
predictor. The encoder fits a small model to the input it is compressing,
using AdamW on the bitwise cross-entropy, and embeds the frozen weights
in the archive. The decoder only runs inference. The bit probabilities
drive a binary range coder (format.md §4).

## Decisions so far

| topic | decision |
|-------|----------|
| fitting | offline, encoder only; the decoder uses frozen weights |
| predictor | bitwise mixer: hashed order-N context features → small MLP → P(bit=1) |
| model budget | `model_size <= raw_size / 32`; a heuristic picks the size |
| level | effort only (epochs, number of candidate sizes), not a size knob |
| weights | stored in a tagged quant format; F16 is standard |
| arithmetic | tagged arithmetic profile; decoders refuse unknown profiles/quants |
| variants | `.aw` stream, `.awseek` seekable with one global model |
| language | C23 inside, CMake + CTest; libtadamw needs libc and pthreads; `taw-nx` also needs OpenSSL 3 and LZ4 |
| library | `libtadamw` (static or shared), `<tadamw/tadamw.h>` usable from C99+/C++, encoder and decoder halves selectable at build time |
| portability | every SIMD path has a generic one with identical results; byte-wise little-endian I/O, so big-endian hosts (ppc64) work too (not yet run on one) |
| license | GPL-3.0-or-later, since `taw-nx` borrows ScarletNX's hactool code |
| Switch dumps | an ordinary `.awseek` with the dump (NCAs and gamecard info decrypted) and `/.meta`; decoding encrypts again and checks a SHA-256 of the original |
| awseek extras | file directory (format.md §7.1), stored blocks/chunks, parallel streaming encoder (`taw_seek_encode`) whose output does not depend on the thread count |
| libm | never used, in the encoder or the decoder; squash/exp go through SIMD intrinsics or tables |
| model shape | table/MLP split chosen heuristically per input, not fixed |
| weight formats (profile 1) | F16 and INT8 (Q8) from the start; then Q4, Q2, FP8 (E4M3), NVFP4, IQ8, IQ4, IQ2 (format.md §3, §5.1) |
| SIMD | scalar reference + AVX2/F16C and AVX-512 paths (ideas: ~/ucas, ~/q2_kernel kernels) |
| GPU | the encoder may fit on a GPU: opt-in (`taw_params.gpu` off/auto/on, `taw c -g`), Vulkan loaded at run time with `dlopen` (no link-time dependency; builds without Vulkan get a stub), GLSL compute shaders compiled to SPIR-V at build time and embedded. The CPU keeps feature collection, candidates, quantization and the final scoring. Decoders are CPU by default and may opt in to a GPU (hybrid or fully). A CPU decoder is always available and is never dropped |
| profile 1 inputs | 12 adaptive integer context models (o0-o4, o6, word, 5 sparse), each giving a direct and an indirect (bit history) estimate, plus a match model: 25 inputs + bias |
| profile 1 mixer | frozen MLP: one optional hidden layer (clamp ±8), weight sets picked by a selector context (the "table" part), fixed `dot4` summation order |
| profile 1 output | integer squash, then two adaptive integer APM stages (c0, c1:c0) |
| profile 2 | profile 1 with the chunk stage of arXiv 2507.07955 ("Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"): a routing table fitted to the input cuts the bytes into chunks and lines, four context models are keyed by them, and the model names which ten of the fourteen hashed models run. Opt-in (`taw c -p chunk`, `TAW_PREDICTOR_CHUNK`). Smaller on every file of the bench below at level 5 (1.5% in total, 20% on the table-like one); the decoder runs profile 1's per-bit loop but is up to 19% slower on text, where the chunk models' lines miss the cache |

## Profile 1 in short

format.md §2.2 is normative; this is the reasoning.

- **Adaptive features, frozen mixer.** The decoder has to adapt to local
  statistics somehow. Profile 1 keeps all adaptation in deterministic
  integer state (counters, bit histories, the match model, APM) and
  freezes only the mixer. Two things follow: the features depend only on
  the data, never on the weights, so the encoder computes them once and
  fits on stored samples; and all float work is one small MLP.
- **Two estimates per context.** A slot holds an adaptive-rate
  probability and the slot's last 7 bits. The bits go through a per-model
  adaptive map (history to probability). Measured on the bench corpus, a
  single counter wanted a different adaptation rate per file (limit 2 to
  30); the pair lets the fitted mixer choose. This saved about 3% over
  the best single-rate counter.
- **Tagged lines.** Hashed models use 64-byte lines (one nibble's bit
  tree per line, so one cache miss per model per nibble) with a 16-bit
  tag. A mismatched line is reset instead of inheriting another context's
  statistics: 2.5% smaller. Two-way lines were not consistently better.
- **Sparse contexts** (bytes 2+3, 3+4, 2+4, 1+4, 1+2+7+8 back) are for
  binaries: libc went from 327 KB to 307 KB, with text unchanged.
- **APM** adds about 2%.
- **Determinism.** The only float code is `dot4` products, clamps and one
  truncation, in an order the spec fixes. Each SIMD lane computes one
  hidden unit in exactly the scalar order. The output layer's `dot4` maps
  onto SIMD exactly too: lane r of each 4-wide half belongs to partial sum
  `s_r`. So scalar, AVX2 and AVX-512 agree by construction, and the tests
  check it. Four partial sums (rather than one running sum) shorten the
  add dependency chain without giving up exactness.

## Profile 2 in short

format.md §2.3 is normative; this is the reasoning.

- **What the paper does.** H-Net reads raw bytes through a small encoder,
  lets a routing module place a chunk boundary wherever the representation
  of a position is dissimilar from the one before it (so the decision
  needs only bytes already seen), runs its large network on the chunks
  alone, and repeats each chunk's output over the bytes up to the next
  boundary. Two stages (chunks of chunks) is its best model. The paper's
  point is that this learned, causal segmentation replaces a tokenizer:
  BPE is its baseline, not a part of it. A BPE vocabulary is fixed and
  greedy, its boundaries depend on bytes that come later (a decoder cannot
  place them from the bytes it has), and it has to be shipped or frozen.
- **What that is here.** A large network per chunk is orders of magnitude
  outside a decoder that spends one step per context model per bit. The
  parts that survive the budget: the routing module becomes a table over
  the last two bytes (a class per byte value and a 16x16 class matrix,
  which is the paper's similarity test with a one-byte encoder and one bit
  of similarity); the chunks become hashes; the network over chunks becomes
  context models keyed by the chunk history, held until the next boundary
  as the paper's upsampler holds a chunk vector; the second stage is a
  second matrix that marks which chunk boundaries start a line, which
  gives every chunk a place in its line and a chunk above it.
- **The table is searched, not trained.** Hash tables carry no gradient.
  The encoder scores a candidate table by the code length of a small
  context mixer over a sample: a fixed base prediction (orders 1-4 and 6
  and the word model, computed once) plus the four chunk contexts under
  the candidate. A table therefore scores by what its chunks add to what
  the mixer has anyway, not by how well they predict alone. Coordinate
  descent from the best of three presets over the class of each frequent
  byte and the bits of both matrices.
- **Lanes, not more models.** A first version ran the chunk models beside
  profile 1's twelve (15 models, 32 mixer inputs): text got 0.7-1% smaller,
  binaries 0.1-0.9% larger (six more weights per mixer row, and noise),
  and decoding was 21% slower (10 MB of enwik8: 7.54 s to 9.14 s). The
  AVX2 kernels have 16 lanes, so the encoder's feature pass can run all
  fourteen hashed candidates at once, and each model's inputs depend on
  its own context alone: a model reads the same values in the 16-model
  pass as in the decoder's lane (`tests/test_p2.c`,
  `lanes_match_candidates`). So the encoder ranks the candidates with a
  linear mixer on the samples, drops the four it misses least, and the
  decoder runs ten hashed models through profile 1's own per-bit loop.
  Text drops sparse models, binaries drop chunk models, tables drop low
  orders.
- **The same instructions are not the same time.** Decoding executes 3%
  more instructions than profile 1 (the chunk state and the lane move, per
  byte), and takes up to 19% longer (table below). On Dickens the user
  cycles rose 12.7% (perf, before the lane move went into registers), and 58% of that rise is inside the per-bit loop,
  which executes the same instructions as before: a sparse model on text
  has few distinct contexts, so its lines stay in the cache, and a chunk
  model that takes its lane has about as many as the high orders. Asking
  Linux for huge pages removed a third of profile 1's TLB misses and none
  of this. Files where every model's lines were cold already (x-ray) lose
  nothing. Not tried: smaller tables for the chunk lanes only, and a
  ranking that charges a chunk model for the lane it takes.
- **Never much worse than profile 1.** The ranking is a linear proxy for
  the MLP. The encoder also fits profile 1's ten lanes on the same samples
  and keeps the smaller estimate, so a misleading ranking costs the ten
  lane bytes. With those lanes the model has no routing table and the
  profile computes what profile 1 does (the golden test checks the coded
  bytes are identical).
- **Where it pays.** Lines and columns first (a chemical database, XML, a
  database dump), then prose, then everything else by a little.

Measured on enwik8 and the twelve files of the Silesia corpus (312 MB),
`-p chunk` against `-p mix`, one thread, Ryzen 5 7500F; every archive was
decoded and compared with its input:

| file | MB | profile 1, level 5 | profile 2, level 5 | change | change at level 1 | profile 2's lanes at level 5 drop |
|------|---:|---:|---:|---:|---:|---|
| enwik8 | 100.0 | 22 238 343 | 21 973 768 | -1.19% | -0.73% | s23 s24 s1278 above |
| mozilla | 51.2 | 13 515 038 | 13 417 915 | -0.72% | -0.26% | o3 o4 o6 s14 |
| webster | 41.5 | 6 675 388 | 6 584 214 | -1.37% | -0.57% | s23 s34 s24 prev |
| nci | 33.6 | 1 595 639 | 1 276 296 | -20.01% | -14.09% | o4 s24 prev col |
| samba | 21.6 | 3 403 068 | 3 355 861 | -1.39% | -1.03% | s23 s24 prev col |
| dickens | 10.2 | 2 347 300 | 2 317 791 | -1.26% | -1.69% | s23 s34 s24 above |
| osdb | 10.1 | 2 410 632 | 2 339 697 | -2.94% | -0.52% | word s14 s1278 above |
| mr | 10.0 | 2 252 046 | 2 239 748 | -0.55% | -0.32% | word s14 s1278 prev |
| x-ray | 8.5 | 3 737 521 | 3 730 434 | -0.19% | -0.09% | o3 word s1278 above |
| sao | 7.3 | 4 537 780 | 4 504 759 | -0.73% | -0.21% | o4 s23 s24 s14 |
| reymont | 6.6 | 1 042 554 | 1 025 682 | -1.62% | +0.18% | o3 o4 s24 s14 |
| ooffice | 6.2 | 2 166 194 | 2 149 675 | -0.76% | +0.00% | o3 o4 o6 above |
| xml | 5.3 | 386 081 | 368 328 | -4.60% | -0.34% | o6 s24 s14 s1278 |
| total | 311.9 | 66 307 584 | 65 284 168 | -1.54% | -0.86% | |

Level 1 uses a preset table without searching it, which is most of the
difference on XML and the database dump.

Decoding, seconds, median of two interleaved runs:

| file | profile 1 before this (e64cc83) | profile 1 | profile 2 | profile 2 against profile 1 |
|------|---:|---:|---:|---:|
| dickens | 7.58 | 7.57 | 8.98 | +18.7% |
| xml | 3.24 | 3.26 | 3.78 | +16.1% |
| nci | 18.80 | 19.04 | 20.02 | +5.1% |
| osdb | 8.41 | 8.46 | 8.70 | +2.8% |
| x-ray | 8.11 | 8.08 | 8.13 | +0.7% |

Encoding at level 5, seconds, median of three runs: 10 MB of enwik8 18.1
with profile 1 and 77.4 with profile 2; the XML file (5 MB) 9.8 and
66.2. The difference, 59 s and 56 s, hardly moves with the input size:
it is mostly the routing search, then the 16-model feature pass and the
second fit.

- **Tried, not kept:** a third chunk n-gram (chunk with the two before
  it) lost to the place-in-line context wherever lines exist and was
  within noise on prose; the chunk above without the place was 2% worse on
  the table-like file; a remembered line of 16 chunks was 1-1.5% worse on
  structured files than 32, and 32 was 4.7% worse than 64 on the
  table-like file (128 and 256 change nothing more there; 128 is 0.1%
  better than 64 on XML). A fixed rule (letters join, every
  other byte alone, the profile 1 word model's rule) gave about two thirds
  of the searched table's gain on English text and nothing on tables.

## Decoder performance

Measured on an i5-9400 (AVX2, no AVX-512), 1 MB inputs at level 1:

| path | words1m | libc1m |
|------|---------|--------|
| scalar | 2.8 s | 2.9 s |
| AVX2 before the context-model kernels | 1.58 s | 1.68 s |
| AVX2 now | 1.20 s (0.83 MB/s) | 1.35 s (0.74 MB/s) |

Per bit, profile 1 runs one arithmetic step per context model, 12 times
over. That shapes the code:

- **Data layout.** All context tables are one arena of u32 slots
  (`[O0][O1][T_2..T_11]`) addressed by u32 indices, not pointers. The
  per-bit hot state is a set of small parallel arrays padded to 16 lanes:
  line bases, slot indices, slot values, history-map indices and values.
  The AVX2 kernels (`p1_avx2.c`) work on those directly: slot indices are
  `line base + nib`, slot values come in by gather, and both counters of
  all 12 models update in two 8-lane halves.
- **Store forwarding mattered most.** Vector loads of arrays just filled
  by scalar stores cannot forward. That cost about 6 blocked loads per bit
  (48.8M on a 1 MB input) until values were built in registers: gather
  slots straight from the arena, assemble history-map entries with
  `set_epi32`, build hash lanes from scalars, and keep the hidden layer in
  registers through the whole mixer. Blocks are now about 0.15 per bit.
- **Gathers vs scalar loads.** Slots are gathered: within a nibble every
  bit uses a different slot, so no pending store aliases them.
  History-map entries are often the ones the previous bit just wrote, and
  gathers cannot forward from pending stores, so those are scalar loads.
- **Exact SIMD shortcuts.** The bit is the same in every lane, so each
  counter needs one product (`bit ? max - p : p`). `R[n]` is
  `cvtt(2^17 / (2n + 3))` in binary32: the quotient's rounding error,
  at most 2^-7/d, is below the 1/d gap to the next integer, so it
  truncates correctly for every `n`. That replaces two dependent gathers.
- **Tried, not kept:** computing the next nibble's or byte's hashes a bit
  early for both outcomes and prefetching their lines was about 3% slower
  than computing them once at the boundary. Scalar table lookups instead
  of the stretch gathers were 7% slower. `-O3` gains 1-2%.

- **Evaluating the next bit's outcomes ahead** (branch
  `exp/two-outcome-decode`). While bit i is decoded, the next bit's
  prediction is prepared from the state before bit i's update. That is
  exact: within a nibble bit i+1 reads other slots and APM rows, and the
  history-map entries and match counter it may share with bit i come from
  bit i's update precomputed for that outcome. Measured (words1m /
  libc1m, level 1): main 1.23 / 1.42 s; both outcomes 1.81 / 2.07 s; only
  the likely outcome (redo on a miss) 1.29 / 1.48 s. The per-candidate
  records cost the sequential path about 5% on their own.

Before the second round below, the decoder ran about 675 cycles and
1060 instructions per bit at IPC 1.6. Evaluating more than one outcome
multiplies the work, so it loses.

### Second round (fused loop)

pmcstat on libc1m (8.4M bits), the same archive throughout:

| step | cycles | instructions |
|------|--------|--------------|
| before | 5.66 G | 8.89 G |
| both-outcome hashes + prefetch at the boundaries | 5.51 G | 9.58 G |
| mixer: partial sums in registers | 5.12 G | 8.31 G |
| counter update on packed fields (identities below) | 5.0 G | 7.96 G |
| one fused AVX2 per-bit loop | 4.27 G | 6.93 G |
| vector line selection, A2 row prefetch, inline tail | 4.0 G | 6.47 G |

Wall clock, `taw d` per 1 MiB file at level 1: 1.24-1.37 s before,
0.93-1.08 s now (20-25% less). Encoding uses the same loop and got about
10% faster.

- **Fused loop** (`p1_run_avx2.c`). One function per mode (encode,
  decode, skip) runs inputs, mixer, squash, APM, coder and counter
  updates. Slot values and history-map entries stay in registers from
  the gathers to the update; the mixer, squash and selectors are inline
  (no function pointer). Only nibble and byte boundaries call out.
- **The mixer kept its partial sums in memory**: `s[i & 3]` with a loop
  index made the compiler keep the four sums in a stack array, so every
  product went through a store and a reload. Four named registers keep
  dot4's exact order.
- **Boundary speculation, kept this time.** One bit before a nibble or
  byte boundary only two outcomes remain. Both outcomes' line hashes
  (and, before a byte end, byte hashes and the match-table index) are
  computed, their lines prefetched, and the boundary then picks one, so
  nothing is computed twice. L3 misses fell from 6.1M to 1.0M. Measured
  against no speculation it is 3-5% faster. (The earlier attempt
  recomputed the hashes at the boundary.)
- **The A2 row** (`c1:c0`, 4.3 MB table) is known at the start of a
  bit; prefetching it there hides its latency behind the mixer.
- **Line selection** (tag check and reset of 10 lines) is one gather and
  a compare, with a loop over the mismatches only.
- **Counter identities**, proved with ~/ucas (`equiv`): `65535 - ps =
  ps ^ 0xFFFF` and `2^22 - 1 - p = p ^ 0x3FFFFF` (exhaustive);
  `bit ? ps + d : ps - d = ps + sign(d, ±1)` (one `vpsignd`; SAT); the
  history step on the packed slot,
  `h' << 8 = ((e << 1) & 0xFE00) | (e & 0x8000) | bit << 8` (SAT). ucas
  found no cheaper scalar form for the squash, APM, rate and mixing
  formulas. For the APM interpolation,
  `(a(128 - w) + bw) >> 7 = a + (((b - a) w) >> 7)` holds by algebra
  (`128a` is a multiple of 128 and `>>` floors), but saves nothing in the
  vector code.
- **Resets.** Line tags carry an epoch (`(hash & 0xFFFF) | epoch << 16`):
  a reset starts a new epoch, and a line of an older epoch is reset when
  it is selected, exactly as if the table had been zeroed. Only O0 and
  O1 are cleared. The arena comes from `mmap` (zero, and pages a unit
  never touches are never faulted in).
- **Tried, not kept:** resetting A2 rows lazily (per-row epoch): the
  check costs about 2.5% per bit and saved nothing measurable even with
  64 KiB blocks. Smaller hashed tables (lines_log2 18 down to 12) were
  only up to 8% faster and cost up to 8% in size: the misses mostly
  overlap.

What remains is mostly the dependency chain of one bit (slot gather,
stretch gather, mixer, squash, APM, coder) plus the 24 scalar write-backs
(no scatter in AVX2). `.awseek` archives also decode their blocks in
parallel (`taw d -j N`, `taw_seek_set_threads`): 5 blocks went from
5.2 s to 1.35 s on 5 threads.

## Encoder pipeline

1. Buffer the input. Fitting is offline, so the encoder sees all of it.
2. Table geometry from the coding-unit size `u` (n for aw, the block size
   for awseek): `lines_log2 = clamp(log2 u - 2, 8, 18)`,
   `hist_log2 = clamp(log2 u, 12, 24)`, `match_log2 = clamp(log2 u - 2, 10, 22)`.
3. One pass of the context models over the input, with the decoder's
   per-unit resets, keeping an evenly spread sample of bits (features,
   selector context, bit). The selection runs over bits in a per-byte
   rotated order (a hash of the byte position), so every bit position is
   sampled: a plain stride kept only bit 7 whenever 8n / samples was a
   multiple of 8, which cost 3-11% on power-of-two inputs.
4. Candidate shapes: hidden ∈ {0, 4, 8, 16} × selectors × the nine
   weight formats, each kept only if `8 + payload <= n/32`. They are
   ranked by a prior fitted to full sweeps (every candidate fitted) at
   levels 1 and 5 on the bench corpus: the first candidate is on average
   0.6% from the best (the earlier hand-set prior: 3%). It prefers sizes
   near n/500 bytes only mildly (capped at 128 KiB: on a 1.3 GB XCI the
   uncapped target chose a 1.8 MB model with a 4096-set selector, whose
   weights fall out of cache, and decoding ran about 3x slower), no
   layer-1 selector and a bit-position output selector, and penalizes Q2,
   IQ2 and F16. Real archives against the previous prior: level 1
   0.6-4.1% smaller, level 5 0.5-1.9% smaller, level 9 within ±0.8%.
5. Fit the top candidates with AdamW (batch 256, lr 0.02 → 0.001
   linearly, β = 0.9/0.999, weight decay 1e-5, lazy updates of only the
   weight sets a batch touches). The first `max(1, epochs/5)` epochs train
   one shared set, which then seeds every set: without this warm-up the
   result depended on the seed by up to 2%. Fitting need not be
   reproducible. On the CPU it is plain scalar C; with `gpu` set, a GPU
   trains the candidates instead (see "GPU fitting").
6. Quantize (F16 round to nearest even; Q8 per-row absmax/127 scale),
   load the bytes through the decoder's own loader, and estimate
   `bits/bit × 8n + 8 × model bytes` on the samples with the real
   inference code. Keep the best.
   Profile 2 (`predictor = chunk`) differs in steps 3 and 4: before the
   feature pass it searches the routing table ("Profile 2 in short"), the
   pass runs all fourteen hashed models (16 lanes), the lanes are chosen
   on the samples, and the candidates are fitted twice, for the chosen
   lanes and for profile 1's. A net reads its inputs from the 33-input
   samples through a map of its lanes, so both fits (and the GPU trainer)
   use the one sample set.
7. Code the input with profile 1. For `.aw` with `predictor = auto`, also
   code it with profile 0 and keep the smaller archive. `.awseek` streams
   through `taw_seek_encode` instead. It codes each block once with the chosen
   model and stores the blocks that do not shrink. Collection, candidate
   fitting and block coding run on `threads` workers. The samples are picked
   by position, so the archive does not depend on the thread count.

Level sets the effort only:

| level | samples (bits) | candidates | epochs |
|-------|----------------|------------|--------|
| 1 | 256 Ki | 1 | 3 |
| 2 | 512 Ki | 1 | 4 |
| 3 | 512 Ki | 2 | 6 |
| 4 | 1 Mi | 2 | 6 |
| 5 | 1 Mi | 3 | 6 |
| 6 | 2 Mi | 3 | 8 |
| 7 | 2 Mi | 4 | 8 |
| 8 | 4 Mi | 6 | 10 |
| 9 | 4 Mi | 8 | 12 |

Profile 2's routing search, by level (sample, how many of the most
frequent bytes get their class searched, sweeps):

| level | sample | bytes | sweeps |
|-------|--------|-------|--------|
| 1 | 256 KiB | presets only | - |
| 2 | 256 KiB | 0 (matrices only) | 1 |
| 3 | 512 KiB | 16 | 1 |
| 4 | 512 KiB | 32 | 1 |
| 5 | 512 KiB | 48 | 1 |
| 6 | 1 MiB | 64 | 1 |
| 7 | 1 MiB | 96 | 2 |
| 8 | 1 MiB | 128 | 2 |
| 9 | 1 MiB | 256 | 2 |

One candidate table costs a pass of the small mixer over the sample;
level 5 tries several hundred, so the search is a fixed
cost of about a minute on one thread (measured above) that small inputs
feel most. Candidates are scored on `threads` workers.

`TAW_FIT_DEBUG=1` in the environment prints every fitted candidate to
stderr, the routing search and the lanes ranking, whether the GPU was used (or why not) and the fitting time;
`TAW_FIT_DEBUG=a` fits all candidates (slow; that is how the prior was
calibrated).

## GPU fitting

`src/fit_gpu.c` drives two compute shaders through Vulkan 1.1, which it
opens at run time (`libvulkan.so.1`). Any failure (no loader, no device,
too little memory, a Vulkan error) makes `TAW_GPU_AUTO` fit on the CPU and
`TAW_GPU_ON` fail with `TAW_E_GPU`; `TAW_FIT_DEBUG=1` says which happened.
`TAW_GPU_DEVICE=n` picks Vulkan device n (even a CPU implementation);
otherwise the first discrete, then integrated GPU that has the limits the
shaders need (45 KiB shared memory, 512-thread workgroups, subgroup
shuffles).

- **Same schedule.** The GPU trainer runs the CPU trainer's algorithm:
  batch 256, the same stride shuffle and start positions, tied warm-up,
  lazy AdamW on the touched sets, the same learning-rate ramp and
  initial weights. Only the float summation order differs (and the forward
  pass reads f16 copies of the weights, see below), so results are close
  to, not equal to, the CPU's. They are reproducible on the same device:
  no float atomics anywhere.
- **What runs where.** The CPU collects the samples (uploaded once,
  14 words each: 4 Mi samples are 224 MiB), enumerates and initializes the
  candidates, then quantizes each trained net and scores it with the real
  decoder inference. The GPU trains every candidate with up to 16 hidden
  units; the others (none at present) stay on CPU workers, which run
  meanwhile. Which candidate goes where is a rule of the shapes, so the
  archive does not depend on the thread count.
- **Batch preparation.** A batch's samples, and so its weight sets, follow
  from the schedule alone. `fit_gpu_prep.comp` therefore lays out the next
  256 steps of every candidate at once, one workgroup per (step,
  candidate): it sorts the batch by set per layer (bitonic) and lists the
  runs of equal sets, splitting runs over 48 samples into pieces.
- **Training.** `fit_gpu.comp` trains one candidate per workgroup, 256
  steps per dispatch. Two threads per sample do the forward pass in sorted
  order (neighbouring threads share a weight set), adding their halves with
  a subgroup shuffle. Then one thread per (run, 4 parameters) sums the
  gradient over the run and applies AdamW at once; pieces of long runs
  leave partial sums that are added after a barrier. The weights of all
  sets a candidate can touch are mirrored in shared memory as f16 when
  they fit (8 sets of 16 hidden units do), because the GT 1030 does not
  cache global loads in L1. The whole fit is one command buffer.
- **Costs on a GT 1030** (3 SMs): one step of one candidate takes 10-30 us
  depending on the shape (the sets a C0 selector touches each step have to
  go through DRAM for AdamW); only three training workgroups are resident
  at a time (one per SM), so eight candidates run in three waves. Opening
  the device and uploading the samples takes 0.2-0.4 s, so `auto` leaves
  fits smaller than 4 Mi sample-epochs (levels 1-2) to the CPU.
- **Hybrid, measured.** Splitting candidates between the GPU and CPU
  workers does not pay on the bench machine: at level 9 a CPU thread needs
  about 22 s for one candidate (17 s alone), the GPU 9 s for all eight.
  Handing the last one, two or three candidates to six CPU workers made
  the level 9 fit of text1m take 22.4 s instead of 8.9 s. So the GPU takes
  every candidate it can; the CPU's share is collection, quantization and
  scoring.
- **Quality.** The float models match the CPU's (on a 16 MiB input, an
  H=8 C0/bit-position net scored 0.416 bits/bit at F16 against the CPU's
  0.418). The 4-bit formats (Q4, NVFP4, IQ4) are sensitive to small
  differences in training either way (Q4: CPU 0.586, GPU 0.476; NVFP4:
  CPU 0.436, GPU 0.459), so per-candidate results there can move by
  several percent; the chosen model's size stays within 0.5%.

Measured on the bench corpus (five 1 MiB files, sums; i5-9400, GT 1030;
`fit` is the fitting stage alone, per file):

| level | fitter | archive bytes | encode | fit per file |
|-------|--------|---------------|--------|--------------|
| 1 | CPU, 1 thread | 888 591 | 10.5 s | 0.20 s |
| 1 | GPU | 888 641 | 12.3 s | 0.47 s |
| 5 | CPU, 1 thread | 862 716 | 33.3 s | 4.7 s |
| 5 | CPU, 6 threads | 862 716 | 19.2 s | 1.9 s |
| 5 | GPU | 862 459 | 14.2 s | 0.9 s |
| 9 | CPU, 6 threads | 853 036 | 228.6 s | 44 s |
| 9 | GPU | 854 052 | 56.3 s | 9.3 s |

On a 16 MiB concatenation at level 9 (`-j 6`): `.aw` 1 013 299 bytes in
62.4 s on the CPU (fit 35.0 s), 1 017 315 bytes in 45.7 s with the GPU
(fit 18.4 s: bigger inputs choose 256-set selectors, whose AdamW traffic
goes to DRAM); `.awseek` 2 826 060 bytes in 39.7 s against 2 822 114 bytes
in 26.5 s.

## Weight formats

All formats are dequantized to binary32 once, when the model loads, so
they cost nothing per bit: they only trade model bytes for accuracy.
Mixer rows are short (26 or H + 1 weights), so every format keeps its
scales per row rather than ggml's 32- or 256-weight blocks, and the
"IQ" formats are plain tables (no E8 lattice or super-blocks):

- `Q4` is ggml's Q4_0 per row; `Q2` is affine (scale and offset per row,
  like Q2_K's min) because two symmetric bits have no zero.
- `FP8` is E4M3 with an F16 row scale. `NVFP4` is NVIDIA's layout
  (E2M1, an E4M3 scale per 16 weights) with an F16 row scale in place of
  the per-tensor FP32 one.
- `IQ4` uses ggml's IQ4_NL values. `IQ2` is the 4-level Lloyd-Max
  quantizer of a Gaussian. `IQ8` is `k (|k| + 128)`: steps of 129 near
  zero widen to 255 at the ends, which suits weights peaked at zero, and
  all 256 codes are valid (Q8 refuses -128).
- The quantizer (encoder only) searches the row scale on a grid around
  `amax / max level` and refines it by least squares on the assignment;
  `Q2` searches the clipped range and solves the 2x2 least squares;
  `NVFP4` tries a few row scales and, per block, the E4M3 codes around the
  ideal one. Every choice is scored with the decoder's own dequantized
  values.

On the bench corpus at level 1 (every candidate fitted, best estimate
per format), the best format is IQ8 on the word list, IQ4 on libc, Q8 on
roff, NVFP4 on text, Q4 on C source; the 4- and 8-bit formats are within
about 1.5% of each other. IQ2 loses 2-13% and Q2 up to 5% (post-training
quantization to 2 bits is too coarse for the hidden layer), except on
libc where the single-layer C1-selector models are within 0.6%.

## Decoder

1. Validate the header and refuse unsupported `(profile, quant)` pairs.
2. Load and dequantize the model once (F16C on x86 when available), and
   pick the layer-1 kernel by CPUID: AVX-512F, AVX2, or scalar.
3. Decode chunks in order (aw). For awseek, decode only the blocks that
   overlap the requested range, after resetting the history at each block.

## Determinism

Fitting may be sloppy, but inference may not. Every probability must be
bit-identical between the encoder and every decoder. The rules:

- Each arithmetic profile fixes its operation order, accumulator widths,
  rounding and squash function.
- The build disables FP contraction and fast-math
  (`-ffp-contract=off -fno-fast-math`).
- Nothing links libm: it is slow, and its results differ between libcs.
  Squash and stretch use SIMD-intrinsic or table formulas defined in the
  spec. Every SIMD path (scalar, AVX2/F16C, AVX-512) must give the scalar
  reference's exact bits.
- The probability handed to the coder is a 16-bit integer, and the
  float-to-integer conversion is defined by the profile.

## Seeking

`.awseek` resets the predictor history at every block but shares the one
model. A seek costs reading the header, model and index, plus decoding
the overlapping blocks. Blocks are independent: the encoder codes them
on `threads` workers, and a reader may decode the blocks of one request
in parallel (`taw_seek_set_threads`, `taw_decompress_mt`, `taw d -j N`).
Each decoding thread holds its own tables (about 175 MB for 1 MiB
blocks), whole blocks are decoded straight into the caller's buffer and
the partial ones at the edges through a per-thread buffer. The source's
`read_at` must then be thread safe (`pread` is).

## Source layout

| path | role |
|------|------|
| `include/tadamw/tadamw.h` | public API (`cmake/config.h.in` gives `<tadamw/config.h>`) |
| `src/api.c` | public entry points, variant dispatch |
| `src/buf.c` | growable output buffer |
| `src/crc32c.c`, `src/crc32c_sse42.c` | CRC-32C: slicing-by-8, SSE4.2 |
| `src/pool.c` | worker threads (`taw_parallel`) |
| `src/rc.c` | binary range coder (format.md §4) |
| `src/predictor.c` | arithmetic profiles: dispatch, profile 0, the APM and per-bit loop of profiles 1 and 2 |
| `src/p1.h` | internal interface and constants of profiles 1 and 2; profile 2's routing rule and chunk state |
| `src/p1_ctx.c` | context models of profiles 1 and 2 (lanes, the 16-model feature pass), squash/stretch tables |
| `src/p1_net.c` | model parsing of profiles 1 and 2, dequantization, scalar mixer, ISA choice |
| `src/p1_avx2.c` | AVX2/F16C kernels: context-model inputs/update, hashes, mixer, F16 load |
| `src/p1_run_avx2.c` | the fused AVX2 per-bit loop (encode, decode, skip) |
| `src/p1_avx512.c` | AVX-512F mixer (used by `p1_net_eval`; the per-bit loop runs the fused AVX2 code) |
| `src/p1_simd_out.h` | mixer pieces shared by the AVX2 and AVX-512 files |
| `src/half.h` | binary16 conversions |
| `src/quant.c` | weight row formats: sizes, validated decoding, the encoder's quantizer |
| `src/fit.c` | encoder: feature pass, candidates, AdamW, quantization, choice, CPU/GPU split; profile 2's routing search and lane choice |
| `src/fit_internal.h` | samples, the trainable net and the training schedule, shared by the two trainers |
| `src/fit_gpu.c` | GPU trainer: Vulkan via `dlopen`, buffers, the command buffer (a stub without `TADAMW_GPU`) |
| `src/fit_gpu_prep.comp`, `src/fit_gpu.comp` | its compute shaders: batch preparation, training |
| `src/header.c` | header (de)serialization and validation |
| `src/aw.c` | `.aw` stream encode/decode |
| `src/awseek.c` | `.awseek` encode/decode/random access |
| `tools/taw.c` | CLI |
| `tools/taw_nx.c` | XCI/NSP transcoder (docs/nx.md) |
| `tools/hactool/` | hactool as ScarletNX carries it, plus `snx_hac_encrypt` (tadamw) |
| `tests/` | CTest executables + CLI roundtrip script |

## Open items

- Profile 2 decodes up to 19% slower than profile 1 on text ("Profile 2
  in short"). The ideas left are a smaller table for the chunk lanes, so
  that their lines stay in the cache, and a lane ranking that counts what
  a cold model costs.
- Profile 2's routing search is brute force: every candidate table is a
  full pass over the sample. Changing one class or matrix bit moves only
  the chunks around the byte pairs it touches, so an incremental score
  would cut it by an order of magnitude. The search also keeps the three
  presets as its only starts.
- Profile 2 has two levels (chunks, lines). The table is a function of
  two bytes, so it cannot find fixed-length binary records, which have no
  delimiter; the sparse models still cover those.
- `predictor = auto` does not try profile 2. It would have to fit and code
  the input a second time.
- The feature pass keeps every `8n / samples`-th bit. When that stride is
  a multiple of a record length in the data (a 4 MiB file of 16-bit
  samples at level 3 keeps one byte of every 8), the samples see one byte
  phase only and the fit, of either profile, is poor: the estimate was
  0.07 bits per bit where the file codes at 2.4. Found while measuring
  profile 2; not fixed here.

- Decoder speed: about 1 MB/s per thread with AVX2 (see "Decoder
  performance"), and `.awseek` scales with threads across blocks. Ideas
  left: fewer context models where some never earn weight, AVX-512
  scatter for the write-back, shortening the slot → stretch → mixer
  chain. Evaluating several outcomes of the next bit was tried and is
  slower.
- Encoder speed: about 2.8 s/MB at level 1, 7.5 s/MB at level 5, and
  around 2 minutes/MB at level 9 on one thread. That is two context
  passes (features, then coding) plus fitting. GPU fitting (`-g on`)
  cuts level 9 by about 4x against six CPU threads (see "GPU fitting");
  the context passes are then most of what is left.
- GPU fitting on small GPUs: the trainer's 512-thread workgroups need
  45 KiB of shared memory and many registers, so a GT 1030 holds one per SM
  (256 threads per workgroup fit two, which helped four to six candidates
  but not eight). A leaner layout would let all eight level 9 candidates
  run in one wave. Sets touched by many-set selectors (C0, C1, C0C1HI)
  are not cached in shared memory, and their AdamW state goes through DRAM
  every step. Candidates with more than 16 hidden units would need the
  shaders extended; the encoder never proposes them yet.
- Low-bit weight formats and GPU fitting: see "GPU fitting", quality.
  Quantization-aware training (on either trainer) would help those formats
  more than the trainer choice does.
- Small awseek blocks are expensive under profile 1: every block resets
  about 5 MB of tables (the APM alone is 4.3 MB). The encoder could size
  tables down for small blocks, but the APM is fixed by the spec.
- Optional GPU decoding path: opt-in only, and it must reproduce the CPU
  reference bit for bit (no contraction, e.g. SPIR-V `NoContraction`, and
  the same `dot4` order). Most of the work is integer context modelling,
  which is serial per coding unit, so GPU decoding mainly makes sense
  across many awseek blocks at once.
- AVX-512 mixer: compiled and dispatched by CPUID, but the development
  machine has no AVX-512, so it has never actually run. The parity tests
  cover it automatically on a machine that has it. Its context models use
  the AVX2 kernels.
- `.aw` streams are still encoded from memory; `.awseek` has the streaming
  encoder (`taw_seek_encode`), which fits on an even sample of the whole input.
- Big-endian: the code is written for it (byte-wise I/O, CRC table path
  byte-swaps), but it has not run on a big-endian machine yet.
- taw-nx: the XCI's update, normal and logo partitions stay encrypted
  (small); personalized tickets' title keys are not decrypted, so those NCAs
  stay encrypted. The encoder writes a temporary plain image the size of the
  dump next to the output; a source that decrypts on the fly would avoid it.
  Incompressible blocks (compressed game assets) pay a full coding pass before
  being stored; an entropy pre-check could skip them.