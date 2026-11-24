# tar_aw container format, version 1

Status: draft. Normative for everything implemented (profiles 0 and 1, and
the awseek variable-length-block container extension in §7.2); reserved values
are listed so that decoders can refuse them. `tests/test_p1.c` pins profile 1
with golden vectors: a change that breaks them changes the format.

All integers are unsigned and little-endian unless stated otherwise.
`crc32c` is CRC-32C (Castagnoli, reflected polynomial `0x82F63B78`, init
`0xFFFFFFFF`, final xor `0xFFFFFFFF`).

Two variants share one header and one model section:

| extension | kind | purpose |
|-----------|------|---------|
| `.aw`     | 0    | stream: chunks are decoded strictly in order |
| `.awseek` | 1    | seekable: independent fixed-size or content-aligned blocks + seek index |

## Allocation registry

This table is the authority for arithmetic-profile IDs and awseek flag bits.
Section numbers follow the rule *section number = profile ID*; profiles
0 and 1 are fixed at §2.1/§2.2 and §5.1.

| id / bit | name | canonical section(s) | status on this base |
|----------|------|----------------------|---------------------|
| profile 0 | `INT_ORDER0` | §2.1, §5 (intro note) | implemented |
| profile 1 | `F32_STRICT_MIX` | §2.2, §5.1 | implemented, golden-pinned |
| profile 6 | `CALIBRATED_MIX` | §2.6, §5.6 | reserved; decoder refuses |
| profile 11 | `ENTROPY_GATE` | §2.11, §5.11 | reserved; decoder refuses |
| profile 14 | `HNET_BOUNDARY` | §2.14, §5.14 | reserved; decoder refuses |
| awseek flag bit 1 | variable-length blocks (container only) | §7.2 | implemented; off by default |
| profile 3, 7 | `SOFT_GATED`, `SOFT_GATED_CAL` | — | reserved; decoder refuses |
| profile 8 | `PAQ_MIX` | — | reserved; decoder refuses |
| profile 10 | `F32_CHAINED_MIX` | — | reserved; decoder refuses |
| profile 12 | `MDL_TREE` | — | reserved; decoder refuses |

Reserved IDs are refused by decoders. No quant ID is added here: quant ids
`5..11` (`Q4`, `Q2`, `FP8`, `NVFP4`, `IQ8`, `IQ4`, `IQ2`) stay exactly as
specified in §3, and every retained profile uses the profile-1 quant set.
§2.3 and §5.2 are sealed (see below).

### Predictor enum (`taw_predictor`, API only)

The public `taw_predictor` selector is an encoder-side choice and is **not**
stored in an archive: header bytes 6 and 7 carry `arith_profile` and `quant`.
Predictor values do not change the arithmetic-profile table above, which
remains the sole authority for profile IDs.

| value | enum name | profile | status |
|-------|-----------|---------|--------|
| 0 | `TAW_PREDICTOR_AUTO` | — | original value |
| 1 | `TAW_PREDICTOR_ORDER0` | 0 | legacy value |
| 2 | `TAW_PREDICTOR_MIX` | 1 | legacy value |
| 6 | `TAW_PREDICTOR_CALIBRATED` | 6 | allocated |
| 11 | `TAW_PREDICTOR_GATE` | 11 | allocated |
| 14 | `TAW_PREDICTOR_HNET` | 14 | allocated |

Values `3..5`, `7..10`, `12..13` and `15..255` are refused:
`taw_params_resolve()` returns `TAW_E_ARG` and the CLI offers no name for
them. Value 11 is the only allocated value in the nominal `7..13` range,
because predictor values equal their profile IDs. Value 3 is **void**. No
predictor value is assigned to the reserved profile IDs 3, 7, 8, 10 or 12.

## 1. File header (32 bytes)

| off | size | field          | notes |
|-----|------|----------------|-------|
| 0   | 4    | magic          | `54 41 57 1A` ("TAW\x1A") |
| 4   | 1    | version        | `1` |
| 5   | 1    | kind           | `0` = aw, `1` = awseek |
| 6   | 1    | arith_profile  | §2 |
| 7   | 1    | quant          | §3 |
| 8   | 1    | level          | encoder effort 1..9, informational |
| 9   | 1    | flags          | bit 0: the awseek index has a file directory (§7.1); bit 1: the awseek index uses variable-length blocks (§7.2); other bits `0` |
| 10  | 2    | reserved       | must be `0` |
| 12  | 4    | block_size     | awseek: raw bytes per block; aw: `0` |
| 16  | 8    | model_size     | bytes in the model section |
| 24  | 4    | model_crc      | crc32c of the model section |
| 28  | 4    | header_crc     | crc32c of bytes 0..27 |

A decoder MUST refuse a file if any of these hold: wrong magic, unknown
version or kind, a flag bit other than 0 or 1 (or either flag bit on an aw
stream), non-zero reserved bytes, bad `header_crc` or
`model_crc`, or an unsupported `(arith_profile, quant)` pair. Decoders never
fall back to a different arithmetic: an archive decodes exactly or not at all.

For awseek, `block_size` MUST be a power of two in `[2^12, 2^30]`. With flag
bit 1 set it is the *nominal* block size used for fitting and for length
bounds; actual block lengths are carried in the index (§7.2). A default
archive (flag bit 1 clear) is byte-identical to the fixed-block layout.

## 2. Arithmetic profiles

The profile fixes how the decoder turns model weights and history into
16-bit bit probabilities. It covers the inference arithmetic, rounding,
and the squash function. Two implementations of one profile MUST produce
identical probabilities for every bit.

| id | name            | model section | status |
|----|-----------------|---------------|--------|
| 0  | `INT_ORDER0`    | empty (`model_size = 0`, `quant = 0`) | implemented, placeholder |
| 1  | `F32_STRICT_MIX`| mixer shape and weights, §5.1; quant `F16`, `Q8`, `Q4`, `Q2`, `FP8`, `NVFP4`, `IQ8`, `IQ4` or `IQ2` | implemented |
| 6  | `CALIBRATED_MIX` | §5.6 | reserved; decoder refuses |
| 11 | `ENTROPY_GATE` | §5.11 | reserved; decoder refuses |
| 14 | `HNET_BOUNDARY` | §5.14 | reserved; decoder refuses |
| 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15..255 | — | — | reserved: refuse |

This is the only profile table. IDs `6`, `11` and `14` are named in the
allocation registry but are not implemented on this base, so a decoder here
refuses them. IDs `3`, `7`, `8`, `10` and `12` are reserved (see the allocation
registry) and MUST NOT be treated as free. §2.3 and §5.2 are sealed.

### 2.1 Profile 0: `INT_ORDER0`

An integer adaptive order-0 bit model. It exists so that the container,
coder and seeking can be tested before the AdamW model lands.

- State: `uint16 p[256]`, every entry `32768` at the start of each coding
  unit (the whole stream for aw, each block for awseek).
- Each byte is coded MSB first as 8 binary decisions. The context is
  `c0`, which starts at `1` and becomes `c0 = 2*c0 + bit` after each bit.
- The probability that bit = 1 is `p[c0]`.
- After coding `bit`: if `bit == 1`, `p += (65536 - p) >> 4`; otherwise
  `p -= p >> 4`. (`p` stays inside `[15, 65521]`.)

### 2.2 Profile 1: `F32_STRICT_MIX`

Adaptive integer context models turn the history into 25 stretched
probabilities. A small frozen MLP, whose shape and weights are in the
model section (§5.1), mixes them in binary32 with a fixed operation order.
An integer squash and two adaptive APM stages produce the final 16-bit
probability. Everything except the MLP is integer arithmetic. The weights
never change while coding.

Arithmetic conventions: integers wrap modulo 2^32 unless stated otherwise,
`>>` is a logical shift of a non-negative value, and products that may
exceed 32 bits are computed in 64 bits. Binary32 operations round to
nearest even, one operation at a time: no fused multiply-add and no
reassociation. (Subnormals never arise: every non-zero product and sum is
at least 2^-102 in magnitude, so flush-to-zero modes cannot change a
result.)

#### 2.2.1 Integer functions

`squash(t)` maps a stretch-domain value `t` (units of 1/256) to
`P(1) * 65536`:

```
K[0..32] = 22 36 60 98 162 267 439 720 1179 1921 3108 4971 7812 11955
           17625 24743 32768 40793 47911 53581 57724 60565 62428 63615
           64357 64816 65097 65269 65374 65438 65476 65500 65514
squash(t): t = clamp(t, -2047, 2047); u = t + 2048
           j = u >> 7; w = u & 127
           return (K[j] * (128 - w) + K[j + 1] * w + 64) >> 7
```

(`K[k]` is `65536 / (1 + e^-(k-16)/2)` rounded, but the table is the
definition.)

`stretch[0..4095]` is the inverse table, built as:

```
pi = 0
for t = -2047 .. 2047: v = squash(t) >> 4; stretch[pi..v] = t; pi = v + 1
stretch[pi..4095] = 2047
```

`R[n] = floor(2^17 / (2n + 3))` for `n = 0..1023` (an update rate of
`1/(n + 1.5)` in 16-bit fixed point).

#### 2.2.2 Coding-unit state

All of the following is reset at the start of every coding unit (the
whole stream for aw, every block for awseek), and carries across aw
chunks. `L`, `HB` and `MB` are `lines_log2`, `hist_log2` and `match_log2`
from the model head.

| state | initial value |
|-------|---------------|
| `c0` partial byte with a leading 1, `bitpos` 0..7, `nib` (§2.2.4) | `1`, `0`, `1` |
| `c4`, `c8`: the last 4 bytes and the 4 before them, newest in the low byte | `0`, `0` |
| `word` (§2.2.5), `pos` (bytes coded in this unit, mod 2^32) | `0`, `0` |
| `hist[2^HB]` byte ring, `MT[2^MB]` u32 match table | all `0` |
| `mlen`, `mptr` (§2.2.6) | `0`, `0` |
| context slots (§2.2.3) in `O0[256]`, `O1[65536]`, `T_k[2^L][16]` | `ps = 32768, h = 1, n = 0`; line tags `0` |
| history maps `HM_k[256]`, k = 0..11 | `p = 2^21, n = 0` |
| match counters `SM[16]` | as a context slot |
| APM tables `A1[256][33]`, `A2[65536][33]` (u16) | `A[c][j] = squash(128 * (j - 16))` |

`c1` below means `c4 & 0xFF`.

#### 2.2.3 Counters

A **context slot** holds `ps` (16 bits, P(1) * 65536), `h` (8 bits, the
last up to 7 bits seen in this slot behind a leading 1) and `n` (update
count). Each of the 12 context models `k` gives two inputs:

```
st[k]      = stretch[ps >> 4]                  direct estimate
st[12 + k] = stretch[HM_k[h].p >> 10]          indirect: history -> P(1)
```

After the bit `b` is known, with `r = R[n]`:

```
slot:  if b: ps += ((65535 - ps) * r) >> 16   else: ps -= (ps * r) >> 16
       h = (h >= 128) ? 128 | ((2h + b) & 127) : 2h + b
       n = min(n + 1, 30)                       (match counters: 63)
HM_k[h_old]:  p is 22 bits, n is 10 bits, r = R[n]
       if b: p += ((4194303 - p) * r) >> 16   else: p -= (p * r) >> 16
       n = min(n + 1, 1023)
```

`HM_k` is indexed with the slot's `h` from before the update.

#### 2.2.4 Context models and hashed tables

```
H(a, b, k): x = a * 0x9E3779B1 + b * 0x85EBCA77 + k * 0xC2B2AE3D
            x ^= x >> 15; x *= 0x2C1B3C6D; x ^= x >> 12
            x *= 0x297A2D39; x ^= x >> 15; return x
```

| k | model | slot |
|---|-------|------|
| 0 | o0 | `O0[c0]` |
| 1 | o1 | `O1[c1 * 256 + c0]` |
| 2..11 | hashed | `T_k[line_k][nib]` |

The hashed models have byte-level hashes `g_k`, computed after every byte
(and at the unit start):

| k | name | `g_k` |
|---|------|-------|
| 2 | o2 | `H(c4 & 0xFFFF, 0, 2)` |
| 3 | o3 | `H(c4 & 0xFFFFFF, 0, 3)` |
| 4 | o4 | `H(c4, 0, 4)` |
| 5 | o6 | `H(c4, c8 & 0xFFFF, 6)` |
| 6 | sparse 2,3 | `H((c4 >> 8) & 0xFFFF, 0, 8)` |
| 7 | word | `H(word, c1, 9)` |
| 8 | sparse 3,4 | `H(c4 & 0xFFFF0000, 0, 11)` |
| 9 | sparse 2,4 | `H(c4 & 0xFF00FF00, 0, 12)` |
| 10 | sparse 1,4 | `H(c4 & 0xFF0000FF, 0, 13)` |
| 11 | sparse 1,2,7,8 | `H(c4 & 0xFFFF, c8 & 0xFFFF0000, 14)` |

A line is 16 slots; slot 0 holds a tag, slots 1..15 serve the 15 nodes of
one nibble's bit tree. At `bitpos` 0 (with `q = 0`) and at `bitpos` 4
(with `q = c0`, then 16..31), every hashed model selects a line:

```
x = H(g_k, q, 0); line_k = x >> (32 - L); tag = (x & 0xFFFF) | 0x10000
if T_k[line_k][0] != tag: reset slots 1..15 to their initial value,
                          T_k[line_k][0] = tag
```

`nib` is `1` at `bitpos` 0 and 4 and becomes `2 * nib + b` after each bit,
so it is 1..15 within a nibble.

#### 2.2.5 Byte end

After the 8th bit of a byte `ch = c0 & 0xFF`, in this order:

1. `hist[pos mod 2^HB] = ch`; `pos += 1`; `c8 = (c8 << 8) | (c4 >> 24)`;
   `c4 = (c4 << 8) | ch`.
2. `word`: if `((ch | 32) - 97) mod 2^32 < 26` then
   `word = (word ^ (ch | 32)) * 0x01000193`, else `word = 0`.
3. Match model update (§2.2.6).
4. `c0 = 1`, `bitpos = 0`, `nib = 1`; recompute every `g_k`; select lines
   with `q = 0`; set `mexp = hist[mptr mod 2^HB]` if `mlen > 0`, otherwise
   no expected byte.

#### 2.2.6 Match model

```
if mlen > 0:
    if mexp == ch: mptr += 1; mlen = min(mlen + 1, 65535)
    else: mlen = 0
i = H(c4, c8 & 0xFF, 10) & (2^MB - 1)
if mlen == 0 and MT[i] != 0:
    e = MT[i]; d = pos - e                       (mod 2^32)
    if 1 <= d <= 2^HB - 1:
        k = 0
        while k < 32 and k < e and
              hist[(e - 1 - k) mod 2^HB] == hist[(pos - 1 - k) mod 2^HB]: k += 1
        if k >= 5: mlen = k; mptr = e
MT[i] = pos
```

For each bit, the match is **active** if there is an expected byte and
`(mexp | 256) >> (8 - bitpos) == c0`. Then:

```
msel = min(floor(log2(mlen)) - 1, 7)            (1..7)
eb   = (mexp >> (7 - bitpos)) & 1
st[24] = stretch[SM[2 * msel + eb].ps >> 4]     and that counter is updated
```

Otherwise `msel = 0`, `st[24] = 0`, and no match counter is updated.

#### 2.2.7 Selectors

A selector picks one weight set per bit:

| id | name | sets | value |
|----|------|------|-------|
| 0 | `NONE` | 1 | 0 |
| 1 | `BITPOS` | 8 | `bitpos` |
| 2 | `MATCH` | 8 | `msel` |
| 3 | `C0` | 256 | `c0` |
| 4 | `C1` | 256 | `c1` |
| 5 | `C0C1HI` | 4096 | `(c1 >> 4) * 256 + c0` |

#### 2.2.8 Mixer

Inputs: `x[i] = st[i] / 256` for `i = 0..24` (exact in binary32), and
`x[25] = 1`. `F = 26`. Dot products have a fixed shape:

```
dot4(w, v, n): s0 = s1 = s2 = s3 = +0
               for i = 0 .. n-1: s[i mod 4] = s[i mod 4] + w[i] * v[i]
               return (s0 + s1) + (s2 + s3)
```

With `H = hidden`, `W1`/`W2` the dequantized weights (§5.1), `a = sel1`'s
value and `c = sel2`'s value:

```
if H > 0:
    for j = 0 .. H-1:  h[j] = min(max(dot4(W1[a][j], x, 26), -8), 8)
    z = dot4(W2[c], h, H) + W2[c][H]
else:
    z = dot4(W2[c], x, 26)
z = min(max(z, -2047/256), 2047/256)
t = trunc(z * 256)                    (toward zero; exact, |t| <= 2047)
p_mix = squash(t)
```

The order of hidden units is free (they are independent); the order inside
each `dot4` is not.

#### 2.2.9 APM stages and the coded probability

```
s  = stretch[p_mix >> 4] + 2048;  lo = s >> 7;  w = s & 127
q1 = (A1[c0][lo] * (128 - w) + A1[c0][lo + 1] * w) >> 7
q2 = (A2[c1 * 256 + c0][lo] * (128 - w) + A2[...][lo + 1] * w) >> 7
p  = (2 * p_mix + q1 + q2) >> 2                (always in [11, 65524])
```

`p` is `P(bit = 1)` for the range coder. After the bit, `A1[c0][lo + (w >> 6)]`
and the same `A2` entry are updated: `e += (65535 - e) >> 6` if `b`, else
`e -= e >> 6`.

#### 2.2.10 Per-bit sequence

1. Compute `st[0..24]` and the selector values from the current state.
2. Mixer, APM, `p`; code the bit.
3. Update the APM entries, every context slot and its history map, and
   the match counter if the match was active. (These touch disjoint
   state, so their order does not matter.)
4. `c0 = 2 * c0 + b`, `nib = 2 * nib + b`, `bitpos += 1`. At `bitpos` 4:
   `nib = 1` and select lines with `q = c0`. At `bitpos` 8: byte end
   (§2.2.5).

### 2.3 (sealed — profile-1-only)

This subsection number is sealed. It is reserved for profile-1 material only
and MUST NOT be used to document any other arithmetic profile. Other profiles
use §2.6, §2.11 and §2.14.

### 2.6 Profile 6: `CALIBRATED_MIX`

Reserved; `arith_profile = 6` is refused. Predictor slot
`TAW_PREDICTOR_CALIBRATED = 6` uses the CLI token `-p cal`.

### 2.11 Profile 11: `ENTROPY_GATE`

Reserved; `arith_profile = 11` is refused. Predictor slot
`TAW_PREDICTOR_GATE = 11` uses the CLI token `-p gate`.

### 2.14 Profile 14: `HNET_BOUNDARY`

Reserved; `arith_profile = 14` is refused. Predictor slot
`TAW_PREDICTOR_HNET = 14` uses the CLI token `-p hnet`. Value 3 is void, and
`3..5`, `7..10` and `12..13` are refused (see the predictor enum above).

## 3. Quantization formats

The format in which model weights are stored. Encoders may pick any format
the profile allows. F16 is the standard format.

Allocation note: no quant ID is added by the registry. Ids `5..11` (`Q4`,
`Q2`, `FP8`, `NVFP4`, `IQ8`, `IQ4`, `IQ2`) and their encodings below are
unchanged on this base, and every retained profile uses the profile-1 quant
set (`F16` plus `Q8`..`IQ2`). Quant ids `12..255` stay reserved, refuse.

| id | name | encoding |
|----|------|----------|
| 0  | `NONE` | no weights |
| 1  | `F16`  | IEEE 754 binary16 |
| 2  | `BF16` | bfloat16 (upper 16 bits of binary32) |
| 3  | `F32`  | IEEE 754 binary32 |
| 4  | `Q8`   | int8 rows with one F16 scale per row (§5.1) |
| 5  | `Q4`   | 4-bit integers, one F16 scale per row |
| 6  | `Q2`   | 2-bit integers, an F16 scale and an F16 offset per row |
| 7  | `FP8`  | FP8 E4M3 values, one F16 scale per row |
| 8  | `NVFP4`| FP4 E2M1 values in blocks of 16 with an E4M3 scale per block, one F16 scale per row |
| 9  | `IQ8`  | 8-bit indices into a non-linear table, one F16 scale per row |
| 10 | `IQ4`  | 4-bit indices into a non-linear table (ggml's IQ4_NL values), one F16 scale per row |
| 11 | `IQ2`  | 2-bit indices into a non-linear table, one F16 scale per row |
| 12..255 | — | reserved, refuse |

Formats 1 and 4..11 are row formats: every row of weights (§5.1) carries
its own scales. Their exact encodings are in §5.1.

Minifloats used by them (both convert to binary32 exactly):

- **E4M3**: sign bit 7, exponent bits 6..3 (bias 7), mantissa bits 2..0.
  Exponent 0 is subnormal: `m × 2^-9`; otherwise `(1 + m/8) × 2^(e-7)`.
  No infinities; `0x7F` and `0xFF` are NaN. The largest value is 448.
- **E2M1**: a 4-bit code, sign bit 3; bits 2..0 index
  `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

Tables:

- `IQ4_TAB[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}`
- `IQ2_TAB[4] = {-127, -38, 38, 127}` (4-level Lloyd-Max for a Gaussian, scaled to 127)
- `IQ8(k) = k × (|k| + 128)` for the signed byte `k` (−32768..32385; the
  step grows from 129 near zero to 255 at the ends).

## 4. Binary range coder

Every coding unit (an aw chunk or an awseek block) is one independently
flushed range-coder segment. The coder is normative:

```
encoder state: x1 = 0, x2 = 0xFFFFFFFF            (uint32)
encode(bit, p):          p = P(bit = 1) in [1, 65535]
    xmid = x1 + (uint32)(((uint64)(x2 - x1) * p) >> 16)
    if bit: x2 = xmid  else: x1 = xmid + 1
    while ((x1 ^ x2) & 0xFF000000) == 0:
        emit(x2 >> 24); x1 <<= 8; x2 = (x2 << 8) | 0xFF
flush:  emit the 4 bytes of x1, most significant first

decoder state: x1, x2 as above; x = first 4 bytes, big-endian
decode(p):
    xmid as above; bit = (x <= xmid)
    update x1/x2 exactly as the encoder
    while ((x1 ^ x2) & 0xFF000000) == 0:
        x1 <<= 8; x2 = (x2 << 8) | 0xFF; x = (x << 8) | next_byte()
```

Reading past the end of a segment yields `0` bytes. A probability outside
`[1, 65535]` is a bug in the profile, not a property of the data.

## 5. Model section

`model_size` bytes immediately follow the header. Their layout belongs to
the arithmetic profile (profile 0: empty). The model is fitted once over the
whole input and is shared by every chunk or block.

Constraint: `model_size <= floor(raw_size / 32)`. Encoders that cannot
beat profile 0 inside that budget use profile 0.

### 5.1 Profile 1 model

An 8-byte head, then the weight payload:

| off | size | field | valid range |
|-----|------|-------|-------------|
| 0 | 1 | `lines_log2` (L) | 8..22 |
| 1 | 1 | `hist_log2` (HB) | 12..26 |
| 2 | 1 | `match_log2` (MB) | 10..24 |
| 3 | 1 | `sel1`, layer-1 selector (§2.2.7) | 0..5; `0` if `hidden = 0` |
| 4 | 1 | `sel2`, output-layer selector | 0..5 |
| 5 | 1 | `hidden` (H) | 0..32 |
| 6 | 2 | reserved | `0` |

The payload is a sequence of rows of weights:

1. Layer 1, only if `H > 0`: for each set `a` of `sel1`, for each hidden
   unit `j = 0..H-1`, one row of 26 weights `W1[a][j][0..25]` in input
   order.
2. Output layer: for each set `c` of `sel2`, one row `W2[c]` of `H + 1`
   weights (hidden units, then the bias), or of 26 weights (input order)
   if `H = 0`.

Row encodings:

- `F16`: each weight is a little-endian binary16. It must be finite.
  Converting it to binary32 is exact.
- `Q8`: a little-endian binary16 `scale` (finite), then one int8 `q` per
  weight, `q != -128`. The weight is the binary32 product
  `float(q) * float(scale)`, rounded once.

The remaining row formats all start with a little-endian binary16 scale
`d` (finite). `float(...)` is exact in every case below, and each `*` and
`+` is one binary32 operation, rounded to nearest even, in the order
written. Sub-byte codes are packed little-end first: weight `i` of a
4-bit format is bits `4(i mod 2)..` of code byte `i / 2`, of a 2-bit
format bits `2(i mod 4)..` of byte `i / 4`. Unused bits in the last code
byte MUST be zero. With `len` the row length:

| quant | after `d` | weight `i` |
|-------|-----------|------------|
| `Q4`  | `ceil(len/2)` bytes of 4-bit `q` | `float(q - 8) * d` |
| `Q2`  | binary16 offset `m` (finite), then `ceil(len/4)` bytes of 2-bit `q` | `float(q) * d + m` |
| `FP8` | `len` E4M3 bytes `q`, not NaN | `e4m3(q) * d` |
| `NVFP4` | `ceil(len/16)` E4M3 block scales `s[b]` (sign bit 0, not NaN), then `ceil(len/2)` bytes of E2M1 codes `q` | `e2m1(q) * (e4m3(s[i/16]) * d)` |
| `IQ8` | `len` bytes, signed `k` | `float(IQ8(k)) * d` |
| `IQ4` | `ceil(len/2)` bytes of 4-bit `q` | `float(IQ4_TAB[q]) * d` |
| `IQ2` | `ceil(len/4)` bytes of 2-bit `q` | `float(IQ2_TAB[q]) * d` |

Bytes per row: `Q4`/`IQ4` `2 + ceil(len/2)`; `Q2` `4 + ceil(len/4)`;
`IQ2` `2 + ceil(len/4)`; `FP8`/`IQ8` `2 + len`; `NVFP4`
`2 + ceil(len/16) + ceil(len/2)`.

`model_size` MUST equal `8` plus the payload size exactly. A decoder MUST
refuse (format error) any field out of range, non-zero reserved bytes, a
size mismatch, a non-finite F16 value, scale or offset, a Q8 value of
-128, a NaN E4M3 code, a negative NVFP4 block scale, or non-zero unused
code bits. Profile 1 accepts the row formats (`quant` 1 and 4..11);
`BF16`, `F32` and reserved values are unsupported.

A decoder needs about `10 * 2^L * 64 + 2^HB + 4 * 2^MB` bytes of tables
plus about 4.6 MB fixed (o1 table and APM), and resets them per coding
unit.

### 5.2 (sealed — profile-1-only)

This subsection number is sealed and reserved for profile-1 material; it
MUST NOT be used for any other profile's model section. Other profiles use
§5.6, §5.11 and §5.14.

### 5.6 Profile 6 model

Reserved; a profile-6 model is refused on this base.

### 5.11 Profile 11 model

Reserved; a profile-11 model is refused on this base.

### 5.14 Profile 14 model

Reserved; a profile-14 model is refused on this base. It reuses the §5.1 head
and row encodings with row width 27 instead of 26.

## 6. aw body (kind 0)

A sequence of chunks follows the model section:

```
chunk:  u32 raw_len   u32 coded_len   u32 raw_crc   coded[coded_len]
end:    u32 0         u32 0           u32 stream_crc
trailer: u64 total_raw_len
```

- `raw_crc` is crc32c of the chunk's decoded bytes, and `stream_crc` is
  crc32c of the whole decoded stream.
- `1 <= raw_len <= 2^26`.
- A chunk with `coded_len == raw_len` is **stored**: `coded` is the raw bytes.
  The predictor state still advances over them exactly as if they had been
  coded (each bit predicted and updated, no range coder). Encoders store a
  chunk whenever coding would not make it smaller, so a coded chunk is always
  shorter than its raw bytes.
- Chunks only frame the data. The predictor state carries across chunks,
  and only the range coder is restarted.
- The decoded size MUST equal `total_raw_len`.

## 7. awseek body (kind 1)

```
header | model | block[0] ... block[n-1] | index | footer
```

Block `i` holds raw bytes `[i*block_size, min((i+1)*block_size, total))`.
Each block is one range-coder segment, and the predictor state is reset at
the start of each block. That is what makes random access possible. A block
whose `coded_len` equals its raw length is **stored**: its bytes are the raw
bytes. Encoders store a block whenever coding would not make it smaller.
Blocks are independent, so they may be coded and decoded in parallel.

Index:

| size | field |
|------|-------|
| 4    | magic `54 41 57 49` ("TAWI") |
| 4    | block_count |
| 8    | total_raw_len |
| 16 × block_count¹ | entries: `u64 coded_offset` (absolute), `u32 coded_len`, `u32 raw_crc` |

¹ With flag bit 1 (§7.2) each entry is **20 bytes** and adds a `u32 raw_len`;
`block_count` is then the number of entries, not `ceil(total/block_size)`.
The table above describes the default fixed-block layout.

`block_count` MUST equal `ceil(total_raw_len / block_size)` whenever flag bit
1 (variable-length blocks, §7.2) is clear — including when flag bit 0 is set
and a file directory follows the entries (§7.1). With flag bit 0 the entries
are followed by the file directory, still inside the index and covered by
`index_crc`.

Footer (the last 16 bytes of the file):

| size | field |
|------|-------|
| 8    | index_offset (absolute) |
| 4    | index_crc (crc32c of the whole index) |
| 4    | magic `54 41 57 45` ("TAWE") |

A random-access read of `[off, off+len)` decodes only the blocks that
overlap it.

### 7.1 File directory

The directory names byte ranges of the decoded data, so one `.awseek` can hold
several files (for example a dump and its `/.meta`, docs/nx.md):

| size | field |
|------|-------|
| 4 | file_count |
| per file | `u64 offset`, `u64 size`, `u16 name_len`, `name[name_len]` |

Rules (a decoder MUST refuse a directory that breaks one):

- Names are UTF-8 paths that start with `/`, 1 to 4096 bytes, without NUL, and
  are unique.
- Files are sorted by offset and do not overlap: `offset + size` of one file
  is at most the next file's offset, and the last one ends by
  `total_raw_len`.
- The directory must end exactly where the footer starts.

Bytes outside every file are allowed but carry no name.

### 7.2 Variable-length blocks (flag bit 1)

Flag bit 1 is an opt-in container feature. A default archive never sets it and
is byte-identical to the fixed-block layout above. It changes only the
container: `.aw` and the profile 0/1 arithmetic, model section and range coder
are unchanged.

With the flag set, the index entries are 20 bytes instead of 16:

| size | field |
|------|-------|
| 8 | `u64 coded_offset` (absolute) |
| 4 | `u32 coded_len` |
| 4 | `u32 raw_crc` (crc32c of the decoded block) |
| 4 | `u32 raw_len` (decoded bytes in this block) |

`block_count` is no longer derived from `block_size`; it is the number of
entries. A decoder MUST refuse an index whose entry sum is not
`total_raw_len`, or that has an entry with `raw_len == 0`, `raw_len > 2^26`,
`coded_len > raw_len`, or an out-of-range `coded_offset`. Block `i` covers raw
bytes `[sum(raw_len[0..i-1]), sum(raw_len[0..i]))`; the same sums drive random
access, so a seek reads exactly the blocks overlapping the requested range.
`coded_len == raw_len` still means stored. `block_size` remains a power of two
and is the nominal size used for model fitting.

The widened index is covered in full by `index_crc`: the footer's `u32
index_crc` is CRC-32C over the index bytes from its magic through the last
entry (and any following file directory, §7.1), including `block_count`,
`total_raw_len` and every new `raw_len`. The footer itself is **unchanged**:
the last 16 bytes are still `u64 index_offset`, `u32 index_crc`, and the magic
`TAWE` as in §7, and they remain outside the CRC. CRCs are checked before any
flag-dependent field is parsed, and a valid CRC does not make an out-of-range
`raw_len`, `coded_len` or `coded_offset` legal.

**Encoder boundary rule (deterministic).** A first pass computes, for each byte
`x` with predecessor `c`, an order-1 surprise
`256 - (((count[c][x]+1)<<16)/(total[c]+256) >> 8)`, where the saturating
`uint16_t` counts are updated strictly in input order. Only deviations of that
surprise from its EWMA (1/256), at least 8 on the 0..256 scale, count as
*change*. A second pass places a boundary whenever the accumulated change
reaches an equal share of the total (`total_change / ceil(total/target)`), or
at a `change_threshold` spike when that knob is set, subject to
`min_len = target/4` and `max_len = min(4*target, 2^26)`. Both passes read the
source sequentially, so the archive does not depend on `threads`. `target`
defaults to `block_size`; the CLI switches are `taw c -D -T N -C N` and
`taw-nx enc -D -T N -C N`.

A decoder that implements only the fixed-block layout of §7 MUST refuse flag
bit 1 with `TAW_E_UNSUPPORTED` rather than misreading the index.
