/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 1, F32_STRICT_MIX (docs/format.md §2.2, §5.1).
 *
 * Adaptive integer context models produce stretched bit probabilities; a
 * small frozen MLP (weights fitted offline by the encoder) mixes them into
 * one 16-bit probability. Everything here is shared by the decoder, the
 * encoder's coding pass, and the encoder's feature extraction for fitting.
 */
#ifndef TAW_P1_H
#define TAW_P1_H

#include <stddef.h>
#include <stdint.h>

#include "tadamw/tadamw.h"

/* ---- shape constants (normative) ---------------------------------------- */

enum {
	P1_NCM = 12,     /* context models (each gives two inputs) */
	P1_NCTX = 25,    /* context-model inputs: 2 per model + match */
	P1_F = 26,       /* mixer inputs in profile 1: P1_NCTX + bias */
	P1_F_HNET = 27,  /* profile 14: profile 1 plus the boundary input */
	P1_FMAX = 27,    /* widest mixer row built here */
	P1_NHASH = 10,   /* hashed context models (all but o0, o1) */
	P1_HMAX = 32,    /* max hidden units */
	P1_MODEL_HEAD = 8,
	P1_MINLEN = 5,   /* match model: minimum verified length */
	P1_LIMIT_SLOW = 30,   /* context slot count limit */
	P1_LIMIT_HMAP = 1023, /* history map count limit */
	P1_MAXVERIFY = 32,
};

/* Arithmetic profile 6 (CALIBRATED_MIX): the profile-1 mixer followed by a
 * handful of fitted integer calibration parameters. The 8-byte calibration
 * extension sits between the profile-1 head and the weight payload.
 *
 *   off 8..9:   temp (u16 LE, Q8 multiplier, 64..512)
 *   off 10..11: bias (i16 LE, stretch units, -1024..1024)
 *   off 12..14: w0, w1, w2 (u8, 0..15, sum != 0) APM mix weights
 *   off 15:     reserved, must be 0
 */
enum {
	P6_CAL_EXT = 8,
	P6_MODEL_HEAD = P1_MODEL_HEAD + P6_CAL_EXT,
	P6_TEMP_MIN = 64, P6_TEMP_MAX = 512,
	P6_BIAS_MAX = 1024,
	P6_W_MAX = 15,
};

typedef struct {
	uint16_t temp;  /* Q8 multiplier of the stretch-domain logit */
	int16_t bias;   /* stretch units added after scaling */
	uint8_t w0, w1, w2; /* weights of p_mix, APM1, APM2 */
} p6_cal;

/* The identity calibration: profile 1's arithmetic exactly. */
#define P6_CAL_IDENTITY ((p6_cal){ .temp = 256, .bias = 0, .w0 = 2, .w1 = 1, .w2 = 1 })

/* Context models; model k gives inputs k (direct) and P1_NCM + k (history).
 * The S models are sparse: S23 = bytes 2 and 3 back, and so on. */
enum {
	P1_CM_O0, P1_CM_O1, P1_CM_O2, P1_CM_O3, P1_CM_O4, P1_CM_O6,
	P1_CM_S23, P1_CM_WORD, P1_CM_S34, P1_CM_S24, P1_CM_S14, P1_CM_S1278
};
enum { P1_IN_MATCH = 2 * P1_NCM, P1_IN_BIAS };

/* Weight-set selectors. */
enum {
	P1_SEL_NONE,   /* 1 set */
	P1_SEL_BITPOS, /* 8: bit index within the byte, 0 = MSB */
	P1_SEL_MATCH,  /* 8: match state, 0 = no active match */
	P1_SEL_C0,     /* 256: partial byte with leading 1 */
	P1_SEL_C1,     /* 256: previous byte */
	P1_SEL_C0C1HI, /* 4096: (c1 >> 4) << 8 | c0 */
	P1_SEL_COUNT
};
extern const uint16_t p1_sel_size[P1_SEL_COUNT];

/* Context slots and history-map entries are stored XOR these, so that
 * zeroed memory is the initial state (p1_ctx.c). */
#define P1_SLOT_INIT 0x80000100u
#define P1_HM_INIT 0x80000000u

/* Geometry limits a decoder accepts. */
enum {
	P1_LINES_MIN = 8, P1_LINES_MAX = 22,
	P1_HIST_MIN = 12, P1_HIST_MAX = 26,
	P1_MATCH_MIN = 10, P1_MATCH_MAX = 24,
};

typedef struct {
	uint8_t lines_log2; /* hashed-table lines per context, log2 */
	uint8_t hist_log2;  /* match history ring, log2 bytes */
	uint8_t match_log2; /* match hash table entries, log2 */
	uint8_t sel1, sel2; /* selectors for layer 1 / output layer */
	uint8_t hidden;     /* hidden units, 0..P1_HMAX (0 = single layer) */
} p1_shape;

/* ---- squash / stretch (integer, normative) ------------------------------ */

/* t in stretch units (1/256), clamped to [-2047, 2047] -> P(1) * 65536. */
uint32_t p1_squash(int32_t t);
/* 12-bit probability -> stretch units, [-2047, 2047]. */
extern int16_t p1_stretch_tab[4096];
/* The same as int32 (for gathers), and the counter rate table R[n]. */
extern int32_t p1_stretch32[4096];
extern uint32_t p1_rate32[1024];
void p1_tables_init(void);

/* ---- context models ----------------------------------------------------- */

typedef struct {
	uint8_t bitpos; /* 0..7 */
	uint8_t c0;     /* 1..255 */
	uint8_t c1;
	uint8_t msel;   /* 0..7 */
} p1_sel_ctx;

typedef struct {
	/* hot per-bit state, parallel arrays padded to 16 lanes */
	alignas(32) uint32_t val[16]; /* slot values read by inputs */
	alignas(32) uint32_t hmi[16]; /* history-map index per model */
	alignas(32) uint32_t hmv[16]; /* history-map entries read by inputs */
	alignas(32) uint32_t si[16];  /* arena index of each model's slot */
	alignas(32) uint32_t lb[16];  /* arena index of each hashed model's line */
	uint32_t c0, bitpos, nib, c4, c8, word, pos;
	uint32_t mptr, mlen, msi;
	int mexp;   /* expected byte, or -1 */
	int mbit;   /* expected bit for the current bit, or -1 */
	bool avx2;  /* use the AVX2 context kernels */
	/* geometry */
	unsigned lines_log2;
	uint32_t hist_mask, match_mask;
	/* tables; slots are stored XOR P1_SLOT_INIT so that zero = initial */
	uint32_t *arena; /* [O0 256][O1 65536][T_k: 2^L x 16] */
	size_t arena_n;
	uint32_t *mtab, sm[16];
	uint8_t *hist;
	alignas(32) uint32_t hmap[P1_NCM * 256]; /* history maps, [k][h] */
	alignas(32) uint32_t hash[16]; /* byte hashes g_k, padded */
	/* One bit before a nibble or byte boundary, the hashes of both
	 * outcomes are computed and their lines prefetched; the boundary
	 * then picks one (argument reuse instead of recomputation). */
	alignas(32) uint32_t spec_g[2][16];  /* byte hashes after the byte */
	alignas(32) uint32_t spec_hh[2][16]; /* line hashes at the boundary */
	uint32_t spec_mi[2];                 /* match-table index */
	/* Line tags are (hash & 0xFFFF) | epoch << 16; a reset starts a new
	 * epoch instead of clearing the hashed tables. */
	uint32_t epoch;
	bool fresh; /* the arena is still all zero */
} p1_ctx;
enum { P1_EPOCH_MAX = 0x7FFF };

taw_status p1_ctx_init(p1_ctx *c, const p1_shape *s);
void p1_ctx_reset(p1_ctx *c);
void p1_ctx_free(p1_ctx *c);
/* Mixer inputs x[0..P1_NCTX-1] = stretch / 256 (exact) and the selector
 * context for the next bit. x[P1_IN_BIAS] is left to the caller. */
void p1_ctx_inputs(p1_ctx *c, float *x, p1_sel_ctx *sc);
void p1_ctx_update(p1_ctx *c, int bit);

/* AVX2 kernels for the 12 context models (p1_avx2.c); identical results
 * to the scalar code in p1_ctx.c. */
void p1_cm_inputs_avx2(p1_ctx *c, float *x);
void p1_cm_update_avx2(p1_ctx *c, int bit);
/* Byte hashes g[0..9] from the history; line hashes hh = H(g, q, 0).
 * g and hh are 16 aligned lanes. */
void p1_byte_hashes_avx2(uint32_t c4, uint32_t c8, uint32_t word,
    uint32_t *g);
void p1_line_hashes_avx2(const uint32_t *g, uint32_t q, uint32_t *hh);
/* Line selection for hh (tag check, reset on mismatch) and prefetch. */
void p1_lines_apply_avx2(p1_ctx *c, const uint32_t *hh);
void p1_lines_prefetch_avx2(const p1_ctx *c, const uint32_t *hh);
/* Arena index of the first hashed table (after O0 and O1). */
enum { P1_ARENA_HX = 256 + 65536 };

static inline uint32_t p1_sel_value(unsigned sel, const p1_sel_ctx *sc)
{
	switch (sel) {
	case P1_SEL_BITPOS: return sc->bitpos;
	case P1_SEL_MATCH: return sc->msel;
	case P1_SEL_C0: return sc->c0;
	case P1_SEL_C1: return sc->c1;
	case P1_SEL_C0C1HI: return (uint32_t)(sc->c1 >> 4) << 8 | sc->c0;
	default: return 0;
	}
}

/* ---- the frozen network -------------------------------------------------- */

/* Mixer input vectors are P1_XPAD floats: x[0..f-1] then zeros, so that
 * SIMD kernels may read whole vectors. f is P1_F (profile 1) or P1_F_HNET
 * (profile 14). */
enum { P1_XPAD = 32 };

typedef struct p1_net p1_net;
struct p1_net {
	p1_shape s;
	unsigned f;    /* mixer inputs: P1_F or P1_F_HNET */
	unsigned hpad; /* hidden rounded up to 8 */
	unsigned w2s;  /* output-layer row stride: hpad, or P1_XPAD if H = 0 */
	float *w1;     /* [S1][f][hpad], transposed, zero padded */
	float *w2;     /* [S2][w2s] zero padded; no bias when H > 0 */
	float *b2;     /* [S2] output bias when H > 0 */
	/* Calibration (profile 6). Profile 1 nets get the identity values. */
	p6_cal cal;
	/* z before the bias: dot4 over the output layer (format.md §2.2.8) */
	float (*mix)(const p1_net *n, const float *x, const float *w1,
	    const float *w2);
};

/* Bytes the weight payload needs for a shape, quant format and row width. */
uint64_t p1_weights_size_ex(const p1_shape *s, uint8_t quant, unsigned f);
static inline uint64_t p1_weights_size(const p1_shape *s, uint8_t quant)
{
	return p1_weights_size_ex(s, quant, P1_F);
}
bool p1_quant_supported(uint8_t quant);

/* Parse a model section (header + weights) and dequantize it. The _ex form
 * chooses the mixer width; profile 1 uses p1_net_load (f = P1_F). */
taw_status p1_net_load_ex(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size, unsigned f);
static inline taw_status p1_net_load(p1_net *n, uint8_t quant,
    const uint8_t *model, size_t size)
{
	return p1_net_load_ex(n, quant, model, size, P1_F);
}
void p1_net_free(p1_net *n);
/* Stretch-domain output, [-2047, 2047]. x has P1_XPAD entries,
 * x[P1_IN_BIAS] = 1 and x[f..] = 0, where f = n->f (26 or 27). */
int32_t p1_net_eval(const p1_net *n, const float *x, const p1_sel_ctx *sc);

/* ---- profile 6: learned probability calibration --------------------------- */

/* Apply the fitted temperature and bias to stretch-domain t (integer only).
 * The result is clamped to [-2047, 2047] and can be passed to p1_squash. */
int32_t p6_cal_t(const p6_cal *cal, int32_t t);
/* Size of a profile 6 model section (head + extension + weights). */
uint64_t p6_model_size(const p1_shape *s, uint8_t quant);
/* Write / parse the 16-byte profile 6 head (the first 8 bytes are the
 * profile 1 head, so p1_head_parse works on them unchanged). */
void p6_head_write(uint8_t out[P6_MODEL_HEAD], const p1_shape *s,
    const p6_cal *cal);
taw_status p6_head_parse(const uint8_t *p, size_t n, p1_shape *s,
    p6_cal *cal);
/* Load a profile 6 model section. The weight payload uses the profile 1
 * encoding; the calibration extension is parsed before the weights. */
taw_status p6_net_load(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size);

/* The mixer kernels; all produce identical bits. */
float p1_mix_scalar(const p1_net *n, const float *x, const float *w1,
    const float *w2);
float p1_mix_avx2(const p1_net *n, const float *x, const float *w1,
    const float *w2);
float p1_mix_avx512(const p1_net *n, const float *x, const float *w1,
    const float *w2);
void p1_f16_to_f32_f16c(const uint16_t *src, float *dst, size_t n);

typedef enum { P1_ISA_SCALAR, P1_ISA_AVX2, P1_ISA_AVX512, P1_ISA_COUNT }
    p1_isa;
/* Best ISA this CPU and build support. */
p1_isa p1_isa_best(void);
bool p1_isa_available(p1_isa isa);
/* Testing: pin the kernels used by later p1_net_load calls (-1 = auto). */
void p1_isa_force(int isa);
/* The ISA that kernels picked now should use (forced or best). */
p1_isa p1_isa_current(void);

/* ---- profile 11: entropy-adaptive confidence gate ------------------------
 * Profile 11 is profile 1's context models, frozen MLP and APM stages plus a
 * four-byte gate appended to the model section (docs/format.md §2.11, §5.11).
 * The gate interpolates the MLP's probability with the raw mixture of the 25
 * stretched context estimates, with a weight driven by their disagreement.
 * All of it is integer arithmetic; fit.c and the decoder share these helpers.
 */

enum {
	P11_GATE_SIZE = 4,
	P11_GATE_SHIFT = 6,   /* fixed: gate_b * mad is floor-divided by 2^6 */
	P11_GATE_A_MIN = -256, P11_GATE_A_MAX = 512,
	P11_GATE_B_MIN = -2048, P11_GATE_B_MAX = 2048,
};

typedef struct {
	int16_t a, b;
} p11_gate;

/* Confidence features from the 25 stretched inputs (st[i] in 1/256 units):
 * mean = round-to-nearest(st[i] sum / 25), mad = mean |st[i] - mean|. */
void p11_features(const int16_t st[P1_NCTX], int32_t *mean, int32_t *mad);
/* p = ((256 - w) * p_mlp + w * p_ctx + 128) >> 8, w from the gate and mad. */
uint32_t p11_gate_p(uint32_t p_mlp, uint32_t p_ctx, int32_t mad,
    const p11_gate *g);

/* Parse/write the four-byte little-endian gate tail (signed int16 a, b). */
taw_status p11_gate_parse(const uint8_t *p, size_t n, p11_gate *g);
void p11_gate_write(uint8_t out[P11_GATE_SIZE], const p11_gate *g);

/* ---- the per-bit loop ------------------------------------------------------- */

/* Profile 1 state. After the mixer, two adaptive APM (SSE) stages refine
 * the probability in the contexts c0 and c1:c0 (format.md §2.2.9).
 * With `gated` set (profile 11) both the generic loop in predictor.c and the
 * fused AVX2 loop insert the entropy gate; the golden vectors pin them. */
enum { P1_APM_RATE = 6 };
struct taw_p1 {
	p1_ctx ctx;
	p1_net net;
	uint16_t apm1[256 * 33];
	uint16_t *apm2;    /* [65536 * 33] */
	uint32_t a1, a2;   /* entries to update after the bit */
	bool calibrated;  /* profile 6: apply cal and learned APM weights */
	p11_gate gate;    /* profile 11 only */
	bool gated;       /* profile 11: interpolate MLP with raw mixture */
};

/* After the context-model counters of a bit are updated: the match
 * counter, the partial byte, and the nibble/byte boundaries. */
void p1_ctx_tail(p1_ctx *c, int bit);
/* Bits after which p1_ctx_tail has boundary work (bitpos 3, 4, 7, 8):
 * speculation for the next nibble, its lines, speculation for the next
 * byte, the byte's end. */
#define P1_BOUNDARY_BITS 0x198u
void p1_ctx_boundary(p1_ctx *c, int bit);
extern const uint16_t p1_squash_knot[33];

/* ---- APM stages (shared by profile 1 and profile 14) --------------------- */

/* Tables and the one-entry update of format.md §2.2.9. Keeping one copy
 * guarantees the two profiles refine probabilities identically. */
void p1_apm_reset(uint16_t apm1[256 * 33], uint16_t *apm2);
static inline uint32_t p1_apm_p(const uint16_t *t, size_t row, int32_t s12,
    uint32_t *upd)
{
	uint32_t u = (uint32_t)(s12 + 2048);
	uint32_t lo = u >> 7, w = u & 127;
	const uint16_t *r = t + row * 33;
	*upd = (uint32_t)(row * 33) + lo + (w >> 6);
	return (r[lo] * (128 - w) + r[lo + 1] * w) >> 7;
}
static inline void p1_apm_update(uint16_t *e, int bit)
{
	*e = bit ? (uint16_t)(*e + ((65535u - *e) >> P1_APM_RATE))
		 : (uint16_t)(*e - (*e >> P1_APM_RATE));
}

/* The whole per-bit loop with the AVX2 kernels fused (p1_run_avx2.c):
 * the same bits as the generic loop in predictor.c. */
void p1_encode_avx2(struct taw_p1 *m, void *enc, const uint8_t *src,
    size_t n);
void p1_decode_avx2(struct taw_p1 *m, void *dec, uint8_t *dst, size_t n);
void p1_skip_avx2(struct taw_p1 *m, const uint8_t *src, size_t n);

/* ---- serialization (encoder) -------------------------------------------- */

/* Write the 8-byte model head. */
void p1_head_write(uint8_t out[P1_MODEL_HEAD], const p1_shape *s);
taw_status p1_head_parse(const uint8_t *p, size_t n, p1_shape *s);

/* ---- encoder: fitting and model choice (fit.c) --------------------------- */

typedef struct {
	uint8_t profile, quant;
	uint8_t *model;
	size_t model_size;
} taw_model_choice;

/* Choose a profile and fit a model for src. unit = coding-unit size (n for
 * .aw, block_size for .awseek). Falls back to profile 0 when profile 1 is
 * not possible inside the budget. */
taw_status taw_model_fit(const uint8_t *src, size_t n, size_t unit,
    const taw_params *p, taw_model_choice *out);
/* The same over any source (read_at may be called from several threads),
 * with params->threads workers; the result does not depend on them. */
taw_status taw_model_fit_source(const taw_source *in, uint64_t unit,
    const taw_params *p, taw_model_choice *out);

#endif
