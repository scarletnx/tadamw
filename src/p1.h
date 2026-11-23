/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profiles 1, F32_STRICT_MIX (docs/format.md §2.2, §5.1), and 2,
 * F32_STRICT_CHUNK (§2.3, §5.2).
 *
 * Adaptive integer context models produce stretched bit probabilities; a
 * small frozen MLP (weights fitted offline by the encoder) mixes them into
 * one 16-bit probability. Profile 2 is profile 1 with a chunk stage: a
 * routing table fitted to the input cuts the bytes into chunks and lines,
 * four more context models are keyed by them, and the model names which
 * ten of the fourteen hashed models run. Everything here is shared by the
 * decoder, the encoder's coding pass, and the encoder's feature extraction
 * for fitting.
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
	P1_F = 26,       /* mixer inputs: P1_NCTX + bias */
	P1_NHASH = 10,   /* hashed context models (all but o0, o1) */
	P2_NCAND = 14,   /* profile 2: the hashed models P1_NHASH are chosen
	                    from: profile 1's ten and four of the chunk stage */
	P2_NCLASS = 16,  /* byte classes of its routing table */
	P2_ROUTE_BYTES = 128 + 4 * P2_NCLASS, /* the table's size in the model */
	P2_LINE_MAX = 128, /* chunks of a line that are remembered */
	P1_NCM_ALL = 2 + P2_NCAND,      /* the encoder runs every candidate */
	P1_NCTX_ALL = 2 * P1_NCM_ALL + 1,
	P1_HMAX = 32,    /* max hidden units */
	P1_MODEL_HEAD = 8,
	P1_MINLEN = 5,   /* match model: minimum verified length */
	P1_LIMIT_SLOW = 30,   /* context slot count limit */
	P1_LIMIT_HMAP = 1023, /* history map count limit */
	P1_MAXVERIFY = 32,
};

/* Context models; model k gives inputs k (direct) and P1_NCM + k (history).
 * The S models are sparse: S23 = bytes 2 and 3 back, and so on. */
enum {
	P1_CM_O0, P1_CM_O1, P1_CM_O2, P1_CM_O3, P1_CM_O4, P1_CM_O6,
	P1_CM_S23, P1_CM_WORD, P1_CM_S34, P1_CM_S24, P1_CM_S14, P1_CM_S1278
};
enum { P1_IN_MATCH = 2 * P1_NCM, P1_IN_BIAS };

/* Hashed context models by id (format.md §2.3.3): profile 1's are 0..9 in
 * its lane order, the chunk stage's follow. Each is the chunk so far with
 * something else: PREV the chunk before it, COL its place in the line,
 * SKIP the chunk two back, ABOVE the chunk at the same place in the line
 * before. */
enum {
	P2_CX_O2, P2_CX_O3, P2_CX_O4, P2_CX_O6, P2_CX_S23, P2_CX_WORD,
	P2_CX_S34, P2_CX_S24, P2_CX_S14, P2_CX_S1278,
	P2_CX_PREV, P2_CX_COL, P2_CX_SKIP, P2_CX_ABOVE
};
static_assert(P2_CX_ABOVE + 1 == P2_NCAND, "context ids");

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

/* Profile 2's routing table (format.md §2.3.1): byte b starts a new chunk
 * after byte a when bit cls[b] of row[cls[a]] is set, and that chunk also
 * starts a new line when the same bit of row2 is set. */
typedef struct {
	uint8_t cls[256];           /* byte -> class, 0..P2_NCLASS-1 */
	uint16_t row[P2_NCLASS];
	uint16_t row2[P2_NCLASS];
} p2_route;

/* 0 no boundary, 1 a chunk boundary, 3 one that also starts a line. */
static inline unsigned p2_boundary(const p2_route *r, uint32_t a, uint32_t b)
{
	const unsigned ca = r->cls[a], cb = r->cls[b];
	const unsigned b1 = r->row[ca] >> cb & 1;
	return b1 | (b1 & (r->row2[ca] >> cb)) << 1;
}

/* Profile 2's chunk state (format.md §2.3.2). */
typedef struct {
	uint32_t ck[3]; /* hashes of the chunk so far, the chunk before it
	                   and the one before that */
	uint32_t idx;   /* chunks closed in this line, at most P2_LINE_MAX */
	uint32_t cl;    /* which of line[2] is the current line */
	uint32_t ab;    /* the chunk at idx in the line before, or 0 */
	bool open;      /* a byte was seen: false only at a unit's start */
	bool cut;       /* the last byte closed a chunk */
} p2_chunk;

/* The chunk state after byte ch, which follows byte prev: a boundary
 * closes the chunk before ch joins the new one (the first byte of a unit
 * has no chunk to close). line holds the closed chunks of the current
 * line (line[k->cl]) and of the one before; the chunk closed here is
 * entered by p2_chunk_commit. */
static inline void p2_chunk_next(const p2_route *r, const p2_chunk *k,
    const uint32_t line[2][P2_LINE_MAX], uint32_t prev, uint32_t ch,
    p2_chunk *out)
{
	*out = *k;
	const unsigned b = k->open ? p2_boundary(r, prev, ch) : 0;
	out->open = true;
	out->cut = b != 0;
	if (b) {
		out->ck[2] = k->ck[1];
		out->ck[1] = k->ck[0];
		out->ck[0] = 0;
		if (b & 2) {
			/* The line that closes is the one before from now on;
			 * its first chunk may be the one closing with it. */
			out->cl = k->cl ^ 1;
			out->idx = 0;
			out->ab = k->idx == 0 ? k->ck[0] : line[k->cl][0];
		} else {
			out->idx = k->idx < P2_LINE_MAX ? k->idx + 1 : P2_LINE_MAX;
			out->ab = out->idx < P2_LINE_MAX ?
			    line[k->cl ^ 1][out->idx] : 0;
		}
	}
	out->ck[0] = (out->ck[0] ^ (ch + 1)) * 0x01000193u;
}

/* Make *next the chunk state; a chunk it closed goes into its line. */
static inline void p2_chunk_commit(p2_chunk *k,
    uint32_t line[2][P2_LINE_MAX], const p2_chunk *next)
{
	if (next->cut && k->idx < P2_LINE_MAX)
		line[k->cl][k->idx] = k->ck[0];
	*k = *next;
}

/* The hash arguments (a, b) of the four chunk contexts, in id order; their
 * byte hashes are H(a, b, 15 + j) (format.md §2.3.3). */
static inline void p2_chunk_args(const p2_chunk *k, uint32_t ab[8])
{
	const uint32_t col = (k->idx + 1) * 0x9E3779B1u;
	ab[0] = k->ck[0];
	ab[1] = k->ck[1];
	ab[2] = k->ck[0];
	ab[3] = col;
	ab[4] = k->ck[0];
	ab[5] = k->ck[2];
	ab[6] = k->ck[0];
	ab[7] = k->ab + col;
}

/* H(a, b, k) of format.md §2.2.4. */
static inline uint32_t p1_hash3(uint32_t a, uint32_t b, uint32_t k)
{
	uint32_t x = a * 0x9E3779B1u + b * 0x85EBCA77u + k * 0xC2B2AE3Du;
	x ^= x >> 15;
	x *= 0x2C1B3C6Du;
	x ^= x >> 12;
	x *= 0x297A2D39u;
	x ^= x >> 15;
	return x;
}

typedef struct {
	uint8_t lines_log2; /* hashed-table lines per context, log2 */
	uint8_t hist_log2;  /* match history ring, log2 bytes */
	uint8_t match_log2; /* match hash table entries, log2 */
	uint8_t sel1, sel2; /* selectors for layer 1 / output layer */
	uint8_t hidden;     /* hidden units, 0..P1_HMAX (0 = single layer) */
	/* Profile 2 only (all zero in profile 1): */
	bool chunk;         /* the model section is profile 2's */
	bool all;           /* encoder: run all P2_NCAND hashed models, in id
	                       order (lane is not used) */
	uint8_t lane[P1_NHASH]; /* the context id of each hashed lane,
	                           increasing */
	p2_route route;     /* used when p2_shape_routes() */
} p1_shape;

/* Whether a profile 2 shape runs the chunk stage: any of its lanes holds
 * a chunk context (then its model section carries the routing table). */
static inline bool p2_shape_routes(const p1_shape *s)
{
	return s->chunk && (s->all || s->lane[P1_NHASH - 1] >= P2_CX_PREV);
}

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
	uint32_t *arena; /* [O0 256][O1 65536][T_k: 2^L x 16], k = 2..ncm-1 */
	size_t arena_n;
	uint32_t *mtab, sm[16];
	uint8_t *hist;
	alignas(32) uint32_t hmap[16 * 256]; /* history maps, [k][h] */
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
	/* What profile 2 and the encoder's feature pass add comes after
	 * everything above, whose places the per-bit loop was measured
	 * with: moving them cost profile 1 up to 5% of its decoding speed. */
	unsigned ncm, nhash; /* context models (P1_NCM, or P1_NCM_ALL in the
	                        encoder's feature pass); the hashed ones */
	bool p2;     /* profile 2 (a decoder, or the feature pass) */
	bool routes; /* profile 2 with the chunk stage running */
	bool picks;  /* profile 2: hashed lane k runs context lane[k] */
	p2_chunk ch;                   /* the chunk stage */
	p2_chunk spec_ch[2];           /* its state after the byte, both ways */
	/* lane k of the picked hashes is lane pick_a[k] of the first eight
	 * candidates, or lane pick_b[k] of the others where pick_s[k] is set */
	alignas(32) int32_t pick_a[16], pick_b[16], pick_s[16];
	uint8_t lane[P1_NHASH];        /* context id per hashed lane */
	p2_route route;
	uint32_t line[2][P2_LINE_MAX]; /* closed chunks, by line */
} p1_ctx;
enum { P1_EPOCH_MAX = 0x7FFF };

taw_status p1_ctx_init(p1_ctx *c, const p1_shape *s);
void p1_ctx_reset(p1_ctx *c);
void p1_ctx_free(p1_ctx *c);
/* Mixer inputs x[0..2 ncm] = stretch / 256 (exact): model k's direct
 * estimate at k, its history estimate at ncm + k, the match model at
 * 2 ncm; and the selector context for the next bit. x has P1_XPAD
 * entries (P1_XALL with P1_NCM_ALL models); the bias x[2 ncm + 1] and the
 * padding after it are left to the caller (the kernels may leave anything
 * there). */
void p1_ctx_inputs(p1_ctx *c, float *x, p1_sel_ctx *sc);
void p1_ctx_update(p1_ctx *c, int bit);

/* AVX2 kernels for the context models (p1_avx2.c); identical results
 * to the scalar code in p1_ctx.c. */
void p1_cm_inputs_avx2(p1_ctx *c, float *x);
void p1_cm_update_avx2(p1_ctx *c, int bit);
/* Byte hashes g[0..9] from the history; line hashes hh = H(g, q, 0).
 * g and hh are 16 aligned lanes. */
void p1_byte_hashes_avx2(uint32_t c4, uint32_t c8, uint32_t word,
    uint32_t *g);
/* Profile 2: the byte hashes of all fourteen candidates (g[10..13] =
 * H(ab[2j], ab[2j + 1], 15 + j) for the chunk contexts), or, when c picks
 * lanes, of the context in each of its ten lanes. */
void p2_byte_hashes_avx2(const p1_ctx *c, uint32_t c4, uint32_t c8,
    uint32_t word, const uint32_t ab[8], uint32_t *g);
void p1_line_hashes_avx2(const uint32_t *g, uint32_t q, uint32_t *hh);
/* Line selection for hh (tag check, reset on mismatch) and prefetch, for
 * the 10 hashed models of a decoder (the encoder's 16-model feature pass
 * uses the scalar ones). */
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

/* Mixer input vectors are P1_XPAD floats: x[0..P1_F-1] then zeros, so
 * that SIMD kernels may read whole vectors. p1_ctx_inputs needs P1_XALL:
 * with P1_NCM_ALL models it writes P1_NCTX_ALL inputs. */
enum { P1_XPAD = 32, P1_XALL = 40 };

typedef struct p1_net p1_net;
struct p1_net {
	p1_shape s;
	unsigned hpad; /* hidden rounded up to 8 */
	unsigned w2s;  /* output-layer row stride: hpad, or P1_XPAD if H = 0 */
	float *w1;     /* [S1][P1_F][hpad], transposed, zero padded */
	float *w2;     /* [S2][w2s] zero padded; no bias when H > 0 */
	float *b2;     /* [S2] output bias when H > 0 */
	/* z before the bias: dot4 over the output layer (format.md §2.2.8) */
	float (*mix)(const p1_net *n, const float *x, const float *w1,
	    const float *w2);
};

/* Bytes the weight payload needs for a shape and quant format. */
uint64_t p1_weights_size(const p1_shape *s, uint8_t quant);
bool p1_quant_supported(uint8_t quant);

/* Parse a model section (head, profile 2's lanes and routing table,
 * weights) and dequantize it. chunk: the section is profile 2's. */
taw_status p1_net_load(p1_net *n, bool chunk, uint8_t quant,
    const uint8_t *model, size_t size);
void p1_net_free(p1_net *n);
/* Stretch-domain output, [-2047, 2047]. x has P1_XPAD entries,
 * x[P1_IN_BIAS] = 1 and x[P1_F..] = 0. */
int32_t p1_net_eval(const p1_net *n, const float *x, const p1_sel_ctx *sc);

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

/* ---- the per-bit loop ------------------------------------------------------- */

/* Profile 1 state. After the mixer, two adaptive APM (SSE) stages refine
 * the probability in the contexts c0 and c1:c0 (format.md §2.2.9). */
enum { P1_APM_RATE = 6 };
struct taw_p1 {
	p1_ctx ctx;
	p1_net net;
	uint16_t apm1[256 * 33];
	uint16_t *apm2;    /* [65536 * 33] */
	uint32_t a1, a2;   /* entries to update after the bit */
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
/* Bytes of a model section before the weights: the head and, in profile
 * 2, the lanes and the routing table when a lane needs it (format.md
 * §5.2). */
enum { P2_HEAD_MAX = P1_MODEL_HEAD + P1_NHASH + P2_ROUTE_BYTES };
static inline size_t p1_head_size(const p1_shape *s)
{
	return !s->chunk ? P1_MODEL_HEAD : P1_MODEL_HEAD + P1_NHASH +
	    (p2_shape_routes(s) ? P2_ROUTE_BYTES : 0);
}
/* Write a profile 2 section's head (p1_head_size bytes). */
void p2_head_write(uint8_t *out, const p1_shape *s);

/* ---- encoder: fitting and model choice (fit.c) --------------------------- */

typedef struct {
	uint8_t profile, quant;
	uint8_t *model;
	size_t model_size;
} taw_model_choice;

/* Choose a profile and fit a model for src. unit = coding-unit size (n for
 * .aw, block_size for .awseek). With predictor AUTO it gives profile 0 when
 * profile 1 is not possible inside the budget; MIX and CHUNK then fail with
 * TAW_E_ARG. */
taw_status taw_model_fit(const uint8_t *src, size_t n, size_t unit,
    const taw_params *p, taw_model_choice *out);
/* The same over any source (read_at may be called from several threads),
 * with params->threads workers; the result does not depend on them. */
taw_status taw_model_fit_source(const taw_source *in, uint64_t unit,
    const taw_params *p, taw_model_choice *out);

#endif
