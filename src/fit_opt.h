/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Optimizer internals shared by the CPU fitter (fit.c) and its unit tests.
 *
 * This is the single home of the Newton-Schulz polynomial and of the
 * optimizer ABI. The GPU trainer (fit_gpu.c, fit_gpu.comp) has its own GLSL
 * Adam and never includes this header; the CPU hooks are NULL there.
 *
 * Bit-exactness rules:
 *  - the coefficients -4.7750f / 2.0315f / 3.4445f and the i/j/k accumulation
 *    order in ns_ortho() are ABI, not style;
 *  - fsqrt() lives here too so there is one implementation, not one per TU;
 *  - the global -ffp-contract=off -fno-fast-math flags must stay.
 */
#ifndef TAW_FIT_OPT_H
#define TAW_FIT_OPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "half.h" /* taw_bits_f32 / taw_f32_bits for the generic fsqrt */

/* ---- small math without libm (moved verbatim from fit.c) ---------------------- */

static inline float fsqrt(float x)
{
#if defined(__x86_64__) || defined(__i386__)
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
	if (x <= 0.0f)
		return 0.0f;
	float r = taw_bits_f32((taw_f32_bits(x) >> 1) + 0x1FC00000u);
	for (int i = 0; i < 4; i++)
		r = 0.5f * (r + x / r);
	return r;
#endif
}

/* ---- the ONE Newton-Schulz polynomial --------------------------------------- */

/* Largest W1 matrix is P1_HMAX x P1_F = 32 x 26, so scratch fits on the stack. */
enum { NS_MAXDIM = 32, NS_MAXEL = NS_MAXDIM * NS_MAXDIM };
/* Keep the old names alive for callers that still use them. */
enum { MUON_MAXDIM = NS_MAXDIM, MUON_MAXEL = NS_MAXEL };

/* Work with the shorter dimension first: tr = rows > cols. */
static inline void ns_geom(size_t rows, size_t cols, bool *tr,
    size_t *mr, size_t *mc)
{
	*tr = rows > cols;
	*mr = *tr ? cols : rows;
	*mc = *tr ? rows : cols;
}

/* Transpose map used when filling x and when mapping the result back:
 * flat index i of a row-major rows x cols matrix lands at this slot. */
static inline size_t ns_at(size_t i, size_t cols, bool tr, size_t mc)
{
	return tr ? (i % cols) * mc + i / cols : i;
}

/* x: mr x mc, already transposed short-first AND normalized to unit Frobenius
 * norm. a, b, y: [NS_MAXEL] caller scratch. This is the whole orthogonalizer.
 * Every coefficient and the i/j/k loop nesting are part of the ABI: moving or
 * contracting these operations changes the fitted weights. */
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

/* ---- optimizer configuration (moved verbatim from fit.c) -------------------- */

typedef enum {
	HYB_ADAM = 0, /* the shipped optimizer: AdamW everywhere */
	HYB_MUON,     /* Muon on W1 matrices, AdamW on W2 vectors */
	HYB_BLEND,    /* W1: alpha*Muon + (1-alpha)*AdamW */
	HYB_ALT,      /* W1: alternate Muon / AdamW by step or by half */
	HYB_ROUTE,    /* W1: Muon for hot sets, AdamW for cold sets */
	HYB_MALT,     /* W1: MALT (two-sided diagonal preconditioned Muon) */
	HYB_MALTER    /* W1: MALTER (MALT + adaptive step-size rescaling) */
} hyb_mode;

/* Norm-grafting source for the MALT update magnitude. */
enum { GRAFT_O = 0, /* paper: ||Orth(M~)||_F */
       GRAFT_D = 1, /* no renormalisation: ||D||_F */
       GRAFT_M = 2  /* ||M~||_F */ };

typedef struct {
	float lr, lrvec, lr1;  /* current LRs: Muon W1, Adam W2, Adam W1 */
	float wd, vecwd, wd1;  /* decoupled weight decay */
	float b1, b2, eps;     /* Adam */
	float mom, gain;       /* Muon */
	float b1t, b2t;        /* b1^t, b2^t */
	float malt_b2, malt_eps;   /* diagonal-preconditioner EMA / damping */
	float malt_b1t, malt_b2t;  /* momentum^t, malt_b2^t (MALTER bias correction) */
	unsigned ns_steps;
	int graft;             /* MALT graft mode */
	bool nesterov;
} optcfg;

/* ---- dispatch / hook ABI -----------------------------------------------------
 * ONE ABI for all fitting-side features. The foundation defines the slots and
 * leaves them NULL on shipped paths; PR #5/#6/#8/#19 supply implementations
 * instead of editing optimizer branches. tnet is opaque here: its definition
 * stays in fit_internal.h (shared with the GPU trainer).
 */

struct tnet;

typedef enum {
	W1_ADAM, W1_MUON, W1_MALT, W1_MALTER, W1_NAMO, W1_NAMOD
} w1_kind;

typedef struct {
	w1_kind kind;         /* geometry for this set/step */
	bool adam_half;       /* HYB_BLEND: also run AdamW on m1/v1 */
	bool adam_primary;    /* HYB_ADAM uses m/v, NOT m1/v1 */
	float scale;          /* 1/nb */
	float lr_geo, wd_geo, mix_geo;    /* Muon/MALT/NAMO half */
	float lr_adam, wd_adam, mix_adam; /* Adam half / pure Adam path */
	unsigned bid;         /* FARO risk block, 0 when the bank is off */
} opt_plan;

/* The single step-shaping slot: C3 sign gate (#8) and NAMO_GATE (#6) are
 * configuration aliases into this, never two implementations. begin() is the
 * first pass (advance sign EMA, fill the per-weight multiplier bank, apply
 * mean-preserving normalization); gate() is the second pass (read multiplier
 * k). eps() is reserved for C2, which staged #8 dropped; NULL when unused. */
typedef struct {
	void  *ctx;
	void  (*begin)(void *ctx, const float *g, size_t len, float scale);
	float (*gate) (void *ctx, size_t k, float gk);
	float (*eps)  (void *ctx, size_t k, float vh, float dflt);
} step_shape;

typedef struct {
	void      *ctx;   /* feature-private state; NULL on shipped paths */
	step_shape shape; /* the only gate / coordinate-eps slot */
	float     *risk;  /* FARO bank [nb]; NULL = no accounting */
	unsigned   nb;    /* bank length (bounds/debug) */
	void (*pre_step )(void *ctx, struct tnet *, optcfg *, uint64_t step,
	                  bool tied);
	void (*pre_set  )(void *ctx, struct tnet *, unsigned layer, unsigned set,
	                  opt_plan *);
	void (*post_set )(void *ctx, struct tnet *, unsigned layer, unsigned set,
	                  size_t off, size_t len, uint64_t step);
	void (*post_step)(void *ctx, struct tnet *, uint64_t step);
	void (*finalize) (void *ctx, struct tnet *);
} opt_hooks;

/* The one sign-consistency gate math, adopted verbatim from staged #8
 * (exp-eps-gate @ af5fd00, fit.c:544-550). sig is the sign EMA of the raw
 * gradient gk; the returned multiplier is |sig|^p (p = 1 or 2). Tests use
 * this directly (0.1 / 0.19 / 0.01 for beta = 0.9). */
float fit_gate_step(float *sig, float gk, float beta, float p);

/* ---- thin wrappers (fit_opt.c) ---------------------------------------------- */

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

/* No ns_score(): staged FARO @ 09311ce removed the routing lane it served.
 * If FARO ever restores TAW_FARO_STAT=lin, re-add it as a thin caller of
 * ns_ortho() rather than a fourth copy of the polynomial. */

#endif
