/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Encoder internals shared by the CPU fitter (fit.c) and the GPU trainer
 * (fit_gpu.c): training samples, the trainable network, the sample order. */
#ifndef TAW_FIT_INTERNAL_H
#define TAW_FIT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"

/* One training bit: the context-model inputs (stretch units), the selector
 * context and the bit. */
typedef struct {
	int16_t st[P1_NCTX];
	int16_t b; /* profile 14: boundary signal sampled at this byte's last bit (0 otherwise) */
	uint8_t c0, c1, msel, bit;
} sample;

/* The trainable network: layer-1 sets then output-layer sets, rows as the
 * model section stores them. Named so fit_opt.h can take an opaque pointer. */
typedef struct tnet {
	p1_shape s;
	unsigned f;               /* mixer width: P1_F or P1_F_HNET */
	size_t s1, s2, row1, row2; /* sets, params per set */
	size_t n1, n2;             /* params per layer */
	float *w, *m, *v, *g;      /* [n1 + n2] each */
	/* Hybrid W1 state: AdamW moments for the W1 rows live separately from
	 * the Muon momentum in m[] (W1 and W2 never overlap in m/v). */
	float *m1, *v1;            /* [n1] */
	/* MALT/MALTER W1 state: row/column diagonal preconditioner EMAs and the
	 * scalar (MALTER) second moment. Only touched when TAW_MALT is enabled,
	 * but allocated unconditionally so the shipped fitter is untouched. */
	float *malt_l, *malt_r;    /* [s1*H], [s1*f] */
	float *malt_nu;            /* [s1] */
	/* NAMO / NAMO-D W1 state: scalar second moment per set, column second
	 * moments per set, plus the shared shrink-gate sign EMA per W1 weight. */
	float *v2s, *v2c, *sig;    /* [s1], [s1*P1_HMAX], [n1] */
	uint8_t *touched1, *touched2;
	bool tied; /* warm-up: every sample uses set 0 */
	uint32_t *list1, *list2;
	uint32_t *cnt1;            /* [s1] samples per W1 set, for routing */
	uint8_t *route1;           /* [s1] 1 = Muon, 0 = AdamW */
	/* Averaging scratch (CPU fitter only): wa is the SWA sum or the live
	 * EMA; last1/last2 record the global step each set joined the EMA. */
	float *wa;
	uint64_t *last1, *last2;
} tnet;

/* ---- deterministic PRNG ------------------------------------------------------ */

static inline uint64_t rng_next(uint64_t *s)
{
	uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

/* ---- the training schedule (both trainers) ------------------------------------ */

enum { FIT_BATCH = 256 };

/* Every epoch strides through the samples with an odd step coprime to ns,
 * from a start drawn with rng_next: a cheap shuffle that keeps neighbouring
 * (correlated) bits in different batches. */
static inline uint64_t fit_stride(uint64_t ns)
{
	uint64_t stride = (uint64_t)((double)ns * 0.6180339887) | 1;
	while (ns > 1) {
		uint64_t a = stride, b = ns;
		while (b) {
			uint64_t r = a % b;
			a = b;
			b = r;
		}
		if (a == 1)
			break;
		stride += 2;
	}
	return stride;
}

/* The first max(1, epochs / 5) epochs train one shared set (tied). */
static inline int fit_warm_epochs(int epochs)
{
	return epochs / 5 > 1 ? epochs / 5 : 1;
}

/* ---- GPU trainer (fit_gpu.c; a stub without TADAMW_GPU) ---------------------- */

typedef struct taw_gpu taw_gpu;

/* Open a GPU and upload the samples. TAW_E_GPU when there is no usable
 * device (or no GPU support in this build). debug: print notes to stderr. */
taw_status taw_gpu_open(const sample *sm, size_t ns, bool debug,
    taw_gpu **out);
/* Whether the GPU trainer handles this shape (hidden <= 16). */
bool taw_gpu_fits(const p1_shape *s);
/* Train nets[0..k) concurrently, all for `epochs` epochs over the samples.
 * Each net comes from tnet_init (its initial weights are used); its w holds
 * the trained weights afterwards. seeds[i] draws net i's epoch start
 * positions, like tnet_train's seed. */
taw_status taw_gpu_train(taw_gpu *g, tnet *nets, size_t k, int epochs,
    const uint64_t *seeds);
void taw_gpu_close(taw_gpu *g);

#endif
