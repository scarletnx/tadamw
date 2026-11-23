/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Encoder internals shared by the CPU fitter (fit.c) and the GPU trainer
 * (fit_gpu.c): training samples, the trainable network, the sample order. */
#ifndef TAW_FIT_INTERNAL_H
#define TAW_FIT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"

/* One training bit: the context-model inputs (stretch units), the selector
 * context and the bit. Profile 1's feature pass fills P1_NCTX inputs;
 * profile 2's fills P1_NCTX_ALL (every candidate model), and a net reads
 * the ones of its lanes through fit_sample_map. */
typedef struct {
	int16_t st[P1_NCTX_ALL];
	uint8_t c0, c1, msel, bit;
} sample;

/* map[i]: where a net of shape s finds its input i in a sample. */
static inline void fit_sample_map(const p1_shape *s, uint8_t map[P1_NCTX])
{
	for (unsigned i = 0; i < P1_NCTX; i++)
		map[i] = (uint8_t)i;
	if (!s->chunk)
		return;
	for (unsigned t = 0; t < P1_NHASH; t++) {
		map[2 + t] = (uint8_t)(2 + s->lane[t]);
		map[P1_NCM + 2 + t] = (uint8_t)(P1_NCM_ALL + 2 + s->lane[t]);
	}
	map[P1_NCM] = P1_NCM_ALL;
	map[P1_NCM + 1] = P1_NCM_ALL + 1;
	map[P1_IN_MATCH] = 2 * P1_NCM_ALL;
}

/* The trainable network: layer-1 sets then output-layer sets, rows as the
 * model section stores them. */
typedef struct {
	p1_shape s;
	uint8_t map[P1_NCTX];      /* fit_sample_map(&s) */
	size_t s1, s2, row1, row2; /* sets, params per set */
	size_t n1, n2;             /* params per layer */
	float *w, *m, *v, *g;      /* [n1 + n2] each */
	uint8_t *touched1, *touched2;
	bool tied; /* warm-up: every sample uses set 0 */
	uint32_t *list1, *list2;
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

/* Open a GPU and upload the samples as nets of shape s read them. TAW_E_GPU
 * when there is no usable device (or no GPU support in this build). debug:
 * print notes to stderr. */
taw_status taw_gpu_open(const sample *sm, size_t ns, const p1_shape *s,
    bool debug, taw_gpu **out);
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
