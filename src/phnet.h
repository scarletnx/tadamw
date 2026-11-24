/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 14, HNET_BOUNDARY (docs/format.md §2.14).
 *
 * Profile 1's 12 context models, selectors, mixer order, squash, APM stages
 * and range coder are unchanged. The only addition is one integer boundary
 * input x[26] = b/256, where b is a clamped sum of absolute differences of
 * four stretched context estimates sampled at the last bit of consecutive
 * bytes (the state just before a byte completes).
 * The generic (non-fused) per-bit loop is used: the AVX2 fused loop, like the
 * AVX2/AVX-512 mixers, hardcodes F = 26.
 */
#ifndef TAW_PHNET_H
#define TAW_PHNET_H

#include "p1.h"
#include "taw_internal.h" /* taw_rc_enc, taw_rc_dec */

enum {
	PHNET_PICK_N = 4,       /* stretched estimates sampled per byte */
	PHNET_IN_BOUNDARY = P1_F, /* x[26] = b/256 */
	/* 4 picks * 2 * 2047: the saturating bound of the 14-bit sum. */
	PHNET_BMAX = 4 * 2 * 2047,
};

struct taw_hnet {
	p1_ctx ctx;             /* byte-identical to a profile 1 context */
	p1_net net;             /* rows are one weight wider than profile 1 */
	uint16_t apm1[256 * 33];
	uint16_t *apm2;         /* [65536 * 33] */
	uint32_t a1, a2;        /* APM entries to update after the bit */
	int16_t prev[PHNET_PICK_N]; /* previous byte's sampled stretch values */
	int32_t b;              /* current boundary signal, [0, PHNET_BMAX] */
};

/* Build the state from a validated model section. */
taw_status phnet_init(struct taw_hnet *m, uint8_t quant, const uint8_t *model,
    size_t size);
/* Release tables held by m (but not m itself). Safe after a failed init. */
void phnet_free(struct taw_hnet *m);
/* Start of a coding unit: reset context, APM and the previous sample. */
void phnet_reset(struct taw_hnet *m);

/* x[0..24] must be the inputs of bit 7 of a byte (from p1_ctx_inputs).
 * Update prev[] to this byte's sample and write the new boundary signal to
 * *b. Integer-exact: x[i] * 256 is the original stretch value. Shared with
 * the encoder's feature pass so both compute the same b. */
void phnet_boundary(const float x[P1_XPAD], int16_t prev[PHNET_PICK_N],
    int32_t *b);

/* Fill a whole input vector (including x[26] and the padding) and the
 * selector context for the next bit. */
void phnet_inputs(struct taw_hnet *m, float x[P1_XPAD], p1_sel_ctx *sc);

/* Generic per-bit loops, the profile 14 counterparts of predictor.c's
 * profile 1 loops. */
void phnet_encode(struct taw_hnet *m, taw_rc_enc *e, const uint8_t *src,
    size_t n);
void phnet_decode(struct taw_hnet *m, taw_rc_dec *d, uint8_t *dst, size_t n);
void phnet_skip(struct taw_hnet *m, const uint8_t *src, size_t n);

#endif
