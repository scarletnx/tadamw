/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 14: profile 1 plus the H-Net-style integer boundary input.
 * No libm, no new weights, no change to the context models or APM: see
 * docs/format.md §2.14 and docs/design.md "H-Net boundary profile". */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "p1.h"
#include "phnet.h"
#include "taw_internal.h"

/* o0 direct, o1 direct, word direct, match. The same four inputs are read by
 * the encoder's feature pass (fit.c), via phnet_boundary below. */
static const uint8_t pick[PHNET_PICK_N] = {
	P1_CM_O0, P1_CM_O1, P1_CM_WORD, P1_IN_MATCH
};

void phnet_boundary(const float x[P1_XPAD], int16_t prev[PHNET_PICK_N],
    int32_t *b)
{
	int16_t cur[PHNET_PICK_N];
	/* x[i] = (float)stretch[i] / 256, exact in binary32. The product by
	 * 256 is exact too, so this recovers the integer stretch value. */
	for (unsigned j = 0; j < PHNET_PICK_N; j++)
		cur[j] = (int16_t)(x[pick[j]] * 256.0f);

	int32_t s = 0;
	for (unsigned j = 0; j < PHNET_PICK_N; j++) {
		int32_t d = (int32_t)cur[j] - (int32_t)prev[j];
		s += d < 0 ? -d : d;
	}
	*b = s > PHNET_BMAX ? PHNET_BMAX : s;
	memcpy(prev, cur, sizeof cur);
}

void phnet_inputs(struct taw_hnet *m, float x[P1_XPAD], p1_sel_ctx *sc)
{
	p1_ctx_inputs(&m->ctx, x, sc);
	/* One sample per byte, taken at bit 7 (the context just before the
	 * byte completes) and held until the next byte's bit 7. The signal
	 * therefore spans the transition between two adjacent bytes. */
	if (sc->bitpos == 7)
		phnet_boundary(x, m->prev, &m->b);
	x[P1_IN_BIAS] = 1.0f;
	x[PHNET_IN_BOUNDARY] = (float)m->b * (1.0f / 256.0f);
	for (unsigned i = P1_F_HNET; i < P1_XPAD; i++)
		x[i] = 0.0f;
}

static inline uint32_t phnet_p16(struct taw_hnet *m)
{
	p1_sel_ctx sc;
	alignas(32) float x[P1_XPAD];
	phnet_inputs(m, x, &sc);
	const uint32_t p = p1_squash(p1_net_eval(&m->net, x, &sc));
	const int32_t s12 = p1_stretch_tab[p >> 4];
	const uint32_t q1 = p1_apm_p(m->apm1, sc.c0, s12, &m->a1);
	const uint32_t q2 =
	    p1_apm_p(m->apm2, (size_t)sc.c1 << 8 | sc.c0, s12, &m->a2);
	/* p >= 22 and q1, q2 >= 0, so the result is in [11, 65524]. */
	return (2 * p + q1 + q2) >> 2;
}

static inline void phnet_update(struct taw_hnet *m, int bit)
{
	p1_apm_update(&m->apm1[m->a1], bit);
	p1_apm_update(&m->apm2[m->a2], bit);
	p1_ctx_update(&m->ctx, bit);
}

taw_status phnet_init(struct taw_hnet *m, uint8_t quant,
    const uint8_t *model, size_t size)
{
	memset(m, 0, sizeof *m);
	m->apm2 = malloc(65536 * 33 * sizeof *m->apm2);
	if (!m->apm2)
		return TAW_E_NOMEM;
	taw_status st = p1_net_load_ex(&m->net, quant, model, size, P1_F_HNET);
	if (st == TAW_OK)
		st = p1_ctx_init(&m->ctx, &m->net.s);
	if (st != TAW_OK) {
		phnet_free(m);
		return st;
	}
	phnet_reset(m);
	return TAW_OK;
}

void phnet_free(struct taw_hnet *m)
{
	p1_ctx_free(&m->ctx);
	p1_net_free(&m->net);
	free(m->apm2);
	m->apm2 = nullptr;
}

void phnet_reset(struct taw_hnet *m)
{
	p1_ctx_reset(&m->ctx);
	p1_apm_reset(m->apm1, m->apm2);
	memset(m->prev, 0, sizeof m->prev);
	m->b = 0;
	m->a1 = m->a2 = 0;
}

void phnet_encode(struct taw_hnet *m, taw_rc_enc *e, const uint8_t *src,
    size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned c = src[i];
		for (int k = 7; k >= 0; k--) {
			int bit = (c >> k) & 1;
			taw_rc_enc_bit(e, bit, phnet_p16(m));
			phnet_update(m, bit);
		}
	}
}

void phnet_skip(struct taw_hnet *m, const uint8_t *src, size_t n)
{
	for (size_t i = 0; i < n; i++)
		for (int k = 7; k >= 0; k--) {
			(void)phnet_p16(m);
			phnet_update(m, (src[i] >> k) & 1);
		}
}

void phnet_decode(struct taw_hnet *m, taw_rc_dec *d, uint8_t *dst, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned c = 1;
		for (int k = 0; k < 8; k++) {
			int bit = taw_rc_dec_bit(d, phnet_p16(m));
			phnet_update(m, bit);
			c = c << 1 | (unsigned)bit;
		}
		dst[i] = (uint8_t)c;
	}
}
