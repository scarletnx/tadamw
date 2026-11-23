/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profiles: 0 = INT_ORDER0 (placeholder), 1 = F32_STRICT_MIX,
 * 2 = F32_STRICT_CHUNK (profile 1 with the chunk stage). */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "p1.h"
#include "taw_internal.h"

enum { APM_RATE = P1_APM_RATE };

static void apm_reset(struct taw_p1 *m)
{
	uint16_t row[33];
	for (int j = 0; j < 33; j++)
		row[j] = (uint16_t)p1_squash((j - 16) * 128);
	for (size_t c = 0; c < 256; c++)
		memcpy(m->apm1 + c * 33, row, sizeof row);
	for (size_t c = 0; c < 65536; c++)
		memcpy(m->apm2 + c * 33, row, sizeof row);
}

/* Interpolated refinement of p; *upd gets the nearer knot. */
static inline uint32_t apm_p(const uint16_t *t, size_t row, int32_t s12,
    uint32_t *upd)
{
	uint32_t u = (uint32_t)(s12 + 2048);
	uint32_t lo = u >> 7, w = u & 127;
	const uint16_t *r = t + row * 33;
	*upd = (uint32_t)(row * 33) + lo + (w >> 6);
	return (r[lo] * (128 - w) + r[lo + 1] * w) >> 7;
}

static inline void apm_update(uint16_t *e, int bit)
{
	*e = bit ? (uint16_t)(*e + ((65535u - *e) >> APM_RATE))
		 : (uint16_t)(*e - (*e >> APM_RATE));
}

bool taw_pred_supported(uint8_t profile, uint8_t quant)
{
	switch (profile) {
	case TAW_PROFILE_INT_ORDER0: return quant == TAW_QUANT_NONE;
	case TAW_PROFILE_F32_STRICT_MIX:
	case TAW_PROFILE_F32_STRICT_CHUNK: return p1_quant_supported(quant);
	default: return false;
	}
}

taw_status taw_pred_init(taw_pred *pr, uint8_t profile, uint8_t quant,
    const uint8_t *model, size_t model_size)
{
	*pr = (taw_pred){.profile = profile};
	if (!taw_pred_supported(profile, quant))
		return TAW_E_UNSUPPORTED;
	if (profile == TAW_PROFILE_INT_ORDER0) {
		if (model_size != 0)
			return TAW_E_FORMAT;
		taw_pred_reset(pr);
		return TAW_OK;
	}
	/* p1_ctx has 32-byte-aligned members, so plain calloc (16-byte
	 * alignment on glibc) is not enough for the AVX2 aligned loads. */
	struct taw_p1 *m = aligned_alloc(_Alignof(struct taw_p1), sizeof *m);
	if (!m)
		return TAW_E_NOMEM;
	memset(m, 0, sizeof *m);
	m->apm2 = malloc(65536 * 33 * sizeof *m->apm2);
	if (!m->apm2) {
		free(m);
		return TAW_E_NOMEM;
	}
	taw_status st = p1_net_load(&m->net,
	    profile == TAW_PROFILE_F32_STRICT_CHUNK, quant, model, model_size);
	if (st == TAW_OK) {
		st = p1_ctx_init(&m->ctx, &m->net.s);
		if (st != TAW_OK)
			p1_net_free(&m->net);
	}
	if (st != TAW_OK) {
		free(m->apm2);
		free(m);
		return st;
	}
	pr->p1 = m;
	apm_reset(m);
	return TAW_OK;
}

void taw_pred_free(taw_pred *pr)
{
	if (pr->p1) {
		p1_ctx_free(&pr->p1->ctx);
		p1_net_free(&pr->p1->net);
		free(pr->p1->apm2);
		free(pr->p1);
		pr->p1 = nullptr;
	}
}

void taw_pred_reset(taw_pred *pr)
{
	if (pr->p1) {
		p1_ctx_reset(&pr->p1->ctx);
		apm_reset(pr->p1);
		return;
	}
	pr->c0 = 1;
	for (int i = 0; i < 256; i++)
		pr->p[i] = 32768;
}

/* ---- profile 0 -------------------------------------------------------------- */

static inline void o0_update(taw_pred *pr, int bit)
{
	uint16_t *p = &pr->p[pr->c0];
	if (bit)
		*p = (uint16_t)(*p + ((65536u - *p) >> 4));
	else
		*p = (uint16_t)(*p - (*p >> 4));
	pr->c0 = 2 * pr->c0 + (uint32_t)bit;
}

/* ---- profiles 1 and 2 ------------------------------------------------------- */

static inline uint32_t p1_p16(struct taw_p1 *m)
{
	p1_sel_ctx sc;
	alignas(32) float x[P1_XPAD];
	p1_ctx_inputs(&m->ctx, x, &sc);
	x[P1_IN_BIAS] = 1.0f;
	for (unsigned i = P1_F; i < P1_XPAD; i++)
		x[i] = 0.0f;
	const uint32_t p = p1_squash(p1_net_eval(&m->net, x, &sc));
	const int32_t s12 = p1_stretch_tab[p >> 4];
	const uint32_t q1 = apm_p(m->apm1, sc.c0, s12, &m->a1);
	const uint32_t q2 =
	    apm_p(m->apm2, (size_t)sc.c1 << 8 | sc.c0, s12, &m->a2);
	/* p >= 22 and q1, q2 >= 0, so the result is in [11, 65524]. */
	return (2 * p + q1 + q2) >> 2;
}

static inline void p1_update(struct taw_p1 *m, int bit)
{
	apm_update(&m->apm1[m->a1], bit);
	apm_update(&m->apm2[m->a2], bit);
	p1_ctx_update(&m->ctx, bit);
}

#if TAW_HAVE_ENCODER
void taw_pred_encode(taw_pred *pr, taw_rc_enc *e, const uint8_t *src,
    size_t n)
{
	if (pr->p1) {
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		if (m->ctx.avx2) {
			p1_encode_avx2(m, e, src, n);
			return;
		}
#endif
		for (size_t i = 0; i < n; i++) {
			unsigned c = src[i];
			for (int k = 7; k >= 0; k--) {
				int bit = (c >> k) & 1;
				taw_rc_enc_bit(e, bit, p1_p16(m));
				p1_update(m, bit);
			}
		}
		return;
	}
	for (size_t i = 0; i < n; i++) {
		unsigned c = src[i];
		for (int k = 7; k >= 0; k--) {
			int bit = (c >> k) & 1;
			taw_rc_enc_bit(e, bit, pr->p[pr->c0]);
			o0_update(pr, bit);
		}
		pr->c0 = 1;
	}
}

#endif

#if TAW_HAVE_DECODER
void taw_pred_skip(taw_pred *pr, const uint8_t *src, size_t n)
{
	if (pr->p1) {
		/* the same predictions and updates as coding, without a coder */
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		if (m->ctx.avx2) {
			p1_skip_avx2(m, src, n);
			return;
		}
#endif
		for (size_t i = 0; i < n; i++)
			for (int k = 7; k >= 0; k--) {
				(void)p1_p16(m);
				p1_update(m, (src[i] >> k) & 1);
			}
		return;
	}
	for (size_t i = 0; i < n; i++) {
		for (int k = 7; k >= 0; k--)
			o0_update(pr, (src[i] >> k) & 1);
		pr->c0 = 1;
	}
}

void taw_pred_decode(taw_pred *pr, taw_rc_dec *d, uint8_t *dst, size_t n)
{
	if (pr->p1) {
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		if (m->ctx.avx2) {
			p1_decode_avx2(m, d, dst, n);
			return;
		}
#endif
		for (size_t i = 0; i < n; i++) {
			unsigned c = 1;
			for (int k = 0; k < 8; k++) {
				int bit = taw_rc_dec_bit(d, p1_p16(m));
				p1_update(m, bit);
				c = c << 1 | (unsigned)bit;
			}
			dst[i] = (uint8_t)c;
		}
		return;
	}
	for (size_t i = 0; i < n; i++) {
		for (int k = 0; k < 8; k++)
			o0_update(pr, taw_rc_dec_bit(d, pr->p[pr->c0]));
		dst[i] = (uint8_t)(pr->c0 - 256);
		pr->c0 = 1;
	}
}

#endif
