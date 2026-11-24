/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profiles: 0 = INT_ORDER0 (placeholder), 1 = F32_STRICT_MIX,
 * 6 = CALIBRATED_MIX, 11 = ENTROPY_GATE and 14 = HNET_BOUNDARY. */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "p1.h"
#include "phnet.h"
#include "taw_internal.h"

/* struct taw_p1 and the APM helpers are shared with profile 14 in p1.h;
 * the gate fields are used only by profile 11. */
bool taw_pred_supported(uint8_t profile, uint8_t quant)
{
	switch (profile) {
	case TAW_PROFILE_INT_ORDER0: return quant == TAW_QUANT_NONE;
	case TAW_PROFILE_F32_STRICT_MIX:
	case TAW_PROFILE_CALIBRATED_MIX:
	case TAW_PROFILE_ENTROPY_GATE:
	case TAW_PROFILE_HNET_BOUNDARY: return p1_quant_supported(quant);
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
	if (profile == TAW_PROFILE_HNET_BOUNDARY) {
		/* taw_hnet also holds a p1_ctx with 32-byte-aligned members. */
		struct taw_hnet *h =
		    aligned_alloc(_Alignof(struct taw_hnet), sizeof *h);
		if (!h)
			return TAW_E_NOMEM;
		taw_status st = phnet_init(h, quant, model, model_size);
		if (st != TAW_OK) {
			free(h);
			return st;
		}
		pr->hn = h;
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
	if (profile == TAW_PROFILE_ENTROPY_GATE) {
		/* Profile 11 = profile 1's head + weight payload + a four-byte
		 * gate tail. The network sees only the first part. */
		if (!p1_quant_supported(quant)) {
			free(m->apm2);
			free(m);
			return TAW_E_UNSUPPORTED;
		}
		p1_shape shape;
		taw_status gs = p1_head_parse(model, model_size, &shape);
		if (gs != TAW_OK) {
			free(m->apm2);
			free(m);
			return gs;
		}
		const size_t wsize = (size_t)p1_weights_size(&shape, quant);
		if (model_size != P1_MODEL_HEAD + wsize + P11_GATE_SIZE) {
			free(m->apm2);
			free(m);
			return TAW_E_FORMAT;
		}
		gs = p11_gate_parse(model + P1_MODEL_HEAD + wsize,
		    P11_GATE_SIZE, &m->gate);
		if (gs != TAW_OK) {
			free(m->apm2);
			free(m);
			return gs;
		}
		m->gated = true;
		model_size = P1_MODEL_HEAD + wsize;
	}
	taw_status st = profile == TAW_PROFILE_CALIBRATED_MIX
	    ? p6_net_load(&m->net, quant, model, model_size)
	    : p1_net_load(&m->net, quant, model, model_size);
	if (st == TAW_OK) {
		m->calibrated = profile == TAW_PROFILE_CALIBRATED_MIX;
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
	p1_apm_reset(m->apm1, m->apm2);
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
	if (pr->hn) {
		phnet_free(pr->hn);
		pr->hn = nullptr;
	}
}

void taw_pred_reset(taw_pred *pr)
{
	if (pr->hn) {
		phnet_reset(pr->hn);
		return;
	}
	if (pr->p1) {
		p1_ctx_reset(&pr->p1->ctx);
		p1_apm_reset(pr->p1->apm1, pr->p1->apm2);
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

/* ---- profile 1 -------------------------------------------------------------- */

static inline uint32_t p1_p16(struct taw_p1 *m)
{
	p1_sel_ctx sc;
	alignas(32) float x[P1_XPAD];
	p1_ctx_inputs(&m->ctx, x, &sc);
	x[P1_IN_BIAS] = 1.0f;
	for (unsigned i = P1_F; i < P1_XPAD; i++)
		x[i] = 0.0f;
	const p6_cal *cal = &m->net.cal;
	const int32_t t = p1_net_eval(&m->net, x, &sc);
	uint32_t p = p1_squash(m->calibrated ? p6_cal_t(cal, t) : t);
	if (m->gated) {
		/* x[i] = st[i] / 256 is exact in binary32, so this recovers
		 * the integer stretch inputs without any rounding. */
		int16_t st[P1_NCTX];
		for (unsigned i = 0; i < P1_NCTX; i++)
			st[i] = (int16_t)(x[i] * 256.0f);
		int32_t mean, mad;
		p11_features(st, &mean, &mad);
		p = p11_gate_p(p, p1_squash(mean), mad, &m->gate);
	}
	const int32_t s12 = p1_stretch_tab[p >> 4];
	const uint32_t q1 = p1_apm_p(m->apm1, sc.c0, s12, &m->a1);
	const uint32_t q2 =
	    p1_apm_p(m->apm2, (size_t)sc.c1 << 8 | sc.c0, s12, &m->a2);
	if (m->calibrated) {
		/* Learned mixture of the mixer and the two APM stages. The
		 * weights are 4-bit, so the products fit in 32 bits. */
		const unsigned den = (unsigned)cal->w0 + cal->w1 + cal->w2;
		return (cal->w0 * p + cal->w1 * q1 + cal->w2 * q2) / den;
	}
	/* p >= 22 and q1, q2 >= 0, so the result is in [11, 65524]. */
	return (2 * p + q1 + q2) >> 2;
}

static inline void p1_update(struct taw_p1 *m, int bit);

uint32_t taw_pred_prob(taw_pred *pr)
{
	return pr->p1 ? p1_p16(pr->p1) : pr->p[pr->c0];
}

void taw_pred_advance(taw_pred *pr, int bit)
{
	if (pr->p1)
		p1_update(pr->p1, bit);
	else
		o0_update(pr, bit);
}

static inline void p1_update(struct taw_p1 *m, int bit)
{
	p1_apm_update(&m->apm1[m->a1], bit);
	p1_apm_update(&m->apm2[m->a2], bit);
	p1_ctx_update(&m->ctx, bit);
}

#if TAW_HAVE_ENCODER
void taw_pred_encode(taw_pred *pr, taw_rc_enc *e, const uint8_t *src,
    size_t n)
{
	if (pr->hn) {
		phnet_encode(pr->hn, e, src, n);
		return;
	}
	if (pr->p1) {
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		/* The fused loop hardcodes profile 1's probability; profile 6
		 * uses the generic loop so its calibration is applied. */
		if (m->ctx.avx2 && !m->calibrated) {
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
	if (pr->hn) {
		phnet_skip(pr->hn, src, n);
		return;
	}
	if (pr->p1) {
		/* the same predictions and updates as coding, without a coder */
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		if (m->ctx.avx2 && !m->calibrated) {
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
	if (pr->hn) {
		phnet_decode(pr->hn, d, dst, n);
		return;
	}
	if (pr->p1) {
		struct taw_p1 *m = pr->p1;
#if TAW_X86_SIMD
		if (m->ctx.avx2 && !m->calibrated) {
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
