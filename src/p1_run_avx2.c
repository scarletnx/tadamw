/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1's (and profile 11's) per-bit loop with the AVX2 kernels fused
 * (built with -mavx2 -mf16c, used only after a CPUID check). It computes
 * exactly what the generic loop (predictor.c over p1_ctx.c, p1_net.c) does, in one
 * function: the slot values and history-map entries a bit reads stay in
 * registers until its update, the mixer and the squash are inline, and
 * the selectors are read once. Only nibble and byte boundaries call out
 * (p1_ctx_tail). The golden vectors and the parity tests pin it. */
#include <immintrin.h>

#include "p1.h"
#include "p1_simd_out.h"
#include "taw_internal.h"

alignas(32) static const int32_t kbase16[16] = {0, 256, 512, 768, 1024,
	1280, 1536, 1792, 2048, 2304, 2560, 2816};

/* R[n] = floor(2^17 / (2n + 3)), exactly (see p1_avx2.c). */
static inline __m256i rate8(__m256i n)
{
	__m256 d = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_add_epi32(n, n),
	    _mm256_set1_epi32(3)));
	return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_set1_ps(131072.0f), d));
}

static inline __m256i mulshr16_8(__m256i a, __m256i r)
{
	__m256i ev = _mm256_srli_epi64(_mm256_mul_epu32(a, r), 16);
	__m256i od = _mm256_srli_epi64(_mm256_mul_epu32(
	    _mm256_srli_epi64(a, 32), _mm256_srli_epi64(r, 32)), 16);
	return _mm256_blend_epi32(ev, _mm256_slli_epi64(od, 32), 0xAA);
}

/* New slot and history-map values for one 8-lane half (see
 * p1_cm_update_avx2 for the identities). */
static inline void counters8(__m256i v, __m256i hv, __m256i sg,
    __m256i flip16, __m256i flip22, __m256i bh, uint32_t *ns, uint32_t *nh)
{
	const __m256i sinit = _mm256_set1_epi32((int)P1_SLOT_INIT);
	const __m256i hinit = _mm256_set1_epi32((int)P1_HM_INIT);
	__m256i e = _mm256_xor_si256(v, sinit);
	__m256i ps = _mm256_srli_epi32(e, 16);
	__m256i n = _mm256_and_si256(e, _mm256_set1_epi32(63));
	__m256i d = _mm256_srli_epi32(_mm256_mullo_epi32(
	    _mm256_xor_si256(ps, flip16), rate8(n)), 16);
	ps = _mm256_add_epi32(ps, _mm256_sign_epi32(d, sg));
	__m256i h = _mm256_or_si256(_mm256_or_si256(
	    _mm256_and_si256(_mm256_slli_epi32(e, 1), _mm256_set1_epi32(0xFE00)),
	    _mm256_and_si256(e, _mm256_set1_epi32(0x8000))), bh);
	n = _mm256_sub_epi32(n, _mm256_cmpgt_epi32(
	    _mm256_set1_epi32(P1_LIMIT_SLOW), n));
	_mm256_store_si256((__m256i *)ns, _mm256_xor_si256(_mm256_or_si256(
	    _mm256_or_si256(_mm256_slli_epi32(ps, 16), h), n), sinit));

	__m256i f = _mm256_xor_si256(hv, hinit);
	__m256i p = _mm256_srli_epi32(f, 10);
	__m256i hn = _mm256_and_si256(f, _mm256_set1_epi32(1023));
	__m256i pd = mulshr16_8(_mm256_xor_si256(p, flip22), rate8(hn));
	p = _mm256_add_epi32(p, _mm256_sign_epi32(pd, sg));
	hn = _mm256_sub_epi32(hn, _mm256_cmpgt_epi32(
	    _mm256_set1_epi32(P1_LIMIT_HMAP), hn));
	_mm256_store_si256((__m256i *)nh, _mm256_xor_si256(_mm256_or_si256(
	    _mm256_slli_epi32(p, 10), hn), hinit));
}

static inline uint32_t squash(int32_t t)
{
	const uint32_t u = (uint32_t)(t + 2048);
	const uint32_t j = u >> 7, w = u & 127;
	return (p1_squash_knot[j] * (128 - w) + p1_squash_knot[j + 1] * w +
	    64) >> 7;
}

static inline uint32_t apm_p(const uint16_t *t, uint32_t row, uint32_t u,
    uint32_t *upd)
{
	const uint32_t lo = u >> 7, w = u & 127;
	const uint16_t *r = t + row * 33;
	*upd = row * 33 + lo + (w >> 6);
	return (r[lo] * (128 - w) + r[lo + 1] * w) >> 7;
}

static inline void apm_update(uint16_t *e, int bit)
{
	*e = bit ? (uint16_t)(*e + ((65535u - *e) >> P1_APM_RATE))
		 : (uint16_t)(*e - (*e >> P1_APM_RATE));
}

/* Sum of eight signed 32-bit lanes (no overflow: |st| <= 2047). */
static inline int hsum8(__m256i v)
{
	__m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
	    _mm256_extracti128_si256(v, 1));
	s = _mm_hadd_epi32(s, s);
	s = _mm_hadd_epi32(s, s);
	return _mm_cvtsi128_si32(s);
}

enum { RUN_ENC, RUN_DEC, RUN_SKIP };

static inline __attribute__((always_inline)) void run(struct taw_p1 *m,
    const int mode, taw_rc_enc *enc, taw_rc_dec *dec, const uint8_t *src,
    uint8_t *dst, size_t len)
{
	p1_ctx *c = &m->ctx;
	const p1_net *net = &m->net;
	const unsigned H = net->s.hidden, hpad = net->hpad, w2s = net->w2s;
	const unsigned sel1 = net->s.sel1, sel2 = net->s.sel2;
	const int *ar = (const int *)c->arena;
	const int *st = (const int *)p1_stretch32;
	const __m256i sinit = _mm256_set1_epi32((int)P1_SLOT_INIT);
	const __m256i hinit = _mm256_set1_epi32((int)P1_HM_INIT);
	const __m256i ff = _mm256_set1_epi32(0xFF);
	const __m256i low4 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
	const __m256 inv = _mm256_set1_ps(1.0f / 256.0f);
	const __m256i kb0 = _mm256_load_si256((const __m256i *)kbase16);
	const __m256i kb1 = _mm256_load_si256((const __m256i *)(kbase16 + 8));
	const float lim = 2047.0f / 256.0f;
	alignas(32) float x[P1_XPAD] = {0};
	x[P1_IN_BIAS] = 1.0f;
	alignas(32) uint32_t ns[16], nh[16];

	for (size_t i = 0; i < len; i++) {
		const unsigned ch = mode == RUN_DEC ? 0 : src[i];
		unsigned acc = 1;
		for (int k = 7; k >= 0; k--) {
			const uint32_t c0 = c->c0, bitpos = c->bitpos;
			const uint32_t c1 = c->c4 & 0xFF;
			/* The APM rows are known now; the mixer hides their
			 * latency (A2 is 4.3 MB). */
			const uint16_t *a2row = m->apm2 + (size_t)(c1 << 8 | c0) * 33;
			_mm_prefetch((const char *)a2row, _MM_HINT_T0);
			_mm_prefetch((const char *)(a2row + 32), _MM_HINT_T0);

			/* ---- inputs (p1_cm_inputs_avx2) ---- */
			const __m256i nib = _mm256_set1_epi32((int)c->nib);
			__m256i si0 = _mm256_add_epi32(
			    _mm256_load_si256((const __m256i *)c->lb), nib);
			si0 = _mm256_blend_epi32(si0, _mm256_setr_epi32((int)c0,
			    (int)(256 + (c1 << 8 | c0)), 0, 0, 0, 0, 0, 0), 0x03);
			const __m256i si1 = _mm256_add_epi32(
			    _mm256_load_si256((const __m256i *)(c->lb + 8)), nib);
			_mm256_store_si256((__m256i *)c->si, si0);
			_mm256_store_si256((__m256i *)(c->si + 8), si1);
			const __m256i v0 = _mm256_i32gather_epi32(ar, si0, 4);
			const __m256i v1 = _mm256_i32gather_epi32(ar, si1, 4);
			const __m256i e0 = _mm256_xor_si256(v0, sinit);
			const __m256i e1 = _mm256_xor_si256(v1, sinit);
			_mm256_store_si256((__m256i *)c->hmi, _mm256_add_epi32(kb0,
			    _mm256_and_si256(_mm256_srli_epi32(e0, 8), ff)));
			_mm256_store_si256((__m256i *)(c->hmi + 8),
			    _mm256_add_epi32(kb1,
			    _mm256_and_si256(_mm256_srli_epi32(e1, 8), ff)));
			const __m256i d0 = _mm256_i32gather_epi32(st,
			    _mm256_srli_epi32(e0, 20), 4);
			const __m256i d1 = _mm256_i32gather_epi32(st,
			    _mm256_srli_epi32(e1, 20), 4);
			const uint32_t *hm = c->hmap, *ix = c->hmi;
			const __m256i hv0 = _mm256_setr_epi32((int)hm[ix[0]],
			    (int)hm[ix[1]], (int)hm[ix[2]], (int)hm[ix[3]],
			    (int)hm[ix[4]], (int)hm[ix[5]], (int)hm[ix[6]],
			    (int)hm[ix[7]]);
			const __m256i hv1 = _mm256_setr_epi32((int)hm[ix[8]],
			    (int)hm[ix[9]], (int)hm[ix[10]], (int)hm[ix[11]], 0, 0,
			    0, 0);
			const __m256i h0 = _mm256_i32gather_epi32(st,
			    _mm256_srli_epi32(_mm256_xor_si256(hv0, hinit), 20), 4);
			const __m256i h1 = _mm256_i32gather_epi32(st,
			    _mm256_srli_epi32(_mm256_xor_si256(hv1, hinit), 20), 4);
			_mm256_store_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(d0), inv));
			_mm_store_ps(x + 8, _mm256_castps256_ps128(
			    _mm256_mul_ps(_mm256_cvtepi32_ps(d1), inv)));
			_mm256_storeu_ps(x + P1_NCM,
			    _mm256_mul_ps(_mm256_cvtepi32_ps(h0), inv));
			_mm_storeu_ps(x + P1_NCM + 8, _mm256_castps256_ps128(
			    _mm256_mul_ps(_mm256_cvtepi32_ps(h1), inv)));

			/* ---- match model input (p1_ctx_inputs) ---- */
			uint32_t msel = 0;
			c->mbit = -1;
			if (c->mexp >= 0 &&
			    ((uint32_t)(c->mexp | 256) >> (8 - bitpos)) == c0) {
				const uint32_t lg = 31 - (uint32_t)__builtin_clz(c->mlen);
				msel = lg - 1 < 7 ? lg - 1 : 7;
				c->mbit = (c->mexp >> (7 - bitpos)) & 1;
				c->msi = msel * 2 + (uint32_t)c->mbit;
				x[P1_IN_MATCH] = (float)p1_stretch_tab[
				    (c->sm[c->msi] ^ P1_SLOT_INIT) >> 20] *
				    (1.0f / 256.0f);
			} else {
				x[P1_IN_MATCH] = 0.0f;
			}

			/* ---- mixer (p1_net_eval) ---- */
			const p1_sel_ctx sc = {.bitpos = (uint8_t)bitpos,
				.c0 = (uint8_t)c0, .c1 = (uint8_t)c1,
				.msel = (uint8_t)msel};
			const uint32_t a2 = p1_sel_value(sel2, &sc);
			const float *w2 = net->w2 + (size_t)a2 * w2s;
			float z;
			if (H) {
				const float *w1 = net->w1 +
				    (size_t)p1_sel_value(sel1, &sc) * P1_F * hpad;
				__m128 s = _mm_setzero_ps();
				for (unsigned j = 0; j < hpad; j += 8)
					s = dot4_chunk(s, _mm256_load_ps(w2 + j),
					    layer1_chunk8(w1, x, hpad, j));
				z = dot4_finish(s) + net->b2[a2];
			} else {
				z = mix_linear8(x, w2);
			}
			z = z > lim ? lim : z < -lim ? -lim : z;
			uint32_t p = squash((int32_t)(z * 256.0f));

			/* ---- profile 11 entropy gate ----
			 * Deliberate duplication of p11_features/p11_gate_p
			 * (p1_ctx.c): the fused loop keeps the 24 context
			 * estimates in registers as integers, so reusing the
			 * scalar helpers would spill them to memory and lose
			 * most of the fused loop's speed. The duplication is
			 * pinned by the profile-11 golden vectors, which compare
			 * this path with the generic p1_p16 path bit-for-bit.
			 * Only the low four lanes of d1/h1 are real models (the
			 * others are padding); the 25th estimate is the match
			 * input, if active. */
			if (m->gated) {
				const int32_t st24 = c->mbit >= 0
				    ? p1_stretch_tab[(c->sm[c->msi] ^
					    P1_SLOT_INIT) >> 20] : 0;
				const int sum = hsum8(d0) + hsum8(_mm256_and_si256(d1, low4)) +
				    hsum8(h0) + hsum8(_mm256_and_si256(h1, low4)) + st24;
				const int mean = (sum >= 0 ? sum + 12 : sum - 12) / 25;
				const __m256i meanv = _mm256_set1_epi32(mean);
				const __m256i ad0 = _mm256_abs_epi32(
				    _mm256_sub_epi32(d0, meanv));
				const __m256i ad1 = _mm256_and_si256(
				    _mm256_abs_epi32(_mm256_sub_epi32(d1, meanv)), low4);
				const __m256i ah0 = _mm256_abs_epi32(
				    _mm256_sub_epi32(h0, meanv));
				const __m256i ah1 = _mm256_and_si256(
				    _mm256_abs_epi32(_mm256_sub_epi32(h1, meanv)), low4);
				const int dev = hsum8(ad0) + hsum8(ad1) + hsum8(ah0) +
				    hsum8(ah1) + (st24 - mean < 0 ? mean - st24
							  : st24 - mean);
				const int mad = (dev + 12) / 25;
				p = p11_gate_p(p, squash(mean), mad, &m->gate);
			}

			/* ---- APM and the coded probability ---- */
			const uint32_t u = (uint32_t)(p1_stretch_tab[p >> 4] + 2048);
			const uint32_t q1 = apm_p(m->apm1, c0, u, &m->a1);
			const uint32_t q2 = apm_p(m->apm2, c1 << 8 | c0, u, &m->a2);
			const uint32_t pc = (2 * p + q1 + q2) >> 2;
			int bit;
			if (mode == RUN_DEC) {
				bit = taw_rc_dec_bit(dec, pc);
			} else {
				bit = (int)(ch >> k) & 1;
				if (mode == RUN_ENC)
					taw_rc_enc_bit(enc, bit, pc);
			}

			/* ---- updates ---- */
			apm_update(&m->apm1[m->a1], bit);
			apm_update(&m->apm2[m->a2], bit);
			const __m256i sg = _mm256_set1_epi32(bit ? 1 : -1);
			const __m256i f16 = _mm256_set1_epi32(bit ? 0xFFFF : 0);
			const __m256i f22 = _mm256_set1_epi32(bit ? 0x3FFFFF : 0);
			const __m256i bh = _mm256_set1_epi32(bit << 8);
			counters8(v0, hv0, sg, f16, f22, bh, ns, nh);
			counters8(v1, hv1, sg, f16, f22, bh, ns + 8, nh + 8);
			uint32_t *arena = c->arena, *hmap = c->hmap;
			for (unsigned q = 0; q < P1_NCM; q++) {
				arena[c->si[q]] = ns[q];
				hmap[c->hmi[q]] = nh[q];
			}
			if (c->mbit >= 0) { /* p1_ctx_tail */
				uint32_t *sm = &c->sm[c->msi];
				const uint32_t ev = *sm ^ P1_SLOT_INIT;
				uint32_t ps = ev >> 16, hh = (ev >> 8) & 0xFF;
				uint32_t n = ev & 63;
				const uint32_t r = p1_rate32[n];
				ps = bit ? ps + (((65535 - ps) * r) >> 16) :
				    ps - ((ps * r) >> 16);
				const uint32_t h2 = hh << 1 | (uint32_t)bit;
				hh = hh >= 128 ? (h2 & 0x7F) | 0x80 : h2;
				n += n < 63;
				*sm = (ps << 16 | hh << 8 | n) ^ P1_SLOT_INIT;
			}
			c->c0 = c0 << 1 | (uint32_t)bit;
			c->nib = c->nib << 1 | (uint32_t)bit;
			c->bitpos = bitpos + 1;
			if (P1_BOUNDARY_BITS >> (bitpos + 1) & 1)
				p1_ctx_boundary(c, bit);
			acc = acc << 1 | (unsigned)bit;
		}
		if (mode == RUN_DEC)
			dst[i] = (uint8_t)acc;
	}
}

void p1_encode_avx2(struct taw_p1 *m, void *enc, const uint8_t *src,
    size_t n)
{
	run(m, RUN_ENC, enc, nullptr, src, nullptr, n);
}

void p1_decode_avx2(struct taw_p1 *m, void *dec, uint8_t *dst, size_t n)
{
	run(m, RUN_DEC, nullptr, dec, nullptr, dst, n);
}

void p1_skip_avx2(struct taw_p1 *m, const uint8_t *src, size_t n)
{
	run(m, RUN_SKIP, nullptr, nullptr, src, nullptr, n);
}
