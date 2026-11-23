/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 and 2 kernels for AVX2 + F16C. Built with -mavx2 -mf16c; only called
 * after a CPUID check. Each lane does the scalar reference's operations
 * (dot4 in p1_net.c) in the same order, so the results are identical. */
#include <immintrin.h>

#include "p1.h"
#include "p1_simd_out.h"

/* The whole mixer; h stays in registers between the layers. Padding
 * lanes have zero weights, so they add only (signed) zeros. */
float p1_mix_avx2(const p1_net *n, const float *x, const float *w1,
    const float *w2)
{
	const unsigned hpad = n->hpad;
	if (!hpad)
		return mix_linear8(x, w2);
	__m128 s = _mm_setzero_ps();
	for (unsigned j = 0; j < hpad; j += 8)
		s = dot4_chunk(s, _mm256_load_ps(w2 + j),
		    layer1_chunk8(w1, x, hpad, j));
	return dot4_finish(s);
}

void p1_f16_to_f32_f16c(const uint16_t *src, float *dst, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(dst + i, _mm256_cvtph_ps(
		    _mm_loadu_si128((const __m128i *)(src + i))));
	if (i < n) {
		uint16_t t[8] = {0};
		float f[8];
		for (size_t k = 0; i + k < n; k++)
			t[k] = src[i + k];
		_mm256_storeu_ps(f, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)t)));
		for (size_t k = 0; i + k < n; k++)
			dst[i + k] = f[k];
	}
}

/* ---- context models -------------------------------------------------------
 * The context models (12; 16 in the encoder's feature pass for profile 2)
 * run as two 8-lane halves. Everything is integer, so the lanes reproduce
 * p1_ctx.c's scalar code exactly. */

/* History-map row of each model (k * 256). */
alignas(32) static const int32_t kbase16[16] = {0, 256, 512, 768, 1024,
	1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840};

/* Lanes 0..ncm-1 are the models, the rest padding (they read arena slots
 * 1..15 and their own history-map rows, which nothing writes, and their
 * results are overwritten or never stored). Slot values
 * come straight from the arena by gather: within a nibble every bit uses a
 * different slot, so no pending store can alias them. History-map entries
 * often are the ones the previous bit just wrote, and gathers cannot
 * forward from pending stores, so those are scalar loads assembled in
 * registers. */
void p1_cm_inputs_avx2(p1_ctx *c, float *x)
{
	const int *st = (const int *)p1_stretch32;
	const int *ar = (const int *)c->arena;
	const __m256i sinit = _mm256_set1_epi32((int)P1_SLOT_INIT);
	const __m256i hinit = _mm256_set1_epi32((int)P1_HM_INIT);
	const __m256i ff = _mm256_set1_epi32(0xFF);
	const __m256 inv = _mm256_set1_ps(1.0f / 256.0f);
	const uint32_t c0 = c->c0;

	const __m256i nib = _mm256_set1_epi32((int)c->nib);
	__m256i si0 = _mm256_add_epi32(
	    _mm256_load_si256((const __m256i *)c->lb), nib);
	si0 = _mm256_blend_epi32(si0, _mm256_setr_epi32((int)c0,
	    (int)(256 + ((c->c4 & 0xFF) << 8 | c0)), 0, 0, 0, 0, 0, 0), 0x03);
	__m256i si1 = _mm256_add_epi32(
	    _mm256_load_si256((const __m256i *)(c->lb + 8)), nib);
	_mm256_store_si256((__m256i *)c->si, si0);
	_mm256_store_si256((__m256i *)(c->si + 8), si1);

	__m256i v0 = _mm256_i32gather_epi32(ar, si0, 4);
	__m256i v1 = _mm256_i32gather_epi32(ar, si1, 4);
	_mm256_store_si256((__m256i *)c->val, v0);
	_mm256_store_si256((__m256i *)(c->val + 8), v1);
	__m256i e0 = _mm256_xor_si256(v0, sinit), e1 = _mm256_xor_si256(v1, sinit);
	__m256i hi0 = _mm256_add_epi32(_mm256_load_si256((const __m256i *)kbase16),
	    _mm256_and_si256(_mm256_srli_epi32(e0, 8), ff));
	__m256i hi1 = _mm256_add_epi32(
	    _mm256_load_si256((const __m256i *)(kbase16 + 8)),
	    _mm256_and_si256(_mm256_srli_epi32(e1, 8), ff));
	_mm256_store_si256((__m256i *)c->hmi, hi0);
	_mm256_store_si256((__m256i *)(c->hmi + 8), hi1);
	__m256i d0 = _mm256_i32gather_epi32(st, _mm256_srli_epi32(e0, 20), 4);
	__m256i d1 = _mm256_i32gather_epi32(st, _mm256_srli_epi32(e1, 20), 4);

	const uint32_t *hm = c->hmap, *ix = c->hmi;
	__m256i hv0 = _mm256_setr_epi32((int)hm[ix[0]], (int)hm[ix[1]],
	    (int)hm[ix[2]], (int)hm[ix[3]], (int)hm[ix[4]], (int)hm[ix[5]],
	    (int)hm[ix[6]], (int)hm[ix[7]]);
	__m256i hv1 = _mm256_setr_epi32((int)hm[ix[8]], (int)hm[ix[9]],
	    (int)hm[ix[10]], (int)hm[ix[11]], (int)hm[ix[12]], (int)hm[ix[13]],
	    (int)hm[ix[14]], (int)hm[ix[15]]);
	_mm256_store_si256((__m256i *)c->hmv, hv0);
	_mm256_store_si256((__m256i *)(c->hmv + 8), hv1);
	__m256i h0 = _mm256_i32gather_epi32(st, _mm256_srli_epi32(
	    _mm256_xor_si256(hv0, hinit), 20), 4);
	__m256i h1 = _mm256_i32gather_epi32(st, _mm256_srli_epi32(
	    _mm256_xor_si256(hv1, hinit), 20), 4);

	/* x[0..ncm-1] direct, x[ncm..2 ncm - 1] history. Both stores are 16
	 * lanes: the history one overwrites the direct padding, and its own
	 * padding lands on x[2 ncm..ncm + 15], which the callers fill next
	 * (match input, bias, zeros). */
	const unsigned ncm = c->ncm;
	_mm256_storeu_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(d0), inv));
	_mm256_storeu_ps(x + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(d1), inv));
	_mm256_storeu_ps(x + ncm, _mm256_mul_ps(_mm256_cvtepi32_ps(h0), inv));
	_mm256_storeu_ps(x + ncm + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(h1), inv));
}

/* (a * r) >> 16 for a < 2^22, r < 2^16, per 32-bit lane. */
static inline __m256i mulshr16_8(__m256i a, __m256i r)
{
	__m256i ev = _mm256_srli_epi64(_mm256_mul_epu32(a, r), 16);
	__m256i od = _mm256_srli_epi64(_mm256_mul_epu32(
	    _mm256_srli_epi64(a, 32), _mm256_srli_epi64(r, 32)), 16);
	return _mm256_blend_epi32(ev, _mm256_slli_epi64(od, 32), 0xAA);
}

/* R[n] = floor(2^17 / (2n + 3)). The binary32 quotient truncates to the
 * same integer for every n: its rounding error is at most 2^-7 / d (d =
 * 2n + 3), less than the 1/d gap between a non-integer quotient and the
 * next integer, and d (odd, > 1) never divides 2^17. */
static inline __m256i rate8(__m256i n)
{
	__m256 d = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_add_epi32(n, n),
	    _mm256_set1_epi32(3)));
	return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_set1_ps(131072.0f), d));
}

/* The models as two 8-lane halves (the lanes after ncm are padding whose
 * results are never stored). The bit is the same in every lane, so each
 * counter needs one product, and the exact identities below (proved with
 * ucas, see docs/design.md) keep the packed fields in place:
 *   65535 - ps = ps ^ 0xFFFF            (ps < 2^16; likewise 2^22 - 1)
 *   bit ? ps + d : ps - d = ps + sign(d, bit ? 1 : -1)
 *   history h' at bits 15..8 = ((e << 1) & 0xFE00) | (e & 0x8000) | bit << 8 */
void p1_cm_update_avx2(p1_ctx *c, int bit)
{
	alignas(32) uint32_t ns[16], nh[16];
	const __m256i sg = _mm256_set1_epi32(bit ? 1 : -1);
	const __m256i flip16 = _mm256_set1_epi32(bit ? 0xFFFF : 0);
	const __m256i flip22 = _mm256_set1_epi32(bit ? 0x3FFFFF : 0);
	const __m256i bh = _mm256_set1_epi32(bit << 8);
	const __m256i sinit = _mm256_set1_epi32((int)P1_SLOT_INIT);
	const __m256i hinit = _mm256_set1_epi32((int)P1_HM_INIT);
	for (unsigned i = 0; i < 16; i += 8) {
		/* context slot: ps 16 | h 8 | n 6 */
		__m256i e = _mm256_xor_si256(
		    _mm256_load_si256((const __m256i *)(c->val + i)), sinit);
		__m256i ps = _mm256_srli_epi32(e, 16);
		__m256i n = _mm256_and_si256(e, _mm256_set1_epi32(63));
		__m256i d = _mm256_srli_epi32(_mm256_mullo_epi32(
		    _mm256_xor_si256(ps, flip16), rate8(n)), 16);
		ps = _mm256_add_epi32(ps, _mm256_sign_epi32(d, sg));
		__m256i h = _mm256_or_si256(_mm256_or_si256(
		    _mm256_and_si256(_mm256_slli_epi32(e, 1),
		    _mm256_set1_epi32(0xFE00)),
		    _mm256_and_si256(e, _mm256_set1_epi32(0x8000))), bh);
		n = _mm256_sub_epi32(n, _mm256_cmpgt_epi32(
		    _mm256_set1_epi32(P1_LIMIT_SLOW), n));
		_mm256_store_si256((__m256i *)(ns + i), _mm256_xor_si256(
		    _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(ps, 16), h),
		    n), sinit));

		/* history map: p 22 | n 10, entries kept from the inputs */
		__m256i f = _mm256_xor_si256(
		    _mm256_load_si256((const __m256i *)(c->hmv + i)), hinit);
		__m256i p = _mm256_srli_epi32(f, 10);
		__m256i hn = _mm256_and_si256(f, _mm256_set1_epi32(1023));
		__m256i pd = mulshr16_8(_mm256_xor_si256(p, flip22), rate8(hn));
		p = _mm256_add_epi32(p, _mm256_sign_epi32(pd, sg));
		hn = _mm256_sub_epi32(hn, _mm256_cmpgt_epi32(
		    _mm256_set1_epi32(P1_LIMIT_HMAP), hn));
		_mm256_store_si256((__m256i *)(nh + i), _mm256_xor_si256(
		    _mm256_or_si256(_mm256_slli_epi32(p, 10), hn), hinit));
	}
	/* No scatter in AVX2: write back one by one. Models never share a
	 * slot or a history-map entry. */
	for (unsigned k = 0; k < c->ncm; k++) {
		c->arena[c->si[k]] = ns[k];
		c->hmap[c->hmi[k]] = nh[k];
	}
}

/* H(a, b, k) per lane (format.md §2.2.4); kk = k * 0xC2B2AE3D. */
static inline __m256i hash8(__m256i a, __m256i b, __m256i kk)
{
	__m256i x = _mm256_add_epi32(_mm256_add_epi32(
	    _mm256_mullo_epi32(a, _mm256_set1_epi32((int)0x9E3779B1u)),
	    _mm256_mullo_epi32(b, _mm256_set1_epi32((int)0x85EBCA77u))), kk);
	x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
	x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x2C1B3C6D));
	x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 12));
	x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x297A2D39));
	return _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
}

/* Byte hashes g_k of profile 1's 10 hashed models and, in lanes 10..13,
 * of profile 2's four chunk contexts with the arguments ab (lanes 14, 15
 * are padding). The lanes are built in registers from the scalars. */
void p1_byte_hashes_avx2(uint32_t c4, uint32_t c8, uint32_t word,
    const uint32_t ab[8], uint32_t *g)
{
	const int C4 = (int)c4;
	__m256i a0 = _mm256_setr_epi32(C4 & 0xFFFF, C4 & 0xFFFFFF, C4, C4,
	    (int)((c4 >> 8) & 0xFFFF), (int)word, (int)(c4 & 0xFFFF0000u),
	    (int)(c4 & 0xFF00FF00u));
	__m256i b0 = _mm256_setr_epi32(0, 0, 0, (int)(c8 & 0xFFFF), 0,
	    C4 & 0xFF, 0, 0);
	__m256i a1 = _mm256_setr_epi32((int)(c4 & 0xFF0000FFu), C4 & 0xFFFF,
	    (int)ab[0], (int)ab[2], (int)ab[4], (int)ab[6], 0, 0);
	__m256i b1 = _mm256_setr_epi32(0, (int)(c8 & 0xFFFF0000u), (int)ab[1],
	    (int)ab[3], (int)ab[5], (int)ab[7], 0, 0);
#define KK(k) ((int)((uint32_t)(k) * 0xC2B2AE3Du))
	__m256i k0 = _mm256_setr_epi32(KK(2), KK(3), KK(4), KK(6), KK(8), KK(9),
	    KK(11), KK(12));
	__m256i k1 = _mm256_setr_epi32(KK(13), KK(14), KK(15), KK(16), KK(17),
	    KK(18), 0, 0);
#undef KK
	_mm256_store_si256((__m256i *)g, hash8(a0, b0, k0));
	_mm256_store_si256((__m256i *)(g + 8), hash8(a1, b1, k1));
}

/* out[k] = g[lane[k]]: the contexts a profile 2 model put in each lane. */
void p2_pick_lanes_avx2(const uint32_t *g, const int32_t *lane,
    uint32_t *out)
{
	_mm256_store_si256((__m256i *)out, _mm256_i32gather_epi32(
	    (const int *)g, _mm256_load_si256((const __m256i *)lane), 4));
	_mm256_store_si256((__m256i *)(out + 8), _mm256_i32gather_epi32(
	    (const int *)g, _mm256_load_si256((const __m256i *)(lane + 8)), 4));
}

/* hh[k] = H(g[k], q, 0) for 16 lanes. */
void p1_line_hashes_avx2(const uint32_t *g, uint32_t q, uint32_t *hh)
{
	const __m256i qv = _mm256_set1_epi32((int)q), z = _mm256_setzero_si256();
	_mm256_store_si256((__m256i *)hh,
	    hash8(_mm256_load_si256((const __m256i *)g), qv, z));
	_mm256_store_si256((__m256i *)(hh + 8),
	    hash8(_mm256_load_si256((const __m256i *)(g + 8)), qv, z));
}

/* Lanes 8..15 that hold a hashed model: 2 of them with 10 hashed models,
 * 6 with the encoder's 14. */
alignas(32) static const int32_t lanes_hi[2][8] = {
	{-1, -1, 0, 0, 0, 0, 0, 0}, {-1, -1, -1, -1, -1, -1, 0, 0}};

/* Arena line index per hashed model k (lanes 0..nhash-1; the rest point
 * at line 0 of the arena, which is never a hashed line, and are masked
 * out). */
static inline void line_bases(const p1_ctx *c, const uint32_t *hh,
    __m256i *lb0, __m256i *lb1)
{
	const __m128i sh = _mm_cvtsi32_si128((int)(32 - c->lines_log2));
	const __m128i lg = _mm_cvtsi32_si128((int)c->lines_log2);
	const __m256i k0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	const __m256i k1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 0, 0);
	const __m256i hx = _mm256_set1_epi32(P1_ARENA_HX);
	__m256i a = _mm256_add_epi32(_mm256_sll_epi32(k0, lg), _mm256_srl_epi32(
	    _mm256_load_si256((const __m256i *)hh), sh));
	__m256i b = _mm256_add_epi32(_mm256_sll_epi32(k1, lg), _mm256_srl_epi32(
	    _mm256_load_si256((const __m256i *)(hh + 8)), sh));
	*lb0 = _mm256_add_epi32(hx, _mm256_slli_epi32(a, 4));
	*lb1 = _mm256_and_si256(_mm256_add_epi32(hx, _mm256_slli_epi32(b, 4)),
	    _mm256_load_si256((const __m256i *)lanes_hi[c->nhash > P1_NHASH]));
}

void p1_lines_apply_avx2(p1_ctx *c, const uint32_t *hh)
{
	__m256i lb0, lb1;
	line_bases(c, hh, &lb0, &lb1);
	const __m256i lo16 = _mm256_set1_epi32(0xFFFF);
	const __m256i one16 = _mm256_set1_epi32((int)(c->epoch << 16));
	const __m256i t0 = _mm256_or_si256(_mm256_and_si256(
	    _mm256_load_si256((const __m256i *)hh), lo16), one16);
	const __m256i t1 = _mm256_or_si256(_mm256_and_si256(
	    _mm256_load_si256((const __m256i *)(hh + 8)), lo16), one16);
	const int *ar = (const int *)c->arena;
	const __m256i g0 = _mm256_i32gather_epi32(ar, lb0, 4);
	const __m256i g1 = _mm256_i32gather_epi32(ar, lb1, 4);
	unsigned miss = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(
	    _mm256_cmpeq_epi32(g0, t0))) |
	    (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(
	    _mm256_cmpeq_epi32(g1, t1))) << 8;
	miss = ~miss & ((1u << c->nhash) - 1);
	/* lb[P1_CM_O2 + k]: lanes 0..nhash-1 of lb0:lb1, moved up by two
	 * (the masked lanes are 0, so the padding of c->lb stays 0). */
	alignas(32) uint32_t lb[16], tg[16];
	_mm256_store_si256((__m256i *)lb, lb0);
	_mm256_store_si256((__m256i *)(lb + 8), lb1);
	_mm256_storeu_si256((__m256i *)(c->lb + P1_CM_O2), lb0);
	_mm_storeu_si128((__m128i *)(c->lb + P1_CM_O2 + 8),
	    _mm256_castsi256_si128(lb1));
	c->lb[P1_CM_O2 + 12] = lb[12];
	c->lb[P1_CM_O2 + 13] = lb[13];
	if (miss) {
		_mm256_store_si256((__m256i *)tg, t0);
		_mm256_store_si256((__m256i *)(tg + 8), t1);
		const __m256i z = _mm256_setzero_si256();
		do {
			const unsigned k = (unsigned)__builtin_ctz(miss);
			uint32_t *ln = c->arena + lb[k];
			_mm256_store_si256((__m256i *)ln, z);
			_mm256_store_si256((__m256i *)(ln + 8), z);
			ln[0] = tg[k];
			miss &= miss - 1;
		} while (miss);
	}
}

void p1_lines_prefetch_avx2(const p1_ctx *c, const uint32_t *hh)
{
	__m256i lb0, lb1;
	line_bases(c, hh, &lb0, &lb1);
	alignas(32) uint32_t lb[16];
	_mm256_store_si256((__m256i *)lb, lb0);
	_mm256_store_si256((__m256i *)(lb + 8), lb1);
	for (unsigned k = 0; k < c->nhash; k++)
		_mm_prefetch((const char *)(c->arena + lb[k]), _MM_HINT_T0);
}
