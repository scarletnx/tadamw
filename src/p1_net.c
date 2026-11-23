/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 and 2 model sections: parsing, dequantization, inference. */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "half.h"
#include "p1.h"
#include "quant.h"
#include "taw_internal.h"

/* ---- model head ------------------------------------------------------------ */

void p1_head_write(uint8_t out[P1_MODEL_HEAD], const p1_shape *s)
{
	out[0] = s->lines_log2;
	out[1] = s->hist_log2;
	out[2] = s->match_log2;
	out[3] = s->sel1;
	out[4] = s->sel2;
	out[5] = s->hidden;
	out[6] = 0;
	out[7] = 0;
}

taw_status p1_head_parse(const uint8_t *p, size_t n, p1_shape *s)
{
	if (n < P1_MODEL_HEAD)
		return TAW_E_FORMAT;
	*s = (p1_shape){
		.lines_log2 = p[0],
		.hist_log2 = p[1],
		.match_log2 = p[2],
		.sel1 = p[3],
		.sel2 = p[4],
		.hidden = p[5],
	};
	if (p[6] || p[7])
		return TAW_E_FORMAT;
	if (s->lines_log2 < P1_LINES_MIN || s->lines_log2 > P1_LINES_MAX ||
	    s->hist_log2 < P1_HIST_MIN || s->hist_log2 > P1_HIST_MAX ||
	    s->match_log2 < P1_MATCH_MIN || s->match_log2 > P1_MATCH_MAX)
		return TAW_E_FORMAT;
	if (s->sel1 >= P1_SEL_COUNT || s->sel2 >= P1_SEL_COUNT ||
	    s->hidden > P1_HMAX)
		return TAW_E_FORMAT;
	if (s->hidden == 0 && s->sel1 != P1_SEL_NONE)
		return TAW_E_FORMAT;
	return TAW_OK;
}

/* Profile 2 (format.md §5.2): after the head, the context id of each
 * hashed lane; then, when a lane holds a chunk context, the routing table:
 * cls as 256 nibbles (byte b in bits 4 (b mod 2).. of byte b / 2), the 16
 * rows and the 16 line rows, little-endian. */
void p2_head_write(uint8_t *out, const p1_shape *s)
{
	p1_head_write(out, s);
	out += P1_MODEL_HEAD;
	for (unsigned k = 0; k < P1_NHASH; k++)
		out[k] = s->lane[k];
	if (!p2_shape_routes(s))
		return;
	out += P1_NHASH;
	const p2_route *r = &s->route;
	for (unsigned b = 0; b < 256; b += 2)
		out[b / 2] = (uint8_t)(r->cls[b] | r->cls[b + 1] << 4);
	for (unsigned k = 0; k < P2_NCLASS; k++) {
		out[128 + 2 * k] = (uint8_t)r->row[k];
		out[128 + 2 * k + 1] = (uint8_t)(r->row[k] >> 8);
		out[160 + 2 * k] = (uint8_t)r->row2[k];
		out[160 + 2 * k + 1] = (uint8_t)(r->row2[k] >> 8);
	}
}

/* The part of a profile 2 section after the 8-byte head and before the
 * weights; s already holds the head. */
static taw_status p2_head_parse(const uint8_t *p, size_t n, p1_shape *s)
{
	s->chunk = true;
	if (n < P1_NHASH)
		return TAW_E_FORMAT;
	for (unsigned k = 0; k < P1_NHASH; k++) {
		s->lane[k] = p[k];
		if (p[k] >= P2_NCAND || (k && p[k] <= p[k - 1]))
			return TAW_E_FORMAT;
	}
	if (!p2_shape_routes(s))
		return TAW_OK;
	p += P1_NHASH;
	if (n - P1_NHASH < P2_ROUTE_BYTES)
		return TAW_E_FORMAT;
	p2_route *r = &s->route;
	for (unsigned b = 0; b < 256; b++)
		r->cls[b] = (p[b / 2] >> (4 * (b & 1))) & 15;
	for (unsigned k = 0; k < P2_NCLASS; k++) {
		r->row[k] = taw_rd16(p + 128 + 2 * k);
		r->row2[k] = taw_rd16(p + 160 + 2 * k);
		/* a line starts only where a chunk does */
		if (r->row2[k] & ~r->row[k])
			return TAW_E_FORMAT;
	}
	return TAW_OK;
}

bool p1_quant_supported(uint8_t quant)
{
	return taw_quant_is_row_format(quant);
}

/* Row geometry: layer 1 has S1 * H rows of P1_F; the output layer has S2
 * rows of H + 1 (or P1_F when H = 0). */
static void rows(const p1_shape *s, uint64_t *r1, uint64_t *l1, uint64_t *r2,
    uint64_t *l2)
{
	*r1 = (uint64_t)p1_sel_size[s->sel1] * s->hidden;
	*l1 = P1_F;
	*r2 = p1_sel_size[s->sel2];
	*l2 = s->hidden ? (uint64_t)s->hidden + 1 : P1_F;
}

uint64_t p1_weights_size(const p1_shape *s, uint8_t quant)
{
	uint64_t r1, l1, r2, l2;
	rows(s, &r1, &l1, &r2, &l2);
	return r1 * taw_quant_row_bytes(quant, l1) +
	    r2 * taw_quant_row_bytes(quant, l2);
}

/* ---- ISA selection ----------------------------------------------------------- */

#if TAW_X86_SIMD
#include <cpuid.h>

static uint64_t xgetbv0(void)
{
	uint32_t lo, hi;
	__asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	return (uint64_t)hi << 32 | lo;
}

static p1_isa detect(void)
{
	unsigned a, b, c, d;
	if (!__get_cpuid(1, &a, &b, &c, &d))
		return P1_ISA_SCALAR;
	const bool osx = c >> 27 & 1, avx = c >> 28 & 1, f16c = c >> 29 & 1;
	if (!osx || !avx || !f16c)
		return P1_ISA_SCALAR;
	uint64_t xcr0 = xgetbv0();
	if ((xcr0 & 6) != 6)
		return P1_ISA_SCALAR;
	if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
		return P1_ISA_SCALAR;
	if (!(b >> 5 & 1))
		return P1_ISA_SCALAR;
	if ((b >> 16 & 1) && (xcr0 & 0xE6) == 0xE6)
		return P1_ISA_AVX512;
	return P1_ISA_AVX2;
}
#else
static p1_isa detect(void)
{
	return P1_ISA_SCALAR;
}
#endif

static int isa_forced = -1;

p1_isa p1_isa_best(void)
{
	static int cached = -1;
	if (cached < 0)
		cached = (int)detect();
	return (p1_isa)cached;
}

bool p1_isa_available(p1_isa isa)
{
	return isa <= p1_isa_best();
}

void p1_isa_force(int isa)
{
	isa_forced = isa;
}

p1_isa p1_isa_current(void)
{
	if (isa_forced >= 0 && p1_isa_available((p1_isa)isa_forced))
		return (p1_isa)isa_forced;
	return p1_isa_best();
}

/* ---- loading ------------------------------------------------------------------ */

/* Decode n_rows rows of len weights into dst[row * stride + col * cstride]. */
static taw_status read_rows(const uint8_t **pp, uint8_t quant, p1_isa isa,
    uint64_t n_rows, unsigned len, float *tmp, float *dst, size_t row_stride,
    size_t col_stride)
{
	const uint8_t *p = *pp;
	const size_t rb = (size_t)taw_quant_row_bytes(quant, len);
	for (uint64_t r = 0; r < n_rows; r++, p += rb) {
#if TAW_X86_SIMD
		if (quant == TAW_QUANT_F16 && isa >= P1_ISA_AVX2) {
			uint16_t hb[P1_HMAX + P1_F];
			for (unsigned i = 0; i < len; i++) {
				hb[i] = taw_rd16(p + 2 * i);
				if ((hb[i] & 0x7C00u) == 0x7C00u)
					return TAW_E_FORMAT;
			}
			p1_f16_to_f32_f16c(hb, tmp, len);
		} else
#endif
		{
			(void)isa;
			taw_status st = taw_quant_row_decode(quant, p, len, tmp);
			if (st != TAW_OK)
				return st;
		}
		for (unsigned i = 0; i < len; i++)
			dst[r * row_stride + i * col_stride] = tmp[i];
	}
	*pp = p;
	return TAW_OK;
}

taw_status p1_net_load(p1_net *n, bool chunk, uint8_t quant,
    const uint8_t *model, size_t size)
{
	memset(n, 0, sizeof *n);
	if (!p1_quant_supported(quant))
		return TAW_E_UNSUPPORTED;
	taw_status st = p1_head_parse(model, size, &n->s);
	if (st == TAW_OK && chunk)
		st = p2_head_parse(model + P1_MODEL_HEAD, size - P1_MODEL_HEAD,
		    &n->s);
	if (st != TAW_OK)
		return st;
	const size_t head = p1_head_size(&n->s);
	if (size - head != p1_weights_size(&n->s, quant))
		return TAW_E_FORMAT;

	const p1_shape *s = &n->s;
	const p1_isa isa = p1_isa_current();
	const unsigned H = s->hidden;
	n->hpad = (H + 7) & ~7u;
	n->w2s = H ? n->hpad : P1_XPAD;
	const size_t s1 = H ? p1_sel_size[s->sel1] : 0;
	const size_t s2 = p1_sel_size[s->sel2];
	const size_t w1n = s1 * P1_F * n->hpad, w2n = s2 * n->w2s;
	n->w1 = aligned_alloc(64, ((w1n ? w1n : 16) * sizeof(float) + 63) &
	    ~(size_t)63);
	n->w2 = aligned_alloc(64, (w2n * sizeof(float) + 63) & ~(size_t)63);
	n->b2 = malloc(s2 * sizeof(float));
	if (!n->w1 || !n->w2 || !n->b2) {
		p1_net_free(n);
		return TAW_E_NOMEM;
	}
	memset(n->w1, 0, w1n * sizeof(float));
	memset(n->w2, 0, w2n * sizeof(float));
	memset(n->b2, 0, s2 * sizeof(float));

	switch (isa) {
#if TAW_X86_SIMD
	case P1_ISA_AVX512: n->mix = p1_mix_avx512; break;
	case P1_ISA_AVX2: n->mix = p1_mix_avx2; break;
#endif
	default: n->mix = p1_mix_scalar; break;
	}

	const uint8_t *p = model + head;
	float tmp[P1_HMAX + P1_F];
	/* Layer 1 row (set, j) lands transposed: w1[set][i][j]. */
	for (size_t set = 0; set < s1; set++) {
		st = read_rows(&p, quant, isa, H, P1_F, tmp,
		    n->w1 + set * P1_F * n->hpad, 1, n->hpad);
		if (st != TAW_OK) {
			p1_net_free(n);
			return st;
		}
	}
	/* Output rows: weights, then (H > 0) the bias, which moves to b2. */
	const unsigned l2 = H ? H + 1 : P1_F;
	for (size_t set = 0; set < s2; set++) {
		float row[P1_HMAX + P1_F];
		st = read_rows(&p, quant, isa, 1, l2, tmp, row, 0, 1);
		if (st != TAW_OK) {
			p1_net_free(n);
			return st;
		}
		/* The row stride can equal H, so the bias never goes in w2. */
		memcpy(n->w2 + set * n->w2s, row, (H ? H : P1_F) * sizeof(float));
		if (H)
			n->b2[set] = row[H];
	}
	return TAW_OK;
}

void p1_net_free(p1_net *n)
{
	free(n->w1);
	free(n->w2);
	free(n->b2);
	memset(n, 0, sizeof *n);
}

/* ---- inference ------------------------------------------------------------------ */

/* The normative dot product: four partial sums over i mod 4, each in
 * increasing i, combined as (s0 + s1) + (s2 + s3). */
static inline float dot4(const float *w, size_t ws, const float *x,
    unsigned n)
{
	float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	for (unsigned i = 0; i < n; i++)
		s[i & 3] += w[i * ws] * x[i];
	return (s[0] + s[1]) + (s[2] + s[3]);
}

float p1_mix_scalar(const p1_net *n, const float *x, const float *w1,
    const float *w2)
{
	const unsigned H = n->s.hidden;
	if (!H)
		return dot4(w2, 1, x, P1_F);
	float h[P1_HMAX];
	for (unsigned j = 0; j < H; j++) {
		float a = dot4(w1 + j, n->hpad, x, P1_F);
		h[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
	}
	return dot4(w2, 1, h, H);
}

int32_t p1_net_eval(const p1_net *n, const float *x, const p1_sel_ctx *sc)
{
	const unsigned H = n->s.hidden;
	const uint32_t c = p1_sel_value(n->s.sel2, sc);
	const float *w1 = H ? n->w1 +
	    (size_t)p1_sel_value(n->s.sel1, sc) * P1_F * n->hpad : n->w1;
	float z = n->mix(n, x, w1, n->w2 + (size_t)c * n->w2s);
	if (H)
		z += n->b2[c];
	const float lim = 2047.0f / 256.0f;
	if (z > lim)
		z = lim;
	else if (z < -lim)
		z = -lim;
	return (int32_t)(z * 256.0f);
}
