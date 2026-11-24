/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 model section: parsing, dequantization, inference. */
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

bool p1_quant_supported(uint8_t quant)
{
	return taw_quant_is_row_format(quant);
}

/* Row geometry: layer 1 has S1 * H rows of f; the output layer has S2
 * rows of H + 1 (or f when H = 0). f is the profile's mixer width. */
static void rows(const p1_shape *s, unsigned f, uint64_t *r1, uint64_t *l1,
    uint64_t *r2, uint64_t *l2)
{
	*r1 = (uint64_t)p1_sel_size[s->sel1] * s->hidden;
	*l1 = f;
	*r2 = p1_sel_size[s->sel2];
	*l2 = s->hidden ? (uint64_t)s->hidden + 1 : f;
}

uint64_t p1_weights_size_ex(const p1_shape *s, uint8_t quant, unsigned f)
{
	uint64_t r1, l1, r2, l2;
	rows(s, f, &r1, &l1, &r2, &l2);
	return r1 * taw_quant_row_bytes(quant, l1) +
	    r2 * taw_quant_row_bytes(quant, l2);
}

/* ---- profile 6 head and calibration ---------------------------------------- */

uint64_t p6_model_size(const p1_shape *s, uint8_t quant)
{
	return P6_MODEL_HEAD + p1_weights_size(s, quant);
}

void p6_head_write(uint8_t out[P6_MODEL_HEAD], const p1_shape *s,
    const p6_cal *cal)
{
	p1_head_write(out, s);
	out[8] = (uint8_t)cal->temp;
	out[9] = (uint8_t)(cal->temp >> 8);
	out[10] = (uint8_t)cal->bias;
	out[11] = (uint8_t)((uint16_t)cal->bias >> 8);
	out[12] = cal->w0;
	out[13] = cal->w1;
	out[14] = cal->w2;
	out[15] = 0;
}

taw_status p6_head_parse(const uint8_t *p, size_t n, p1_shape *s,
    p6_cal *cal)
{
	if (n < P6_MODEL_HEAD || p1_head_parse(p, n, s) != TAW_OK)
		return TAW_E_FORMAT;
	*cal = (p6_cal){
		.temp = taw_rd16(p + 8),
		.bias = (int16_t)taw_rd16(p + 10),
		.w0 = p[12], .w1 = p[13], .w2 = p[14],
	};
	if (cal->temp < P6_TEMP_MIN || cal->temp > P6_TEMP_MAX ||
	    cal->bias < -P6_BIAS_MAX || cal->bias > P6_BIAS_MAX ||
	    cal->w0 > P6_W_MAX || cal->w1 > P6_W_MAX || cal->w2 > P6_W_MAX ||
	    (cal->w0 == 0 && cal->w1 == 0 && cal->w2 == 0) || p[15])
		return TAW_E_FORMAT;
	return TAW_OK;
}

int32_t p6_cal_t(const p6_cal *cal, int32_t t)
{
	/* Round the Q8 product half away from zero, then add the bias. All
	 * quantities are integers; |temp * t| < 2^21. */
	int32_t q = (int32_t)cal->temp * t;
	int32_t s = q >= 0 ? (q + 128) / 256 : (q - 128) / 256;
	s += cal->bias;
	return s < -2047 ? -2047 : s > 2047 ? 2047 : s;
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
			uint16_t hb[P1_HMAX + P1_FMAX];
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

/* Shared loader: profiles 1/11/14 use an 8-byte head, profile 6 a 16-byte
 * head; f is the mixer row width (P1_F or P1_F_HNET). */
static taw_status net_load(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size, size_t head_len, unsigned f)
{
	memset(n, 0, sizeof *n);
	if (!p1_quant_supported(quant) || (f != P1_F && f != P1_F_HNET))
		return TAW_E_UNSUPPORTED;
	if (size < head_len)
		return TAW_E_FORMAT;
	taw_status st = p1_head_parse(model, size, &n->s);
	if (st != TAW_OK)
		return st;
	if (size - head_len != p1_weights_size_ex(&n->s, quant, f))
		return TAW_E_FORMAT;
	n->cal = P6_CAL_IDENTITY;

	const p1_shape *s = &n->s;
	const p1_isa isa = p1_isa_current();
	const unsigned H = s->hidden;
	n->f = f;
	n->hpad = (H + 7) & ~7u;
	n->w2s = H ? n->hpad : P1_XPAD;
	const size_t s1 = H ? p1_sel_size[s->sel1] : 0;
	const size_t s2 = p1_sel_size[s->sel2];
	const size_t w1n = s1 * f * n->hpad, w2n = s2 * n->w2s;
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
	/* The SIMD mixers unroll F = 26 (p1_simd_out.h) and the fused per-bit
	 * loop hardcodes it too. Profile 14 therefore always uses the scalar
	 * mixer; its per-bit loop is the generic one in phnet.c. */
	if (f != P1_F)
		n->mix = p1_mix_scalar;

	const uint8_t *p = model + head_len;
	float tmp[P1_HMAX + P1_FMAX];
	/* Layer 1 row (set, j) lands transposed: w1[set][i][j]. */
	for (size_t set = 0; set < s1; set++) {
		st = read_rows(&p, quant, isa, H, f, tmp,
		    n->w1 + set * f * n->hpad, 1, n->hpad);
		if (st != TAW_OK) {
			p1_net_free(n);
			return st;
		}
	}
	/* Output rows: weights, then (H > 0) the bias, which moves to b2. */
	const unsigned l2 = H ? H + 1 : f;
	for (size_t set = 0; set < s2; set++) {
		float row[P1_HMAX + P1_FMAX];
		st = read_rows(&p, quant, isa, 1, l2, tmp, row, 0, 1);
		if (st != TAW_OK) {
			p1_net_free(n);
			return st;
		}
		/* The row stride can equal H, so the bias never goes in w2. */
		memcpy(n->w2 + set * n->w2s, row, (H ? H : f) * sizeof(float));
		if (H)
			n->b2[set] = row[H];
	}
	return TAW_OK;
}

taw_status p1_net_load_ex(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size, unsigned f)
{
	return net_load(n, quant, model, size, P1_MODEL_HEAD, f);
}

taw_status p6_net_load(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size)
{
	p1_shape s;
	p6_cal cal;
	taw_status st = p6_head_parse(model, size, &s, &cal);
	if (st != TAW_OK)
		return st;
	st = net_load(n, quant, model, size, P6_MODEL_HEAD, P1_F);
	if (st == TAW_OK)
		n->cal = cal;
	return st;
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
		return dot4(w2, 1, x, n->f);
	float h[P1_HMAX];
	for (unsigned j = 0; j < H; j++) {
		float a = dot4(w1 + j, n->hpad, x, n->f);
		h[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
	}
	return dot4(w2, 1, h, H);
}

int32_t p1_net_eval(const p1_net *n, const float *x, const p1_sel_ctx *sc)
{
	const unsigned H = n->s.hidden;
	const uint32_t c = p1_sel_value(n->s.sel2, sc);
	const float *w1 = H ? n->w1 +
	    (size_t)p1_sel_value(n->s.sel1, sc) * n->f * n->hpad : n->w1;
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

/* ---- APM tables (format.md §2.2.9), shared with profile 14 --------------- */

void p1_apm_reset(uint16_t apm1[256 * 33], uint16_t *apm2)
{
	uint16_t row[33];
	for (int j = 0; j < 33; j++)
		row[j] = (uint16_t)p1_squash((j - 16) * 128);
	for (size_t c = 0; c < 256; c++)
		memcpy(apm1 + c * 33, row, sizeof row);
	for (size_t c = 0; c < 65536; c++)
		memcpy(apm2 + c * 33, row, sizeof row);
}
