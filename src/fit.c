/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Encoder: choose a profile, fit the profile 1 mixer with AdamW, quantize.
 *
 * Fitting may be sloppy (it is encoder-only and need not be reproducible);
 * the chosen weights are quantized and dequantized exactly as the decoder
 * does it, and the final estimate uses the real inference code. */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "fit_internal.h"
#include "format.h"
#include "half.h"
#include "p1.h"
#include "quant.h"
#include "taw_internal.h"

/* ---- small math without libm ---------------------------------------------- */

static float fsqrt(float x)
{
#if defined(__x86_64__) || defined(__i386__)
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
	if (x <= 0.0f)
		return 0.0f;
	float r = taw_bits_f32((taw_f32_bits(x) >> 1) + 0x1FC00000u);
	for (int i = 0; i < 4; i++)
		r = 0.5f * (r + x / r);
	return r;
#endif
}

/* log2(x) for x > 0, good to ~1e-9; used only for size estimates. */
static double dlog2(double x)
{
	int e = 0;
	while (x >= 2.0) {
		x *= 0.5;
		e++;
	}
	while (x < 1.0) {
		x *= 2.0;
		e--;
	}
	/* ln(x) = 2 atanh(y), y = (x-1)/(x+1) in [0, 1/3) */
	double y = (x - 1.0) / (x + 1.0), y2 = y * y, t = y, s = 0.0;
	for (int k = 1; k < 40; k += 2) {
		s += t / k;
		t *= y2;
	}
	return e + 2.0 * s * 1.4426950408889634;
}

/* cost_tab[p] = -log2(p / 65536) bits, p = 1..65535 */
static float *cost_tab(void)
{
	static _Atomic(float *) tab;
	float *cur = atomic_load(&tab);
	if (cur)
		return cur;
	float *t = malloc(65536 * sizeof *t);
	if (!t)
		return nullptr;
	t[0] = 16.0f;
	for (uint32_t p = 1; p < 65536; p++)
		t[p] = (float)(16.0 - dlog2((double)p));
	if (!atomic_compare_exchange_strong(&tab, &cur, t)) {
		free(t); /* another thread won */
		return cur;
	}
	return t;
}

/* ---- deterministic PRNG (rng_next: fit_internal.h) ----------------------------- */

static float rng_unif(uint64_t *s) /* [-1, 1) */
{
	return (float)(int32_t)(uint32_t)(rng_next(s) >> 32) * (1.0f / 2147483648.0f);
}

/* ---- samples (the type: fit_internal.h) ---------------------------------------- */

static inline p1_sel_ctx sample_sc(const sample *s)
{
	return (p1_sel_ctx){
		.bitpos = (uint8_t)(31 - __builtin_clz(s->c0)),
		.c0 = s->c0,
		.c1 = s->c1,
		.msel = s->msel,
	};
}

/* The mixer inputs of a net whose sample map is map (fit_sample_map). */
static inline void sample_x(const sample *s, const uint8_t map[P1_NCTX],
    float x[P1_XPAD])
{
	for (unsigned i = P1_F; i < P1_XPAD; i++)
		x[i] = 0.0f;
	for (unsigned i = 0; i < P1_NCTX; i++)
		x[i] = (float)s->st[map[i]] * (1.0f / 256.0f);
	x[P1_IN_BIAS] = 1.0f;
}

static unsigned ceil_log2(uint64_t v)
{
	unsigned k = 0;
	while (k < 63 && (1ull << k) < v)
		k++;
	return k;
}

static unsigned clampu(unsigned v, unsigned lo, unsigned hi)
{
	return v < lo ? lo : v > hi ? hi : v;
}

/* Table geometry from the coding-unit size. */
static void geometry(size_t unit, p1_shape *s)
{
	unsigned lu = ceil_log2(unit ? unit : 1);
	s->lines_log2 = (uint8_t)clampu(lu >= 2 ? lu - 2 : 0, P1_LINES_MIN, 18);
	s->hist_log2 = (uint8_t)clampu(lu, P1_HIST_MIN, 24);
	s->match_log2 = (uint8_t)clampu(lu >= 2 ? lu - 2 : 0, P1_MATCH_MIN, 22);
}

/* One pass of the context models over the input with the decoder's unit
 * resets, keeping an evenly spread set of about `want` bits as training
 * samples. Bit i (of 8n) is kept when floor((i+1) want / 8n) passes an
 * integer, and lands at index floor(i want / 8n): a rule of the position
 * alone, so units can run in parallel and the samples do not depend on the
 * thread count. */
typedef struct {
	const taw_source *in;
	const uint8_t *view; /* the input itself, for memory sources */
	uint64_t n, unit, total, want;
	const p1_shape *geo;
	sample *sm;
	p1_ctx *ctx;    /* [threads], set up on first use */
	uint8_t **buf;  /* [threads] unit buffers (non-memory sources) */
	taw_status *st; /* [units] */
} collect_job;

static inline uint64_t sample_index(uint64_t bit, uint64_t want,
    uint64_t total)
{
#if defined(__SIZEOF_INT128__)
	__extension__ typedef unsigned __int128 u128; /* -Wpedantic */
	return (uint64_t)(((u128)bit * want) / total);
#else
	/* want <= 2^22 and bit <= total: exact while total < 2^42 bits */
	return bit / total * want + bit % total * want / total;
#endif
}

static void collect_unit(void *arg, size_t u, unsigned w)
{
	collect_job *J = arg;
	const uint64_t start = (uint64_t)u * J->unit;
	const uint64_t len = J->n - start < J->unit ? J->n - start : J->unit;
	p1_ctx *c = &J->ctx[w];
	if (!c->arena) {
		if ((J->st[u] = p1_ctx_init(c, J->geo)) != TAW_OK)
			return;
	} else {
		p1_ctx_reset(c);
	}
	const unsigned nctx = 2 * c->ncm + 1;
	const uint8_t *src = J->view ? J->view + start : nullptr;
	if (!src) {
		if (!J->buf[w] && !(J->buf[w] = malloc((size_t)J->unit))) {
			J->st[u] = TAW_E_NOMEM;
			return;
		}
		if ((J->st[u] = J->in->read_at(J->in->ctx, start, J->buf[w],
			 (size_t)len)) != TAW_OK)
			return;
		src = J->buf[w];
	}
	/* Inside each byte the bits are visited in a rotated order for the
	 * selection (the rotation is a hash of the byte's position): a plain
	 * stride over bit positions keeps the same bit of every byte whenever
	 * 8n / want is a multiple of 8, e.g. for any power-of-two input size,
	 * and the mixer then never sees the other seven bit positions. */
	for (uint64_t i = 0; i < len; i++) {
		const uint64_t byte = start + i;
		uint64_t idx[9];
		for (unsigned j = 0; j <= 8; j++)
			idx[j] = sample_index(byte * 8 + j, J->want, J->total);
		const unsigned rot =
		    (unsigned)((byte * 0x9E3779B97F4A7C15ull) >> 61);
		const unsigned ch = src[i];
		for (unsigned bp = 0; bp < 8; bp++) {
			const int v = (ch >> (7 - bp)) & 1;
			const unsigned j = (bp + rot) & 7;
			float xv[P1_XALL];
			p1_sel_ctx sc;
			p1_ctx_inputs(c, xv, &sc); /* the update needs them */
			if (idx[j + 1] != idx[j]) {
				sample *s = &J->sm[idx[j]];
				for (unsigned q = 0; q < nctx; q++) /* exact */
					s->st[q] = (int16_t)(xv[q] * 256.0f);
				s->c0 = sc.c0;
				s->c1 = sc.c1;
				s->msel = sc.msel;
				s->bit = (uint8_t)v;
			}
			p1_ctx_update(c, v);
		}
	}
}

static taw_status collect(const taw_source *in, uint64_t unit,
    const p1_shape *geo, uint64_t want, unsigned threads, sample **out,
    size_t *out_n)
{
	const uint64_t n = in->size, total = n * 8;
	if (want > total)
		want = total;
	const uint64_t units = n ? (n + unit - 1) / unit : 0;
	if (threads == 0)
		threads = 1;
	if (threads > units)
		threads = units ? (unsigned)units : 1;
	collect_job J = {.in = in, .view = taw_source_view(in), .n = n,
		.unit = unit, .total = total, .want = want, .geo = geo};
	J.sm = malloc((size_t)(want ? want : 1) * sizeof *J.sm);
	J.ctx = aligned_alloc(_Alignof(p1_ctx), (size_t)threads * sizeof *J.ctx);
	if (J.ctx)
		memset(J.ctx, 0, (size_t)threads * sizeof *J.ctx);
	J.buf = calloc(threads, sizeof *J.buf);
	J.st = calloc(units ? (size_t)units : 1, sizeof *J.st);
	taw_status st = TAW_E_NOMEM;
	if (J.sm && J.ctx && J.buf && J.st) {
		st = taw_parallel(threads, (size_t)units, 0, collect_unit, nullptr,
		    &J);
		for (uint64_t u = 0; u < units && st == TAW_OK; u++)
			st = J.st[u];
	}
	for (unsigned w = 0; J.ctx && w < threads; w++)
		if (J.ctx[w].arena)
			p1_ctx_free(&J.ctx[w]);
	for (unsigned w = 0; J.buf && w < threads; w++)
		free(J.buf[w]);
	free(J.ctx);
	free(J.buf);
	free(J.st);
	if (st != TAW_OK) {
		free(J.sm);
		return st;
	}
	*out = J.sm;
	*out_n = (size_t)want;
	return TAW_OK;
}

/* ---- the trainable network (tnet: fit_internal.h) ------------------------------- */

static void tnet_free(tnet *t)
{
	free(t->w);
	free(t->m);
	free(t->v);
	free(t->g);
	free(t->touched1);
	free(t->touched2);
	free(t->list1);
	free(t->list2);
	memset(t, 0, sizeof *t);
}

static taw_status tnet_init(tnet *t, const p1_shape *s, uint64_t seed)
{
	memset(t, 0, sizeof *t);
	t->s = *s;
	fit_sample_map(s, t->map);
	const unsigned H = s->hidden;
	t->s1 = H ? p1_sel_size[s->sel1] : 0;
	t->s2 = p1_sel_size[s->sel2];
	t->row1 = (size_t)H * P1_F;
	t->row2 = H ? H + 1u : P1_F;
	t->n1 = t->s1 * t->row1;
	t->n2 = t->s2 * t->row2;
	size_t np = t->n1 + t->n2;
	t->w = calloc(np, sizeof(float));
	t->m = calloc(np, sizeof(float));
	t->v = calloc(np, sizeof(float));
	t->g = calloc(np, sizeof(float));
	t->touched1 = calloc(t->s1 + 1, 1);
	t->touched2 = calloc(t->s2, 1);
	t->list1 = malloc((t->s1 + 1) * sizeof(uint32_t));
	t->list2 = malloc(t->s2 * sizeof(uint32_t));
	if (!t->w || !t->m || !t->v || !t->g || !t->touched1 || !t->touched2 ||
	    !t->list1 || !t->list2) {
		tnet_free(t);
		return TAW_E_NOMEM;
	}
	uint64_t rs = seed;
	float *w1 = t->w, *w2 = t->w + t->n1;
	if (H) {
		/* Every set starts from the same random init. */
		float base[P1_HMAX * P1_F];
		for (unsigned j = 0; j < H; j++)
			for (unsigned i = 0; i < P1_F; i++)
				base[j * P1_F + i] = (i == P1_IN_BIAS ? 0.0f : 0.15f) +
				    0.3f * rng_unif(&rs);
		float base2[P1_HMAX + 1];
		for (unsigned j = 0; j < H; j++)
			base2[j] = (0.5f + 0.5f * rng_unif(&rs)) / (float)H *
			    (j & 1 ? -1.0f : 1.0f);
		base2[H] = 0.0f;
		for (size_t set = 0; set < t->s1; set++)
			memcpy(w1 + set * t->row1, base, t->row1 * sizeof(float));
		for (size_t set = 0; set < t->s2; set++)
			memcpy(w2 + set * t->row2, base2, t->row2 * sizeof(float));
	} else {
		for (size_t set = 0; set < t->s2; set++)
			for (unsigned i = 0; i < P1_F; i++)
				w2[set * P1_F + i] = i == P1_IN_BIAS ? 0.0f : 0.3f;
	}
	return TAW_OK;
}

static inline float clamp8(float a)
{
	return a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
}

/* Forward + backward for one sample, accumulating into t->g. */
static void tnet_step(tnet *t, const sample *smp)
{
	const unsigned H = t->s.hidden;
	float x[P1_XPAD];
	sample_x(smp, t->map, x);
	const p1_sel_ctx sc = sample_sc(smp);
	const uint32_t s2 = t->tied ? 0 : p1_sel_value(t->s.sel2, &sc);
	float *w2 = t->w + t->n1 + s2 * t->row2;
	float *g2 = t->g + t->n1 + s2 * t->row2;
	float z = 0.0f, h[P1_HMAX], a[P1_HMAX];
	const float *w1 = nullptr;
	float *g1 = nullptr;
	if (H) {
		const uint32_t s1 = t->tied ? 0 : p1_sel_value(t->s.sel1, &sc);
		w1 = t->w + s1 * t->row1;
		g1 = t->g + s1 * t->row1;
		for (unsigned j = 0; j < H; j++) {
			float acc = 0.0f;
			const float *r = w1 + j * P1_F;
			for (unsigned i = 0; i < P1_F; i++)
				acc += r[i] * x[i];
			a[j] = acc;
			h[j] = clamp8(acc);
			z += w2[j] * h[j];
		}
		z += w2[H];
	} else {
		for (unsigned i = 0; i < P1_F; i++)
			z += w2[i] * x[i];
	}
	const float lim = 2047.0f / 256.0f;
	float zc = z > lim ? lim : z < -lim ? -lim : z;
	float p = (float)p1_squash((int32_t)(zc * 256.0f)) * (1.0f / 65536.0f);
	float gz = p - (float)smp->bit;
	if (H) {
		for (unsigned j = 0; j < H; j++) {
			g2[j] += gz * h[j];
			float ga = (a[j] > -8.0f && a[j] < 8.0f) ? gz * w2[j] : 0.0f;
			if (ga != 0.0f) {
				float *gr = g1 + j * P1_F;
				for (unsigned i = 0; i < P1_F; i++)
					gr[i] += ga * x[i];
			}
		}
		g2[H] += gz;
	} else {
		for (unsigned i = 0; i < P1_F; i++)
			g2[i] += gz * x[i];
	}
}

/* End of warm-up: copy set 0 (weights and moments) to every set. */
static void tnet_untie(tnet *t)
{
	float *arr[3] = {t->w, t->m, t->v};
	for (int a = 0; a < 3; a++) {
		float *b = arr[a];
		for (size_t set = 1; set < t->s1; set++)
			memcpy(b + set * t->row1, b, t->row1 * sizeof(float));
		for (size_t set = 1; set < t->s2; set++)
			memcpy(b + t->n1 + set * t->row2, b + t->n1,
			    t->row2 * sizeof(float));
	}
	t->tied = false;
}

typedef struct {
	float lr, b1, b2, eps, wd;
	float b1t, b2t; /* b1^t, b2^t */
} adamw;

static void adam_rows(tnet *t, adamw *o, float *w, float *m, float *v,
    float *g, size_t len, float scale)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		g[k] = 0.0f;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		float mh = m[k] * c1, vh = v[k] * c2;
		w[k] -= o->lr * (mh / (fsqrt(vh) + o->eps) + o->wd * w[k]);
	}
	(void)t;
}

/* Train on samples for the given epochs. Batches touch few weight sets, so
 * only those rows are updated ("lazy" AdamW). */
static void tnet_train(tnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed)
{
	enum { BATCH = FIT_BATCH };
	adamw o = {.b1 = 0.9f, .b2 = 0.999f, .eps = 1e-6f, .wd = 1e-5f,
		.b1t = 1.0f, .b2t = 1.0f};
	const float lr0 = 0.02f, lr1 = 0.001f;
	const uint64_t steps_total = (uint64_t)epochs * ((ns + BATCH - 1) / BATCH);
	uint64_t step = 0;
	/* Stride through the samples (fit_stride). */
	uint64_t rs = seed;
	const unsigned H = t->s.hidden;
	/* Warm-up: the first epochs train one shared set, which then seeds
	 * every selected set. */
	const int warm = fit_warm_epochs(epochs);
	t->tied = t->s1 > 1 || t->s2 > 1;
	for (int ep = 0; ep < epochs; ep++) {
		if (t->tied && ep >= warm)
			tnet_untie(t);
		const uint64_t stride = fit_stride(ns);
		uint64_t pos = rng_next(&rs) % (ns ? ns : 1);
		for (size_t done = 0; done < ns;) {
			size_t nb = ns - done < BATCH ? ns - done : BATCH;
			size_t n1 = 0, n2 = 0;
			for (size_t k = 0; k < nb; k++) {
				const sample *s = &sm[pos];
				pos += stride;
				if (pos >= ns)
					pos %= ns;
				const p1_sel_ctx sc = sample_sc(s);
				uint32_t a2 = t->tied ? 0 : p1_sel_value(t->s.sel2, &sc);
				if (!t->touched2[a2]) {
					t->touched2[a2] = 1;
					t->list2[n2++] = a2;
				}
				if (H) {
					uint32_t a1 = t->tied ? 0 : p1_sel_value(t->s.sel1, &sc);
					if (!t->touched1[a1]) {
						t->touched1[a1] = 1;
						t->list1[n1++] = a1;
					}
				}
				tnet_step(t, s);
			}
			done += nb;
			step++;
			float frac = (float)step / (float)steps_total;
			o.lr = lr0 + (lr1 - lr0) * frac;
			o.b1t *= o.b1;
			o.b2t *= o.b2;
			const float scale = 1.0f / (float)nb;
			for (size_t k = 0; k < n1; k++) {
				size_t off = t->list1[k] * t->row1;
				adam_rows(t, &o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row1, scale);
				t->touched1[t->list1[k]] = 0;
			}
			for (size_t k = 0; k < n2; k++) {
				size_t off = t->n1 + t->list2[k] * t->row2;
				adam_rows(t, &o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row2, scale);
				t->touched2[t->list2[k]] = 0;
			}
		}
	}
}

/* ---- quantization / serialization -------------------------------------------- */

static void put_row(taw_buf *b, uint8_t quant, const float *w, size_t len)
{
	uint8_t row[2 * (P1_HMAX + P1_F)];
	const size_t n = (size_t)taw_quant_row_bytes(quant, len);
	taw_quant_row_encode(quant, w, (unsigned)len, row);
	taw_buf_put(b, row, n);
}

static taw_status serialize(const tnet *t, uint8_t quant, taw_buf *b)
{
	uint8_t head[P2_HEAD_MAX];
	if (t->s.chunk)
		p2_head_write(head, &t->s);
	else
		p1_head_write(head, &t->s);
	taw_buf_put(b, head, p1_head_size(&t->s));
	const unsigned H = t->s.hidden;
	for (size_t set = 0; set < t->s1; set++)
		for (unsigned j = 0; j < H; j++)
			put_row(b, quant, t->w + set * t->row1 + j * P1_F, P1_F);
	for (size_t set = 0; set < t->s2; set++)
		put_row(b, quant, t->w + t->n1 + set * t->row2, t->row2);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

/* Mean cost in bits per coded bit, with the decoder's exact inference. */
static taw_status evaluate(const uint8_t *model, size_t size, bool chunk,
    uint8_t quant, const sample *sm, size_t ns, double *bits)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	p1_net net;
	taw_status st = p1_net_load(&net, chunk, quant, model, size);
	if (st != TAW_OK)
		return st;
	uint8_t map[P1_NCTX];
	fit_sample_map(&net.s, map);
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float x[P1_XPAD];
		sample_x(&sm[k], map, x);
		const p1_sel_ctx sc = sample_sc(&sm[k]);
		uint32_t p = p1_squash(p1_net_eval(&net, x, &sc));
		sum += ct[sm[k].bit ? p : 65536 - p];
	}
	p1_net_free(&net);
	*bits = ns ? sum / (double)ns : 8.0;
	return TAW_OK;
}

/* ---- candidates and the size heuristic ----------------------------------------- */

typedef struct {
	p1_shape s;
	uint8_t quant;
	uint64_t size; /* model section bytes */
} cand;

static const uint8_t hidden_opts[] = {0, 4, 8, 16};
static const uint8_t sel1_opts[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0,
	P1_SEL_C0C1HI};
static const uint8_t sel2_opts_h[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0};
static const uint8_t sel2_opts_0[] = {P1_SEL_NONE, P1_SEL_BITPOS,
	P1_SEL_MATCH, P1_SEL_C0, P1_SEL_C1, P1_SEL_C0C1HI};

/* Prior preference (higher = fitted first), fitted to full candidate
 * sweeps at levels 1 and 5 over text, a word list, roff, C source and an
 * ELF binary (1 MiB each): the first candidate is on average 0.6% from
 * the best one (it was 3% with the earlier hand-set prior). Shapes near
 * n/500 bytes (at most 128 KiB) are preferred only mildly; no layer-1
 * selector and a bit-position output selector win most often; the 2-bit
 * formats and F16 rarely do. The 4- and 8-bit formats are close, so the
 * size term breaks their ties. */
static double prior(const cand *c, size_t n)
{
	/* Capped: past ~128 KiB the weight tables (a 4096-set selector is
	 * megabytes of floats) fall out of cache and decoding slows down; a
	 * 1.3 GB dump decoded about 3x slower with a 1.8 MB model. */
	double target = (double)n / 500.0 + 96.0;
	if (target > 131072.0)
		target = 131072.0;
	double r = dlog2((double)c->size / target);
	double pr = -0.25 * (r < 0 ? -r : r);
	if (c->s.sel1 == P1_SEL_NONE)
		pr += 0.5;
	if (c->s.sel2 == P1_SEL_BITPOS)
		pr += 0.6;
	switch (c->quant) {
	case TAW_QUANT_Q2: pr -= 0.5; break;
	case TAW_QUANT_IQ2: pr -= 1.0; break;
	case TAW_QUANT_F16: pr -= 0.3; break;
	default: break;
	}
	return pr;
}

static int cmp_prior(const void *a, const void *b)
{
	double x = ((const double *)a)[0], y = ((const double *)b)[0];
	return (x < y) - (x > y);
}

static size_t enumerate(const p1_shape *geo, uint64_t budget, int quant_pref,
    size_t n, cand *out, size_t cap)
{
	size_t k = 0;
	static const uint8_t quants[] = {TAW_QUANT_F16, TAW_QUANT_Q8,
		TAW_QUANT_Q4, TAW_QUANT_Q2, TAW_QUANT_FP8, TAW_QUANT_NVFP4,
		TAW_QUANT_IQ8, TAW_QUANT_IQ4, TAW_QUANT_IQ2};
	const uint8_t want = taw_quant_from_weights((taw_weights)quant_pref);
	for (size_t qi = 0; qi < sizeof quants; qi++) {
		uint8_t q = quants[qi];
		if (want != TAW_QUANT_NONE && q != want)
			continue;
		for (size_t hi = 0; hi < sizeof hidden_opts; hi++) {
			uint8_t H = hidden_opts[hi];
			size_t n1 = H ? sizeof sel1_opts : 1;
			size_t n2 = H ? sizeof sel2_opts_h : sizeof sel2_opts_0;
			for (size_t a = 0; a < n1; a++)
				for (size_t b = 0; b < n2; b++) {
					cand c = {.s = *geo, .quant = q};
					c.s.hidden = H;
					c.s.sel1 = H ? sel1_opts[a] : P1_SEL_NONE;
					c.s.sel2 = H ? sel2_opts_h[b] : sel2_opts_0[b];
					c.size = p1_head_size(&c.s) + p1_weights_size(&c.s, q);
					if (c.size <= budget && k < cap)
						out[k++] = c;
				}
		}
	}
	/* Sort by prior, best first. */
	struct {
		double pr;
		cand c;
	} *tmp = malloc(k * sizeof *tmp);
	if (!tmp)
		return k;
	for (size_t i = 0; i < k; i++) {
		tmp[i].pr = prior(&out[i], n);
		tmp[i].c = out[i];
	}
	qsort(tmp, k, sizeof *tmp, cmp_prior);
	for (size_t i = 0; i < k; i++)
		out[i] = tmp[i].c;
	free(tmp);
	return k;
}

/* Level -> effort. Epochs matter more than samples: an under-trained net
 * loses more than a smaller sample does. */
static const struct {
	uint32_t samples; /* training bits kept */
	uint8_t cands;    /* candidates fitted */
	uint8_t epochs;   /* the first max(1, epochs / 5) are tied warm-up */
} effort[10] = {
	{0, 0, 0},
	{1u << 18, 1, 3},
	{1u << 19, 1, 4},
	{1u << 19, 2, 6},
	{1u << 20, 2, 6},
	{1u << 20, 3, 6},
	{1u << 21, 3, 8},
	{1u << 21, 4, 8},
	{1u << 22, 6, 10},
	{1u << 22, 8, 12},
};

/* Fitting one candidate: train, quantize, score with the real inference. */
typedef struct {
	const cand *cs;
	const sample *sm;
	size_t ns;
	int epochs;
	uint64_t n;
	taw_buf *model; /* [tries] */
	double *est, *bpb;
	taw_status *st;
	tnet *pre; /* [tries] or nullptr: nets already trained (on the GPU) */
} fit_job;

static void fit_one(void *arg, size_t ci, unsigned w)
{
	(void)w;
	fit_job *F = arg;
	tnet t;
	taw_status st = TAW_OK;
	if (F->pre && F->pre[ci].w) {
		t = F->pre[ci];
		F->pre[ci] = (tnet){0};
	} else if ((st = tnet_init(&t, &F->cs[ci].s, 0x7A57A11ull + ci)) == TAW_OK) {
		tnet_train(&t, F->sm, F->ns, F->epochs, 0xC0FFEEull + ci);
	}
	if (st == TAW_OK) {
		st = serialize(&t, F->cs[ci].quant, &F->model[ci]);
		tnet_free(&t);
	}
	if (st == TAW_OK)
		st = evaluate(F->model[ci].p, F->model[ci].n, F->cs[ci].s.chunk,
		    F->cs[ci].quant, F->sm, F->ns, &F->bpb[ci]);
	F->est[ci] = F->bpb[ci] * 8.0 * (double)F->n + 8.0 * (double)F->model[ci].n;
	F->st[ci] = st;
}

/* ---- hybrid fitting: the GPU trains some candidates, CPU workers the rest ---- */

typedef struct {
	fit_job *F;
	const size_t *idx; /* the candidates of this list */
} fit_list;

static void fit_listed(void *arg, size_t i, unsigned w)
{
	const fit_list *L = arg;
	fit_one(L->F, L->idx[i], w);
}

typedef struct {
	taw_gpu *g;
	tnet *nets;
	size_t k;
	int epochs;
	const uint64_t *seeds;
	taw_status st;
} gpu_job;

static void *gpu_main(void *arg)
{
	gpu_job *J = arg;
	J->st = taw_gpu_train(J->g, J->nets, J->k, J->epochs, J->seeds);
	return nullptr;
}

static double now_s(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Fit candidates [0, tries). Which ones the GPU trains is a rule of the
 * candidates alone (every shape it supports), so the archive does not depend
 * on the thread count or on timing; CPU workers fit the others meanwhile. A
 * failing GPU falls back to the CPU (TAW_GPU_AUTO) or fails (TAW_GPU_ON). */
static taw_status fit_all(fit_job *F, size_t tries, unsigned threads,
    taw_gpu_mode mode, bool dbg)
{
	const double t0 = dbg ? now_s() : 0.0;
	/* AUTO leaves small fits to the CPU: opening a GPU takes about 0.2 s,
	 * more than levels 1-2 spend fitting. A rule of the effort alone. */
	if (mode == TAW_GPU_AUTO &&
	    (uint64_t)F->ns * (uint64_t)F->epochs * tries < (1u << 22)) {
		if (dbg)
			fprintf(stderr, "taw fit: gpu: not worth it for this "
					"little fitting\n");
		mode = TAW_GPU_OFF;
	}
	size_t *idx = malloc(2 * tries * sizeof *idx), ng = 0, nc = 0;
	if (!idx)
		return TAW_E_NOMEM;
	size_t *gi = idx, *ci = idx + tries;
	for (size_t c = 0; c < tries; c++) {
		if (mode != TAW_GPU_OFF && taw_gpu_fits(&F->cs[c].s))
			gi[ng++] = c;
		else
			ci[nc++] = c;
	}
	taw_gpu *g = nullptr;
	taw_status st = TAW_OK;
	if (ng && (st = taw_gpu_open(F->sm, F->ns, &F->cs[0].s, dbg, &g)) !=
	    TAW_OK) {
		if (mode == TAW_GPU_ON) {
			free(idx);
			return st;
		}
		nc = tries; /* auto: all on the CPU */
		for (size_t c = 0; c < tries; c++)
			ci[c] = c;
		ng = 0;
		st = TAW_OK;
	}
	tnet *nets = ng ? calloc(ng, sizeof *nets) : nullptr;
	uint64_t *seeds = ng ? malloc(ng * sizeof *seeds) : nullptr;
	F->pre = ng ? calloc(tries, sizeof *F->pre) : nullptr;
	if (ng && (!nets || !seeds || !F->pre))
		st = TAW_E_NOMEM;
	for (size_t j = 0; j < ng && st == TAW_OK; j++) {
		st = tnet_init(&nets[j], &F->cs[gi[j]].s, 0x7A57A11ull + gi[j]);
		seeds[j] = 0xC0FFEEull + gi[j];
	}
	gpu_job J = {.g = g, .nets = nets, .k = ng, .epochs = F->epochs,
		.seeds = seeds, .st = TAW_OK};
	pthread_t th;
	bool spawned = false;
	if (st == TAW_OK && ng) {
		spawned = pthread_create(&th, nullptr, gpu_main, &J) == 0;
		if (!spawned)
			gpu_main(&J);
	}
	if (st == TAW_OK)
		st = taw_parallel(threads, nc, 0, fit_listed, nullptr,
		    &(fit_list){F, ci});
	if (spawned)
		pthread_join(th, nullptr);
	taw_gpu_close(g);
	const double t1 = dbg ? now_s() : 0.0;
	if (st == TAW_OK && ng) {
		if (J.st == TAW_OK) {
			for (size_t j = 0; j < ng; j++) {
				F->pre[gi[j]] = nets[j];
				nets[j] = (tnet){0};
			}
		} else if (mode == TAW_GPU_ON) {
			st = J.st;
		} else if (dbg) {
			fprintf(stderr, "taw fit: gpu: training failed, "
					"fitting on the CPU\n");
		}
		if (st == TAW_OK) /* quantize and score (or fit, after a failure) */
			st = taw_parallel(threads, ng, 0, fit_listed, nullptr,
			    &(fit_list){F, gi});
	}
	if (dbg)
		fprintf(stderr, "taw fit: %zu candidates (%zu on the GPU): "
				"training %.2f s, total %.2f s\n",
		    tries, J.st == TAW_OK ? ng : 0, t1 - t0, now_s() - t0);
	for (size_t j = 0; j < ng; j++)
		tnet_free(&nets[j]);
	for (size_t c = 0; F->pre && c < tries; c++)
		tnet_free(&F->pre[c]);
	free(F->pre);
	F->pre = nullptr;
	free(nets);
	free(seeds);
	free(idx);
	return st;
}

/* ---- profile 2: the routing table and the lanes ------------------------------- */

/* The routing table is searched, not trained: the chunk contexts go
 * through hash tables, which no gradient crosses. The search minimizes the
 * code length a small context mixer gets on a sample of the input: a base
 * prediction that stays the same for every table (orders 1-4 and 6 and the
 * word model, mixed online, computed once) plus the four chunk contexts
 * under the candidate table. So a table scores by what its chunks add to
 * the contexts the mixer has anyway. Coordinate descent from the best
 * preset: the class of each frequent byte, then the bits of the two class
 * matrices. Candidates of a step are scored in parallel and the first best
 * wins, so the result does not depend on the thread count. */

enum {
	RT_NBASE = 6,  /* context models of the base prediction */
	RT_NCK = 4,    /* chunk contexts */
	RT_PIECES = 16,
	RT_BATCH = 2 * P2_NCLASS,
};

/* slot: P(1) in 12 bits << 4 | count, adaptive rate 1 / (n + 1.5) */
constexpr uint16_t RT_SLOT_INIT = 2048 << 4;

static inline void rt_slot_up(uint16_t *s, int bit)
{
	static const uint16_t rate[16] = {43690, 26214, 18724, 14563, 11915,
		10082, 8738, 7710, 6898, 6241, 5698, 5242, 4854, 4519, 4228,
		3972}; /* 65536 / (n + 1.5) */
	uint32_t p = *s >> 4, n = *s & 15;
	if (bit)
		p += ((4095 - p) * rate[n]) >> 16;
	else
		p -= (p * rate[n]) >> 16;
	n += n < 15;
	*s = (uint16_t)(p << 4 | n);
}

typedef struct {
	const uint8_t *src;
	size_t n;
	unsigned lines_log2;
	int16_t *base;     /* [8 n] stretch of the base prediction per bit */
	const float *cost; /* cost_tab */
} rt_sample;

static inline int32_t rt_clamp(float z)
{
	const int32_t t = (int32_t)(z * 256.0f);
	return t > 2047 ? 2047 : t < -2047 ? -2047 : t;
}

/* One context mixer pass over the sample: nk hashed contexts whose byte
 * hashes next() gives, plus the stored base prediction when base is set.
 * Returns the code length in bits; with out, stores each bit's stretched
 * prediction there. */
typedef struct {
	uint32_t c4, c8, word; /* base contexts */
	p2_chunk ch;           /* chunk contexts */
	uint32_t line[2][P2_LINE_MAX];
	const p2_route *route;
} rt_state;

static void rt_hashes(rt_state *q, uint32_t ch, bool chunks, uint32_t *g)
{
	if (chunks) {
		p2_chunk nx;
		uint32_t ab[8];
		p2_chunk_next(q->route, &q->ch, q->line, q->c4 & 0xFF, ch, &nx);
		p2_chunk_commit(&q->ch, q->line, &nx);
		q->c4 = ch;
		p2_chunk_args(&q->ch, ab);
		for (unsigned j = 0; j < RT_NCK; j++)
			g[j] = p1_hash3(ab[2 * j], ab[2 * j + 1], 15 + j);
		return;
	}
	q->c8 = q->c8 << 8 | q->c4 >> 24;
	q->c4 = q->c4 << 8 | ch;
	q->word = (ch | 32) - 'a' < 26 ? (q->word ^ (ch | 32)) * 0x01000193u : 0;
	g[0] = p1_hash3(q->c4 & 0xFF, 0, 1);
	g[1] = p1_hash3(q->c4 & 0xFFFF, 0, 2);
	g[2] = p1_hash3(q->c4 & 0xFFFFFF, 0, 3);
	g[3] = p1_hash3(q->c4, 0, 4);
	g[4] = p1_hash3(q->c4, q->c8 & 0xFFFF, 6);
	g[5] = p1_hash3(q->word, q->c4 & 0xFF, 9);
}

static double rt_pass(const rt_sample *S, const p2_route *route,
    uint16_t *tab, int16_t *out)
{
	const bool chunks = route != nullptr;
	const unsigned nk = chunks ? RT_NCK : RT_NBASE;
	const unsigned nx = nk + (chunks ? 2 : 1); /* + base, + bias */
	const size_t lines = (size_t)1 << S->lines_log2;
	const unsigned sh = 32 - S->lines_log2;
	for (size_t i = 0; i < nk * lines * 16; i++)
		tab[i] = RT_SLOT_INIT;
	float w[8][RT_NBASE + 2];
	for (unsigned b = 0; b < 8; b++) {
		for (unsigned i = 0; i < nk; i++)
			w[b][i] = chunks ? 0.1f : 0.3f;
		w[b][nk] = chunks ? 1.0f : 0.0f; /* base, or the bias */
		w[b][nk + 1] = 0.0f;
	}
	rt_state q = {.route = route};
	uint32_t g[RT_NBASE];
	if (chunks) {
		uint32_t ab[8];
		p2_chunk_args(&q.ch, ab);
		for (unsigned j = 0; j < RT_NCK; j++)
			g[j] = p1_hash3(ab[2 * j], ab[2 * j + 1], 15 + j);
	} else {
		for (unsigned j = 0; j < RT_NBASE; j++)
			g[j] = p1_hash3(0, 0, j < 4 ? j + 1 : j == 4 ? 6 : 9);
	}
	double bits = 0.0;
	for (size_t i = 0; i < S->n; i++) {
		const uint32_t ch = S->src[i];
		uint16_t *ln[RT_NBASE];
		uint32_t c0 = 1, nib = 1;
		for (unsigned bp = 0; bp < 8; bp++) {
			if ((bp & 3) == 0) {
				for (unsigned k = 0; k < nk; k++)
					ln[k] = tab + (k * lines +
					    (p1_hash3(g[k], bp ? c0 : 0, 0) >> sh)) * 16;
				nib = 1;
			}
			const int bit = (int)(ch >> (7 - bp)) & 1;
			float x[RT_NBASE + 2], z = 0.0f;
			for (unsigned k = 0; k < nk; k++)
				x[k] = (float)p1_stretch_tab[ln[k][nib] >> 4] *
				    (1.0f / 256.0f);
			if (chunks) {
				x[nk] = (float)S->base[i * 8 + bp] * (1.0f / 256.0f);
				x[nk + 1] = 1.0f;
			} else {
				x[nk] = 1.0f;
			}
			float *wr = w[bp];
			for (unsigned k = 0; k < nx; k++)
				z += wr[k] * x[k];
			const int32_t t = rt_clamp(z);
			const uint32_t p = p1_squash(t);
			if (out)
				out[i * 8 + bp] = (int16_t)t;
			bits += S->cost[bit ? p : 65536 - p];
			const float err =
			    ((float)bit - (float)p * (1.0f / 65536.0f)) * 0.02f;
			for (unsigned k = 0; k < nx; k++)
				wr[k] += err * x[k];
			for (unsigned k = 0; k < nk; k++)
				rt_slot_up(&ln[k][nib], bit);
			c0 = c0 << 1 | (uint32_t)bit;
			nib = nib << 1 | (uint32_t)bit;
		}
		rt_hashes(&q, ch, chunks, g);
	}
	return bits;
}

typedef struct {
	const rt_sample *S;
	const p2_route *tabs;
	double *score;
	uint16_t **tab; /* [threads], made on first use */
	size_t tab_n;
	_Atomic bool oom;
} rt_batch;

static void rt_work(void *arg, size_t i, unsigned w)
{
	rt_batch *B = arg;
	if (!B->tab[w] && !(B->tab[w] = malloc(B->tab_n * sizeof **B->tab))) {
		atomic_store(&B->oom, true);
		return;
	}
	B->score[i] = rt_pass(B->S, &B->tabs[i], B->tab[w], nullptr);
}

/* Score tabs[0..k); the best one that beats *bs by more than eps replaces
 * *best. */
static taw_status rt_step(rt_batch *B, unsigned threads, p2_route *tabs,
    size_t k, p2_route *best, double *bs, double eps, size_t *evals)
{
	double score[RT_BATCH];
	B->tabs = tabs;
	B->score = score;
	taw_status st = taw_parallel(threads, k, 0, rt_work, nullptr, B);
	if (st != TAW_OK)
		return st;
	if (atomic_load(&B->oom))
		return TAW_E_NOMEM;
	*evals += k;
	size_t bi = k;
	for (size_t j = 0; j < k; j++)
		if (score[j] < *bs - eps && (bi == k || score[j] < score[bi]))
			bi = j;
	if (bi < k) {
		*bs = score[bi];
		*best = tabs[bi];
	}
	return TAW_OK;
}

/* Starting tables. 0: letters join into words, every other byte is a chunk
 * of its own, no lines. 1: letters with the bytes from 0x80 up, digits,
 * blanks, line ends and the other control bytes each join their own kind,
 * the rest stand alone; a line starts after the line ends. 2: the same
 * classes with the zero byte as the line end (binary records). */
static void rt_preset(p2_route *r, unsigned which)
{
	*r = (p2_route){0};
	for (unsigned b = 0; b < 256; b++) {
		const bool letter = (b | 32) - 'a' < 26u, digit = b - '0' < 10u;
		if (which == 0)
			r->cls[b] = letter;
		else
			r->cls[b] = (letter || b >= 128) ? 1 : digit ? 2 :
			    (b == ' ' || b == '\t') ? 3 :
			    (b == '\n' || b == '\r') ? 4 : b == 0 ? 6 : b < 32 ? 5 : 0;
	}
	for (unsigned a = 0; a < P2_NCLASS; a++)
		r->row[a] = 0xFFFF;
	if (which == 0) {
		r->row[1] = 0xFFFF & ~2u;
		return;
	}
	for (unsigned a = 1; a <= 6; a++)
		r->row[a] = (uint16_t)(0xFFFF & ~(1u << a));
	const unsigned eol = which == 1 ? 4 : 6;
	r->row2[eol] = r->row[eol];
}

/* Search effort by level: sample bytes, how many of the most frequent
 * bytes get their class searched, sweeps. */
static const struct {
	uint32_t sample;
	uint16_t bytes;
	uint8_t sweeps;
} rt_effort[10] = {
	{0, 0, 0},
	{1u << 18, 0, 0},
	{1u << 18, 0, 1},
	{1u << 19, 16, 1},
	{1u << 19, 32, 1},
	{1u << 19, 48, 1},
	{1u << 20, 64, 1},
	{1u << 20, 96, 2},
	{1u << 20, 128, 2},
	{1u << 20, 256, 2},
};

static taw_status route_fit(const taw_source *in, const taw_params *p,
    p2_route *out)
{
	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	const char *dbg = getenv("TAW_FIT_DEBUG");
	rt_sample S = {.cost = cost_tab()};
	if (!S.cost)
		return TAW_E_NOMEM;
	p1_tables_init();

	/* the sample: the whole input, or RT_PIECES evenly spread pieces */
	const uint64_t n = in->size;
	size_t want = rt_effort[lv].sample;
	uint8_t *buf = nullptr;
	taw_status st = TAW_OK;
	if (n <= want) {
		want = (size_t)n;
		S.src = taw_source_view(in);
		if (!S.src) {
			if (!(buf = malloc(want ? want : 1)))
				return TAW_E_NOMEM;
			st = in->read_at(in->ctx, 0, buf, want);
			S.src = buf;
		}
	} else {
		const size_t piece = want / RT_PIECES;
		want = piece * RT_PIECES;
		if (!(buf = malloc(want)))
			return TAW_E_NOMEM;
		for (unsigned k = 0; k < RT_PIECES && st == TAW_OK; k++)
			st = in->read_at(in->ctx, (n - piece) / (RT_PIECES - 1) * k,
			    buf + k * piece, piece);
		S.src = buf;
	}
	S.n = want;
	S.lines_log2 = clampu(ceil_log2(want ? want : 1), 10, 18);
	const size_t lines = (size_t)1 << S.lines_log2;
	rt_batch B = {.S = &S, .tab_n = RT_NCK * lines * 16};
	uint16_t *btab = nullptr;
	if (st == TAW_OK) {
		S.base = malloc((want ? want : 1) * 8 * sizeof *S.base);
		btab = malloc(RT_NBASE * lines * 16 * sizeof *btab);
		B.tab = calloc(threads, sizeof *B.tab);
		if (!S.base || !btab || !B.tab)
			st = TAW_E_NOMEM;
	}
	double base_bits = 0.0;
	if (st == TAW_OK)
		base_bits = rt_pass(&S, nullptr, btab, S.base);
	free(btab);

	size_t freq[256] = {0};
	for (size_t i = 0; st == TAW_OK && i < S.n; i++)
		freq[S.src[i]]++;
	uint8_t order[256];
	for (unsigned i = 0; i < 256; i++)
		order[i] = (uint8_t)i;
	for (unsigned i = 1; i < 256; i++) /* by falling frequency, stable */
		for (unsigned j = i; j > 0 && freq[order[j]] > freq[order[j - 1]]; j--) {
			const uint8_t t = order[j];
			order[j] = order[j - 1];
			order[j - 1] = t;
		}

	p2_route tabs[RT_BATCH], best;
	double bs = 0.0;
	size_t evals = 0;
	if (st == TAW_OK) {
		for (unsigned k = 0; k < 3; k++)
			rt_preset(&tabs[k], k);
		best = tabs[0];
		bs = 1e300;
		st = rt_step(&B, threads, tabs, 3, &best, &bs, 0.0, &evals);
	}
	if (st == TAW_OK && dbg)
		fprintf(stderr, "taw fit: route: %zu sample bytes, base %.4f bpc, "
				"best preset %.4f bpc\n", S.n,
		    base_bits / (double)(S.n ? S.n : 1), bs / (double)(S.n ? S.n : 1));
	/* A step must gain 0.01% of the code length: less is noise. */
	for (unsigned sw = 0; st == TAW_OK && sw < rt_effort[lv].sweeps; sw++) {
		const double before = bs, eps = bs * 1e-4;
		/* (a) the class of each frequent byte */
		for (unsigned oi = 0; st == TAW_OK && oi < rt_effort[lv].bytes; oi++) {
			const unsigned v = order[oi];
			if (freq[v] * 2000 < S.n) /* under 0.05% of the sample */
				break;
			size_t k = 0;
			bool fresh = false;
			for (unsigned cl = 0; cl < P2_NCLASS; cl++) {
				if (cl == best.cls[v])
					continue;
				bool used = false;
				for (unsigned b = 0; b < 256 && !used; b++)
					used = best.cls[b] == cl && b != v;
				if (!used && fresh)
					continue; /* one empty class is enough */
				tabs[k] = best;
				tabs[k].cls[v] = (uint8_t)cl;
				if (!used) {
					/* v alone: a chunk of its own, no line */
					fresh = true;
					tabs[k].row[cl] = 0xFFFF;
					tabs[k].row2[cl] = 0;
					for (unsigned a = 0; a < P2_NCLASS; a++) {
						tabs[k].row[a] |= (uint16_t)(1u << cl);
						tabs[k].row2[a] &= (uint16_t)~(1u << cl);
					}
				}
				k++;
			}
			st = rt_step(&B, threads, tabs, k, &best, &bs, eps, &evals);
		}
		/* (b) the bits of the chunk matrix and of the line matrix, a
		 * row at a time, between the classes in use */
		bool used[P2_NCLASS] = {false};
		for (unsigned b = 0; b < 256; b++)
			if (freq[b])
				used[best.cls[b]] = true;
		for (unsigned a = 0; st == TAW_OK && a < P2_NCLASS; a++) {
			if (!used[a])
				continue;
			/* the best single change of the row, up to four times */
			for (unsigned round = 0; st == TAW_OK && round < 4; round++) {
				const double was = bs;
				size_t k = 0;
				for (unsigned b = 0; b < P2_NCLASS; b++) {
					if (!used[b])
						continue;
					const uint16_t bit = (uint16_t)(1u << b);
					tabs[k] = best; /* chunk boundary on or off */
					tabs[k].row[a] ^= bit;
					tabs[k].row2[a] &= tabs[k].row[a];
					k++;
					tabs[k] = best; /* line start on or off */
					tabs[k].row[a] |= bit;
					tabs[k].row2[a] ^= bit;
					k++;
				}
				st = rt_step(&B, threads, tabs, k, &best, &bs, eps,
				    &evals);
				if (bs == was)
					break;
			}
		}
		if (dbg && st == TAW_OK)
			fprintf(stderr, "taw fit: route: sweep %u: %.4f bpc "
					"(%zu tables tried)\n", sw,
			    bs / (double)(S.n ? S.n : 1), evals);
		if (bs > before * 0.9995)
			break;
	}
	if (st == TAW_OK) {
		if (dbg) {
			for (unsigned a = 0; a < P2_NCLASS; a++) {
				bool any = false;
				for (unsigned b = 0; b < 256 && !any; b++)
					any = best.cls[b] == a && freq[b];
				if (!any)
					continue;
				fprintf(stderr, "taw fit: route: class %2u chunk %04x "
						"line %04x:", a, best.row[a], best.row2[a]);
				for (unsigned b = 0; b < 256; b++)
					if (best.cls[b] == a && freq[b])
						fprintf(stderr, b > 32 && b < 127 ? " %c" : " %02x",
						    b);
				fprintf(stderr, "\n");
			}
		}
		*out = best;
	}
	for (unsigned w = 0; B.tab && w < threads; w++)
		free(B.tab[w]);
	free(B.tab);
	free(S.base);
	free(buf);
	return st;
}

/* A linear mixer over every candidate's inputs, one weight set per bit
 * position: cheap to fit, and enough to rank the candidate models. */
enum { LN_F = P1_NCTX_ALL + 1, LN_SUB = 1 << 17 };

static inline float lanes_z(const float w[8][LN_F], const sample *s,
    float x[LN_F])
{
	const float *wr = w[31 - __builtin_clz(s->c0)];
	float z = wr[LN_F - 1];
	x[LN_F - 1] = 1.0f;
	for (unsigned i = 0; i < P1_NCTX_ALL; i++) {
		x[i] = (float)s->st[i] * (1.0f / 256.0f);
		z += wr[i] * x[i];
	}
	return z;
}

static inline uint32_t lanes_p(float z)
{
	const float lim = 2047.0f / 256.0f;
	z = z > lim ? lim : z < -lim ? -lim : z;
	return p1_squash((int32_t)(z * 256.0f));
}

/* One pass of stochastic gradient steps over the subsample; inputs of
 * dropped models stay out. */
static void lanes_train(float w[8][LN_F], const sample *sm, size_t ns,
    size_t step, const bool *live, float lr0, float lr1)
{
	const size_t cnt = (ns + step - 1) / step;
	size_t done = 0;
	for (size_t k = 0; k < ns; k += step, done++) {
		float x[LN_F];
		const sample *s = &sm[k];
		const float p = (float)lanes_p(lanes_z(w, s, x)) * (1.0f / 65536.0f);
		const float lr = lr0 + (lr1 - lr0) * (float)done / (float)cnt;
		const float e = ((float)s->bit - p) * lr;
		float *wr = w[31 - __builtin_clz(s->c0)];
		for (unsigned i = 0; i < LN_F; i++)
			if (live[i])
				wr[i] += e * x[i];
	}
}

/* Choose the ten hashed models of a profile 2 shape from the fourteen the
 * feature pass ran: drop, four times, the model whose inputs the linear
 * mixer misses least. A rule of the samples alone. */
static taw_status lanes_choose(const sample *sm, size_t ns, p1_shape *s)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	const size_t step = ns > LN_SUB ? ns / LN_SUB : 1;
	float w[8][LN_F];
	bool live[LN_F], kept[P2_NCAND];
	for (unsigned i = 0; i < LN_F; i++)
		live[i] = true;
	for (unsigned j = 0; j < P2_NCAND; j++)
		kept[j] = true;
	for (unsigned b = 0; b < 8; b++)
		for (unsigned i = 0; i < LN_F; i++)
			w[b][i] = i < P1_NCTX_ALL ? 0.15f : 0.0f;
	lanes_train(w, sm, ns, step, live, 0.02f, 0.005f);
	const char *dbg = getenv("TAW_FIT_DEBUG");
	for (unsigned drop = 0; drop < P2_NCAND - P1_NHASH; drop++) {
		lanes_train(w, sm, ns, step, live, 0.005f, 0.002f);
		/* cost without each model, the others as fitted */
		double cost[P2_NCAND] = {0}, full = 0;
		for (size_t k = 0; k < ns; k += step) {
			float x[LN_F];
			const sample *sp = &sm[k];
			const float z = lanes_z(w, sp, x);
			const float *wr = w[31 - __builtin_clz(sp->c0)];
			uint32_t p = lanes_p(z);
			full += ct[sp->bit ? p : 65536 - p];
			for (unsigned j = 0; j < P2_NCAND; j++) {
				if (!kept[j])
					continue;
				const unsigned d = 2 + j, h = P1_NCM_ALL + 2 + j;
				p = lanes_p(z - wr[d] * x[d] - wr[h] * x[h]);
				cost[j] += ct[sp->bit ? p : 65536 - p];
			}
		}
		unsigned worst = P2_NCAND;
		for (unsigned j = 0; j < P2_NCAND; j++)
			if (kept[j] && (worst == P2_NCAND || cost[j] < cost[worst]))
				worst = j;
		if (dbg) {
			fprintf(stderr, "taw fit: lanes: full %.0f;", full);
			for (unsigned j = 0; j < P2_NCAND; j++)
				if (kept[j])
					fprintf(stderr, " %u:%+.0f", j, cost[j] - full);
			fprintf(stderr, "; dropping context %u\n", worst);
		}
		kept[worst] = false;
		live[2 + worst] = live[P1_NCM_ALL + 2 + worst] = false;
		for (unsigned b = 0; b < 8; b++)
			w[b][2 + worst] = w[b][P1_NCM_ALL + 2 + worst] = 0.0f;
	}
	unsigned t = 0;
	for (unsigned j = 0; j < P2_NCAND; j++)
		if (kept[j])
			s->lane[t++] = (uint8_t)j;
	s->all = false;
	return TAW_OK;
}

/* Fit the best `tries` candidates of the shape geo (its lanes included) and
 * leave the one with the smallest estimate in *best, if it beats *best_est
 * (in bits; start it at infinity). */
static taw_status fit_shape(const p1_shape *geo, uint64_t budget,
    const taw_params *p, uint64_t n, const sample *sm, size_t ns,
    taw_buf *best, double *best_est, uint8_t *best_quant)
{
	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	cand cs[1024];
	const size_t ncand = enumerate(geo, budget, p->weights, (size_t)n, cs,
	    1024);
	if (ncand == 0)
		return TAW_OK;
	const char *dbg = getenv("TAW_FIT_DEBUG");
	size_t tries = effort[lv].cands;
	if (dbg && strchr(dbg, 'a'))
		tries = ncand;
	if (tries > ncand)
		tries = ncand;

	fit_job F = {.cs = cs, .sm = sm, .ns = ns, .epochs = effort[lv].epochs,
		.n = n};
	F.model = calloc(tries, sizeof *F.model);
	F.est = calloc(tries, sizeof *F.est);
	F.bpb = calloc(tries, sizeof *F.bpb);
	F.st = calloc(tries, sizeof *F.st);
	taw_status st = TAW_OK;
	if (!F.model || !F.est || !F.bpb || !F.st)
		st = TAW_E_NOMEM;
	else
		st = fit_all(&F, tries, threads, p->gpu, dbg != nullptr);
	for (size_t ci = 0; st == TAW_OK && ci < tries; ci++) {
		if ((st = F.st[ci]) != TAW_OK)
			break;
		if (dbg)
			fprintf(stderr,
			    "taw fit: H=%u sel1=%u sel2=%u q=%u model=%zu "
			    "bpc=%.4f est=%.0f B\n",
			    cs[ci].s.hidden, cs[ci].s.sel1, cs[ci].s.sel2,
			    cs[ci].quant, F.model[ci].n, F.bpb[ci] * 8.0,
			    F.est[ci] / 8.0);
		if (F.est[ci] < *best_est) {
			*best_est = F.est[ci];
			*best_quant = cs[ci].quant;
			taw_buf_free(best);
			*best = F.model[ci];
			F.model[ci] = (taw_buf){0};
		}
	}
	for (size_t ci = 0; F.model && ci < tries; ci++)
		taw_buf_free(&F.model[ci]);
	free(F.model);
	free(F.est);
	free(F.bpb);
	free(F.st);
	return st;
}

taw_status taw_model_fit(const uint8_t *src, size_t n, size_t unit,
    const taw_params *p, taw_model_choice *out)
{
	taw_source in;
	taw_source_memory(&in, src, n);
	return taw_model_fit_source(&in, unit, p, out);
}

taw_status taw_model_fit_source(const taw_source *in, uint64_t unit,
    const taw_params *p, taw_model_choice *out)
{
	*out = (taw_model_choice){.profile = TAW_PROFILE_INT_ORDER0,
		.quant = TAW_QUANT_NONE};
	if (p->predictor == TAW_PREDICTOR_ORDER0)
		return TAW_OK;
	const uint64_t n = in->size;
	const bool chunk = p->predictor == TAW_PREDICTOR_CHUNK;
	const bool forced = chunk || p->predictor == TAW_PREDICTOR_MIX;
	const uint64_t budget = n / 32;
	if (unit == 0 || unit > n)
		unit = n;

	p1_shape geo = {0};
	geometry((size_t)unit, &geo);
	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	cand cs[1024];
	taw_status st = TAW_OK;
	if (chunk) {
		/* The feature pass runs every candidate model. Profile 2 is
		 * refused unless a model with its largest head (the routing
		 * table) fits the budget. */
		geo.chunk = geo.all = true;
		if (enumerate(&geo, budget, p->weights, (size_t)n, cs, 1024) == 0)
			return TAW_E_ARG;
		st = route_fit(in, p, &geo.route);
		if (st != TAW_OK)
			return st;
	} else if (enumerate(&geo, budget, p->weights, (size_t)n, cs, 1024) == 0) {
		return forced ? TAW_E_ARG : TAW_OK;
	}
	sample *sm;
	size_t ns;
	st = collect(in, unit, &geo, effort[lv].samples, threads, &sm, &ns);
	if (st != TAW_OK)
		return st;

	taw_buf best = {0};
	double best_est = 1e300;
	uint8_t best_quant = TAW_QUANT_NONE;
	if (chunk) {
		/* Two sets of lanes are fitted and the smaller estimate wins:
		 * the ten models that serve this input best by the linear
		 * ranking, and profile 1's ten, so that a ranking that
		 * misleads costs only the ten bytes of the lanes. */
		p1_shape own = geo;
		if ((st = lanes_choose(sm, ns, &own)) == TAW_OK)
			st = fit_shape(&own, budget, p, n, sm, ns, &best, &best_est,
			    &best_quant);
		if (st == TAW_OK && own.lane[P1_NHASH - 1] != P1_NHASH - 1) {
			p1_shape plain = geo;
			plain.all = false;
			for (unsigned k = 0; k < P1_NHASH; k++)
				plain.lane[k] = (uint8_t)k;
			st = fit_shape(&plain, budget, p, n, sm, ns, &best,
			    &best_est, &best_quant);
		}
	} else {
		st = fit_shape(&geo, budget, p, n, sm, ns, &best, &best_est,
		    &best_quant);
	}
	free(sm);
	if (st == TAW_OK && !best.p)
		st = forced ? TAW_E_ARG : TAW_OK; /* nothing fits the budget */
	else if (st == TAW_OK) {
		out->profile = chunk ? TAW_PROFILE_F32_STRICT_CHUNK :
				       TAW_PROFILE_F32_STRICT_MIX;
		out->quant = best_quant;
		out->model = best.p;
		out->model_size = best.n;
		best = (taw_buf){0};
	}
	taw_buf_free(&best);
	return st;
}
