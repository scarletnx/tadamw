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


#include "fit_internal.h"
#include "fit_opt.h"
#include "format.h"
#include "half.h"
#include "p1.h"
#include "quant.h"
#include "taw_internal.h"

/* fsqrt and the Newton-Schulz polynomial live in fit_opt.h (one copy). */

/* ---- weight averaging (encoder-only fitting experiment) --------------------
 * Averaging happens after the optimizer's per-set update and before the
 * existing quantize/serialize path; format and decoder are untouched.
 * A/B: TAW_FIT_AVG={off,swa,ema9,ema99,ema999,late,...}; the compile-time
 * default is OFF, so no override is byte-for-byte upstream. "late" (EMA
 * 0.999 for >= 6 epochs) is explicit opt-in only.
 *
 * This feature uses the shared optimizer ABI: pre_step initialises the
 * averaging state at un-tie, post_set folds in each touched set, finalize
 * writes the averaged weights back. No EMA/SWA state is threaded through
 * tnet_init/tnet_train signatures. */
enum { TAW_AVG_OFF = 0, TAW_AVG_SWA = 1, TAW_AVG_EMA9 = 2,
	TAW_AVG_EMA99 = 3, TAW_AVG_SWA_ALL = 4, TAW_AVG_EMA95 = 5,
	TAW_AVG_EMA999 = 6, TAW_AVG_LATE = 7 };
#ifndef TAW_FIT_AVG_DEFAULT
#define TAW_FIT_AVG_DEFAULT TAW_AVG_OFF
#endif

static int parse_avg_mode(const char *s)
{
	if (!s || !*s)
		return TAW_FIT_AVG_DEFAULT;
	if (!strcmp(s, "0") || !strcmp(s, "off"))
		return TAW_AVG_OFF;
	if (!strcmp(s, "1") || !strcmp(s, "swa"))
		return TAW_AVG_SWA;
	if (!strcmp(s, "2") || !strcmp(s, "ema9") || !strcmp(s, "ema09"))
		return TAW_AVG_EMA9;
	if (!strcmp(s, "3") || !strcmp(s, "ema99"))
		return TAW_AVG_EMA99;
	if (!strcmp(s, "4") || !strcmp(s, "swa_all"))
		return TAW_AVG_SWA_ALL;
	if (!strcmp(s, "5") || !strcmp(s, "ema95"))
		return TAW_AVG_EMA95;
	if (!strcmp(s, "6") || !strcmp(s, "ema999"))
		return TAW_AVG_EMA999;
	if (!strcmp(s, "7") || !strcmp(s, "late"))
		return TAW_AVG_LATE;
	return TAW_FIT_AVG_DEFAULT;
}

/* Averaging needs a long enough run: levels 1-2 (3-4 epochs) are slightly
 * hurt by EMA, levels 3+ (6-12 epochs) benefit. */
static int resolve_avg_mode(int mode, int epochs)
{
	if (mode == TAW_AVG_LATE)
		return epochs >= 6 ? TAW_AVG_EMA999 : TAW_AVG_OFF;
	return mode;
}

static bool mode_is_ema(int m)
{
	return m == TAW_AVG_EMA9 || m == TAW_AVG_EMA95 || m == TAW_AVG_EMA99 ||
	       m == TAW_AVG_EMA999;
}

static float mode_ema_decay(int m)
{
	switch (m) {
	case TAW_AVG_EMA9:  return 0.9f;
	case TAW_AVG_EMA95: return 0.95f;
	case TAW_AVG_EMA99: return 0.99f;
	default:            return 0.999f;
	}
}

/* d2[k] = decay^(2^k): binary exponentiation of the number of steps since a
 * set joined the EMA. No libm, no fast-math, no SIMD divergence. */
static inline float ema_pow(const float d2[64], uint64_t gap)
{
	float f = 1.0f;
	int k = 0;
	while (gap) {
		if (gap & 1u)
			f *= d2[k];
		gap >>= 1;
		k++;
	}
	return f;
}

/* Hook context: stack-local in fit_one, reached through opt_hooks.ctx. */
typedef struct {
	int mode;
	float d2[64];
	bool live;
	uint64_t nswa;
} avg_ctx;

/* Called at training start and again at un-tie. The tied flag is false when
 * averaging state must (re)initialise: untouched sets hold stale weights
 * before un-tie, and the EMA gaps must start from the un-tie step. */
static void avg_pre_step(void *v, struct tnet *t, optcfg *o, uint64_t step,
    bool tied)
{
	avg_ctx *a = v;
	(void)o;
	if (!a || a->mode == TAW_AVG_OFF || tied)
		return;
	size_t np = t->n1 + t->n2;
	if (mode_is_ema(a->mode)) {
		memcpy(t->wa, t->w, np * sizeof(float));
		for (size_t k = 0; k < t->s1 + 1; k++)
			t->last1[k] = step;
		for (size_t k = 0; k < t->s2; k++)
			t->last2[k] = step;
		a->live = true;
	} else {
		/* SWA: discard anything accumulated before un-tie. */
		memset(t->wa, 0, np * sizeof(float));
		a->nswa = 0;
	}
}

/* One touched set's post-update fold-in. layer is 1 (W1) or 2 (W2); off/len
 * are the row's slice in t->w. Untouched weights are constant over the gap,
 * so decaying toward the current w is the time-weighted EMA. */
static void avg_post_set(void *v, struct tnet *t, unsigned layer, unsigned set,
    size_t off, size_t len, uint64_t step)
{
	avg_ctx *a = v;
	if (!a || !a->live || !mode_is_ema(a->mode))
		return;
	uint64_t *last = layer == 1 ? &t->last1[set] : &t->last2[set];
	float d = ema_pow(a->d2, step - *last);
	float *av = t->wa + off, *w = t->w + off;
	for (size_t i = 0; i < len; i++)
		av[i] = d * av[i] + (1.0f - d) * w[i];
	*last = step;
}

/* Final write-back: the quantizer sees the averaged weights. */
static void avg_finalize(void *v, struct tnet *t)
{
	avg_ctx *a = v;
	if (!a || a->mode == TAW_AVG_OFF)
		return;
	size_t np = t->n1 + t->n2;
	if (mode_is_ema(a->mode)) {
		if (a->live)
			memcpy(t->w, t->wa, np * sizeof(float));
	} else if (a->nswa) {
		float inv = 1.0f / (float)a->nswa;
		for (size_t i = 0; i < np; i++)
			t->w[i] = t->wa[i] * inv;
	}
}

static void avg_hooks_init(opt_hooks *h, avg_ctx *a, int mode)
{
	memset(a, 0, sizeof *a);
	a->mode = mode;
	if (mode_is_ema(mode)) {
		const float decay = mode_ema_decay(mode);
		a->d2[0] = decay;
		for (int k = 1; k < 64; k++)
			a->d2[k] = a->d2[k - 1] * a->d2[k - 1];
	}
	memset(h, 0, sizeof *h);
	h->ctx = a;
	h->pre_step = avg_pre_step;
	h->post_set = avg_post_set;
	h->finalize = avg_finalize;
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

static inline void sample_x(const sample *s, float x[P1_XPAD])
{
	for (unsigned i = P1_F; i < P1_XPAD; i++)
		x[i] = 0.0f;
	for (unsigned i = 0; i < P1_NCTX; i++)
		x[i] = (float)s->st[i] * (1.0f / 256.0f);
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
			float xv[32];
			p1_sel_ctx sc;
			p1_ctx_inputs(c, xv, &sc); /* the update needs them */
			if (idx[j + 1] != idx[j]) {
				sample *s = &J->sm[idx[j]];
				for (unsigned q = 0; q < P1_NCTX; q++) /* exact */
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
	free(t->m1);
	free(t->v1);
	free(t->malt_l);
	free(t->malt_r);
	free(t->malt_nu);
	free(t->v2s);
	free(t->v2c);
	free(t->sig);
	free(t->touched1);
	free(t->touched2);
	free(t->list1);
	free(t->list2);
	free(t->cnt1);
	free(t->route1);
	free(t->wa);
	free(t->last1);
	free(t->last2);
	memset(t, 0, sizeof *t);
}

static taw_status tnet_init(tnet *t, const p1_shape *s, uint64_t seed)
{
	memset(t, 0, sizeof *t);
	t->s = *s;
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
	t->m1 = calloc(t->n1 ? t->n1 : 1, sizeof(float));
	t->v1 = calloc(t->n1 ? t->n1 : 1, sizeof(float));
	t->malt_l = calloc(t->s1 * H + 1, sizeof(float));
	t->malt_r = calloc(t->s1 * P1_F + 1, sizeof(float));
	t->malt_nu = calloc(t->s1 + 1, sizeof(float));
	t->v2s = calloc(t->s1 ? t->s1 : 1, sizeof(float));
	t->v2c = calloc(t->n1 ? t->s1 * P1_HMAX : 1, sizeof(float));
	t->sig = calloc(t->n1 ? t->n1 : 1, sizeof(float));
	t->touched1 = calloc(t->s1 + 1, 1);
	t->touched2 = calloc(t->s2, 1);
	t->list1 = malloc((t->s1 + 1) * sizeof(uint32_t));
	t->list2 = malloc(t->s2 * sizeof(uint32_t));
	t->cnt1 = calloc(t->s1 ? t->s1 : 1, sizeof(uint32_t));
	t->route1 = calloc(t->s1 ? t->s1 : 1, 1);
	/* Averaging scratch: allocated unconditionally (like upstream's MALT
	 * state) so tnet_init keeps its shipped signature. It stays untouched
	 * unless the hooks are enabled. */
	t->wa = calloc(np, sizeof(float));
	t->last1 = calloc(t->s1 + 1, sizeof(uint64_t));
	t->last2 = calloc(t->s2, sizeof(uint64_t));
	if (!t->w || !t->m || !t->v || !t->g || !t->m1 || !t->v1 ||
	    !t->malt_l || !t->malt_r || !t->malt_nu ||
	    !t->v2s || !t->v2c || !t->sig ||
	    !t->touched1 || !t->touched2 || !t->list1 || !t->list2 ||
	    !t->cnt1 || !t->route1 || !t->wa || !t->last1 || !t->last2) {
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
	sample_x(smp, x);
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
	float *arr[5] = {t->w, t->m, t->v, t->m1, t->v1};
	for (int a = 0; a < 5; a++) {
		float *b = arr[a];
		for (size_t set = 1; set < t->s1; set++)
			memcpy(b + set * t->row1, b, t->row1 * sizeof(float));
		if (a >= 3)
			continue; /* W1-only buffers */
		for (size_t set = 1; set < t->s2; set++)
			memcpy(b + t->n1 + set * t->row2, b + t->n1,
			    t->row2 * sizeof(float));
	}
	/* MALT W1 state: copy set 0's preconditioner EMAs and scalar to all sets. */
	const unsigned H = t->s.hidden;
	for (size_t set = 1; set < t->s1; set++) {
		memcpy(t->malt_l + set * H, t->malt_l, H * sizeof(float));
		memcpy(t->malt_r + set * P1_F, t->malt_r,
		    P1_F * sizeof(float));
		t->malt_nu[set] = t->malt_nu[0];
	}
	/* NAMO / NAMO-D second moments and the shared sign EMA are W1-only state. */
	for (size_t set = 1; set < t->s1; set++) {
		t->v2s[set] = t->v2s[0];
		memcpy(t->v2c + set * P1_HMAX, t->v2c, P1_HMAX * sizeof(float));
		memcpy(t->sig + set * t->row1, t->sig,
		    t->row1 * sizeof(float));
	}
	t->tied = false;
}

/* ---- optimizers: AdamW, Muon and hybrids -------------------------------------
 *
 * W1 of the fitted mixer is a per-set H x P1_F matrix (one row per hidden
 * unit); W2 rows are vectors (and include the output bias). Muon keeps one
 * momentum buffer and orthogonalizes the (Nesterov-extrapolated) matrix
 * update with Newton-Schulz; W2 stays on AdamW. Hybrid modes either blend the
 * two W1 updates, alternate them per step, or route each W1 set to one of
 * them. AdamW state for W1 is kept in t->m1/t->v1 so it never aliases Muon's
 * momentum in t->m.
 *
 * The configuration types (hyb_mode, optcfg), the dispatch ABI (opt_plan,
 * opt_hooks) and the thin wrappers all live in fit_opt.h / fit_opt.c. This
 * file keeps only env parsing, the LR schedule and the per-set dispatch.
 */
/* Tuning hooks (encoder-only; unset = shipped behaviour). */
static float fit_env_float(const char *name, float def)
{
	const char *s = getenv(name);
	if (!s || !*s)
		return def;
	char *end;
	float v = strtof(s, &end);
	return end != s ? v : def;
}

static unsigned fit_env_uint(const char *name, unsigned def)
{
	const char *s = getenv(name);
	if (!s || !*s)
		return def;
	char *end;
	unsigned long v = strtoul(s, &end, 10);
	return end != s ? (unsigned)v : def;
}

/* The optimizer wrappers (adam_rows, muon_rows, malt_rows), ns_score() and
 * the single Newton-Schulz polynomial now live in fit_opt.c / fit_opt.h.
 * Their FP expressions were moved verbatim; see tests/test_fit_opt.c. */

static const char *fit_env_str(const char *name, const char *def)
{
	const char *s = getenv(name);
	return s && *s ? s : def;
}

/* ---- FARO: function-space adaptive return-to-risk allocation -----------------
 *
 * An opt-in, encoder-only approximation of arXiv:2610.40858. Two decisions on
 * the fitted mixer's parameter blocks (W1 matrices = spectral/Muon candidate,
 * W2 vectors = coordinate/AdamW candidate):
 *
 *   route  - during a short post-untie exploration, compare the two candidate
 *            geometries on a deterministic in-batch probe and freeze after
 *            un-tie. Default (TAW_FARO_STAT=probe): apply each geometry's real
 *            update to a shadow copy of the block and score the one-step loss
 *            decrease in bits (Eq. 1). TAW_FARO_STAT=lin uses the cheaper
 *            first-order proxy ||g||_1/sqrt(N) vs <g, NS(g)>/||NS(g)||_F,
 *            both per unit update (Appendix G's linearized return).
 *   alloc  - after the freeze, per-block risk is the EWMA of the update-to-
 *            weight ratio ||Δ_b|| / (||θ_b|| + eps), measured on the *previous*
 *            step. Winsorize against the cross-block median, then solve
 *            Λ(λ) = Σ β̄_b clip(λ / r̃_b, 1/c, c) = Σ β̄_b by bisection and
 *            apply μ_b = clip(λ/r̃_b, 1/c, c) to the block's LR. Total budget
 *            is preserved exactly (up to the optional μ clamp).
 *
 * The paper's counterfactual uses a disjoint held-out probe shard and follows
 * the full training trajectory; here the probe is in-batch, one step long, and
 * only W1 has a second candidate geometry.
 */
typedef struct {
	bool on, route_ret, alloc, frozen, debug;
	bool blocks_set;   /* false: 2 layer blocks, true: one block per set */
	int stat;          /* 0 = probe loss (default), 1 = linearized return */
	unsigned explore;  /* post-untie batches spent measuring returns */
	int nb, nw1;       /* total blocks, W1 blocks (first nw1) */
	uint64_t untie_step, freeze_step;
	float gam, kappa, cap, floor, ceilv;
	float *usum;       /* Σ Δ² accumulated in the current step, [nb] */
	float *risk;       /* pointer to usum while allocating, else NULL */
	float *r;          /* risk EWMA [nb] */
	float *mu;         /* budget multiplier [nb] */
	float *scratch;    /* [3*nb]: median, winsorized risk, reference budgets */
	float *ret0, *ret1;/* coordinate vs spectral return [nw1] */
	float *wtmp, *da0, *da1; /* shadow update scratch [P1_HMAX*P1_F] */
	uint8_t *geo;      /* 1 = Muon/spectral, 0 = Adam/coordinate [nw1] */
} faro_ctx;

static inline int faro_b1(const faro_ctx *f, unsigned set)
{
	return f->blocks_set ? (int)set : 0;
}

static inline int faro_b2(const faro_ctx *f, unsigned set)
{
	return f->blocks_set ? (int)(f->nw1 + set) : 1;
}

static void faro_free(faro_ctx *f)
{
	free(f->usum); free(f->r); free(f->mu); free(f->scratch);
	free(f->ret0); free(f->ret1); free(f->geo);
	free(f->wtmp); free(f->da0); free(f->da1);
	memset(f, 0, sizeof *f);
}

static void faro_init(faro_ctx *f, const tnet *t, int epochs, size_t ns)
{
	memset(f, 0, sizeof *f);
	const char *on = fit_env_str("TAW_FARO", "0");
	if (strcmp(on, "1") && strcmp(on, "on") && strcmp(on, "yes") &&
	    strcmp(on, "true"))
		return;
	const char *route = fit_env_str("TAW_FARO_ROUTE", "ret");
	f->alloc = strcmp(fit_env_str("TAW_FARO_ALLOC", "erc"), "none") != 0;
	f->route_ret = strcmp(route, "ret") == 0;
	f->stat = strcmp(fit_env_str("TAW_FARO_STAT", "probe"), "lin") == 0;
	f->debug = fit_env_uint("TAW_FARO_DEBUG", 0) != 0;
	f->blocks_set = strcmp(fit_env_str("TAW_FARO_BLOCKS", "set"), "layer") != 0;
	f->nw1 = f->blocks_set ? (int)t->s1 : (t->s1 ? 1 : 0);
	f->nb = f->blocks_set ? (int)(t->s1 + t->s2) : (t->n1 ? 2 : 1);
	if (f->nb < 1 || f->nb > 8192)
		return;
	if (!f->route_ret) /* forced muon/adam: nothing to measure */
		f->explore = 0;
	else
		f->explore = fit_env_uint("TAW_FARO_EXPLORE", 20);
	f->gam = fit_env_float("TAW_FARO_GAMMA", 0.9f);
	f->kappa = fit_env_float("TAW_FARO_KAPPA", 3.0f);
	f->cap = fit_env_float("TAW_FARO_CAP", 1.4f);
	f->floor = fit_env_float("TAW_FARO_FLOOR", 0.5f);
	f->ceilv = fit_env_float("TAW_FARO_CEIL", 2.0f);
	if (f->gam < 0.0f || f->gam >= 1.0f) f->gam = 0.9f;
	if (f->kappa < 1.0f) f->kappa = 3.0f;
	if (f->cap < 1.0f) f->cap = 1.4f;
	if (f->floor <= 0.0f) f->floor = 1.0f / f->cap;
	if (f->ceilv < 1.0f) f->ceilv = f->cap;
	f->untie_step = (uint64_t)fit_warm_epochs(epochs) *
	    ((ns + FIT_BATCH - 1) / FIT_BATCH);
	f->freeze_step = f->untie_step + f->explore;
	f->usum = calloc((size_t)f->nb, sizeof *f->usum);
	f->r = calloc((size_t)f->nb, sizeof *f->r);
	f->mu = malloc((size_t)f->nb * sizeof *f->mu);
	f->scratch = malloc(3u * (size_t)f->nb * sizeof *f->scratch);
	f->ret0 = calloc((size_t)(f->nw1 ? f->nw1 : 1), sizeof *f->ret0);
	f->ret1 = calloc((size_t)(f->nw1 ? f->nw1 : 1), sizeof *f->ret1);
	f->geo = malloc((size_t)(f->nw1 ? f->nw1 : 1));
	const size_t nw = (size_t)P1_HMAX * P1_F;
	f->wtmp = malloc(nw * sizeof *f->wtmp);
	f->da0 = malloc(nw * sizeof *f->da0);
	f->da1 = malloc(nw * sizeof *f->da1);
	if (!f->usum || !f->r || !f->mu || !f->scratch || !f->ret0 ||
	    !f->ret1 || !f->geo || !f->wtmp || !f->da0 || !f->da1) {
		faro_free(f);
		return;
	}
	for (int b = 0; b < f->nb; b++)
		f->mu[b] = 1.0f; /* identity until the freeze */
	/* The forced arms pick their geometry up front. */
	for (int b = 0; b < f->nw1; b++)
		f->geo[b] = (uint8_t)(strcmp(route, "adam") != 0);
	f->on = true;
}

/* Linearized return of the spectral candidate on one W1 matrix: <g, NS(g)>
 * divided by ||NS(g)||_F. The NS polynomial is not duplicated here: FARO's
 * TAW_FARO_STAT=lin lane calls the shared ns_score() in fit_opt.c. */
static inline float faro_ns_score(const optcfg *o, const float *g,
    unsigned rows, unsigned cols)
{
	return ns_score(o, g, rows, cols);
}

/* Both scores are per-unit-update: ||g||_1/sqrt(N) is the RMS absolute
 * gradient (coordinate geometry), the other the mean singular value. */
static void faro_explore(faro_ctx *f, const optcfg *o, const float *g,
    int b, unsigned H)
{
	if (b < 0 || b >= f->nw1)
		return;
	double l1 = 0.0;
	const size_t n = (size_t)H * P1_F;
	for (size_t i = 0; i < n; i++) {
		float v = g[i];
		l1 += v < 0.0f ? -v : v;
	}
	f->ret0[b] += (float)l1 / fsqrt((float)n);
	f->ret1[b] += faro_ns_score(o, g, H, P1_F);
}

/* Forward-only cost of one probe sample, in the same bits the fitter uses.
 * Samples routed to another W1 set are skipped so the return stays per-block. */
static double faro_probe_loss(const tnet *t, const sample *s, unsigned set)
{
	const unsigned H = t->s.hidden;
	const p1_sel_ctx sc = sample_sc(s);
	if (!t->tied && p1_sel_value(t->s.sel1, &sc) != set)
		return 0.0;
	float x[P1_XPAD];
	sample_x(s, x);
	const uint32_t s2 = t->tied ? 0 : p1_sel_value(t->s.sel2, &sc);
	const float *w2 = t->w + t->n1 + (size_t)s2 * t->row2;
	float z = 0.0f;
	if (H) {
		const float *w1 = t->w + (size_t)set * t->row1;
		for (unsigned j = 0; j < H; j++) {
			float acc = 0.0f;
			const float *r = w1 + j * P1_F;
			for (unsigned i = 0; i < P1_F; i++)
				acc += r[i] * x[i];
			z += w2[j] * clamp8(acc);
		}
		z += w2[H];
	} else {
		for (unsigned i = 0; i < P1_F; i++)
			z += w2[i] * x[i];
	}
	const float lim = 2047.0f / 256.0f;
	float zc = z > lim ? lim : z < -lim ? -lim : z;
	uint32_t p = p1_squash((int32_t)(zc * 256.0f));
	const float *ct = cost_tab();
	if (!ct)
		return 0.0;
	unsigned idx = s->bit ? p : 65536u - p;
	if (idx > 65535)
		idx = 65535;
	return ct[idx];
}

/* Eq. (1) on the collected samples: apply each candidate geometry's real
 * update to a shadow copy of this block, measure the probe loss change, undo.
 * The two states are copied, so exploration never perturbs the training path. */
static void faro_cf_probe(faro_ctx *f, tnet *t, const optcfg *o, unsigned set,
    const float *g, float scale, const sample *const *probe, size_t nprobe)
{
	int b = faro_b1(f, set);
	if (b < 0 || b >= f->nw1 || !t->s.hidden)
		return;
	const size_t len = (size_t)t->s.hidden * P1_F;
	const size_t off = (size_t)set * t->row1;
	float ms[P1_HMAX * P1_F], m1s[P1_HMAX * P1_F], vs[P1_HMAX * P1_F];
	float ws[P1_HMAX * P1_F];
	if (len > sizeof ms / sizeof *ms)
		return;
	memcpy(ms, t->m + off, len * sizeof *ms);
	memcpy(m1s, t->m1 + off, len * sizeof *m1s);
	memcpy(vs, t->v1 + off, len * sizeof *vs);
	memcpy(ws, t->w + off, len * sizeof *ws);
	memset(f->wtmp, 0, len * sizeof *f->wtmp);
	muon_rows(o, f->wtmp, ms, g, t->s.hidden, P1_F, scale, o->lr, 0.0f,
	    1.0f, nullptr, 0);
	memcpy(f->da1, f->wtmp, len * sizeof *f->da1); /* -Δ spectral */
	memset(f->wtmp, 0, len * sizeof *f->wtmp);
	optcfg local = *o; /* adam_rows takes a mutable cfg; keep o const */
	adam_rows(&local, f->wtmp, m1s, vs, g, len, scale, o->lr1, 0.0f, 1.0f,
	    nullptr, 0);
	memcpy(f->da0, f->wtmp, len * sizeof *f->da0); /* -Δ coordinate */
	double before = 0.0;
	for (size_t k = 0; k < nprobe; k++)
		before += faro_probe_loss(t, probe[k], set);
	for (int cand = 0; cand < 2; cand++) {
		const float *da = cand ? f->da1 : f->da0;
		for (size_t i = 0; i < len; i++)
			t->w[off + i] += da[i]; /* wtmp holds -Δ, so += applies Δ */
		double after = 0.0;
		for (size_t k = 0; k < nprobe; k++)
			after += faro_probe_loss(t, probe[k], set);
		if (cand)
			f->ret1[b] += (float)(before - after);
		else
			f->ret0[b] += (float)(before - after);
		memcpy(t->w + off, ws, len * sizeof *ws);
	}
}

static void faro_freeze(faro_ctx *f)
{
	const char *route = fit_env_str("TAW_FARO_ROUTE", "ret");
	for (int b = 0; b < f->nw1; b++)
		f->geo[b] = (uint8_t)(strcmp(route, "adam") == 0 ? 0 :
		    strcmp(route, "muon") == 0 ? 1 : f->ret1[b] >= f->ret0[b]);
	f->frozen = true;
	if (f->debug) {
		fprintf(stderr, "FARO: frozen at step %llu, blocks %d/%d\n",
		    (unsigned long long)f->freeze_step, f->nw1, f->nb);
		for (int b = 0; b < f->nw1; b++)
			fprintf(stderr, "FARO: W1 block %d return coord=%.9g "
			    "spectral=%.9g -> %s\n", b, f->ret0[b], f->ret1[b],
			    f->geo[b] ? "spectral" : "coordinate");
	}
}

/* Update-to-weight ratio from the applied Δ and the current weights. The
 * allocation at step t only ever reads r(t-1), so no gradient leaks into its
 * own budget. */
static void faro_measure(faro_ctx *f, const tnet *t)
{
	for (int b = 0; b < f->nb; b++) {
		const float *w;
		size_t len;
		if (f->blocks_set && b >= f->nw1) {
			size_t set = (size_t)(b - f->nw1);
			w = t->w + t->n1 + set * t->row2;
			len = t->row2;
		} else if (f->blocks_set) {
			w = t->w + (size_t)b * t->row1;
			len = t->row1;
		} else if (b == 0) {
			w = t->w;
			len = t->n1;
		} else {
			w = t->w + t->n1;
			len = t->n2;
		}
		double sw = 0.0;
		for (size_t i = 0; i < len; i++)
			sw += (double)w[i] * w[i];
		float uwr = fsqrt(f->usum[b]) / (fsqrt((float)sw) + 1e-6f);
		f->r[b] = f->r[b] > 0.0f ? f->gam * f->r[b] + (1.0f - f->gam) * uwr
					 : uwr;
		f->usum[b] = 0.0f;
	}
}

static double faro_lambda_of(double lam, int nb, const float *rt,
    const float *ref, double invc, double cap)
{
	double s = 0.0;
	for (int b = 0; b < nb; b++) {
		double u = lam / (double)rt[b];
		if (u < invc) u = invc;
		if (u > cap) u = cap;
		s += (double)ref[b] * u;
	}
	return s;
}

/* Cross-block ERC, Algorithm 1 of the paper: winsorized risk, capped
 * multiplier, budget-neutral bisection. β̄_b is the block's reference LR. */
static void faro_allocate(faro_ctx *f, const optcfg *o)
{
	float *med = f->scratch, *rt = f->scratch + f->nb;
	float *ref = f->scratch + 2 * f->nb;
	for (int b = 0; b < f->nb; b++)
		med[b] = f->r[b];
	for (int i = 1; i < f->nb; i++) { /* insertion sort, nb is tiny */
		float v = med[i];
		int j = i;
		while (j && med[j - 1] > v) {
			med[j] = med[j - 1];
			j--;
		}
		med[j] = v;
	}
	const float m = med[f->nb / 2];
	double bud = 0.0;
	for (int b = 0; b < f->nb; b++) {
		bool is_w1 = f->blocks_set ? b < f->nw1 : b == 0;
		ref[b] = is_w1 ? (f->geo && f->geo[b] ? o->lr : o->lr1) : o->lrvec;
		bud += ref[b];
		float lo = m > 0.0f ? m / f->kappa : 1.0f;
		float hi = m > 0.0f ? m * f->kappa : 1.0f;
		rt[b] = f->r[b] < lo ? lo : f->r[b] > hi ? hi : f->r[b];
	}
	if (!(bud > 0.0))
		return;
	const double c = f->cap, invc = 1.0 / c;
	double lo = 0.0, hi = (double)rt[0] * c * 2.0;
	for (int b = 1; b < f->nb; b++) {
		double t = (double)rt[b] * c * 2.0;
		if (t > hi) hi = t;
	}
	while (faro_lambda_of(hi, f->nb, rt, ref, invc, c) < bud && hi < 1e30)
		hi *= 2.0;
	for (int it = 0; it < 48; it++) {
		double mid = 0.5 * (lo + hi);
		if (faro_lambda_of(mid, f->nb, rt, ref, invc, c) < bud)
			lo = mid;
		else
			hi = mid;
	}
	const double lam = 0.5 * (lo + hi);
	for (int b = 0; b < f->nb; b++) {
		double u = lam / (double)rt[b];
		if (u < invc) u = invc;
		if (u > c) u = c;
		if (u < f->floor) u = f->floor;
		if (u > f->ceilv) u = f->ceilv;
		f->mu[b] = (float)u;
	}
	if (f->debug) { /* neutrality diagnostic: Σ β̄ μ should equal Σ β̄ */
		double chk = 0.0;
		for (int b = 0; b < f->nb; b++)
			chk += (double)ref[b] * f->mu[b];
		double err = chk - bud;
		if (err < 0.0) err = -err;
		if (err > 1e-4 * bud)
			fprintf(stderr, "FARO: budget drift %.3g / %.3g\n", err, bud);
	}
}

static void faro_report(const faro_ctx *f)
{
	if (!f->on || !f->debug)
		return;
	float lo = 1.0f, hi = 1.0f;
	for (int b = 0; b < f->nb; b++) {
		if (f->mu[b] < lo) lo = f->mu[b];
		if (f->mu[b] > hi) hi = f->mu[b];
	}
	fprintf(stderr, "FARO: final multiplier range %.4g..%.4g, risks:", lo, hi);
	for (int b = 0; b < f->nb; b++)
		fprintf(stderr, " %.4g", f->r[b]);
	fprintf(stderr, "\n");
}

/* One shared step_shape adapter for NAMO's optional shrink gate. The gate
 * arithmetic is fit_gate_step() in fit_opt.c; there is no second owner. */
typedef struct {
	float *sig;  /* shared sign EMA (t->sig) */
	float *mul;  /* per-coordinate bank filled by begin(), read by gate() */
	float beta, p;
} namo_gate;

static void namo_gate_begin(void *v, const float *g, size_t len, float scale)
{
	namo_gate *c = v;
	for (size_t k = 0; k < len; k++)
		c->mul[k] = fit_gate_step(&c->sig[k], g[k] * scale,
		    c->beta, c->p);
}

static float namo_gate_at(void *v, size_t k, float gk)
{
	namo_gate *c = v;
	(void)gk; /* the multiplier bank was filled by begin() */
	return c->mul[k];
}

/* ---- C3 sign-consistency gate (opt-in, CPU only) ------------------------------
 * The gate arithmetic itself belongs to the foundation: fit_gate_step() in
 * fit_opt.c is the single implementation. Here we only parse knobs and wire
 * the foundation's step_shape slot into the AdamW vector updates. Muon/MALT/
 * MALTER W1 paths are never shaped. Default TAW_EPS_MODE=none is bit-exact
 * upstream behaviour. */

enum { SCOPE_ALL = 0, SCOPE_COLD, SCOPE_WARM };

typedef struct {
	bool enabled;
	float beta, p;
	bool mean_pres;  /* mean-preserving vs shrink-only */
	int layers;      /* bit 0: layer 1 (AdamW path), bit 1: layer 2 */
	int scope;       /* SCOPE_* (touch-count stratification) */
	bool debug;
} eps_cfg;

static bool eps_mode_off(const char *m)
{
	return !m || !*m || !strcmp(m, "none") || !strcmp(m, "off") ||
	    !strcmp(m, "0");
}

/* Only the literal "c3" enables the gate; c2/nc9/junk are errors, never an
 * implicit enable. */
static bool eps_mode_enabled(void)
{
	const char *m = getenv("TAW_EPS_MODE");
	return m && !strcmp(m, "c3");
}

/* NAMO-only knobs (including the shared TAW_EPS_MODE gate) are inert off-NAMO,
 * while the C3 gate is enabled on plain AdamW. The NAMO leak guard takes
 * precedence whenever NAMO-only knobs are set but NAMO geometry is not
 * selected; configurations with no NAMO knobs keep the gate. */
static bool namo_knobs_present(void)
{
	static const char *const names[] = {"TAW_NAMO_BETA", "TAW_NAMO_SCALE",
		"TAW_NAMO_CLAMP", "TAW_NAMO_WD", "TAW_NAMO_D_AXIS",
		"TAW_NAMO_NEST"};
	for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
		const char *v = getenv(names[i]);
		if (v && *v)
			return true;
	}
	return false;
}

static taw_status eps_mode_validate(void)
{
	const char *m = getenv("TAW_EPS_MODE");
	if (eps_mode_off(m) || eps_mode_enabled())
		return TAW_OK;
	fprintf(stderr, "taw fit: TAW_EPS_MODE=%s is invalid (use none|c3)\n",
	    m ? m : "");
	return TAW_E_ARG;
}

static eps_cfg eps_config(void)
{
	eps_cfg c = {.enabled = eps_mode_enabled(), .beta = 0.9f, .p = 1.0f,
		.mean_pres = true, .layers = 2, .scope = SCOPE_ALL};
	c.beta = fit_env_float("TAW_EPS_SIGN_BETA", c.beta);
	c.p = fit_env_float("TAW_EPS_GATE_P", c.p);
	const char *gm = getenv("TAW_EPS_GATE_MODE");
	if (gm && !strcmp(gm, "shrink")) c.mean_pres = false;
	const char *l = getenv("TAW_EPS_LAYERS");
	if (l) {
		c.layers = 0;
		if (strchr(l, '1')) c.layers |= 1;
		if (strchr(l, '2')) c.layers |= 2;
	}
	const char *sc = getenv("TAW_EPS_SCOPE");
	if (sc) {
		if (!strcmp(sc, "cold")) c.scope = SCOPE_COLD;
		else if (!strcmp(sc, "warm")) c.scope = SCOPE_WARM;
	}
	c.debug = getenv("TAW_EPS_DEBUG") != nullptr;
	return c;
}

/* Scratch bank covers any mixer width (P1_XPAD >= P1_F); out-of-range rows
 * simply bypass the gate instead of reading out of bounds. */
enum { EPS_GATE_CAP = P1_HMAX * P1_XPAD + 1 };

typedef struct {
	float *sig;      /* sign EMA, one float per weight */
	size_t off;      /* current weight-row offset into sig */
	float beta, p;
	bool mean_pres;
	float bank[EPS_GATE_CAP];
	float gmean;
	bool over;
	float acc;       /* debug: accumulated row-mean gate */
} c3_ctx;

static void c3_begin(void *vctx, const float *g, size_t len, float scale)
{
	c3_ctx *c = vctx;
	c->over = len > EPS_GATE_CAP;
	if (c->over) {
		c->gmean = 1.0f;
		return;
	}
	float sum = 0.0f;
	float *sig = c->sig + c->off;
	for (size_t k = 0; k < len; k++) {
		c->bank[k] = fit_gate_step(sig + k, g[k] * scale, c->beta, c->p);
		sum += c->bank[k];
	}
	c->gmean = c->mean_pres ? sum / (float)len : 1.0f;
	if (c->gmean < 1e-12f) /* sign EMA not established yet: no gating */
		c->gmean = 1.0f;
}

static float c3_gate(void *vctx, size_t k, float gk)
{
	c3_ctx *c = vctx;
	(void)gk;
	if (c->over)
		return 1.0f;
	float fac = c->mean_pres ? c->bank[k] / c->gmean : c->bank[k];
	c->acc += fac;
	return fac;
}

/* Per-set diagnostics for the touch-count stratification. */
typedef struct {
	double touches, sum_gate;
} eps_stats;

/* Median of the positive entries of a[] (0 if none), for the cold/warm split.
 * Frequencies come from the sample selector values, not the training order. */
static size_t positive_median(const size_t *a, size_t n)
{
	size_t *v = malloc((n ? n : 1) * sizeof *v);
	if (!v)
		return 0;
	size_t m = 0;
	for (size_t i = 0; i < n; i++)
		if (a[i])
			v[m++] = a[i];
	if (!m) {
		free(v);
		return 0;
	}
	for (size_t i = 1; i < m; i++) { /* insertion sort; n <= 4096 */
		size_t x = v[i], j = i;
		while (j && v[j - 1] > x) {
			v[j] = v[j - 1];
			j--;
		}
		v[j] = x;
	}
	size_t med = v[m / 2];
	free(v);
	return med;
}

static inline bool in_scope(int scope, size_t freq, size_t med)
{
	if (scope == SCOPE_ALL || !freq)
		return scope == SCOPE_ALL;
	return scope == SCOPE_COLD ? freq < med : freq >= med;
}

/* Train on samples for the given epochs. Batches touch few weight sets, so
 * only those rows are updated ("lazy" optimizer state). With no environment
 * overrides this is the shipped AdamW fitter, bit for bit. */
static void tnet_train(tnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed, const opt_hooks *hooks_in)
{
	enum { BATCH = FIT_BATCH };
	/* One hook ABI. FARO publishes its per-block risk bank through hooks.risk;
	 * with TAW_FARO off the bank stays NULL and every shipped path is bit-exact. */
	opt_hooks hooks = hooks_in ? *hooks_in : (opt_hooks){0};
	opt_hooks *h = &hooks; /* EMA hooks copied in; C3 fills shape */
	const size_t np = t->n1 + t->n2;
	/* One hook ABI: C3 fills shape, EMA fills the pre/post hooks in fit_one,
	 * NAMO uses its own W1-only copy, and the base FARO owns risk/nb. */
	opt_hooks namo_hooks = {0};
	namo_gate ng = {0};
	float nmul[NS_MAXEL]; /* shrink bank, reused for each W1 set */
	const eps_cfg ecfg = eps_config();
	c3_ctx c3 = {0};
	const char *early_hs = fit_env_str("TAW_HYBRID", "adam");
	const bool early_namo = !strcmp(early_hs, "namo") ||
	    !strcmp(early_hs, "namod") || !strcmp(early_hs, "namo-d");
	/* NAMO owns its own gate through namo_hooks; never also gate W2 here. */
	if (ecfg.enabled && !early_namo && !namo_knobs_present()) {
		c3.beta = ecfg.beta;
		c3.p = ecfg.p;
		c3.mean_pres = ecfg.mean_pres;
		/* shape has its own ctx; hooks.ctx stays owned by EMA. */
		hooks.shape = (step_shape){&c3, c3_begin, c3_gate, nullptr};
	}
	const char *hs = fit_env_str("TAW_HYBRID", "adam");
	hyb_mode mode = HYB_ADAM;
	if (!strcmp(hs, "muon"))
		mode = HYB_MUON;
	else if (!strcmp(hs, "blend"))
		mode = HYB_BLEND;
	else if (!strcmp(hs, "alt"))
		mode = HYB_ALT;
	else if (!strcmp(hs, "route"))
		mode = HYB_ROUTE;
	else if (!strcmp(hs, "namo"))
		mode = HYB_NAMO;
	else if (!strcmp(hs, "namod") || !strcmp(hs, "namo-d"))
		mode = HYB_NAMOD;
	/* MALT is a separate opt-in: unset (or "off"/"0") keeps every shipped
	 * path bit-identical. "1"/"on"/"malt" = MALT, "malter" = adaptive. */
	const char *ms = fit_env_str("TAW_MALT", "off");
	if (!strcmp(ms, "1") || !strcmp(ms, "on") || !strcmp(ms, "malt"))
		mode = HYB_MALT;
	else if (!strcmp(ms, "malter"))
		mode = HYB_MALTER;
	if (mode == HYB_MALT && fit_env_uint("TAW_MALTER", 0) != 0)
		mode = HYB_MALTER;

	/* LR endpoints. Plain AdamW defaults to the shipped schedule; each
	 * other component has its own knob so the two LRs can be tuned apart. */
	const float lr0 = fit_env_float("TAW_LR0", 0.02f);
	/* TAW_LR1=flat keeps LR constant at TAW_LR0. */
	const char *lr1s = getenv("TAW_LR1");
	float lr1 = fit_env_float("TAW_LR1", lr0 / 20.0f);
	if (lr1s && strcmp(lr1s, "flat") == 0)
		lr1 = lr0;
	const float vec_lr0 = fit_env_float("TAW_VEC_LR0", lr0);
	const float vec_lr1 = fit_env_float("TAW_VEC_LR1", lr1);
	const float m_lr0 = fit_env_float("TAW_MUON_LR0", 0.01f);
	const float m_lr1 = fit_env_float("TAW_MUON_LR1", m_lr0 / 20.0f);
	const float a1_lr0 = fit_env_float("TAW_ADAM1_LR0", vec_lr0);
	const float a1_lr1 = fit_env_float("TAW_ADAM1_LR1", vec_lr1);
	/* MALT/MALTER W1 LR endpoints default to the tuned Muon schedule. */
	const float malt_lr0 = fit_env_float("TAW_MALT_LR0", 0.055f);
	const float malt_lr1 = fit_env_float("TAW_MALT_LR1", 0.0011f);
	const bool is_namo = mode == HYB_NAMO || mode == HYB_NAMOD;
	optcfg o = {.b1 = fit_env_float("TAW_B1", 0.9f),
		.b2 = fit_env_float("TAW_B2", 0.999f),
		.eps = 1e-6f,
		.mom = fit_env_float("TAW_MUON_MOM", 0.95f),
		.gain = fit_env_float("TAW_MUON_GAIN", 0.2f),
		.b1t = 1.0f, .b2t = 1.0f,
		.n1t = 1.0f, .n2t = 1.0f,
		.nb2 = fit_env_float("TAW_NAMO_BETA", 0.99f),
		.nscale = fit_env_float("TAW_NAMO_SCALE", 1.0f),
		.nclamp = fit_env_float("TAW_NAMO_CLAMP", 0.1f),
		.ns_steps = fit_env_uint("TAW_MUON_NS", 5),
		.nesterov = fit_env_uint("TAW_MUON_NEST", 1) != 0,
		.drow = !strcmp(fit_env_str("TAW_NAMO_D_AXIS", "col"), "row")};
	/* NAMO has its own NEST switch (default 0: the paper's M_t direction). */
	if (is_namo)
		o.nesterov = fit_env_uint("TAW_NAMO_NEST", 0) != 0;
	/* Shared gate: TAW_EPS_MODE=c3, with the beta/p knobs. It is
	 * wired only into the NAMO hook, never into AdamW/Muon/MALTER. */
	if (is_namo && !strcmp(fit_env_str("TAW_EPS_MODE", "none"), "c3")) {
		ng.sig = t->sig;
		ng.mul = nmul;
		ng.beta = fit_env_float("TAW_EPS_SIGN_BETA", 0.9f);
		ng.p = fit_env_float("TAW_EPS_GATE_P", 1.0f);
		namo_hooks.shape.ctx = &ng;
		namo_hooks.shape.begin = namo_gate_begin;
		namo_hooks.shape.gate = namo_gate_at;
	}
	/* Diagonal preconditioner: l/r EMA decay and damping (paper eps). */
	o.malt_b2 = fit_env_float("TAW_MALT_B2", 0.999f);
	o.malt_eps = fit_env_float("TAW_MALT_EPS", 1e-8f);
	o.malt_b1t = 1.0f;
	o.malt_b2t = 1.0f;
	const char *gs = fit_env_str("TAW_MALT_GRAFT", "o");
	o.graft = !strcmp(gs, "d") ? GRAFT_D
		: !strcmp(gs, "m") ? GRAFT_M : GRAFT_O;
	o.vecwd = fit_env_float("TAW_WD", 1e-5f);
	o.wd = fit_env_float("TAW_MUON_WD", mode == HYB_ADAM ? o.vecwd : 1e-3f);
	if (is_namo)
		o.wd = fit_env_float("TAW_NAMO_WD", o.wd);
	o.wd1 = fit_env_float("TAW_ADAM1_WD", o.vecwd);
	const float alpha = fit_env_float("TAW_ALPHA", 0.5f);
	const bool warm_adam = !strcmp(fit_env_str("TAW_WARM", "muon"), "adam");
	const bool alt_step = strcmp(fit_env_str("TAW_ALT", "step"), "half") != 0;
	const bool alt_first_muon =
	    strcmp(fit_env_str("TAW_ALT_FIRST", "muon"), "adam") != 0;
	faro_ctx fr;
	faro_init(&fr, t, epochs, ns);

	/* Per-set frequency for routing: count the same samples the epoch loop
	 * strides through (order does not matter for a count). */
	if (mode == HYB_ROUTE && t->s1 > 1) {
		for (size_t k = 0; k < ns; k++) {
			const p1_sel_ctx sc = sample_sc(&sm[k]);
			t->cnt1[p1_sel_value(t->s.sel1, &sc)]++;
		}
		uint32_t *sorted = malloc(t->s1 * sizeof *sorted);
		if (sorted) {
			memcpy(sorted, t->cnt1, t->s1 * sizeof *sorted);
			/* comparator: ascending count, then lower index first */
			/* qsort via a small insertion sort (s1 <= 4096) */
			for (size_t i = 1; i < t->s1; i++) {
				uint32_t v = sorted[i];
				size_t j = i;
				while (j && sorted[j - 1] > v) {
					sorted[j] = sorted[j - 1];
					j--;
				}
				sorted[j] = v;
			}
			float q = fit_env_float("TAW_ROUTE_Q", 0.5f);
			if (q <= 0.0f)
				memset(t->route1, 0, t->s1);
			else if (q >= 1.0f)
				memset(t->route1, 1, t->s1);
			else {
				double f = (1.0 - (double)q) * (double)(t->s1 - 1);
				size_t idx = (size_t)(f + 0.5);
				uint32_t thr = sorted[idx];
				for (size_t set = 0; set < t->s1; set++)
					t->route1[set] = t->cnt1[set] >= thr;
			}
			free(sorted);
		} else {
			memset(t->route1, 1, t->s1);
		}
	} else if (mode == HYB_ROUTE) {
		memset(t->route1, 1, t->s1 ? t->s1 : 1);
	}

	const uint64_t steps_total = (uint64_t)epochs * ((ns + BATCH - 1) / BATCH);
	uint64_t step = 0;
	uint64_t rs = seed;
	const unsigned H = t->s.hidden;
	const int warm = fit_warm_epochs(epochs);
	t->tied = t->s1 > 1 || t->s2 > 1;

	/* C3 sign-EMA state and touch-count strata. */
	c3.sig = ecfg.enabled ? calloc(np, sizeof *c3.sig) : nullptr;
	eps_stats *st1 = ecfg.debug && H ? calloc(t->s1 + 1, sizeof *st1) : nullptr;
	eps_stats *st2 = ecfg.debug ? calloc(t->s2 ? t->s2 : 1, sizeof *st2) : nullptr;
	size_t *f1 = (H && ecfg.scope != SCOPE_ALL)
	    ? calloc(t->s1 + 1, sizeof *f1) : nullptr;
	size_t *f2 = ecfg.scope != SCOPE_ALL ? calloc(t->s2 ? t->s2 : 1,
	    sizeof *f2) : nullptr;
	size_t med1 = 0, med2 = 0;
	if (f1 || f2) {
		for (size_t i = 0; i < ns; i++) {
			const p1_sel_ctx sc = sample_sc(&sm[i]);
			if (f2)
				f2[p1_sel_value(t->s.sel2, &sc)]++;
			if (f1)
				f1[p1_sel_value(t->s.sel1, &sc)]++;
		}
		med1 = f1 ? positive_median(f1, t->s1 + 1) : 0;
		med2 = f2 ? positive_median(f2, t->s2) : 0;
	}
	/* EMA/SWA: initialise averaging state before the first epoch. */
	if (h->pre_step)
		h->pre_step(h->ctx, t, &o, step, t->tied);
	for (int ep = 0; ep < epochs; ep++) {
		if (t->tied && ep >= warm) {
			tnet_untie(t);
			if (c3.sig) { /* mirror the seed set's sign state like m/v */
				for (size_t set = 1; set < t->s1; set++)
					memcpy(c3.sig + set * t->row1, c3.sig,
					    t->row1 * sizeof *c3.sig);
				for (size_t set = 1; set < t->s2; set++)
					memcpy(c3.sig + t->n1 + set * t->row2,
					    c3.sig + t->n1,
					    t->row2 * sizeof *c3.sig);
			}
			if (h->pre_step)
				h->pre_step(h->ctx, t, &o, step, t->tied);
		}
		const uint64_t stride = fit_stride(ns);
		uint64_t pos = rng_next(&rs) % (ns ? ns : 1);
		for (size_t done = 0; done < ns;) {
			size_t nb = ns - done < BATCH ? ns - done : BATCH;
			size_t n1 = 0, n2 = 0;
			/* Deterministic probe slice within this batch: 16 strided
			 * samples, used only for counterfactual return measurement. */
			const sample *probe[16];
			size_t nprobe = 0;
			const size_t pevery = (nb + 15) / 16;
			for (size_t k = 0; k < nb; k++) {
				const sample *s = &sm[pos];
				if (fr.on && pevery && k % pevery == 0 && nprobe < 16)
					probe[nprobe++] = s;
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
			bool malty = mode == HYB_MALT || mode == HYB_MALTER;
			o.lr = malty ? malt_lr0 + (malt_lr1 - malt_lr0) * frac
				     : m_lr0 + (m_lr1 - m_lr0) * frac;
			o.lrvec = vec_lr0 + (vec_lr1 - vec_lr0) * frac;
			o.lr1 = a1_lr0 + (a1_lr1 - a1_lr0) * frac;
			o.b1t *= o.b1;
			o.b2t *= o.b2;
			o.malt_b1t *= o.mom;
			o.malt_b2t *= o.malt_b2;
			o.n1t *= o.mom;
			o.n2t *= o.nb2;
			bool alt_muon = alt_step ? (step & 1) == 1
						 : (frac < 0.5f);
			if (!alt_first_muon)
				alt_muon = !alt_muon;
			const float scale = 1.0f / (float)nb;
			/* FARO: geometry routes settle after the untie + exploration
			 * window; the ERC solve only reads the previous step's risk. */
			if (fr.on && !t->tied && !fr.frozen && step >= fr.freeze_step)
				faro_freeze(&fr);
			if (fr.on && fr.frozen && fr.alloc)
				faro_allocate(&fr, &o);
			fr.risk = fr.on && fr.frozen && fr.alloc ? fr.usum : nullptr;
			hooks.risk = fr.risk;
			hooks.nb = fr.on ? (unsigned)fr.nb : 0u;
			/* Mask C3's shape for sets outside its scope while leaving every
			 * other hook slot (notably FARO's risk bank) untouched. */
			opt_hooks nogate = hooks;
			nogate.shape = (step_shape){0};
			for (size_t k = 0; k < n1; k++) {
				size_t set = t->list1[k], off = set * t->row1;
				const bool g1 = ecfg.enabled && (ecfg.layers & 1) &&
				    in_scope(ecfg.scope, f1 ? f1[set] : 1, med1);
				if (g1)
					c3.off = off;
				bool muon = mode == HYB_MUON ||
				    (mode == HYB_ALT && alt_muon) ||
				    (mode == HYB_ROUTE && t->route1[set]);
				if (t->tied && warm_adam)
					muon = false;
				int fb = fr.on ? faro_b1(&fr, (unsigned)set) : 0;
				float lr_w1 = o.lr, lr_a1 = o.lr1;
				if (fr.on) {
					lr_w1 *= fr.mu[fb];
					lr_a1 *= fr.mu[fb];
					if (!t->tied && mode == HYB_MUON) {
						if (!fr.frozen && fr.route_ret) {
							/* alternate candidates to measure returns */
							if (fr.stat == 0)
								faro_cf_probe(&fr, t, &o,
								    (unsigned)set,
								    t->g + off, scale,
								    probe, nprobe);
							else
								faro_explore(&fr, &o,
								    t->g + off, fb, H);
							muon = (step & 1) != 0;
						} else if (fr.frozen) {
							muon = fr.geo[fb] != 0;
						}
					}
				}
				unsigned bid = fr.on ? (unsigned)fb : 0;
				if (mode == HYB_ADAM) {
					/* The shipped path, using the original m/v. */
					float lrvec_w1 = o.lrvec;
					if (fr.on)
						lrvec_w1 *= fr.mu[fb];
					adam_rows(&o, t->w + off, t->m + off,
					    t->v + off, t->g + off, t->row1,
					    scale, lrvec_w1, o.vecwd, 1.0f,
					    g1 ? &hooks : &nogate, bid);
				} else if (malty && !(t->tied && warm_adam)) {
					malt_rows(&o, t->w + off, t->m + off,
					    t->malt_l + set * H,
					    t->malt_r + set * P1_F,
					    t->malt_nu + set, t->g + off,
					    H, P1_F, scale, lr_w1, o.wd, 1.0f,
					    mode == HYB_MALTER, &hooks, bid);
				} else if (is_namo && !(t->tied && warm_adam)) {
					/* P6-3: match MALT's tied AdamW warm-up guard. */
					namo_rows(&o, t->w + off, t->m + off,
					    t->g + off, t->v2s, t->v2c, set,
					    H, P1_F, scale, o.lr, o.wd,
					    mode == HYB_NAMOD, &namo_hooks, 0);
				} else if (mode == HYB_BLEND) {
					adam_rows(&o, t->w + off, t->m1 + off,
					    t->v1 + off, t->g + off, t->row1,
					    scale, lr_a1, o.wd1, 1.0f - alpha,
					    g1 ? &hooks : &nogate, bid);
					muon_rows(&o, t->w + off, t->m + off,
					    t->g + off, H, P1_F, scale, lr_w1,
					    o.wd, alpha, &hooks, bid);
				} else if (muon) {
					muon_rows(&o, t->w + off, t->m + off,
					    t->g + off, H, P1_F, scale, lr_w1,
					    o.wd, 1.0f, &hooks, bid);
				} else {
					adam_rows(&o, t->w + off, t->m1 + off,
					    t->v1 + off, t->g + off, t->row1,
					    scale, lr_a1, o.wd1, 1.0f,
					    g1 ? &hooks : &nogate, bid);
				}
				if (g1 && st1) {
					st1[set].touches += 1.0;
					st1[set].sum_gate += c3.acc / (float)t->row1;
				}
				c3.acc = 0.0f;
				if (h->post_set)
					h->post_set(h->ctx, t, 1, (unsigned)set, off,
					    t->row1, step);
				t->touched1[set] = 0;
			}
			for (size_t k = 0; k < n2; k++) {
				const size_t s2 = t->list2[k];
				size_t off = t->n1 + s2 * t->row2;
				int fb = fr.on ? faro_b2(&fr, (unsigned)s2) : 0;
				float lrvec_b = o.lrvec;
				if (fr.on)
					lrvec_b *= fr.mu[fb];
				const bool g2 = ecfg.enabled && (ecfg.layers & 2) &&
				    in_scope(ecfg.scope, f2 ? f2[s2] : 1, med2);
				if (g2)
					c3.off = off;
				adam_rows(&o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row2, scale, lrvec_b,
				    o.vecwd, 1.0f, g2 ? &hooks : &nogate,
				    fr.on ? (unsigned)fb : 0);
				if (g2 && st2) {
					st2[s2].touches += 1.0;
					st2[s2].sum_gate += c3.acc / (float)t->row2;
				}
				c3.acc = 0.0f;
				if (h->post_set)
					h->post_set(h->ctx, t, 2, (unsigned)s2, off,
					    t->row2, step);
				t->touched2[s2] = 0;
			}
			if (fr.on && fr.frozen && fr.alloc)
				faro_measure(&fr, t);
			memset(t->g, 0, (t->n1 + t->n2) * sizeof(float));
		}
		/* SWA is epoch-granular; use the hook's feature context for the
		 * snapshot window instead of threading a mode parameter. */
		avg_ctx *ac = h->ctx;
		if (ac && ac->mode != TAW_AVG_OFF && !mode_is_ema(ac->mode) &&
		    (ac->mode == TAW_AVG_SWA_ALL || ep >= epochs / 2)) {
			for (size_t i = 0; i < np; i++)
				t->wa[i] += t->w[i];
			ac->nswa++;
		}
	}
	if (h->finalize)
		h->finalize(h->ctx, t);
	if ((mode == HYB_MALT || mode == HYB_MALTER) && getenv("TAW_MALT_DEBUG")) {
		float lmin = 1e30f, lmax = 0.0f, lsum = 0.0f;
		float rmin = 1e30f, rmax = 0.0f, rsum = 0.0f;
		for (unsigned i = 0; i < H; i++) {
			float v = t->malt_l[i];
			lmin = lmin < v ? lmin : v;
			lmax = lmax > v ? lmax : v;
			lsum += v;
		}
		for (unsigned j = 0; j < P1_F; j++) {
			float v = t->malt_r[j];
			rmin = rmin < v ? rmin : v;
			rmax = rmax > v ? rmax : v;
			rsum += v;
		}
		fprintf(stderr,
		    "MALT state: l[min %.4g max %.4g mean %.4g] "
		    "r[min %.4g max %.4g mean %.4g] nu %.4g\n",
		    lmin, lmax, lsum / (float)(H ? H : 1), rmin, rmax,
		    rsum / (float)P1_F, t->malt_nu[0]);
	}
	faro_report(&fr);
	faro_free(&fr);
	if (ecfg.debug) {
		fprintf(stderr, "[eps] gate beta=%.3g p=%.3g mode=%s "
			"layers=%d scope=%d\n", ecfg.beta, ecfg.p,
			ecfg.mean_pres ? "mean" : "shrink", ecfg.layers,
			ecfg.scope);
		const char *ln[2] = {"layer1", "layer2"};
		for (int L = 0; L < 2; L++) {
			const eps_stats *s = L ? st2 : st1;
			const size_t nset = L ? t->s2 : (H ? t->s1 : 0);
			if (!s || !nset)
				continue;
			size_t nt = 0;
			double tmin = 1e300, tmax = 0, tsum = 0, gsum = 0;
			for (size_t i = 0; i < nset; i++) {
				if (s[i].touches == 0.0)
					continue;
				nt++;
				double tv = s[i].touches;
				if (tv < tmin) tmin = tv;
				if (tv > tmax) tmax = tv;
				tsum += tv;
				gsum += s[i].sum_gate / tv;
			}
			fprintf(stderr, "[eps] %s: touched=%zu min/med/max touches="
				"%.0f/%.0f/%.0f mean_gate=%.4g\n",
				ln[L], nt, tmin, nt ? tsum / (double)nt : 0.0, tmax,
				nt ? gsum / (double)nt : 0.0);
		}
	}
	free(c3.sig);
	free(st1);
	free(st2);
	free(f1);
	free(f2);
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
	uint8_t head[P1_MODEL_HEAD];
	p1_head_write(head, &t->s);
	taw_buf_put(b, head, sizeof head);
	const unsigned H = t->s.hidden;
	for (size_t set = 0; set < t->s1; set++)
		for (unsigned j = 0; j < H; j++)
			put_row(b, quant, t->w + set * t->row1 + j * P1_F, P1_F);
	for (size_t set = 0; set < t->s2; set++)
		put_row(b, quant, t->w + t->n1 + set * t->row2, t->row2);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

/* Mean cost in bits per coded bit, with the decoder's exact inference. */
static taw_status evaluate(const uint8_t *model, size_t size, uint8_t quant,
    const sample *sm, size_t ns, double *bits)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	p1_net net;
	taw_status st = p1_net_load(&net, quant, model, size);
	if (st != TAW_OK)
		return st;
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float x[P1_XPAD];
		sample_x(&sm[k], x);
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
					c.size = P1_MODEL_HEAD + p1_weights_size(&c.s, q);
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
	int avg_mode; /* TAW_FIT_AVG, resolved once here per candidate */
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
	/* EMA/SWA state lives in tnet; the behaviour is supplied by the shared
	 * opt_hooks ABI. GPU-trained nets have no CPU trajectory: their hooks
	 * stay NULL (fit_one never calls tnet_train for them). */
	avg_ctx ac;
	opt_hooks hooks;
	avg_hooks_init(&hooks, &ac,
	    resolve_avg_mode(F->avg_mode, F->epochs));
	taw_status st = TAW_OK;
	if (F->pre && F->pre[ci].w) {
		t = F->pre[ci];
		F->pre[ci] = (tnet){0};
	} else if ((st = tnet_init(&t, &F->cs[ci].s, 0x7A57A11ull + ci)) == TAW_OK) {
		tnet_train(&t, F->sm, F->ns, F->epochs, 0xC0FFEEull + ci, &hooks);
	}
	if (st == TAW_OK) {
		st = serialize(&t, F->cs[ci].quant, &F->model[ci]);
		tnet_free(&t);
	}
	if (st == TAW_OK)
		st = evaluate(F->model[ci].p, F->model[ci].n, F->cs[ci].quant,
		    F->sm, F->ns, &F->bpb[ci]);
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
	if (ng && (st = taw_gpu_open(F->sm, F->ns, dbg, &g)) != TAW_OK) {
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
	/* c2/nc9/junk are errors, never an implicit gate enable. */
	taw_status es = eps_mode_validate();
	if (es != TAW_OK)
		return es;
	/* The sign gate is CPU-only. Reject instead of silently training GPU
	 * candidates with plain AdamW. */
	if (eps_mode_enabled() && p->gpu != TAW_GPU_OFF) {
		fprintf(stderr, "taw fit: TAW_EPS_MODE=c3 is CPU-only; set -g off\n");
		return TAW_E_ARG;
	}
	if (p->predictor == TAW_PREDICTOR_ORDER0)
		return TAW_OK;
	const uint64_t n = in->size;
	const bool forced = p->predictor == TAW_PREDICTOR_MIX;
	const uint64_t budget = n / 32;
	if (unit == 0 || unit > n)
		unit = n;

	p1_shape geo = {0};
	geometry((size_t)unit, &geo);
	cand cs[1024];
	size_t ncand = enumerate(&geo, budget, p->weights, (size_t)n, cs, 1024);
	if (ncand == 0)
		return forced ? TAW_E_ARG : TAW_OK;

	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	sample *sm;
	size_t ns;
	taw_status st = collect(in, unit, &geo, effort[lv].samples, threads,
	    &sm, &ns);
	if (st != TAW_OK)
		return st;

	const char *dbg = getenv("TAW_FIT_DEBUG");
	size_t tries = effort[lv].cands;
	if (dbg && strchr(dbg, 'a'))
		tries = ncand;
	if (tries > ncand)
		tries = ncand;

	fit_job F = {.cs = cs, .sm = sm, .ns = ns, .epochs = effort[lv].epochs,
		.avg_mode = parse_avg_mode(getenv("TAW_FIT_AVG")), .n = n};
	F.model = calloc(tries, sizeof *F.model);
	F.est = calloc(tries, sizeof *F.est);
	F.bpb = calloc(tries, sizeof *F.bpb);
	F.st = calloc(tries, sizeof *F.st);
	if (!F.model || !F.est || !F.bpb || !F.st)
		st = TAW_E_NOMEM;
	else
		st = fit_all(&F, tries, threads, p->gpu, dbg != nullptr);
	size_t best = SIZE_MAX;
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
		if (best == SIZE_MAX || F.est[ci] < F.est[best])
			best = ci;
	}
	if (st == TAW_OK) {
		out->profile = TAW_PROFILE_F32_STRICT_MIX;
		out->quant = cs[best].quant;
		out->model = F.model[best].p;
		out->model_size = F.model[best].n;
		F.model[best] = (taw_buf){0};
	}
	for (size_t ci = 0; F.model && ci < tries; ci++)
		taw_buf_free(&F.model[ci]);
	free(F.model);
	free(F.est);
	free(F.bpb);
	free(F.st);
	free(sm);
	return st;
}
