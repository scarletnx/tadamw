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
	free(t->touched1);
	free(t->touched2);
	free(t->list1);
	free(t->list2);
	free(t->cnt1);
	free(t->route1);
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
	t->touched1 = calloc(t->s1 + 1, 1);
	t->touched2 = calloc(t->s2, 1);
	t->list1 = malloc((t->s1 + 1) * sizeof(uint32_t));
	t->list2 = malloc(t->s2 * sizeof(uint32_t));
	t->cnt1 = calloc(t->s1 ? t->s1 : 1, sizeof(uint32_t));
	t->route1 = calloc(t->s1 ? t->s1 : 1, 1);
	if (!t->w || !t->m || !t->v || !t->g || !t->m1 || !t->v1 ||
	    !t->malt_l || !t->malt_r || !t->malt_nu ||
	    !t->touched1 || !t->touched2 || !t->list1 || !t->list2 ||
	    !t->cnt1 || !t->route1) {
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
 */
enum { MUON_MAXDIM = 32, MUON_MAXEL = MUON_MAXDIM * MUON_MAXDIM };

typedef enum {
	HYB_ADAM = 0, /* the shipped optimizer: AdamW everywhere */
	HYB_MUON,     /* Muon on W1 matrices, AdamW on W2 vectors */
	HYB_BLEND,    /* W1: alpha*Muon + (1-alpha)*AdamW */
	HYB_ALT,      /* W1: alternate Muon / AdamW by step or by half */
	HYB_ROUTE,    /* W1: Muon for hot sets, AdamW for cold sets */
	HYB_MALT,     /* W1: MALT (two-sided diagonal preconditioned Muon) */
	HYB_MALTER    /* W1: MALTER (MALT + adaptive step-size rescaling) */
} hyb_mode;

/* Norm-grafting source for the MALT update magnitude. */
enum { GRAFT_O = 0, /* paper: ||Orth(M~)||_F */
       GRAFT_D = 1, /* no renormalisation: ||D||_F */
       GRAFT_M = 2 /*||M~||_F */ };

typedef struct {
	float lr, lrvec, lr1;  /* current LRs: Muon W1, Adam W2, Adam W1 */
	float wd, vecwd, wd1;  /* decoupled weight decay */
	float b1, b2, eps;     /* Adam */
	float mom, gain;       /* Muon */
	float b1t, b2t;        /* b1^t, b2^t */
	float malt_b2, malt_eps;   /* diagonal-preconditioner EMA / damping */
	float malt_b1t, malt_b2t;  /* momentum^t, malt_b2^t (MALTER bias correction) */
	unsigned ns_steps;
	int graft;             /* MALT graft mode */
	bool nesterov;
} optcfg;

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

/* AdamW on a flat vector. `mix` scales the whole update (1 for plain AdamW,
 * 1-alpha for the Adam half of a blend). Gradients are cleared by caller. */
/* `risk` (FARO only) accumulates the squared applied update of block `bid`;
 * NULL keeps the shipped update bit-for-bit. The w update itself is never
 * rewritten: the extra multiply is done separately, so rounding is unchanged. */
static void adam_rows(optcfg *o, float *w, float *m, float *v, const float *g,
    size_t len, float scale, float lr, float wd, float mix, float *risk,
    unsigned bid)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		float mh = m[k] * c1, vh = v[k] * c2;
		w[k] -= mix * lr * (mh / (fsqrt(vh) + o->eps) + wd * w[k]);
		if (risk) {
			float upd = mix * lr * (mh / (fsqrt(vh) + o->eps));
			risk[bid] += upd * upd; /* decoupled wd is not part of Δ_b */
		}
	}
}

/* Muon on a row-major rows x cols matrix. Largest shape here is
 * P1_HMAX x P1_F = 32 x 26, so scratch fits on the stack. */
static void muon_rows(const optcfg *o, float *w, float *m, const float *g,
    size_t rows, size_t cols, float scale, float lr, float wd, float mix,
    float *risk, unsigned bid)
{
	float x[MUON_MAXEL], a[MUON_MAXEL], b[MUON_MAXEL], y[MUON_MAXEL];
	const bool tr = rows > cols; /* work with the shorter dimension first */
	const size_t mr = tr ? cols : rows, mc = tr ? rows : cols;
	if (rows * cols > MUON_MAXEL || mr == 0 || mc == 0)
		return;
	for (size_t i = 0; i < rows * cols; i++) {
		float gi = g[i] * scale;
		m[i] = o->mom * m[i] + gi;
		float u = o->nesterov ? o->mom * m[i] + gi : m[i];
		size_t at = tr ? (i % cols) * mc + i / cols : i;
		x[at] = u;
	}
	float ss = 0.0f;
	for (size_t i = 0; i < mr * mc; i++)
		ss += x[i] * x[i];
	float norm = fsqrt(ss);
	if (!(norm > 1e-12f))
		return; /* zero gradient: leave weights alone */
	for (size_t i = 0; i < mr * mc; i++)
		x[i] *= 1.0f / norm;
	for (unsigned it = 0; it < o->ns_steps; it++) {
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mc; k++)
					s += x[i * mc + k] * x[j * mc + k];
				a[i * mr + j] = s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mr; k++)
					s += a[i * mr + k] * a[k * mr + j];
				b[i * mr + j] = -4.7750f * a[i * mr + j] + 2.0315f * s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t k = 0; k < mc; k++) {
				float s = 0.0f;
				for (size_t j = 0; j < mr; j++)
					s += b[i * mr + j] * x[j * mc + k];
				y[i * mc + k] = s;
			}
		for (size_t i = 0; i < mr * mc; i++)
			x[i] = 3.4445f * x[i] + y[i];
	}
	const size_t dmax = rows > cols ? rows : cols;
	const float factor = mix * lr * o->gain * fsqrt((float)dmax);
	for (size_t i = 0; i < rows * cols; i++) {
		size_t at = tr ? (i % cols) * mc + i / cols : i;
		w[i] -= factor * x[at] + mix * lr * wd * w[i];
		if (risk) {
			float upd = factor * x[at];
			risk[bid] += upd * upd; /* only the geometry-induced Δ_b */
		}
	}
}

/* x^{-1/8} without libm; x > 0 thanks to the damping term. */
static inline float inv_root8(float x)
{
	return 1.0f / fsqrt(fsqrt(fsqrt(x)));
}

/* MALT / MALTER (arXiv:2608.05088, Algorithms 1-2) on a row-major
 * rows x cols matrix. l/r are this set's row/column EMA vectors, nu the
 * scalar MALTER second moment. With `adaptive` false this is plain MALT:
 *   M = mom*M + G          (Nesterov extrapolation when enabled)
 *   l = b2*l + (1-b2)*sum_j G_ij^2 ;  r = b2*r + (1-b2)*sum_i G_ij^2
 *   L = diag((l+eps)^-1/8), R = diag((r+eps)^-1/8)
 *   M~ = L M R ;  O = Orth(M~) ;  D = L O R
 *   W -= lr * (||O||_F / (||D||_F + eps)) * D
 * MALTER adds nu = b2*nu + (1-b2)*||L G R||^2 and rescales by
 *   alpha = sqrt(1-b2^t)/(1-mu1^t+eps) * ||M~||_F / (sqrt(nu)+eps).
 * Orthogonalization reuses Muon's Newton-Schulz polynomial. */
static void malt_rows(const optcfg *o, float *w, float *m, float *l, float *r,
    float *nu, const float *g, size_t rows, size_t cols, float scale,
    float lr, float wd, float mix, bool adaptive, float *risk, unsigned bid)
{
	float x[MUON_MAXEL], a[MUON_MAXEL], b[MUON_MAXEL], y[MUON_MAXEL];
	float Lc[MUON_MAXDIM], Rc[MUON_MAXDIM];
	const bool tr = rows > cols; /* work with the shorter dimension first */
	const size_t mr = tr ? cols : rows, mc = tr ? rows : cols;
	if (rows * cols > MUON_MAXEL || mr == 0 || mc == 0)
		return;
	const float b2 = o->malt_b2, one_b2 = 1.0f - b2, eps = o->malt_eps;

	/* Momentum and EMA of squared row / column gradient norms. */
	for (size_t i = 0; i < rows; i++)
		l[i] *= b2;
	for (size_t j = 0; j < cols; j++)
		r[j] *= b2;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t idx = i * cols + j;
			const float gi = g[idx] * scale;
			const float mm = o->mom * m[idx] + gi;
			m[idx] = mm;
			const float u = o->nesterov ? o->mom * mm + gi : mm;
			l[i] += one_b2 * gi * gi;
			r[j] += one_b2 * gi * gi;
			const size_t at = tr ? j * mc + i : idx;
			x[at] = u;
		}
	}
	for (size_t i = 0; i < rows; i++)
		Lc[i] = inv_root8(l[i] + eps);
	for (size_t j = 0; j < cols; j++)
		Rc[j] = inv_root8(r[j] + eps);

	/* Preconditioned momentum M~ = L M R; also accumulate ||L G R||^2. */
	float pg2 = 0.0f, ss = 0.0f;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t idx = i * cols + j;
			const size_t at = tr ? j * mc + i : idx;
			x[at] *= Lc[i] * Rc[j];
			ss += x[at] * x[at];
			if (adaptive) {
				const float gv = Lc[i] * Rc[j] * g[idx] * scale;
				pg2 += gv * gv;
			}
		}
	}
	float norm = fsqrt(ss); /* ||M~||_F, also the MALTER signal magnitude */
	if (!(norm > 1e-12f))
		return;
	const float recip = 1.0f / norm;
	for (size_t i = 0; i < mr * mc; i++)
		x[i] *= recip;
	for (unsigned it = 0; it < o->ns_steps; it++) {
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mc; k++)
					s += x[i * mc + k] * x[j * mc + k];
				a[i * mr + j] = s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mr; k++)
					s += a[i * mr + k] * a[k * mr + j];
				b[i * mr + j] = -4.7750f * a[i * mr + j] + 2.0315f * s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t k = 0; k < mc; k++) {
				float s = 0.0f;
				for (size_t j = 0; j < mr; j++)
					s += b[i * mr + j] * x[j * mc + k];
				y[i * mc + k] = s;
			}
		for (size_t i = 0; i < mr * mc; i++)
			x[i] = 3.4445f * x[i] + y[i];
	}

	/* O = x; D = L O R (map the orthogonal direction back). */
	float o2 = 0.0f, d2 = 0.0f;
	for (size_t i = 0; i < mr * mc; i++)
		o2 += x[i] * x[i];
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t at = tr ? j * mc + i : i * cols + j;
			x[at] *= Lc[i] * Rc[j];
			d2 += x[at] * x[at];
		}
	}
	const float normD = fsqrt(d2);
	if (!(normD > 1e-12f))
		return;
	/* Graft the magnitude from O (paper), M~ (ablation) or leave D as-is. */
	float graft = o->graft == GRAFT_M ? norm
		    : o->graft == GRAFT_D ? normD : fsqrt(o2);
	graft /= normD + eps;
	float fac = mix * lr * graft;
	if (adaptive) {
		*nu = b2 * *nu + one_b2 * pg2;
		/* The codebase's momentum is M = mu*M + G, while the paper's is
		 * M = mu*M + (1-mu)*G. Rescale ||M~|| by (1-mu) so alpha is the
		 * paper's signal-to-noise ratio (scale-invariant to G). */
		const float sig = norm * (1.0f - o->mom);
		const float alpha = fsqrt(1.0f - o->malt_b2t) /
		    (1.0f - o->malt_b1t + eps) * sig / (fsqrt(*nu) + eps);
		fac *= alpha;
	}
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t idx = i * cols + j;
			const size_t at = tr ? j * mc + i : idx;
			w[idx] -= fac * x[at] + mix * lr * wd * w[idx];
			if (risk) { /* FARO: update-to-weight risk, wd excluded */
				float upd = fac * x[at];
				risk[bid] += upd * upd;
			}
		}
	}
}

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
 * only W1 has a second candidate geometry. See RESULTS.md: this approximation
 * does not recover the hand-designed Muon route, so the ERC lane is the part
 * that is validated here.
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
 * divided by ||NS(g)||_F. Duplicates Muon's Newton-Schulz loop (only used with
 * FARO enabled) so the shipped muon_rows stays byte-identical. */
static float faro_ns_score(const optcfg *o, const float *g, unsigned rows,
    unsigned cols)
{
	float x[MUON_MAXEL], a[MUON_MAXEL], b[MUON_MAXEL], y[MUON_MAXEL];
	const size_t n = (size_t)rows * cols;
	if (n > MUON_MAXEL || !n)
		return 0.0f;
	const bool tr = rows > cols;
	const size_t mr = tr ? cols : rows, mc = tr ? rows : cols;
	for (size_t i = 0; i < n; i++) {
		size_t at = tr ? (i % cols) * mc + i / cols : i;
		x[at] = g[i];
	}
	float ss = 0.0f;
	for (size_t i = 0; i < n; i++)
		ss += x[i] * x[i];
	if (!(ss > 1e-12f))
		return 0.0f;
	float inv = 1.0f / fsqrt(ss);
	for (size_t i = 0; i < n; i++)
		x[i] *= inv;
	for (unsigned it = 0; it < o->ns_steps; it++) {
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mc; k++)
					s += x[i * mc + k] * x[j * mc + k];
				a[i * mr + j] = s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t j = 0; j < mr; j++) {
				float s = 0.0f;
				for (size_t k = 0; k < mr; k++)
					s += a[i * mr + k] * a[k * mr + j];
				b[i * mr + j] = -4.7750f * a[i * mr + j] + 2.0315f * s;
			}
		for (size_t i = 0; i < mr; i++)
			for (size_t k = 0; k < mc; k++) {
				float s = 0.0f;
				for (size_t j = 0; j < mr; j++)
					s += b[i * mr + j] * x[j * mc + k];
				y[i * mc + k] = s;
			}
		for (size_t i = 0; i < n; i++)
			x[i] = 3.4445f * x[i] + y[i];
	}
	double dot = 0.0, xn = 0.0;
	for (size_t i = 0; i < n; i++) {
		size_t at = tr ? (i % cols) * mc + i / cols : i;
		dot += (double)g[i] * x[at];
		xn += (double)x[at] * x[at];
	}
	return xn > 0.0 ? (float)(dot / (double)fsqrt((float)xn)) : 0.0f;
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

/* Train on samples for the given epochs. Batches touch few weight sets, so
 * only those rows are updated ("lazy" optimizer state). With no environment
 * overrides this is the shipped AdamW fitter, bit for bit. */
static void tnet_train(tnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed)
{
	enum { BATCH = FIT_BATCH };
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
	const float lr1 = fit_env_float("TAW_LR1", lr0 / 20.0f);
	const float vec_lr0 = fit_env_float("TAW_VEC_LR0", lr0);
	const float vec_lr1 = fit_env_float("TAW_VEC_LR1", lr1);
	const float m_lr0 = fit_env_float("TAW_MUON_LR0", 0.01f);
	const float m_lr1 = fit_env_float("TAW_MUON_LR1", m_lr0 / 20.0f);
	const float a1_lr0 = fit_env_float("TAW_ADAM1_LR0", vec_lr0);
	const float a1_lr1 = fit_env_float("TAW_ADAM1_LR1", vec_lr1);
	/* MALT/MALTER W1 LR endpoints default to the tuned Muon schedule. */
	const float malt_lr0 = fit_env_float("TAW_MALT_LR0", 0.055f);
	const float malt_lr1 = fit_env_float("TAW_MALT_LR1", 0.0011f);
	optcfg o = {.b1 = fit_env_float("TAW_B1", 0.9f),
		.b2 = fit_env_float("TAW_B2", 0.999f),
		.eps = 1e-6f,
		.mom = fit_env_float("TAW_MUON_MOM", 0.95f),
		.gain = fit_env_float("TAW_MUON_GAIN", 0.2f),
		.b1t = 1.0f, .b2t = 1.0f,
		.ns_steps = fit_env_uint("TAW_MUON_NS", 5),
		.nesterov = fit_env_uint("TAW_MUON_NEST", 1) != 0};
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
	for (int ep = 0; ep < epochs; ep++) {
		if (t->tied && ep >= warm)
			tnet_untie(t);
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
			for (size_t k = 0; k < n1; k++) {
				size_t set = t->list1[k], off = set * t->row1;
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
					    fr.risk, bid);
				} else if (malty && !(t->tied && warm_adam)) {
					malt_rows(&o, t->w + off, t->m + off,
					    t->malt_l + set * H,
					    t->malt_r + set * P1_F,
					    t->malt_nu + set, t->g + off,
					    H, P1_F, scale, lr_w1, o.wd, 1.0f,
					    mode == HYB_MALTER, fr.risk, bid);
				} else if (mode == HYB_BLEND) {
					adam_rows(&o, t->w + off, t->m1 + off,
					    t->v1 + off, t->g + off, t->row1,
					    scale, lr_a1, o.wd1, 1.0f - alpha,
					    fr.risk, bid);
					muon_rows(&o, t->w + off, t->m + off,
					    t->g + off, H, P1_F, scale, lr_w1,
					    o.wd, alpha, fr.risk, bid);
				} else if (muon) {
					muon_rows(&o, t->w + off, t->m + off,
					    t->g + off, H, P1_F, scale, lr_w1,
					    o.wd, 1.0f, fr.risk, bid);
				} else {
					adam_rows(&o, t->w + off, t->m1 + off,
					    t->v1 + off, t->g + off, t->row1,
					    scale, lr_a1, o.wd1, 1.0f,
					    fr.risk, bid);
				}
				t->touched1[set] = 0;
			}
			for (size_t k = 0; k < n2; k++) {
				size_t set = t->list2[k];
				size_t off = t->n1 + set * t->row2;
				int fb = fr.on ? faro_b2(&fr, (unsigned)set) : 0;
				float lrvec_b = o.lrvec;
				if (fr.on)
					lrvec_b *= fr.mu[fb];
				adam_rows(&o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row2, scale, lrvec_b,
				    o.vecwd, 1.0f, fr.risk,
				    fr.on ? (unsigned)fb : 0);
				t->touched2[set] = 0;
			}
			if (fr.on && fr.frozen && fr.alloc)
				faro_measure(&fr, t);
			memset(t->g, 0, (t->n1 + t->n2) * sizeof(float));
		}
	}
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
		.n = n};
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
