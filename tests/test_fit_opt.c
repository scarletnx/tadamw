/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Foundation refactor: bit-exactness of the shared Newton-Schulz polynomial.
 *
 * The `ref_*` functions below are VERBATIM copies of main's inline loops
 * (fit.c @ 1aa3dd7). They are the oracle: run the ref and the new wrappers on
 * identical inputs and compare the raw float bytes. This catches any change
 * in coefficient, accumulation order, transpose map or normalization. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "fit_opt.h"
#include "taw_internal.h"

/* ---- oracle: main's original loops, copied verbatim ------------------------- */

static void ref_ns(float *x, float *a, float *b, float *y,
    size_t mr, size_t mc, unsigned steps)
{
	for (unsigned it = 0; it < steps; it++) {
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
}

static void ref_adam_rows(optcfg *o, float *w, float *m, float *v,
    const float *g, size_t len, float scale, float lr, float wd, float mix)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		float mh = m[k] * c1, vh = v[k] * c2;
		w[k] -= mix * lr * (mh / (fsqrt(vh) + o->eps) + wd * w[k]);
	}
}

static void ref_muon_rows(const optcfg *o, float *w, float *m, const float *g,
    size_t rows, size_t cols, float scale, float lr, float wd, float mix)
{
	float x[MUON_MAXEL], a[MUON_MAXEL], b[MUON_MAXEL], y[MUON_MAXEL];
	const bool tr = rows > cols;
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
		return;
	for (size_t i = 0; i < mr * mc; i++)
		x[i] *= 1.0f / norm;
	ref_ns(x, a, b, y, mr, mc, o->ns_steps);
	const size_t dmax = rows > cols ? rows : cols;
	const float factor = mix * lr * o->gain * fsqrt((float)dmax);
	for (size_t i = 0; i < rows * cols; i++) {
		size_t at = tr ? (i % cols) * mc + i / cols : i;
		w[i] -= factor * x[at] + mix * lr * wd * w[i];
	}
}

static inline float ref_inv_root8(float x)
{
	return 1.0f / fsqrt(fsqrt(fsqrt(x)));
}

static void ref_malt_rows(const optcfg *o, float *w, float *m, float *l,
    float *r, float *nu, const float *g, size_t rows, size_t cols,
    float scale, float lr, float wd, float mix, bool adaptive)
{
	float x[MUON_MAXEL], a[MUON_MAXEL], b[MUON_MAXEL], y[MUON_MAXEL];
	float Lc[MUON_MAXDIM], Rc[MUON_MAXDIM];
	const bool tr = rows > cols;
	const size_t mr = tr ? cols : rows, mc = tr ? rows : cols;
	if (rows * cols > MUON_MAXEL || mr == 0 || mc == 0)
		return;
	const float b2 = o->malt_b2, one_b2 = 1.0f - b2, eps = o->malt_eps;

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
		Lc[i] = ref_inv_root8(l[i] + eps);
	for (size_t j = 0; j < cols; j++)
		Rc[j] = ref_inv_root8(r[j] + eps);

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
	float norm = fsqrt(ss);
	if (!(norm > 1e-12f))
		return;
	const float recip = 1.0f / norm;
	for (size_t i = 0; i < mr * mc; i++)
		x[i] *= recip;
	ref_ns(x, a, b, y, mr, mc, o->ns_steps);

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
	float graft = o->graft == GRAFT_M ? norm
		    : o->graft == GRAFT_D ? normD : fsqrt(o2);
	graft /= normD + eps;
	float fac = mix * lr * graft;
	if (adaptive) {
		*nu = b2 * *nu + one_b2 * pg2;
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
		}
	}
}

/* ---- deterministic inputs --------------------------------------------------- */

static uint32_t seq;

static float nextf(void)
{
	seq = seq * 1664525u + 1013904223u;
	return (float)((int32_t)(seq >> 16) - 32768) * (1.0f / 32768.0f);
}

static void fill(float *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = nextf();
}

static void fill_pos(float *p, size_t n, float base)
{
	for (size_t i = 0; i < n; i++) {
		float v = nextf();
		p[i] = base + (v < 0.0f ? -v : v) * base;
	}
}

static optcfg cfg(unsigned steps, bool nesterov, int graft)
{
	optcfg o = {.b1 = 0.9f, .b2 = 0.999f, .eps = 1e-6f,
		.mom = 0.95f, .gain = 0.2f, .b1t = 0.99f, .b2t = 0.9999f,
		.malt_b2 = 0.999f, .malt_eps = 1e-8f,
		.malt_b1t = 0.99f, .malt_b2t = 0.9999f,
		.ns_steps = steps, .graft = graft, .nesterov = nesterov};
	return o;
}

/* ---- the shared polynomial, both transpose cases, H up to 32 ---------------- */

static void polynomial(void)
{
	static const struct { unsigned rows, cols; } shapes[] = {
		{8, 26}, {26, 8}, {16, 26}, {32, 26}, {26, 32}
	};
	for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++) {
		unsigned rows = shapes[s].rows, cols = shapes[s].cols;
		size_t n = (size_t)rows * cols;
		bool tr;
		size_t mr, mc;
		ns_geom(rows, cols, &tr, &mr, &mc);
		for (unsigned steps = 1; steps <= 5; steps += 4) {
			float src[NS_MAXEL], xn[NS_MAXEL], xr[NS_MAXEL];
			float an[NS_MAXEL], ar[NS_MAXEL], bn[NS_MAXEL], br[NS_MAXEL];
			float yn[NS_MAXEL], yr[NS_MAXEL];
			seq = 0x1234u + (uint32_t)s * 7919u + steps;
			fill(src, n);
			for (size_t i = 0; i < n; i++) {
				size_t at = ns_at(i, cols, tr, mc);
				xr[at] = xn[at] = src[i];
			}
			float ss = 0.0f;
			for (size_t i = 0; i < mr * mc; i++)
				ss += xn[i] * xn[i];
			if (!(ss > 1e-12f))
				continue;
			float inv = 1.0f / fsqrt(ss);
			for (size_t i = 0; i < mr * mc; i++) {
				xr[i] *= inv;
				xn[i] *= inv;
			}
			ref_ns(xr, ar, br, yr, mr, mc, steps);
			ns_ortho(xn, an, bn, yn, mr, mc, steps);
			CHECK(memcmp(xn, xr, mr * mc * sizeof *xn) == 0);
	 	}
	}
}

/* ---- wrappers vs the oracle ------------------------------------------------- */

static void wrappers(void)
{
	static const struct { unsigned rows, cols; } shapes[] = {
		{8, 26}, {32, 26}, {26, 8}
	};
	for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++) {
		unsigned rows = shapes[s].rows, cols = shapes[s].cols;
		size_t n = (size_t)rows * cols;
		for (unsigned steps = 1; steps <= 5; steps += 4) {
			for (int nest = 0; nest < 2; nest++) {
				for (int graft = 0; graft < 3; graft++) {
					optcfg o = cfg(steps, nest != 0, graft);
					/* AdamW */
					{
						float wn[n], wr[n], mn[n], mr_[n], vr[n], vn[n], g[n];
						seq = 0xA11CEu + (uint32_t)s * 101u + steps + (unsigned)nest;
						fill(g, n); fill(wn, n); fill_pos(vr, n, 1e-3f);
						memcpy(wr, wn, sizeof wn);
						for (size_t i = 0; i < n; i++) mn[i] = mr_[i] = g[i] * 0.1f;
						memcpy(vn, vr, sizeof vr);
						ref_adam_rows(&o, wr, mr_, vr, g, n, 1.0f / 16, 0.02f, 1e-5f, 1.0f);
						adam_rows(&o, wn, mn, vn, g, n, 1.0f / 16, 0.02f, 1e-5f, 1.0f,
						    nullptr, 0);
						CHECK(memcmp(wn, wr, sizeof wn) == 0);
						CHECK(memcmp(mn, mr_, sizeof mn) == 0);
						CHECK(memcmp(vn, vr, sizeof vn) == 0);
					}
					/* Muon */
					{
						float wn[n], wr[n], mn[n], mr_[n], g[n];
						seq = 0xB0B0u + (uint32_t)s * 211u + steps + (unsigned)nest;
						fill(g, n); fill(wn, n);
						memcpy(wr, wn, sizeof wn);
						for (size_t i = 0; i < n; i++) mn[i] = mr_[i] = g[i] * 0.05f;
						ref_muon_rows(&o, wr, mr_, g, rows, cols, 1.0f / 16,
						    0.01f, 1e-3f, 1.0f);
						muon_rows(&o, wn, mn, g, rows, cols, 1.0f / 16,
						    0.01f, 1e-3f, 1.0f, nullptr, 0);
						CHECK(memcmp(wn, wr, sizeof wn) == 0);
						CHECK(memcmp(mn, mr_, sizeof mn) == 0);
					}
					/* MALT and MALTER */
					for (int adaptive = 0; adaptive < 2; adaptive++) {
						float wn[n], wr[n], mn[n], mr_[n], g[n];
						float ln[rows], lr_[rows], rn[cols], rr[cols], nun, nur;
						seq = 0xC0FFu + (uint32_t)s * 307u + steps + (unsigned)nest +
						    (unsigned)graft * 17u + (unsigned)adaptive;
						fill(g, n); fill(wn, n);
						memcpy(wr, wn, sizeof wn);
						for (size_t i = 0; i < n; i++) mn[i] = mr_[i] = g[i] * 0.05f;
						fill_pos(ln, rows, 1e-3f);
						fill_pos(rn, cols, 1e-3f);
						memcpy(lr_, ln, sizeof lr_);
						memcpy(rr, rn, sizeof rr);
						nun = nur = 1e-3f;
						ref_malt_rows(&o, wr, mr_, lr_, rr, &nur, g, rows, cols,
						    1.0f / 16, 0.055f, 1e-3f, 1.0f, adaptive != 0);
						malt_rows(&o, wn, mn, ln, rn, &nun, g, rows, cols,
						    1.0f / 16, 0.055f, 1e-3f, 1.0f, adaptive != 0,
						    nullptr, 0);
						CHECK(memcmp(wn, wr, sizeof wn) == 0);
						CHECK(memcmp(mn, mr_, sizeof mn) == 0);
						CHECK(memcmp(ln, lr_, sizeof ln) == 0);
						CHECK(memcmp(rn, rr, sizeof rn) == 0);
						CHECK(memcmp(&nun, &nur, sizeof nun) == 0);
					}
				}
			}
		}
	}
}

/* ---- seven optimizer profiles: trajectory goldens --------------------------- */

typedef struct {
	const char *name;
	hyb_mode mode;
	bool adaptive;
	float lr, wd, mix;
} profile_desc;

static const profile_desc profiles[] = {
	{"adam",   HYB_ADAM,   false, 0.02f,  1e-5f, 1.0f},
	{"muon",   HYB_MUON,   false, 0.01f,  1e-3f, 1.0f},
	{"blend",  HYB_BLEND,  false, 0.01f,  1e-3f, 0.5f},
	{"alt",    HYB_ALT,    false, 0.01f,  1e-3f, 1.0f},
	{"route",  HYB_ROUTE,  false, 0.01f,  1e-3f, 1.0f},
	{"malt",   HYB_MALT,   false, 0.055f, 1e-3f, 1.0f},
	{"malter", HYB_MALTER, true,  0.055f, 1e-3f, 1.0f},
};

/* Golden CRCs recorded from the oracle (verbatim main) over the raw float
 * bytes of the full trajectory state. Filled from the reference run below;
 * any change to the shipped arithmetic breaks these. */
static const uint32_t profile_golden[] = {
	0x963B6289u, /* adam   */
	0x9A197E2Cu, /* muon   */
	0x61AF6C05u, /* blend  */
	0xD5A36E5Au, /* alt    */
	0xA57FF9E8u, /* route  */
	0xEDB16729u, /* malt   */
	0xB9F32C0Eu, /* malter */
};

static uint32_t trajectory(const profile_desc *p, bool use_ref)
{
	enum { H = 8, C = 26, NSTEPS = 12 };
	optcfg o = cfg(5, true, GRAFT_O);
	float w[H * C] = {0}, m[H * C] = {0}, v[H * C] = {0};
	float m1[H * C] = {0}, v1[H * C] = {0};
	float l[H] = {0}, r[C] = {0}, nu = 0.0f, g[H * C];
	seq = 0x5EEDu;
	fill_pos(l, H, 1e-3f);
	fill_pos(r, C, 1e-3f);
	for (int step = 0; step < NSTEPS; step++) {
		fill(g, H * C);
		bool muon = p->mode == HYB_MUON ||
		    (p->mode == HYB_ALT && (step & 1) != 0) ||
		    (p->mode == HYB_ROUTE && (step & 3) != 0);
		bool malty = p->mode == HYB_MALT || p->mode == HYB_MALTER;
		if (p->mode == HYB_ADAM) {
			if (use_ref) ref_adam_rows(&o, w, m, v, g, H * C, 1.0f / 16,
					   p->lr, p->wd, 1.0f);
			else adam_rows(&o, w, m, v, g, H * C, 1.0f / 16,
				       p->lr, p->wd, 1.0f, nullptr, 0);
		} else if (malty) {
			if (use_ref) ref_malt_rows(&o, w, m, l, r, &nu, g, H, C,
					   1.0f / 16, p->lr, p->wd, 1.0f, p->adaptive);
			else malt_rows(&o, w, m, l, r, &nu, g, H, C,
				       1.0f / 16, p->lr, p->wd, 1.0f, p->adaptive,
				       nullptr, 0);
		} else if (p->mode == HYB_BLEND) {
			if (use_ref) {
				ref_adam_rows(&o, w, m1, v1, g, H * C, 1.0f / 16,
					      p->lr, p->wd, 1.0f - p->mix);
				ref_muon_rows(&o, w, m, g, H, C, 1.0f / 16,
					      p->lr, p->wd, p->mix);
			} else {
				adam_rows(&o, w, m1, v1, g, H * C, 1.0f / 16,
					  p->lr, p->wd, 1.0f - p->mix, nullptr, 0);
				muon_rows(&o, w, m, g, H, C, 1.0f / 16,
					  p->lr, p->wd, p->mix, nullptr, 0);
			}
		} else if (muon) {
			if (use_ref) ref_muon_rows(&o, w, m, g, H, C, 1.0f / 16,
					   p->lr, p->wd, 1.0f);
			else muon_rows(&o, w, m, g, H, C, 1.0f / 16,
				       p->lr, p->wd, 1.0f, nullptr, 0);
		} else {
			if (use_ref) ref_adam_rows(&o, w, m1, v1, g, H * C, 1.0f / 16,
					   p->lr, p->wd, 1.0f);
			else adam_rows(&o, w, m1, v1, g, H * C, 1.0f / 16,
				       p->lr, p->wd, 1.0f, nullptr, 0);
		}
	}
	uint32_t crc = taw_crc32c(0, w, sizeof w);
	crc = taw_crc32c(crc, m, sizeof m);
	crc = taw_crc32c(crc, v, sizeof v);
	crc = taw_crc32c(crc, m1, sizeof m1);
	crc = taw_crc32c(crc, v1, sizeof v1);
	crc = taw_crc32c(crc, l, sizeof l);
	crc = taw_crc32c(crc, r, sizeof r);
	crc = taw_crc32c(crc, &nu, sizeof nu);
	return crc;
}

static void profiles_test(void)
{
	for (size_t i = 0; i < sizeof profiles / sizeof *profiles; i++) {
		uint32_t ref = trajectory(&profiles[i], true);
		uint32_t got = trajectory(&profiles[i], false);
#ifdef PRINT_GOLDENS
		printf("profile %-6s ref=0x%08X new=0x%08X\n",
		    profiles[i].name, ref, got);
#endif
		if (got != ref)
			fprintf(stderr, "profile %s: ref 0x%08X new 0x%08X\n",
			    profiles[i].name, ref, got);
		CHECK(got == ref);
		CHECK(ref == profile_golden[i]);
	}
}

/* ---- hooks ABI: one gate owner, risk bank, NULL = shipped path ------------- */

typedef struct {
	float *sig, *mul;
	float beta, p;
	bool mean_pres;
} test_gate;

static void tg_begin(void *ctx, const float *g, size_t len, float scale)
{
	test_gate *t = ctx;
	float sum = 0.0f;
	for (size_t k = 0; k < len; k++) {
		t->mul[k] = fit_gate_step(t->sig + k, g[k] * scale,
		    t->beta, t->p);
		sum += t->mul[k];
	}
	if (t->mean_pres) {
		float mean = sum / (float)len;
		if (!(mean > 1e-12f))
			mean = 1.0f;
		for (size_t k = 0; k < len; k++)
			t->mul[k] /= mean;
	}
}

static float tg_gate(void *ctx, size_t k, float gk)
{
	(void)gk; /* the bank is already normalized by tg_begin */
	test_gate *t = ctx;
	return t->mul[k];
}

/* Oracle: staged #8's adam_rows_gated math, written out locally so the test
 * does not depend on a second copy living in fit.c. */
static void ref_gated_adam(optcfg *o, float *w, float *m, float *v,
    const float *g, size_t len, float scale, float lr, float wd, float mix,
    float *sig, float beta, float p, bool mean_pres)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	float gate[NS_MAXEL], gsum = 0.0f;
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		gate[k] = fit_gate_step(sig + k, gk, beta, p);
		gsum += gate[k];
	}
	float gmean = mean_pres ? gsum / (float)len : 1.0f;
	if (gmean < 1e-12f)
		gmean = 1.0f;
	for (size_t k = 0; k < len; k++) {
		float upd = (m[k] * c1) / (fsqrt(v[k] * c2) + o->eps) +
		    wd * w[k];
		upd *= mean_pres ? gate[k] / gmean : gate[k];
		w[k] -= mix * lr * upd;
	}
}

static bool gate_near(float a, float b)
{
	float d = a - b;
	return d < 0.0f ? -d < 1e-6f : d < 1e-6f;
}

static void hooks_abi(void)
{
	/* The gate's direct numbers are #8's unit test (af5fd00). */
	float s = 0.0f;
	CHECK(gate_near(fit_gate_step(&s, 1.0f, 0.9f, 1.0f), 0.1f));
	CHECK(gate_near(fit_gate_step(&s, 1.0f, 0.9f, 1.0f), 0.19f));
	float sn = 0.0f;
	CHECK(gate_near(fit_gate_step(&sn, -1.0f, 0.9f, 1.0f), 0.1f));
	CHECK(fit_gate_step(&sn, 1.0f, 0.9f, 1.0f) < 0.1f);
	float s2 = 0.0f;
	CHECK(gate_near(fit_gate_step(&s2, 1.0f, 0.9f, 2.0f), 0.01f));

	enum { N = 64 };
	for (int mean_pres = 0; mean_pres < 2; mean_pres++) {
		optcfg o = cfg(5, true, GRAFT_O);
		float w[N], wr[N], m[N], mr_[N], v[N], vr[N], g[N];
		float sig[N] = {0}, sigr[N] = {0}, mul[N];
		test_gate tg = {.sig = sig, .mul = mul, .beta = 0.9f,
		    .p = mean_pres ? 1.0f : 2.0f,
		    .mean_pres = mean_pres != 0};
		opt_hooks h = {0};
		h.ctx = &tg;
		h.shape.ctx = &tg; /* step_shape owns the gate state */
		h.shape.begin = tg_begin;
		h.shape.gate = tg_gate;
		h.nb = 1;
		float risk[1] = {0};
		h.risk = risk;

		seq = 0xABBAu + (uint32_t)mean_pres;
		fill(g, N); fill(w, N);
		for (size_t k = 0; k < N; k++) {
			m[k] = mr_[k] = g[k] * 0.1f;
			v[k] = vr[k] = 1e-3f + (float)(k % 7) * 1e-4f;
		}
		memcpy(wr, w, sizeof w);

		adam_rows(&o, w, m, v, g, N, 1.0f / 16, 0.02f, 1e-5f, 1.0f,
		    &h, 0);
		ref_gated_adam(&o, wr, mr_, vr, g, N, 1.0f / 16, 0.02f,
		    1e-5f, 1.0f, sigr, tg.beta, tg.p, tg.mean_pres != 0);
		CHECK(memcmp(w, wr, sizeof w) == 0);
		CHECK(memcmp(m, mr_, sizeof m) == 0);
		CHECK(memcmp(v, vr, sizeof v) == 0);
		CHECK(memcmp(sig, sigr, sizeof sig) == 0);

		/* FARO risk: Σ (mix*lr*mh/(sqrt(vh)+eps))², wd excluded. */
		float want = 0.0f;
		const float c1 = 1.0f / (1.0f - o.b1t), c2 = 1.0f / (1.0f - o.b2t);
		for (size_t k = 0; k < N; k++) {
			float upd = 1.0f * 0.02f *
			    (mr_[k] * c1 / (fsqrt(vr[k] * c2) + o.eps));
			want += upd * upd;
		}
		CHECK(risk[0] == want);
	}
}

int main(void)
{
	polynomial();
	wrappers();
	profiles_test();
	hooks_abi();
	return CHECK_DONE();
}
