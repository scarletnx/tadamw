/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Thin optimizer wrappers. The Newton-Schulz loop itself lives exactly once,
 * in ns_ortho() in fit_opt.h; everything here is the per-optimizer
 * pre/post-processing that was already in fit.c. The FP expression order is
 * preserved character-for-character so shipped profiles stay bit-identical. */
#include "fit_opt.h"

/* x^{-1/8} without libm; x > 0 thanks to the damping term. */
static inline float inv_root8(float x)
{
	return 1.0f / fsqrt(fsqrt(fsqrt(x)));
}

/* The single sign-consistency gate, verbatim from staged #8 (af5fd00). Both
 * #8's TAW_EPS_MODE=c3 and #6's TAW_NAMO_GATE=c3 must call this, not copy it. */
float fit_gate_step(float *sig, float gk, float beta, float p)
{
	const float sg = gk > 0.0f ? 1.0f : gk < 0.0f ? -1.0f : 0.0f;
	*sig = beta * *sig + (1.0f - beta) * sg;
	const float a = *sig < 0.0f ? -*sig : *sig;
	return p == 2.0f ? a * a : a;
}

/* AdamW on a flat vector. `mix` scales the whole update (1 for plain AdamW,
 * 1-alpha for the Adam half of a blend). Gradients are cleared by caller.
 * The ungated branch is main's expression character-for-character; a NULL
 * h (or empty shape) therefore stays bit-identical. */
void adam_rows(optcfg *o, float *w, float *m, float *v, const float *g,
    size_t len, float scale, float lr, float wd, float mix,
    const opt_hooks *h, unsigned bid)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	const bool gated = h && h->shape.begin && h->shape.gate;
	if (gated)
		h->shape.begin(h->shape.ctx, g, len, scale);
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		float mh = m[k] * c1, vh = v[k] * c2;
		if (gated) {
			/* #8's rounding order: shape upd first, then apply lr. */
			float upd = mh / (fsqrt(vh) + o->eps) + wd * w[k];
			upd *= h->shape.gate(h->shape.ctx, k, gk);
			w[k] -= mix * lr * upd;
		} else {
			w[k] -= mix * lr * (mh / (fsqrt(vh) + o->eps) + wd * w[k]);
		}
		if (h && h->risk) {
			/* FARO's Δ_b: decoupled weight decay excluded. */
			float upd = mix * lr * (mh / (fsqrt(vh) + o->eps));
			h->risk[bid] += upd * upd;
		}
	}
}

/* Muon on a row-major rows x cols matrix. */
void muon_rows(const optcfg *o, float *w, float *m, const float *g,
    size_t rows, size_t cols, float scale, float lr, float wd, float mix,
    const opt_hooks *h, unsigned bid)
{
	float x[NS_MAXEL], a[NS_MAXEL], b[NS_MAXEL], y[NS_MAXEL];
	bool tr;
	size_t mr, mc;
	ns_geom(rows, cols, &tr, &mr, &mc);
	if (rows * cols > NS_MAXEL || mr == 0 || mc == 0)
		return;
	for (size_t i = 0; i < rows * cols; i++) {
		float gi = g[i] * scale;
		m[i] = o->mom * m[i] + gi;
		float u = o->nesterov ? o->mom * m[i] + gi : m[i];
		x[ns_at(i, cols, tr, mc)] = u;
	}
	float ss = 0.0f;
	for (size_t i = 0; i < mr * mc; i++)
		ss += x[i] * x[i];
	float norm = fsqrt(ss);
	if (!(norm > 1e-12f))
		return; /* zero gradient: leave weights alone */
	for (size_t i = 0; i < mr * mc; i++)
		x[i] *= 1.0f / norm;
	ns_ortho(x, a, b, y, mr, mc, o->ns_steps);
	const size_t dmax = rows > cols ? rows : cols;
	const float factor = mix * lr * o->gain * fsqrt((float)dmax);
	for (size_t i = 0; i < rows * cols; i++) {
		size_t at = ns_at(i, cols, tr, mc);
		w[i] -= factor * x[at] + mix * lr * wd * w[i];
		if (h && h->risk) {
			float upd = factor * x[at];
			h->risk[bid] += upd * upd;
		}
	}
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
 * Orthogonalization uses the same ns_ortho() as Muon. */
void malt_rows(const optcfg *o, float *w, float *m, float *l, float *r,
    float *nu, const float *g, size_t rows, size_t cols, float scale,
    float lr, float wd, float mix, bool adaptive,
    const opt_hooks *h, unsigned bid)
{
	float x[NS_MAXEL], a[NS_MAXEL], b[NS_MAXEL], y[NS_MAXEL];
	float Lc[NS_MAXDIM], Rc[NS_MAXDIM];
	bool tr;
	size_t mr, mc;
	ns_geom(rows, cols, &tr, &mr, &mc);
	if (rows * cols > NS_MAXEL || mr == 0 || mc == 0)
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
			x[ns_at(idx, cols, tr, mc)] = u;
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
			const size_t at = ns_at(idx, cols, tr, mc);
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
	ns_ortho(x, a, b, y, mr, mc, o->ns_steps);

	/* O = x; D = L O R (map the orthogonal direction back). */
	float o2 = 0.0f, d2 = 0.0f;
	for (size_t i = 0; i < mr * mc; i++)
		o2 += x[i] * x[i];
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			const size_t at = ns_at(i * cols + j, cols, tr, mc);
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
			const size_t at = ns_at(idx, cols, tr, mc);
			w[idx] -= fac * x[at] + mix * lr * wd * w[idx];
			if (h && h->risk) {
				float upd = fac * x[at];
				h->risk[bid] += upd * upd;
			}
		}
	}
}

/* FARO's TAW_FARO_STAT=lin linearized return. Upstream's copy repeated the
 * NS loop; here it shares ns_ortho(), so the polynomial has one home. The
 * normalization/contraction order is unchanged (verified byte-identical). */
float ns_score(const optcfg *o, const float *g, unsigned rows, unsigned cols)
{
	float x[NS_MAXEL], a[NS_MAXEL], b[NS_MAXEL], y[NS_MAXEL];
	const size_t n = (size_t)rows * cols;
	if (n > NS_MAXEL || !n)
		return 0.0f;
	bool tr;
	size_t mr, mc;
	ns_geom(rows, cols, &tr, &mr, &mc);
	for (size_t i = 0; i < n; i++)
		x[ns_at(i, cols, tr, mc)] = g[i];
	float ss = 0.0f;
	for (size_t i = 0; i < n; i++)
		ss += x[i] * x[i];
	if (!(ss > 1e-12f))
		return 0.0f;
	float inv = 1.0f / fsqrt(ss);
	for (size_t i = 0; i < n; i++)
		x[i] *= inv;
	ns_ortho(x, a, b, y, mr, mc, o->ns_steps);
	double dot = 0.0, xn = 0.0;
	for (size_t i = 0; i < n; i++) {
		size_t at = ns_at(i, cols, tr, mc);
		dot += (double)g[i] * x[at];
		xn += (double)x[at] * x[at];
	}
	return xn > 0.0 ? (float)(dot / (double)fsqrt((float)xn)) : 0.0f;
}
