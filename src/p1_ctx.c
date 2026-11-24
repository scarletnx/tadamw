/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 context models and the integer squash/stretch (format.md §5.1). */
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "p1.h"
#include "taw_internal.h"

const uint16_t p1_sel_size[P1_SEL_COUNT] = {1, 8, 8, 256, 256, 4096};

/* ---- squash / stretch ----------------------------------------------------- */

/* 65536 / (1 + e^-(k-16)/2), rounded, k = 0..32. */
const uint16_t p1_squash_knot[33] = {
	22, 36, 60, 98, 162, 267, 439, 720, 1179, 1921, 3108, 4971, 7812,
	11955, 17625, 24743, 32768, 40793, 47911, 53581, 57724, 60565, 62428,
	63615, 64357, 64816, 65097, 65269, 65374, 65438, 65476, 65500, 65514,
};

uint32_t p1_squash(int32_t t)
{
	if (t > 2047)
		t = 2047;
	if (t < -2047)
		t = -2047;
	uint32_t u = (uint32_t)(t + 2048); /* 1..4095 */
	uint32_t j = u >> 7, w = u & 127;
	return (p1_squash_knot[j] * (128 - w) + p1_squash_knot[j + 1] * w + 64) >> 7;
}

int16_t p1_stretch_tab[4096];
int32_t p1_stretch32[4096];
/* rate[n] = floor(2^17 / (2n + 3)) = 2^16 / (n + 1.5) */
uint32_t p1_rate32[1024];
#define rate_tab p1_rate32

static void tables_build(void)
{
	int pi = 0;
	for (int t = -2047; t <= 2047; t++) {
		int v = (int)(p1_squash(t) >> 4);
		for (int j = pi; j <= v; j++)
			p1_stretch_tab[j] = (int16_t)t;
		pi = v + 1;
	}
	for (int j = pi; j < 4096; j++)
		p1_stretch_tab[j] = 2047;
	for (int j = 0; j < 4096; j++)
		p1_stretch32[j] = p1_stretch_tab[j];
	for (uint32_t n = 0; n < 1024; n++)
		rate_tab[n] = (1u << 17) / (2 * n + 3);
}

static atomic_int tables_state; /* 0 none, 1 building, 2 ready */

void p1_tables_init(void)
{
	if (atomic_load(&tables_state) == 2)
		return;
	int expect = 0;
	if (atomic_compare_exchange_strong(&tables_state, &expect, 1)) {
		tables_build();
		atomic_store(&tables_state, 2);
		return;
	}
	while (atomic_load(&tables_state) != 2)
		;
}

/* ---- counters --------------------------------------------------------------
 * A context slot holds
 *   bits 31..16  ps  P(1), 16 bits, adaptive rate 1/(n + 1.5)
 *   bits 15..8   h   bit history: the last up to 7 bits behind a leading 1
 *   bits  7..6   0
 *   bits  5..0   n   update count, saturating at LIMIT_SLOW
 * stored XOR SLOT_INIT, so an all-zero table means ps = 2^15, h = 1, n = 0.
 * Each context model also owns an adaptive map h -> P(1) (indirect model).
 * The match model uses ps and n only. */

constexpr uint32_t LIMIT_SLOW = P1_LIMIT_SLOW;
constexpr uint32_t LIMIT_MATCH = 63;
constexpr uint32_t LIMIT_HMAP = P1_LIMIT_HMAP;
constexpr uint32_t SLOT_INIT = P1_SLOT_INIT;

static inline uint32_t slot_ps12(uint32_t v)
{
	return (v ^ SLOT_INIT) >> 20;
}

static inline uint32_t slot_hist(uint32_t v)
{
	return ((v ^ SLOT_INIT) >> 8) & 0xFF;
}

static inline void slot_update(uint32_t *s, int bit, uint32_t limit)
{
	uint32_t e = *s ^ SLOT_INIT;
	uint32_t ps = e >> 16, h = (e >> 8) & 0xFF, n = e & 63;
	uint32_t r = rate_tab[n];
	uint32_t up = ((65535 - ps) * r) >> 16, dn = (ps * r) >> 16;
	ps = bit ? ps + up : ps - dn;
	uint32_t h2 = h << 1 | (uint32_t)bit;
	h = h >= 128 ? (h2 & 0x7F) | 0x80 : h2;
	n += n < limit;
	*s = (ps << 16 | h << 8 | n) ^ SLOT_INIT;
}

/* History map entry: P(1) in 22 bits, count in 10, XOR 0x80000000. */
static inline uint32_t hm_p12(uint32_t v)
{
	return (v ^ 0x80000000u) >> 20;
}

static inline void hm_update(uint32_t *s, int bit, uint32_t limit)
{
	uint32_t e = *s ^ 0x80000000u;
	uint32_t p = e >> 10, n = e & 1023;
	uint64_t r = rate_tab[n];
	uint32_t up = (uint32_t)(((uint64_t)(0x3FFFFFu - p) * r) >> 16);
	uint32_t dn = (uint32_t)(((uint64_t)p * r) >> 16);
	p = bit ? p + up : p - dn;
	n += n < limit;
	*s = (p << 10 | n) ^ 0x80000000u;
}

/* ---- hashing -------------------------------------------------------------- */

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t k)
{
	uint32_t x = a * 0x9E3779B1u + b * 0x85EBCA77u + k * 0xC2B2AE3Du;
	x ^= x >> 15;
	x *= 0x2C1B3C6Du;
	x ^= x >> 12;
	x *= 0x297A2D39u;
	x ^= x >> 15;
	return x;
}

/* ---- context state ---------------------------------------------------------
 * Data layout: every context table lives in one arena of u32 slots,
 * [O0: 256][O1: 65536][T_2 .. T_11: 2^L lines of 16 each], and the per-bit
 * hot state is a set of small parallel arrays (line bases, slot indices,
 * slot values, history-map indices) that the scalar loops and the AVX2
 * kernels both walk. (Computing the next nibble's hashes a bit early for
 * both outcomes, with prefetches, measured slower than computing them once
 * at the boundary: the out-of-order core already hides that latency.) */

enum { ARENA_O1 = 256, ARENA_HX = 256 + 65536 };

taw_status p1_ctx_init(p1_ctx *c, const p1_shape *s)
{
	p1_tables_init();
	memset(c, 0, sizeof *c);
	c->avx2 = p1_isa_current() >= P1_ISA_AVX2;
	c->lines_log2 = s->lines_log2;
	c->hist_mask = (1u << s->hist_log2) - 1;
	c->match_mask = (1u << s->match_log2) - 1;
	c->arena_n = ARENA_HX + ((size_t)P1_NHASH << (s->lines_log2 + 4));
	/* Fresh anonymous memory is zero, and pages a unit never touches are
	 * never faulted in (the tables are sized for the coding unit, so many
	 * lines stay unused). */
	int fl = MAP_PRIVATE | MAP_ANON;
#ifdef MAP_ALIGNED_SUPER
	fl |= MAP_ALIGNED_SUPER;
#endif
	void *a = mmap(nullptr, c->arena_n * sizeof *c->arena,
	    PROT_READ | PROT_WRITE, fl, -1, 0);
	c->arena = a == MAP_FAILED ? nullptr : a;
	c->epoch = 1;
	c->fresh = true;
	c->mtab = calloc((size_t)c->match_mask + 1, sizeof *c->mtab);
	c->hist = calloc((size_t)c->hist_mask + 1, 1);
	if (!c->arena || !c->mtab || !c->hist) {
		p1_ctx_free(c);
		return TAW_E_NOMEM;
	}
	p1_ctx_reset(c);
	return TAW_OK;
}

void p1_ctx_free(p1_ctx *c)
{
	if (c->arena)
		munmap(c->arena, c->arena_n * sizeof *c->arena);
	free(c->mtab);
	free(c->hist);
	memset(c, 0, sizeof *c);
}

/* g[k] = H(a_k, b_k, kk_k) for the hashed models O2..S1278, given the
 * history after a byte (format.md §2.2.4). */
static void byte_hashes_for(const p1_ctx *c, uint32_t c4, uint32_t c8,
    uint32_t word, uint32_t *g)
{
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_byte_hashes_avx2(c4, c8, word, g);
		return;
	}
#endif
	(void)c;
	g[0] = hash3(c4 & 0xFFFF, 0, 2);
	g[1] = hash3(c4 & 0xFFFFFF, 0, 3);
	g[2] = hash3(c4, 0, 4);
	g[3] = hash3(c4, c8 & 0xFFFF, 6);
	g[4] = hash3((c4 >> 8) & 0xFFFF, 0, 8);
	g[5] = hash3(word, c4 & 0xFF, 9);
	g[6] = hash3(c4 & 0xFFFF0000u, 0, 11);
	g[7] = hash3(c4 & 0xFF00FF00u, 0, 12);
	g[8] = hash3(c4 & 0xFF0000FFu, 0, 13);
	g[9] = hash3(c4 & 0xFFFF, c8 & 0xFFFF0000u, 14);
}

/* hh[k] = H(g[k], q, 0): the line hash for a nibble (q = 0 at bit 0, the
 * partial byte 16..31 at bit 4). */
static void line_hashes(const p1_ctx *c, const uint32_t *g, uint32_t q,
    uint32_t *hh)
{
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_line_hashes_avx2(g, q, hh);
	} else
#endif
	{
		for (unsigned k = 0; k < P1_NHASH; k++)
			hh[k] = hash3(g[k], q, 0);
	}
}

/* Make the lines named by hh current: check tags, reset mismatches. */
static void lines_apply(p1_ctx *c, const uint32_t *hh)
{
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_lines_apply_avx2(c, hh);
		return;
	}
#endif
	const unsigned sh = 32 - c->lines_log2;
	for (unsigned k = 0; k < P1_NHASH; k++) {
		uint32_t lb = ARENA_HX +
		    (uint32_t)(((k << c->lines_log2) + (hh[k] >> sh)) * 16);
		uint32_t tag = (hh[k] & 0xFFFFu) | c->epoch << 16;
		uint32_t *ln = c->arena + lb;
		c->lb[P1_CM_O2 + k] = lb;
		if (ln[0] != tag) {
			memset(ln, 0, 64);
			ln[0] = tag;
		}
	}
}

/* Prefetch the lines that hh would select. */
static inline void lines_prefetch(const p1_ctx *c, const uint32_t *hh)
{
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_lines_prefetch_avx2(c, hh);
		return;
	}
#endif
	const unsigned sh = 32 - c->lines_log2;
	for (unsigned k = 0; k < P1_NHASH; k++)
		__builtin_prefetch(c->arena + ARENA_HX +
		    (((size_t)k << c->lines_log2) + (hh[k] >> sh)) * 16, 1);
}

static inline uint32_t word_next(uint32_t word, uint32_t ch)
{
	return (ch | 32) - 'a' < 26 ? (word ^ (ch | 32)) * 0x01000193u : 0;
}

static inline uint32_t match_index(const p1_ctx *c, uint32_t c4, uint32_t c8)
{
	return hash3(c4, c8 & 0xFF, 10) & c->match_mask;
}

void p1_ctx_reset(p1_ctx *c)
{
	/* O0 and O1 are cleared; the hashed lines are invalidated by a new
	 * epoch in their tags, so each is reset when it is next selected,
	 * exactly as if the table had been zeroed. */
	memset(c->arena, 0, ARENA_HX * sizeof *c->arena);
	if (!c->fresh && ++c->epoch > P1_EPOCH_MAX) {
		memset(c->arena + ARENA_HX, 0,
		    (c->arena_n - ARENA_HX) * sizeof *c->arena);
		c->epoch = 1;
	}
	c->fresh = false;
	memset(c->mtab, 0, ((size_t)c->match_mask + 1) * sizeof *c->mtab);
	memset(c->hist, 0, (size_t)c->hist_mask + 1);
	memset(c->sm, 0, sizeof c->sm);
	memset(c->hmap, 0, sizeof c->hmap);
	memset(c->val, 0, sizeof c->val);
	memset(c->hmi, 0, sizeof c->hmi);
	c->c0 = 1;
	c->nib = 1;
	c->bitpos = 0;
	c->c4 = c->c8 = c->word = c->pos = 0;
	c->mptr = c->mlen = 0;
	c->mexp = -1;
	alignas(32) uint32_t hh[16];
	byte_hashes_for(c, 0, 0, 0, c->hash);
	line_hashes(c, c->hash, 0, hh);
	lines_apply(c, hh);
}

void p1_ctx_inputs(p1_ctx *c, float *x, p1_sel_ctx *sc)
{
	const uint32_t c0 = c->c0, nib = c->nib;
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_cm_inputs_avx2(c, x);
	} else
#endif
	{
		uint32_t *si = c->si;
		si[P1_CM_O0] = c0;
		si[P1_CM_O1] = ARENA_O1 + ((c->c4 & 0xFF) << 8 | c0);
		for (unsigned k = P1_CM_O2; k < P1_NCM; k++)
			si[k] = c->lb[k] + nib;
		for (unsigned k = 0; k < P1_NCM; k++)
			c->val[k] = c->arena[si[k]];
		for (unsigned k = 0; k < P1_NCM; k++) {
			const uint32_t v = c->val[k];
			c->hmi[k] = k * 256 + slot_hist(v);
			x[k] = (float)p1_stretch_tab[slot_ps12(v)] * (1.0f / 256.0f);
			x[P1_NCM + k] = (float)p1_stretch_tab[hm_p12(c->hmap[c->hmi[k]])] *
			    (1.0f / 256.0f);
		}
	}

	uint32_t msel = 0;
	c->mbit = -1;
	if (c->mexp >= 0 &&
	    ((uint32_t)(c->mexp | 256) >> (8 - c->bitpos)) == c0) {
		uint32_t lg = 31 - (uint32_t)__builtin_clz(c->mlen);
		msel = lg - 1 < 7 ? lg - 1 : 7;
		c->mbit = (c->mexp >> (7 - c->bitpos)) & 1;
		c->msi = msel * 2 + (uint32_t)c->mbit;
		x[P1_IN_MATCH] = (float)p1_stretch_tab[slot_ps12(c->sm[c->msi])] *
		    (1.0f / 256.0f);
	} else {
		x[P1_IN_MATCH] = 0.0f;
	}
	*sc = (p1_sel_ctx){
		.bitpos = (uint8_t)c->bitpos,
		.c0 = (uint8_t)c0,
		.c1 = (uint8_t)c->c4,
		.msel = (uint8_t)msel,
	};
}

static void byte_end(p1_ctx *c)
{
	const uint32_t ch = c->c0 & 0xFF;
	const uint32_t hm = c->hist_mask;
	c->hist[c->pos & hm] = (uint8_t)ch;
	c->pos++;
	c->c8 = c->c8 << 8 | c->c4 >> 24;
	c->c4 = c->c4 << 8 | ch;
	c->word = word_next(c->word, ch);

	/* match model */
	if (c->mlen) {
		if (c->mexp == (int)ch) {
			c->mptr++;
			if (c->mlen < 65535)
				c->mlen++;
		} else {
			c->mlen = 0;
		}
	}
	const unsigned b = ch & 1; /* the outcome speculated at bit 7 */
	uint32_t *me = &c->mtab[c->spec_mi[b]];
	if (c->mlen == 0 && *me) {
		uint32_t e = *me, d = c->pos - e;
		if (d >= 1 && d <= hm) {
			uint32_t k = 0;
			while (k < P1_MAXVERIFY && k < e &&
			    c->hist[(e - 1 - k) & hm] == c->hist[(c->pos - 1 - k) & hm])
				k++;
			if (k >= P1_MINLEN) {
				c->mlen = k;
				c->mptr = e;
			}
		}
	}
	*me = c->pos;

	c->c0 = 1;
	c->nib = 1;
	c->bitpos = 0;
	memcpy(c->hash, c->spec_g[b], sizeof c->hash);
	lines_apply(c, c->spec_hh[b]);
	c->mexp = c->mlen ? c->hist[c->mptr & c->hist_mask] : -1;
}

/* Bit 3 done: the next nibble's lines for q = 2 c0 + {0, 1}. */
static void speculate_nibble(p1_ctx *c)
{
	for (unsigned b = 0; b < 2; b++) {
		line_hashes(c, c->hash, c->c0 << 1 | b, c->spec_hh[b]);
		lines_prefetch(c, c->spec_hh[b]);
	}
}

/* Bit 7 done: the next byte's hashes, lines and match entry for both
 * values of its last bit. */
static void speculate_byte(p1_ctx *c)
{
	for (unsigned b = 0; b < 2; b++) {
		const uint32_t ch = (c->c0 << 1 | b) & 0xFF;
		const uint32_t c8 = c->c8 << 8 | c->c4 >> 24;
		const uint32_t c4 = c->c4 << 8 | ch;
		byte_hashes_for(c, c4, c8, word_next(c->word, ch), c->spec_g[b]);
		line_hashes(c, c->spec_g[b], 0, c->spec_hh[b]);
		lines_prefetch(c, c->spec_hh[b]);
		c->spec_mi[b] = match_index(c, c4, c8);
		__builtin_prefetch(c->mtab + c->spec_mi[b], 1);
	}
}

void p1_ctx_update(p1_ctx *c, int bit)
{
#if TAW_X86_SIMD
	if (c->avx2) {
		p1_cm_update_avx2(c, bit);
	} else
#endif
	{
		for (unsigned k = 0; k < P1_NCM; k++) {
			slot_update(&c->arena[c->si[k]], bit, LIMIT_SLOW);
			hm_update(&c->hmap[c->hmi[k]], bit, LIMIT_HMAP);
		}
	}
	p1_ctx_tail(c, bit);
}

void p1_ctx_tail(p1_ctx *c, int bit)
{
	if (c->mbit >= 0)
		slot_update(&c->sm[c->msi], bit, LIMIT_MATCH);
	c->c0 = c->c0 << 1 | (uint32_t)bit;
	c->nib = c->nib << 1 | (uint32_t)bit;
	c->bitpos++;
	if (P1_BOUNDARY_BITS >> c->bitpos & 1)
		p1_ctx_boundary(c, bit);
}

void p1_ctx_boundary(p1_ctx *c, int bit)
{
	switch (c->bitpos) {
	case 3: speculate_nibble(c); break;
	case 4:
		c->nib = 1;
		lines_apply(c, c->spec_hh[bit]);
		break;
	case 7: speculate_byte(c); break;
	case 8: byte_end(c); break;
	default: break;
	}
}

/* ---- profile 11: entropy-adaptive gate helpers ---------------------------
 * Shared by the decoder's inference and the encoder's gate fitting so both
 * use exactly the same integer arithmetic. */

void p11_features(const int16_t st[P1_NCTX], int32_t *mean_out,
    int32_t *mad_out)
{
	int32_t sum = 0;
	for (unsigned i = 0; i < P1_NCTX; i++)
		sum += st[i];
	/* sum is in [-25*2047, 25*2047], so both values fit in int32. */
	int32_t mean = (sum >= 0 ? sum + 12 : sum - 12) / 25;
	int32_t dev = 0;
	for (unsigned i = 0; i < P1_NCTX; i++) {
		int32_t d = st[i] - mean;
		dev += d < 0 ? -d : d;
	}
	*mean_out = mean;
	*mad_out = (dev + 12) / 25;
}

uint32_t p11_gate_p(uint32_t p_mlp, uint32_t p_ctx, int32_t mad,
    const p11_gate *g)
{
	int32_t prod = (int32_t)g->b * mad;
	/* floor(prod / 2^P11_GATE_SHIFT), also for negative prod. */
	int32_t q = prod >= 0 ? prod >> P11_GATE_SHIFT
			      : -((-prod + (1 << P11_GATE_SHIFT) - 1) >>
				  P11_GATE_SHIFT);
	int32_t v = (int32_t)g->a + q;
	uint32_t w = v < 0 ? 0u : v > 256 ? 256u : (uint32_t)v;
	return ((256u - w) * p_mlp + w * p_ctx + 128u) >> 8;
}

taw_status p11_gate_parse(const uint8_t *p, size_t n, p11_gate *g)
{
	if (n < P11_GATE_SIZE)
		return TAW_E_FORMAT;
	int16_t a = (int16_t)taw_rd16(p), b = (int16_t)taw_rd16(p + 2);
	if (a < P11_GATE_A_MIN || a > P11_GATE_A_MAX ||
	    b < P11_GATE_B_MIN || b > P11_GATE_B_MAX)
		return TAW_E_FORMAT;
	*g = (p11_gate){.a = a, .b = b};
	return TAW_OK;
}

void p11_gate_write(uint8_t out[P11_GATE_SIZE], const p11_gate *g)
{
	out[0] = (uint8_t)(uint16_t)g->a;
	out[1] = (uint8_t)((uint16_t)g->a >> 8);
	out[2] = (uint8_t)(uint16_t)g->b;
	out[3] = (uint8_t)((uint16_t)g->b >> 8);
}
