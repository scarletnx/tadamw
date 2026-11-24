/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 (F32_STRICT_MIX): integer tables, F16, ISA parity, model
 * validation, and a golden vector that pins the arithmetic. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "half.h"
#include "header.h"
#include "p1.h"
#include "quant.h"
#include "taw_internal.h"
#include "testdata.h"

/* ---- hand-built models ------------------------------------------------------ */

typedef struct {
	p1_shape s;
	uint8_t quant;
	uint64_t seed;
	float scale; /* weight magnitude */
} model_spec;

static float wgen(uint64_t *r, float scale)
{
	return scale * ((float)(int32_t)(uint32_t)(td_next(r) >> 32) *
	    (1.0f / 2147483648.0f));
}

static void put_h(taw_buf *b, float w)
{
	uint16_t h = taw_f2h(w);
	taw_buf_put_u8(b, (uint8_t)h);
	taw_buf_put_u8(b, (uint8_t)(h >> 8));
}

static void put_row(taw_buf *b, uint8_t quant, uint64_t *r, float scale,
    size_t len)
{
	if (quant == TAW_QUANT_Q8) {
		put_h(b, scale / 127.0f);
		for (size_t i = 0; i < len; i++)
			taw_buf_put_u8(b, (uint8_t)(int8_t)(
			    (int)(td_next(r) % 255) - 127));
	} else if (quant == TAW_QUANT_F16) {
		for (size_t i = 0; i < len; i++)
			put_h(b, wgen(r, scale));
	} else {
		/* Random valid codes, so that the goldens pin the decoding
		 * rules independently of the encoder's quantizer. */
		uint8_t row[128];
		const size_t n = (size_t)taw_quant_row_bytes(quant, len);
		const unsigned nb = (unsigned)(len + 15) / 16;
		for (size_t i = 0; i < n; i++)
			row[i] = (uint8_t)td_next(r);
		const float div = quant == TAW_QUANT_Q4 ? 8.0f :
		    quant == TAW_QUANT_Q2 ? 3.0f :
		    quant == TAW_QUANT_FP8 ? 448.0f :
		    quant == TAW_QUANT_NVFP4 ? 6.0f * 448.0f :
		    quant == TAW_QUANT_IQ8 ? 32768.0f : 127.0f;
		uint16_t h = taw_f2h(scale / div);
		row[0] = (uint8_t)h;
		row[1] = (uint8_t)(h >> 8);
		size_t codes = 2, bits = 8;
		switch (quant) {
		case TAW_QUANT_Q2:
			h = taw_f2h(-scale);
			row[2] = (uint8_t)h;
			row[3] = (uint8_t)(h >> 8);
			codes = 4;
			bits = 2;
			break;
		case TAW_QUANT_IQ2: bits = 2; break;
		case TAW_QUANT_Q4: case TAW_QUANT_IQ4: bits = 4; break;
		case TAW_QUANT_FP8:
			for (size_t i = 2; i < n; i++)
				if ((row[i] & 0x7F) == 0x7F)
					row[i] = 0x38;
			break;
		case TAW_QUANT_NVFP4:
			for (unsigned k = 0; k < nb; k++) {
				row[2 + k] &= 0x7F;
				if (row[2 + k] >= 0x70) /* keep scales moderate */
					row[2 + k] = 0x38;
			}
			codes = 2 + nb;
			bits = 4;
			break;
		default: break;
		}
		const size_t per = 8 / bits, used = len % per;
		if (used)
			row[codes + len / per] &= (uint8_t)((1u << (used * bits)) - 1);
		taw_buf_put(b, row, n);
	}
}

static taw_buf build_model(const model_spec *m)
{
	taw_buf b = {0};
	uint8_t head[P1_MODEL_HEAD];
	p1_head_write(head, &m->s);
	taw_buf_put(&b, head, sizeof head);
	uint64_t r = m->seed;
	const unsigned H = m->s.hidden;
	for (size_t set = 0; H && set < p1_sel_size[m->s.sel1]; set++)
		for (unsigned j = 0; j < H; j++)
			put_row(&b, m->quant, &r, m->scale, P1_F);
	for (size_t set = 0; set < p1_sel_size[m->s.sel2]; set++)
		put_row(&b, m->quant, &r, m->scale, H ? H + 1 : P1_F);
	REQUIRE(!b.oom);
	REQUIRE(b.n == P1_MODEL_HEAD + p1_weights_size(&m->s, m->quant));
	return b;
}

static taw_buf build_gate_model(const model_spec *m, p11_gate gate)
{
	taw_buf b = build_model(m);
	uint8_t tail[P11_GATE_SIZE];
	p11_gate_write(tail, &gate);
	taw_buf_put(&b, tail, sizeof tail);
	REQUIRE(!b.oom);
	return b;
}

static const p1_shape geo = {
	.lines_log2 = 12, .hist_log2 = 16, .match_log2 = 12,
	.sel1 = P1_SEL_BITPOS, .sel2 = P1_SEL_C0, .hidden = 12,
};

/* ---- profile 6 hand-built models -------------------------------------------- */

typedef struct {
	p1_shape s;
	p6_cal cal;
	uint8_t quant;
	uint64_t seed;
	float scale;
} model6_spec;

static taw_buf build_model6(const model6_spec *m)
{
	taw_buf b = {0};
	uint8_t head[P6_MODEL_HEAD];
	p6_head_write(head, &m->s, &m->cal);
	taw_buf_put(&b, head, sizeof head);
	uint64_t r = m->seed;
	const unsigned H = m->s.hidden;
	for (size_t set = 0; H && set < p1_sel_size[m->s.sel1]; set++)
		for (unsigned j = 0; j < H; j++)
			put_row(&b, m->quant, &r, m->scale, P1_F);
	for (size_t set = 0; set < p1_sel_size[m->s.sel2]; set++)
		put_row(&b, m->quant, &r, m->scale, H ? H + 1 : P1_F);
	REQUIRE(!b.oom);
	REQUIRE(b.n == P6_MODEL_HEAD + p1_weights_size(&m->s, m->quant));
	return b;
}

static const p6_cal cal_fixed = {
	.temp = 247, .bias = 19, .w0 = 1, .w1 = 2, .w2 = 3,
};

/* ---- squash / stretch ------------------------------------------------------- */

static void tables(void)
{
	p1_tables_init();
	CHECK(p1_squash(0) == 32768);
	CHECK(p1_squash(-2047) >= 22 && p1_squash(2047) <= 65514);
	CHECK(p1_squash(-5000) == p1_squash(-2047));
	CHECK(p1_squash(5000) == p1_squash(2047));
	for (int t = -2047; t < 2047; t++)
		CHECK(p1_squash(t) <= p1_squash(t + 1));
	for (int p = 0; p < 4095; p++)
		CHECK(p1_stretch_tab[p] <= p1_stretch_tab[p + 1]);
	CHECK(p1_stretch_tab[0] == -2047 && p1_stretch_tab[4095] == 2047);
	/* stretch inverts squash to within one 12-bit step (in the middle;
	 * near the ends one step of p spans more than 64 units of t). */
	for (int t = -1500; t <= 1500; t += 25) {
		int back = p1_stretch_tab[p1_squash(t) >> 4];
		CHECK(back <= t && back >= t - 64);
	}
}

/* ---- F16 ---------------------------------------------------------------------- */

static void halves(void)
{
	for (uint32_t h = 0; h < 65536; h++) {
		bool nan = (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu);
		if (nan)
			continue;
		float f = taw_h2f((uint16_t)h);
		CHECK(taw_f2h(f) == h);
#if defined(__x86_64__)
		if (p1_isa_available(P1_ISA_AVX2)) {
			uint16_t in = (uint16_t)h;
			float out;
			p1_f16_to_f32_f16c(&in, &out, 1);
			CHECK(taw_f32_bits(out) == taw_f32_bits(f));
		}
#endif
	}
	CHECK(taw_f2h(65520.0f) == 0x7C00u); /* rounds to inf */
	CHECK(taw_f2h(65519.0f) == 0x7BFFu);
	CHECK(taw_f2h(1e-8f) == 0x0000u);
	CHECK(taw_f2h(-3.0e-8f) == 0x8001u);
}

/* ---- ISA parity ---------------------------------------------------------------- */

static void parity_one(const model_spec *m)
{
	taw_buf mb = build_model(m);
	p1_net ref;
	p1_isa_force(P1_ISA_SCALAR);
	REQUIRE(p1_net_load(&ref, m->quant, mb.p, mb.n) == TAW_OK);
	for (int isa = P1_ISA_AVX2; isa < P1_ISA_COUNT; isa++) {
		if (!p1_isa_available((p1_isa)isa))
			continue;
		p1_isa_force(isa);
		p1_net net;
		REQUIRE(p1_net_load(&net, m->quant, mb.p, mb.n) == TAW_OK);
		uint64_t r = m->seed ^ 0x55;
		int bad = 0;
		for (int k = 0; k < 20000; k++) {
			alignas(32) float x[P1_XPAD] = {0};
			for (unsigned i = 0; i < P1_NCTX; i++)
				x[i] = (float)((int)(td_next(&r) % 4095) - 2047) /
				    256.0f;
			x[P1_IN_BIAS] = 1.0f;
			uint8_t c0 = (uint8_t)(td_next(&r) | 1);
			p1_sel_ctx sc = {
				.bitpos = (uint8_t)(31 - __builtin_clz(c0)),
				.c0 = c0,
				.c1 = (uint8_t)td_next(&r),
				.msel = (uint8_t)(td_next(&r) & 7),
			};
			bad += p1_net_eval(&ref, x, &sc) != p1_net_eval(&net, x, &sc);
		}
		CHECK(bad == 0);
		p1_net_free(&net);
	}
	p1_isa_force(-1);
	p1_net_free(&ref);
	taw_buf_free(&mb);
}

static void parity(void)
{
	static const uint8_t hs[] = {0, 1, 7, 8, 9, 16, 24, 32};
	for (size_t i = 0; i < sizeof hs; i++) {
		model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = i,
			.scale = 0.5f};
		m.s.hidden = hs[i];
		if (!hs[i])
			m.s.sel1 = P1_SEL_NONE;
		parity_one(&m);
		for (uint8_t q = TAW_QUANT_Q8; q < TAW_QUANT_COUNT; q++) {
			m.quant = q;
			parity_one(&m);
		}
		/* Large weights: every hidden unit saturates. */
		m.quant = TAW_QUANT_F16;
		m.scale = 60000.0f;
		parity_one(&m);
	}
}

/* Whole archives: every ISA encodes the same bytes and decodes them. */
static void archive_parity(void)
{
	enum { N = 60000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 21);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_MIX};
	void *ref = nullptr;
	size_t ref_n = 0;
	for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
		if (!p1_isa_available((p1_isa)isa))
			continue;
		p1_isa_force(isa);
		void *arc, *dec;
		size_t arc_n, dec_n;
		REQUIRE(taw_compress(src, N, &p, &arc, &arc_n) == TAW_OK);
		if (!ref) {
			ref = arc;
			ref_n = arc_n;
		} else {
			CHECK(arc_n == ref_n && memcmp(arc, ref, arc_n) == 0);
			free(arc);
		}
		CHECK(taw_decompress(ref, ref_n, &dec, &dec_n) == TAW_OK);
		CHECK(dec_n == N && memcmp(dec, src, N) == 0);
		free(dec);
	}
	p1_isa_force(-1);
	free(ref);
	free(src);
}

/* The context models alone: scalar and AVX2 produce the same inputs and
 * selector context at every bit, across unit resets. */
static void ctx_parity(void)
{
	if (!p1_isa_available(P1_ISA_AVX2))
		return;
	enum { N = 200000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N / 2, TD_TEXT, 31);
	td_fill(src + N / 2, N / 4, TD_SKEWED, 32);
	td_fill(src + 3 * N / 4, N / 4, TD_RANDOM, 33);
	for (size_t i = 0; i + 300 < N; i += 7919) /* repeats for the match model */
		memcpy(src + i + 150, src + i, 100);
	p1_ctx a, b;
	p1_isa_force(P1_ISA_SCALAR);
	REQUIRE(p1_ctx_init(&a, &geo) == TAW_OK);
	p1_isa_force(P1_ISA_AVX2);
	REQUIRE(p1_ctx_init(&b, &geo) == TAW_OK);
	p1_isa_force(-1);
	CHECK(!a.avx2 && b.avx2);
	long bad = 0;
	for (size_t i = 0; i < N; i++) {
		if (i == N / 3) {
			p1_ctx_reset(&a);
			p1_ctx_reset(&b);
		}
		for (int k = 7; k >= 0; k--) {
			float xa[32], xb[32];
			p1_sel_ctx sa, sb;
			p1_ctx_inputs(&a, xa, &sa);
			p1_ctx_inputs(&b, xb, &sb);
			bad += memcmp(xa, xb, P1_NCTX * sizeof(float)) != 0 ||
			    memcmp(&sa, &sb, sizeof sa) != 0;
			int bit = (src[i] >> k) & 1;
			p1_ctx_update(&a, bit);
			p1_ctx_update(&b, bit);
		}
	}
	CHECK(bad == 0);
	p1_ctx_free(&a);
	p1_ctx_free(&b);
	free(src);
}

/* ---- model validation ------------------------------------------------------------ */

static taw_status load(const taw_buf *mb, uint8_t quant)
{
	taw_pred pr;
	taw_status st = taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_MIX, quant,
	    mb->p, mb->n);
	taw_pred_free(&pr);
	return st;
}

static taw_status load_patched(const taw_buf *mb, uint8_t quant, size_t off,
    uint8_t v)
{
	uint8_t saved = mb->p[off];
	mb->p[off] = v;
	taw_status st = load(mb, quant);
	mb->p[off] = saved;
	return st;
}

static void validation(void)
{
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 9,
		.scale = 1.0f};
	taw_buf mb = build_model(&m);
	CHECK(load(&mb, TAW_QUANT_F16) == TAW_OK);
	CHECK(load(&mb, TAW_QUANT_Q8) == TAW_E_FORMAT);   /* size mismatch */
	CHECK(load(&mb, TAW_QUANT_BF16) == TAW_E_UNSUPPORTED);
	CHECK(load(&mb, TAW_QUANT_F32) == TAW_E_UNSUPPORTED);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 0, P1_LINES_MAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 0, P1_LINES_MIN - 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 1, P1_HIST_MAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 2, P1_MATCH_MAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 3, P1_SEL_COUNT) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 4, P1_SEL_COUNT) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 5, P1_HMAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 6, 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 7, 1) == TAW_E_FORMAT);
	/* A different shape no longer matches the payload size. */
	CHECK(load_patched(&mb, TAW_QUANT_F16, 5, 11) == TAW_E_FORMAT);
	/* Non-finite F16 weights: inf (0x7C00) and NaN (0x7E00). */
	CHECK(load_patched(&mb, TAW_QUANT_F16, P1_MODEL_HEAD + 101, 0x7C) ==
	    TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_F16, mb.n - 1, 0xFE) == TAW_E_FORMAT);
	mb.n--;
	CHECK(load(&mb, TAW_QUANT_F16) == TAW_E_FORMAT);
	taw_buf_free(&mb);

	/* Hidden = 0 needs sel1 = NONE. */
	m.s.hidden = 0;
	m.s.sel1 = P1_SEL_NONE;
	mb = build_model(&m);
	CHECK(load(&mb, TAW_QUANT_F16) == TAW_OK);
	CHECK(load_patched(&mb, TAW_QUANT_F16, 3, P1_SEL_BITPOS) == TAW_E_FORMAT);
	taw_buf_free(&mb);

	/* Q8: -128 and a non-finite scale are refused. */
	m.s = geo;
	m.quant = TAW_QUANT_Q8;
	mb = build_model(&m);
	CHECK(load(&mb, TAW_QUANT_Q8) == TAW_OK);
	CHECK(load_patched(&mb, TAW_QUANT_Q8, P1_MODEL_HEAD + 2 + 3, 0x80) ==
	    TAW_E_FORMAT);
	CHECK(load_patched(&mb, TAW_QUANT_Q8, P1_MODEL_HEAD + 1, 0x7C) ==
	    TAW_E_FORMAT);
	taw_buf_free(&mb);
}

/* ---- profile 6 validation --------------------------------------------------- */

static taw_status load6(const taw_buf *mb, uint8_t quant)
{
	taw_pred pr;
	taw_status st = taw_pred_init(&pr, TAW_PROFILE_CALIBRATED_MIX, quant,
	    mb->p, mb->n);
	taw_pred_free(&pr);
	return st;
}

static taw_status load6_patched(const taw_buf *mb, uint8_t quant, size_t off,
    uint8_t v)
{
	uint8_t saved = mb->p[off];
	mb->p[off] = v;
	taw_status st = load6(mb, quant);
	mb->p[off] = saved;
	return st;
}

static void validation6(void)
{
	model6_spec m = {.s = geo, .cal = cal_fixed, .quant = TAW_QUANT_F16,
		.seed = 9, .scale = 1.0f};
	taw_buf mb = build_model6(&m);
	CHECK(load6(&mb, TAW_QUANT_F16) == TAW_OK);
	CHECK(load6(&mb, TAW_QUANT_Q8) == TAW_E_FORMAT);   /* size mismatch */
	CHECK(load6(&mb, TAW_QUANT_BF16) == TAW_E_UNSUPPORTED);
	CHECK(load(&mb, TAW_QUANT_F16) == TAW_E_FORMAT);   /* 8 extra bytes */
	/* temp, bias and APM weights have narrow valid ranges; reserved must be 0. */
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 8, 0) == TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 9, 3) == TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 11, 4) == TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 12, P6_W_MAX + 1) ==
	    TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 14, P6_W_MAX + 1) ==
	    TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 15, 1) == TAW_E_FORMAT);
	uint8_t w[3] = {mb.p[12], mb.p[13], mb.p[14]};
	mb.p[12] = mb.p[13] = mb.p[14] = 0;
	CHECK(load6(&mb, TAW_QUANT_F16) == TAW_E_FORMAT);
	mb.p[12] = w[0];
	mb.p[13] = w[1];
	mb.p[14] = w[2];
	/* The profile 1 head fields are still validated. */
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 5, P1_HMAX + 1) ==
	    TAW_E_FORMAT);
	CHECK(load6_patched(&mb, TAW_QUANT_F16, 6, 1) == TAW_E_FORMAT);
	taw_buf_free(&mb);

	model_spec m1 = {.s = geo, .quant = TAW_QUANT_F16, .seed = 9,
		.scale = 1.0f};
	taw_buf mb1 = build_model(&m1);
	CHECK(load6(&mb1, TAW_QUANT_F16) == TAW_E_FORMAT);
	taw_buf_free(&mb1);
}

/* The identity calibration must reproduce profile 1 bit for bit: only the
 * model section is 8 bytes longer. */
static void identity_matches_p1(void)
{
	enum { N = 8192 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 77);
	model_spec m1 = {.s = geo, .quant = TAW_QUANT_F16, .seed = 5,
		.scale = 0.4f};
	model6_spec m6 = {.s = geo, .cal = P6_CAL_IDENTITY,
		.quant = TAW_QUANT_F16, .seed = 5, .scale = 0.4f};
	taw_buf b1 = build_model(&m1), b6 = build_model6(&m6);
	taw_pred p1, p6;
	REQUIRE(taw_pred_init(&p1, TAW_PROFILE_F32_STRICT_MIX, m1.quant,
	    b1.p, b1.n) == TAW_OK);
	REQUIRE(taw_pred_init(&p6, TAW_PROFILE_CALIBRATED_MIX, m6.quant,
	    b6.p, b6.n) == TAW_OK);
	taw_buf o1 = {0}, o6 = {0};
	taw_rc_enc e1, e6;
	taw_rc_enc_init(&e1, &o1);
	taw_rc_enc_init(&e6, &o6);
	taw_pred_encode(&p1, &e1, src, N);
	taw_pred_encode(&p6, &e6, src, N);
	taw_rc_enc_flush(&e1);
	taw_rc_enc_flush(&e6);
	CHECK(o1.n == o6.n && memcmp(o1.p, o6.p, o1.n) == 0);
	taw_pred_free(&p1);
	taw_pred_free(&p6);
	taw_buf_free(&o1);
	taw_buf_free(&o6);
	taw_buf_free(&b1);
	taw_buf_free(&b6);
	free(src);
}

/* Model damage inside an archive: the model CRC catches raw flips; a
 * consistent but invalid model (CRCs fixed) is a format error. */
static void archive_model_damage(void)
{
	enum { N = 40000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 8);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_MIX};
	void *arc_v, *dec = nullptr;
	size_t n, dec_n;
	REQUIRE(taw_compress(src, N, &p, &arc_v, &n) == TAW_OK);
	uint8_t *a = arc_v;
	taw_info info;
	REQUIRE(taw_read_info(a, n, &info) == TAW_OK);
	REQUIRE(info.arith_profile == TAW_PROFILE_F32_STRICT_MIX);
	REQUIRE(info.model_size > P1_MODEL_HEAD && info.model_size <= N / 32);

	a[TAW_HEADER_SIZE + P1_MODEL_HEAD] ^= 1;
	CHECK(taw_decompress(a, n, &dec, &dec_n) == TAW_E_CHECKSUM);
	a[TAW_HEADER_SIZE + P1_MODEL_HEAD] ^= 1;

	a[TAW_HEADER_SIZE + 5] = P1_HMAX + 1; /* hidden */
	taw_wr32(a + 24, taw_crc32c(0, a + TAW_HEADER_SIZE,
	    (size_t)info.model_size));
	taw_wr32(a + 28, taw_crc32c(0, a, 28));
	CHECK(taw_decompress(a, n, &dec, &dec_n) == TAW_E_FORMAT);

	free(a);
	free(src);
}

/* ---- golden vector ------------------------------------------------------------------
 * A fixed hand-built model over fixed text. Any change to profile 1's
 * arithmetic (context models, hashing, mixer order, squash, APM) changes
 * these CRCs: that is a format break and needs a new profile id. */

static uint32_t golden_crc(uint8_t quant, uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < N; i += 4096) /* some match-model work */
		memcpy(src + i, src, 64);
	model_spec m = {.s = geo, .quant = quant, .seed = 42, .scale = 0.4f};
	taw_buf mb = build_model(&m);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_MIX, quant, mb.p,
	    mb.n) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, N);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);

	taw_pred_reset(&pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	taw_pred_decode(&pr, &d, dst, N);
	CHECK(memcmp(src, dst, N) == 0);

	uint32_t crc = taw_crc32c(0, out.p, out.n);
	*coded_len = (uint32_t)out.n;
	taw_pred_free(&pr);
	taw_buf_free(&out);
	taw_buf_free(&mb);
	free(src);
	free(dst);
	return crc;
}

static void golden(void)
{
	static const struct {
		uint8_t quant;
		uint32_t len, crc;
	} want[] = {
		{TAW_QUANT_F16, 59885, 0x74309CD4u},
		{TAW_QUANT_Q8, 48112, 0x899F4057u},
		{TAW_QUANT_Q4, 47020, 0xA78FE21Du},
		{TAW_QUANT_Q2, 23304, 0x2D02777Au},
		{TAW_QUANT_FP8, 43808, 0xF5A4AC7Bu},
		{TAW_QUANT_NVFP4, 44641, 0xD47885B8u},
		{TAW_QUANT_IQ8, 62559, 0x2AC391B8u},
		{TAW_QUANT_IQ4, 45922, 0x84FA96FDu},
		{TAW_QUANT_IQ2, 48256, 0xAF5179FAu},
	};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = golden_crc(want[i].quant, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr, "golden quant=%u isa=%d: len %u crc 0x%08X\n",
				    want[i].quant, isa, len, crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
}

/* ---- profile 6 golden vector -------------------------------------------------- */

static uint32_t golden6_crc(uint8_t quant, uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < N; i += 4096) /* some match-model work */
		memcpy(src + i, src, 64);
	model6_spec m = {.s = geo, .cal = cal_fixed, .quant = quant,
		.seed = 42, .scale = 0.4f};
	taw_buf mb = build_model6(&m);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_CALIBRATED_MIX, quant, mb.p,
	    mb.n) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, N);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);

	taw_pred_reset(&pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	taw_pred_decode(&pr, &d, dst, N);
	CHECK(memcmp(src, dst, N) == 0);

	uint32_t crc = taw_crc32c(0, out.p, out.n);
	*coded_len = (uint32_t)out.n;
	taw_pred_free(&pr);
	taw_buf_free(&out);
	taw_buf_free(&mb);
	free(src);
	free(dst);
	return crc;
}

static void golden6(void)
{
	static const struct {
		uint8_t quant;
		uint32_t len, crc;
	} want[] = {
		{TAW_QUANT_F16, 43914, 0xD593C4D1u},
		{TAW_QUANT_Q8, 39189, 0xEBF4AF55u},
	};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = golden6_crc(want[i].quant, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr,
				    "golden6 quant=%u isa=%d: len %u crc 0x%08X\n",
				    want[i].quant, isa, len, crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
}

/* Predictor values are profile IDs, with 0/1/2
 * grandfathered. Value 3 is void; 6/11/14 are allocated. */
static void gate_arithmetic(void)
{
	int16_t st[P1_NCTX];
	int32_t mean, mad;
	for (unsigned i = 0; i < P1_NCTX; i++)
		st[i] = 0;
	p11_features(st, &mean, &mad);
	CHECK(mean == 0 && mad == 0);
	for (unsigned i = 0; i < P1_NCTX - 1; i++)
		st[i] = i & 1 ? 100 : -100;
	st[P1_NCTX - 1] = 0;
	p11_features(st, &mean, &mad);
	CHECK(mean == 0 && mad == 96);
	for (unsigned i = 0; i < P1_NCTX; i++)
		st[i] = 1000;
	p11_features(st, &mean, &mad);
	CHECK(mean == 1000 && mad == 0);

	const p11_gate zero = {0, 0}, full = {256, 0}, neg = {-1, 1};
	CHECK(p11_gate_p(12345, 54321, 7, &zero) == 12345);
	CHECK(p11_gate_p(12345, 54321, 7, &full) == 54321);
	CHECK(p11_gate_p(12345, 54321, 0, &neg) == 12345);
	CHECK(p11_gate_p(12345, 54321, 1, &neg) == 12345); /* floor(1/64)=0 */
	CHECK(p11_gate_p(0, 25600, 256, &neg) == 300);     /* w = 3 */
}

static void gate_validation(void)
{
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 7,
		.scale = 1.0f};
	taw_buf mb = build_gate_model(&m, (p11_gate){208, -56});
	p11_gate g;
	CHECK(p11_gate_parse(mb.p + mb.n - P11_GATE_SIZE, P11_GATE_SIZE, &g) ==
	    TAW_OK);
	CHECK(g.a == 208 && g.b == -56);
	CHECK(p11_gate_parse(mb.p + mb.n - P11_GATE_SIZE, 3, &g) ==
	    TAW_E_FORMAT);

	taw_pred pr;
	CHECK(taw_pred_supported(TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_F16));
	CHECK(!taw_pred_supported(TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_BF16));
	CHECK(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_OK);
	taw_pred_free(&pr);
	CHECK(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_Q8,
	    mb.p, mb.n) == TAW_E_FORMAT); /* the payload is F16 */

	/* Out-of-range gate parameters and a short model are refused. */
	mb.p[mb.n - 4] = 0;
	mb.p[mb.n - 3] = 0x7F; /* a = 32512 */
	CHECK(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_E_FORMAT);
	mb.p[mb.n - 4] = (uint8_t)(uint16_t)208;
	mb.p[mb.n - 3] = 0;
	mb.p[mb.n - 2] = 0;
	mb.p[mb.n - 1] = 0x7F; /* b = 32512 */
	CHECK(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_E_FORMAT);
	mb.p[mb.n - 2] = (uint8_t)(uint16_t)-56;
	mb.p[mb.n - 1] = 0xFF;
	mb.n--;
	CHECK(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_E_FORMAT);
	taw_pred_free(&pr);
	taw_buf_free(&mb);
}

/* A zero gate must be bit-identical to profile 1: the new profile is a
 * strict extension, not a change to the existing arithmetic. */
static void gate_identity(void)
{
	enum { N = 8192 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 77);
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 42,
		.scale = 0.4f};
	taw_buf m1 = build_model(&m);
	taw_buf m11 = build_gate_model(&m, (p11_gate){0, 0});
	taw_pred a, b;
	REQUIRE(taw_pred_init(&a, TAW_PROFILE_F32_STRICT_MIX, m.quant,
	    m1.p, m1.n) == TAW_OK);
	REQUIRE(taw_pred_init(&b, TAW_PROFILE_ENTROPY_GATE, m.quant,
	    m11.p, m11.n) == TAW_OK);
	taw_buf o1 = {0}, o11 = {0};
	taw_rc_enc e1, e11;
	taw_rc_enc_init(&e1, &o1);
	taw_rc_enc_init(&e11, &o11);
	taw_pred_encode(&a, &e1, src, N);
	taw_pred_encode(&b, &e11, src, N);
	taw_rc_enc_flush(&e1);
	taw_rc_enc_flush(&e11);
	CHECK(!o1.oom && !o11.oom && o1.n == o11.n &&
	    memcmp(o1.p, o11.p, o1.n) == 0);
	taw_pred_free(&a);
	taw_pred_free(&b);
	taw_buf_free(&m1);
	taw_buf_free(&m11);
	taw_buf_free(&o1);
	taw_buf_free(&o11);
	free(src);
}

/* The encoder's gate fitting, header and decoder all agree end to end. */
static void gate_archive(void)
{
	enum { N = 50000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 77);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_GATE};
	void *arc;
	size_t an;
	REQUIRE(taw_compress(src, N, &p, &arc, &an) == TAW_OK);
	taw_info info;
	REQUIRE(taw_read_info(arc, an, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_ENTROPY_GATE);
	CHECK(info.model_size > P1_MODEL_HEAD + P11_GATE_SIZE);
	void *dst;
	size_t dn;
	CHECK(taw_decompress(arc, an, &dst, &dn) == TAW_OK);
	CHECK(dn == N && memcmp(dst, src, N) == 0);
	free(dst);
	free(arc);
	free(src);
}

/* A fixed gated model over fixed text, pinned for every ISA. */
static uint32_t gate_golden_crc(uint8_t quant, p11_gate gate,
    uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < N; i += 4096)
		memcpy(src + i, src, 64);
	model_spec m = {.s = geo, .quant = quant, .seed = 42, .scale = 0.4f};
	taw_buf mb = build_gate_model(&m, gate);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_ENTROPY_GATE, quant, mb.p,
	    mb.n) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, N);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);
	taw_pred_reset(&pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	taw_pred_decode(&pr, &d, dst, N);
	CHECK(memcmp(src, dst, N) == 0);
	uint32_t crc = taw_crc32c(0, out.p, out.n);
	*coded_len = (uint32_t)out.n;
	taw_pred_free(&pr);
	taw_buf_free(&out);
	taw_buf_free(&mb);
	free(src);
	free(dst);
	return crc;
}

static void gate_golden(void)
{
	static const struct {
		uint8_t quant;
		uint32_t len, crc;
	} want[] = {
		{TAW_QUANT_F16, 59403, 0xE70A25C1u},
		{TAW_QUANT_Q8, 47616, 0x4310D2DBu},
	};
	const p11_gate gate = {208, -56};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = gate_golden_crc(want[i].quant, gate, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr,
				    "gate golden quant=%u isa=%d: len %u crc 0x%08X\n",
				    want[i].quant, isa, len, crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
}

/* Match-active golden: a short repeating pattern keeps the match model
 * (input 24) live on nearly every bit, so this pins the st[24] path for
 * both the profile-1 and the gated profile-11 arithmetic. */
static uint32_t match_golden_crc(uint8_t profile, const p11_gate *gate,
    uint8_t quant, uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	static const uint8_t pat[] = "the quick brown fox jumps over 13 dogs!";
	const size_t plen = sizeof pat - 1;
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	for (size_t i = 0; i < N; i++)
		src[i] = pat[i % plen];
	model_spec m = {.s = geo, .quant = quant, .seed = 42, .scale = 0.4f};
	taw_buf mb = gate ? build_gate_model(&m, *gate) : build_model(&m);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, profile, quant, mb.p, mb.n) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, N);
	taw_rc_enc_flush(&e);
	CHECK(pr.p1->ctx.mlen > 0); /* the match was really active */
	taw_pred_reset(&pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	taw_pred_decode(&pr, &d, dst, N);
	CHECK(memcmp(src, dst, N) == 0);
	uint32_t crc = taw_crc32c(0, out.p, out.n);
	*coded_len = (uint32_t)out.n;
	taw_pred_free(&pr);
	taw_buf_free(&out);
	taw_buf_free(&mb);
	free(src);
	free(dst);
	return crc;
}

static void match_golden(void)
{
	static const p11_gate gate = {208, -56};
	static const struct {
		uint8_t profile;
		const p11_gate *gate; /* nullptr = profile 1 */
		uint8_t quant;
		uint32_t len, crc;
	} want[] = {
		{TAW_PROFILE_F32_STRICT_MIX, nullptr, TAW_QUANT_F16,
			51467, 0x57FA86EBu},
		{TAW_PROFILE_F32_STRICT_MIX, nullptr, TAW_QUANT_Q8,
			45171, 0x64204868u},
		{TAW_PROFILE_ENTROPY_GATE, &gate, TAW_QUANT_F16,
			41249, 0x7A171FBBu},
		{TAW_PROFILE_ENTROPY_GATE, &gate, TAW_QUANT_Q8,
			36314, 0xBD4902A5u},
	};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = match_golden_crc(want[i].profile,
			    want[i].gate, want[i].quant, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr,
				    "match golden p=%u quant=%u isa=%d: "
				    "len %u crc 0x%08X\n",
				    want[i].profile, want[i].quant, isa, len, crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
}

static void predictor_slots(void)
{
	_Static_assert(TAW_PREDICTOR_CALIBRATED == 6, "predictor slot 6");
	static const unsigned ok[] = {0, 1, 2, 6, 11, 14};
	for (size_t i = 0; i < sizeof ok / sizeof *ok; i++) {
		taw_params p = {.predictor = (taw_predictor)ok[i]}, r;
		CHECK(taw_params_resolve(&p, &r) == TAW_OK);
		CHECK(r.predictor == (taw_predictor)ok[i]);
	}
	static const unsigned bad[] = {3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 255};
	for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
		taw_params p = {.predictor = (taw_predictor)bad[i]}, r;
		CHECK(taw_params_resolve(&p, &r) == TAW_E_ARG);
	}
	/* In the coalesced build the gate predictor is implemented. */
	taw_params g = {.predictor = TAW_PREDICTOR_GATE}, r;
	CHECK(taw_params_resolve(&g, &r) == TAW_OK);
	void *arc;
	size_t an;
	uint8_t src[4096];
	td_fill(src, sizeof src, TD_TEXT, 5);
	REQUIRE(taw_compress(src, sizeof src, &g, &arc, &an) == TAW_OK);
	taw_info gi;
	REQUIRE(taw_read_info(arc, an, &gi) == TAW_OK);
	CHECK(gi.arith_profile == TAW_PROFILE_ENTROPY_GATE);
	free(arc);
}

static void cal_predictor_roundtrip(void)
{
	enum { N = 32768 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 77);
	const taw_params p = {.level = 1,
		.predictor = TAW_PREDICTOR_CALIBRATED};
	void *arc = nullptr, *dec = nullptr;
	size_t an = 0, dn = 0;
	REQUIRE(taw_compress(src, N, &p, &arc, &an) == TAW_OK);
	taw_info info;
	REQUIRE(taw_read_info(arc, an, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_CALIBRATED_MIX);
	REQUIRE(taw_decompress(arc, an, &dec, &dn) == TAW_OK);
	CHECK(dn == N && memcmp(dec, src, N) == 0);
	free(dec);
	free(arc);
	free(src);
}

int main(void)
{
	predictor_slots();
	cal_predictor_roundtrip();
	tables();
	halves();
	parity();
	ctx_parity();
	validation();
	validation6();
	identity_matches_p1();
	golden();
	golden6();
	gate_arithmetic();
	gate_validation();
	gate_identity();
	gate_archive();
	gate_golden();
	match_golden();
	archive_parity();
	archive_model_damage();
	return CHECK_DONE();
}
