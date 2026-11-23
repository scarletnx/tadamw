/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 1 (F32_STRICT_MIX): integer tables, F16, ISA parity, model
 * validation, and a golden vector that pins the arithmetic. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "half.h"
#include "header.h"
#include "old_p1_archive.h"
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

static const p1_shape geo = {
	.lines_log2 = 12, .hist_log2 = 16, .match_log2 = 12,
	.sel1 = P1_SEL_BITPOS, .sel2 = P1_SEL_C0, .hidden = 12,
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
	REQUIRE(p1_net_load(&ref, m->s.chunk, m->quant, mb.p, mb.n) == TAW_OK);
	for (int isa = P1_ISA_AVX2; isa < P1_ISA_COUNT; isa++) {
		if (!p1_isa_available((p1_isa)isa))
			continue;
		p1_isa_force(isa);
		p1_net net;
		REQUIRE(p1_net_load(&net, m->s.chunk, m->quant, mb.p, mb.n) == TAW_OK);
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

/* An archive from before profile 2 decodes to the same bytes on every
 * ISA: the profile 1 format did not move. */
static void old_archive(void)
{
	uint8_t want[OLD_P1_RAW_SIZE];
	td_fill(want, sizeof want, TD_TEXT, 5);
	taw_info info;
	REQUIRE(taw_read_info(old_p1_archive, sizeof old_p1_archive, &info) ==
	    TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_F32_STRICT_MIX);
	for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
		if (!p1_isa_available((p1_isa)isa))
			continue;
		p1_isa_force(isa);
		void *dec = nullptr;
		size_t dec_n = 0;
		CHECK(taw_decompress(old_p1_archive, sizeof old_p1_archive, &dec,
		    &dec_n) == TAW_OK);
		CHECK(dec_n == sizeof want && memcmp(dec, want, dec_n) == 0);
		free(dec);
	}
	p1_isa_force(-1);
}

int main(void)
{
	tables();
	halves();
	parity();
	ctx_parity();
	validation();
	golden();
	archive_parity();
	archive_model_damage();
	old_archive();
	return CHECK_DONE();
}
