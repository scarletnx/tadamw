/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 14 (HNET_BOUNDARY): boundary arithmetic, the wider weight rows,
 * loader validation and golden vectors pinning the exact probabilities.
 * Profile 1's own goldens live in test_p1.c and must stay unchanged. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "half.h"
#include "header.h"
#include "p1.h"
#include "phnet.h"
#include "quant.h"
#include "taw_internal.h"
#include "testdata.h"

static const p1_shape geo = {
	.lines_log2 = 12, .hist_log2 = 16, .match_log2 = 12,
	.sel1 = P1_SEL_BITPOS, .sel2 = P1_SEL_C0, .hidden = 12,
};

/* ---- hand-built models with ROW WIDTH 27 ---------------------------------- */

typedef struct {
	p1_shape s;
	uint8_t quant;
	uint64_t seed;
	float scale;
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
	if (quant == TAW_QUANT_F16) {
		for (size_t i = 0; i < len; i++)
			put_h(b, wgen(r, scale));
		return;
	}
	REQUIRE(quant == TAW_QUANT_Q8);
	put_h(b, scale / 127.0f);
	for (size_t i = 0; i < len; i++)
		taw_buf_put_u8(b, (uint8_t)(int8_t)(
		    (int)(td_next(r) % 255) - 127));
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
			put_row(&b, m->quant, &r, m->scale, P1_F_HNET);
	for (size_t set = 0; set < p1_sel_size[m->s.sel2]; set++)
		put_row(&b, m->quant, &r, m->scale, H ? H + 1 : P1_F_HNET);
	REQUIRE(!b.oom);
	REQUIRE(b.n == P1_MODEL_HEAD +
	    p1_weights_size_ex(&m->s, m->quant, P1_F_HNET));
	return b;
}

static void rows_and_loader(void)
{
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 3,
		.scale = 1.0f};
	taw_buf b = build_model(&m);
	/* 8 header + (8 sets * 12 hidden * 27 + 256 * 13) f16 weights. */
	CHECK(b.n == 8 + 2 * (8u * 12u * 27u + 256u * 13u));
	CHECK(p1_weights_size_ex(&geo, TAW_QUANT_F16, P1_F) <
	    p1_weights_size_ex(&geo, TAW_QUANT_F16, P1_F_HNET));

	taw_pred pr;
	CHECK(taw_pred_init(&pr, TAW_PROFILE_HNET_BOUNDARY, TAW_QUANT_F16,
	    b.p, b.n) == TAW_OK);
	if (pr.hn) {
		CHECK(pr.hn->net.f == P1_F_HNET);
		/* The width is profile-specific; the 26-wide model must not load. */
		taw_buf b26 = {0};
		uint8_t head[P1_MODEL_HEAD];
		p1_head_write(head, &geo);
		taw_buf_put(&b26, head, sizeof head);
		uint64_t r = 3;
		const unsigned H = geo.hidden;
		for (size_t set = 0; set < p1_sel_size[geo.sel1]; set++)
			for (unsigned j = 0; j < H; j++) {
				float w[P1_F];
				for (unsigned i = 0; i < P1_F; i++)
					w[i] = wgen(&r, 1.0f);
				uint8_t row[2 * P1_F];
				taw_quant_row_encode(TAW_QUANT_F16, w, P1_F,
				    row);
				taw_buf_put(&b26, row, sizeof row);
			}
		for (size_t set = 0; set < p1_sel_size[geo.sel2]; set++) {
			float w[H + 1];
			for (unsigned i = 0; i < H + 1; i++)
				w[i] = wgen(&r, 1.0f);
			uint8_t row[2 * (H + 1)];
			taw_quant_row_encode(TAW_QUANT_F16, w, H + 1, row);
			taw_buf_put(&b26, row, sizeof row);
		}
		CHECK(b26.n == 8 + 2 * (8u * 12u * 26u + 256u * 13u));
		taw_pred q;
		CHECK(taw_pred_init(&q, TAW_PROFILE_HNET_BOUNDARY,
		    TAW_QUANT_F16, b26.p, b26.n) == TAW_E_FORMAT);
		CHECK(taw_pred_init(&q, TAW_PROFILE_F32_STRICT_MIX,
		    TAW_QUANT_F16, b.p, b.n) == TAW_E_FORMAT);
		taw_pred_free(&q);
		taw_buf_free(&b26);
	}
	taw_pred_free(&pr);

	/* Bad head fields and quant formats are refused like profile 1. */
	CHECK(taw_pred_init(&pr, TAW_PROFILE_HNET_BOUNDARY, TAW_QUANT_BF16,
	    b.p, b.n) == TAW_E_UNSUPPORTED);
	uint8_t saved = b.p[5];
	b.p[5] = P1_HMAX + 1;
	CHECK(taw_pred_init(&pr, TAW_PROFILE_HNET_BOUNDARY, TAW_QUANT_F16,
	    b.p, b.n) == TAW_E_FORMAT);
	b.p[5] = saved;
	b.n--;
	CHECK(taw_pred_init(&pr, TAW_PROFILE_HNET_BOUNDARY, TAW_QUANT_F16,
	    b.p, b.n) == TAW_E_FORMAT);
	taw_buf_free(&b);

	/* Central registry: predictor values are profile IDs. Value 3 is void;
	 * 6/11/14 are implemented here. All six names must resolve. */
	uint8_t tiny[64] = {0};
	void *arc = nullptr;
	size_t an = 0;
	taw_params bad = {.predictor = (taw_predictor)3};
	CHECK(taw_compress(tiny, sizeof tiny, &bad, &arc, &an) == TAW_E_ARG);
	CHECK(arc == nullptr);
	taw_params p6 = {.predictor = TAW_PREDICTOR_CALIBRATED};
	taw_params resolved = {0};
	CHECK(taw_params_resolve(&p6, &resolved) == TAW_OK);
	CHECK(resolved.predictor == TAW_PREDICTOR_CALIBRATED);
	/* Both are implemented in the coalesced build; a tiny input only
	 * fails because no model fits the n/32 budget. */
	CHECK(taw_compress(tiny, sizeof tiny, &p6, &arc, &an) ==
	    TAW_E_ARG);
	taw_params p11 = {.predictor = TAW_PREDICTOR_GATE};
	CHECK(taw_compress(tiny, sizeof tiny, &p11, &arc, &an) ==
	    TAW_E_ARG);
	taw_params p14 = {.predictor = TAW_PREDICTOR_HNET};
	CHECK(taw_params_resolve(&p14, &resolved) == TAW_OK);
	CHECK(resolved.predictor == 14);
}

/* ---- boundary signal ------------------------------------------------------- */

/* The encoder's feature pass calls exactly this function on the last bit of
 * every byte; pinning its outputs pins the feature the fitter trains on. */
static void boundary(void)
{
	enum { N = 4096 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 777);
	for (size_t i = 0; i + 200 < N; i += 1024)
		memcpy(src + i + 100, src + i, 64);
	struct taw_hnet *h = aligned_alloc(_Alignof(struct taw_hnet),
	    sizeof *h);
	REQUIRE(h);
	/* The boundary arithmetic does not depend on the weights, but the
	 * context geometry does, so use a real (small) model. */
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 1,
		.scale = 0.01f};
	taw_buf mb = build_model(&m);
	REQUIRE(phnet_init(h, TAW_QUANT_F16, mb.p, mb.n) == TAW_OK);

	uint32_t sum = 0, sum2 = 0;
	int bmin = PHNET_BMAX + 1, bmax = -1, nzero = 0;
	for (size_t i = 0; i < N; i++) {
		for (int k = 7; k >= 0; k--) {
			int bit = (src[i] >> k) & 1;
			p1_sel_ctx sc;
			alignas(32) float x[P1_XPAD];
			p1_ctx_inputs(&h->ctx, x, &sc);
			if (k == 0) { /* last bit of this byte */
				phnet_boundary(x, h->prev, &h->b);
				CHECK(h->b >= 0 && h->b <= PHNET_BMAX);
				sum += (uint32_t)h->b;
				sum2 += (uint32_t)h->b * (uint32_t)h->b;
				nzero += h->b == 0;
				if (h->b < bmin)
					bmin = h->b;
				if (h->b > bmax)
					bmax = h->b;
			}
			p1_ctx_update(&h->ctx, bit);
		}
	}
	CHECK(sum == 8352596u);
	CHECK(sum2 == 1113733144u);
	CHECK(bmin == 0 && bmax == 11517);
	CHECK(nzero == 9);

	/* x[26] is the signal, scaled exactly; padding is zero. */
	{
		alignas(32) float xv[P1_XPAD];
		p1_sel_ctx sc;
		phnet_inputs(h, xv, &sc);
		CHECK(xv[P1_IN_BIAS] == 1.0f);
		CHECK(xv[PHNET_IN_BOUNDARY] == (float)h->b * (1.0f / 256.0f));
		for (unsigned i = P1_F_HNET; i < P1_XPAD; i++)
			CHECK(xv[i] == 0.0f);
	}
	phnet_free(h);
	free(h);
	taw_buf_free(&mb);
	free(src);
}

/* The fitter's feature pass (fit.c) and the decoder (phnet_inputs) must
 * produce the same b for every byte, including after a coding-unit reset. */
static void fit_decode_parity(void)
{
	enum { N = 2048, RESET = 1024 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_SKEWED, 91);
	int32_t *bf = malloc(N / 8 * sizeof *bf), *bd = malloc(N / 8 * sizeof *bd);
	REQUIRE(bf && bd);
	model_spec m = {.s = geo, .quant = TAW_QUANT_F16, .seed = 5,
		.scale = 0.01f};
	taw_buf mb = build_model(&m);

	/* Feature pass: a bare p1_ctx, as collect() uses. */
	p1_ctx c;
	REQUIRE(p1_ctx_init(&c, &geo) == TAW_OK);
	int16_t prev[PHNET_PICK_N] = {0};
	for (size_t i = 0; i < N; i++) {
		if (i == RESET) {
			p1_ctx_reset(&c);
			memset(prev, 0, sizeof prev);
		}
		for (int k = 7; k >= 0; k--) {
			alignas(32) float x[P1_XPAD];
			p1_sel_ctx sc;
			p1_ctx_inputs(&c, x, &sc);
			if (k == 0)
				phnet_boundary(x, prev, &bf[i / 8]);
			p1_ctx_update(&c, (src[i] >> k) & 1);
		}
	}
	p1_ctx_free(&c);

	/* Decoder pass: the profile 14 per-bit loop's input preparation. */
	struct taw_hnet *h = aligned_alloc(_Alignof(struct taw_hnet),
	    sizeof *h);
	REQUIRE(h);
	REQUIRE(phnet_init(h, TAW_QUANT_F16, mb.p, mb.n) == TAW_OK);
	for (size_t i = 0; i < N; i++) {
		if (i == RESET)
			phnet_reset(h);
		for (int k = 7; k >= 0; k--) {
			alignas(32) float x[P1_XPAD];
			p1_sel_ctx sc;
			phnet_inputs(h, x, &sc);
			if (k == 0)
				bd[i / 8] = h->b;
			p1_ctx_update(&h->ctx, (src[i] >> k) & 1);
		}
	}
	for (size_t i = 0; i < N / 8; i++)
		CHECK(bf[i] == bd[i]);
	phnet_free(h);
	free(h);
	taw_buf_free(&mb);
	free(bf);
	free(bd);
	free(src);
}

/* ---- golden vectors --------------------------------------------------------- */

static uint32_t golden_crc(uint8_t quant, uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < N; i += 4096)
		memcpy(src + i, src, 64);
	model_spec m = {.s = geo, .quant = quant, .seed = 42, .scale = 0.4f};
	taw_buf mb = build_model(&m);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_HNET_BOUNDARY, quant, mb.p,
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
		{TAW_QUANT_F16, 57761, 0xEE2926BDu},
		{TAW_QUANT_Q8, 48573, 0x482211FBu},
	};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = golden_crc(want[i].quant, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr,
				    "hnet golden q=%u isa=%d: len=%u crc=%08X "
				    "(want %u %08X)\n",
				    want[i].quant, isa, len, crc, want[i].len,
				    want[i].crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
}

/* Whole archives: every ISA must produce the same archive and decode it. */
static void archive_parity(void)
{
	enum { N = 60000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 21);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_HNET};
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

int main(void)
{
	rows_and_loader();
	boundary();
	fit_decode_parity();
	golden();
	archive_parity();
	return CHECK_DONE();
}
