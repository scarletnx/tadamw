/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 2 (F32_STRICT_CHUNK): the chunk stage, lanes, ISA parity, model
 * validation, archives, and a golden vector that pins the arithmetic. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "half.h"
#include "header.h"
#include "p1.h"
#include "taw_internal.h"
#include "testdata.h"

/* The golden vector (see golden() below). */
enum : uint32_t {
	GOLDEN_LEN_CHUNK = 63203, GOLDEN_CRC_CHUNK = 0x742617E4u,
	GOLDEN_LEN_PLAIN = 62897, GOLDEN_CRC_PLAIN = 0x80709101u,
};

/* ---- inputs ----------------------------------------------------------------- */

/* Lines of fields, as a table has them: the same column repeats its kind
 * of value from line to line, which only the line contexts can see. */
static void fill_table(uint8_t *p, size_t n, uint64_t seed)
{
	static const char *const names[] = {"alpha", "beta", "gamma", "delta",
		"epsilon", "zeta", "eta", "theta"};
	static const char *const units[] = {"kg", "m", "s", "mol"};
	uint64_t s = seed;
	size_t i = 0, row = 0;
	unsigned prev = 0; /* the unit follows the name of the line before */
	while (i < n) {
		char line[96];
		const unsigned a = (unsigned)(td_next(&s) % 8);
		const int len = snprintf(line, sizeof line, "%s %u %s %u.%02u\n",
		    names[a], (unsigned)(1000 + row % 97), units[prev % 4],
		    (unsigned)(td_next(&s) % 50), (unsigned)(row * 7 % 100));
		prev = a;
		for (int k = 0; k < len && i < n; k++)
			p[i++] = (uint8_t)line[k];
		row++;
	}
}

/* ---- hand-built models ------------------------------------------------------ */

/* Letters, digits, blanks and line ends each join their own kind, every
 * other byte is a chunk of its own, and a line starts after a line end. */
static void table_route(p2_route *r)
{
	*r = (p2_route){0};
	for (unsigned b = 0; b < 256; b++)
		r->cls[b] = (b | 32) - 'a' < 26u ? 1 : b - '0' < 10u ? 2 :
		    b == ' ' ? 3 : b == '\n' ? 4 : 0;
	r->row[0] = 0xFFFF;
	for (unsigned a = 1; a <= 4; a++)
		r->row[a] = (uint16_t)(0xFFFF & ~(1u << a));
	r->row2[4] = r->row[4];
}

static const p1_shape geo = {
	.lines_log2 = 12, .hist_log2 = 16, .match_log2 = 12,
	.sel1 = P1_SEL_BITPOS, .sel2 = P1_SEL_C0, .hidden = 12,
	.chunk = true,
	.lane = {P2_CX_O2, P2_CX_O3, P2_CX_O4, P2_CX_O6, P2_CX_WORD,
		P2_CX_S1278, P2_CX_PREV, P2_CX_COL, P2_CX_SKIP, P2_CX_ABOVE},
};

static void put_h(taw_buf *b, float w)
{
	uint16_t h = taw_f2h(w);
	taw_buf_put_u8(b, (uint8_t)h);
	taw_buf_put_u8(b, (uint8_t)(h >> 8));
}

/* A profile 2 model section: s's head, lanes and route, F16 weights from
 * the seed. */
static taw_buf build_model(const p1_shape *s, uint64_t seed, float scale)
{
	taw_buf b = {0};
	uint8_t head[P2_HEAD_MAX];
	p2_head_write(head, s);
	taw_buf_put(&b, head, p1_head_size(s));
	uint64_t r = seed;
	const unsigned H = s->hidden;
	const size_t rows1 = H ? (size_t)p1_sel_size[s->sel1] * H : 0;
	const size_t n = rows1 * P1_F +
	    (size_t)p1_sel_size[s->sel2] * (H ? H + 1u : P1_F);
	for (size_t i = 0; i < n; i++)
		put_h(&b, scale * ((float)(int32_t)(uint32_t)(td_next(&r) >> 32) *
		    (1.0f / 2147483648.0f)));
	REQUIRE(!b.oom);
	REQUIRE(b.n == p1_head_size(s) + p1_weights_size(s, TAW_QUANT_F16));
	return b;
}

/* ---- the chunk stage by hand ------------------------------------------------- */

/* "ab cd\nab xy\n" under table_route: chunks, places in the line and the
 * chunk above each one. */
static void chunk_stage(void)
{
	p2_route r;
	table_route(&r);
	static const char text[] = "ab cd\nab xy\n";
	/* after each byte: place in the line, and whether the chunk above
	 * (same place, line before) is known (0 = nothing there) */
	static const uint8_t want_idx[] = {0, 0, 1, 2, 2, 3, 0, 0, 1, 2, 2, 3};
	p2_chunk k = {0};
	uint32_t line[2][P2_LINE_MAX] = {0};
	uint32_t prev = 0, first_ab = 0, first_cd = 0;
	for (size_t i = 0; i + 1 < sizeof text; i++) {
		p2_chunk nx;
		const uint32_t ch = (uint8_t)text[i];
		p2_chunk_next(&r, &k, line, prev, ch, &nx);
		p2_chunk_commit(&k, line, &nx);
		prev = ch;
		CHECK(k.idx == want_idx[i]);
		if (i == 1)
			first_ab = k.ck[0]; /* "ab" */
		if (i == 4)
			first_cd = k.ck[0]; /* "cd" */
		if (i < 6)
			CHECK(k.ab == 0); /* no line before the first */
		if (i == 2)
			CHECK(k.ck[1] == first_ab && k.cut);
		if (i == 3)
			CHECK(k.cut && k.ck[2] == first_ab);
		/* second line: "ab" sits under "ab", "xy" under "cd" */
		if (i == 6 || i == 7)
			CHECK(k.ab == first_ab);
		if (i == 7)
			CHECK(k.ck[0] == first_ab);
		if (i == 9 || i == 10)
			CHECK(k.ab == first_cd && k.ck[0] != first_cd);
	}
	/* no boundary without the row bit, and a line needs a chunk */
	CHECK(p2_boundary(&r, 'a', 'b') == 0);
	CHECK(p2_boundary(&r, 'a', ' ') == 1);
	CHECK(p2_boundary(&r, '\n', 'a') == 3);
	r.row[4] = 0;
	CHECK(p2_boundary(&r, '\n', 'a') == 0);
}

/* ---- ISA parity of the context models ----------------------------------------- */

static void ctx_parity_shape(const p1_shape *s)
{
	if (!p1_isa_available(P1_ISA_AVX2))
		return;
	enum { N = 120000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	fill_table(src, N / 2, 5);
	td_fill(src + N / 2, N / 4, TD_TEXT, 31);
	td_fill(src + 3 * N / 4, N / 4, TD_RANDOM, 33);
	p1_ctx a, b;
	p1_isa_force(P1_ISA_SCALAR);
	REQUIRE(p1_ctx_init(&a, s) == TAW_OK);
	p1_isa_force(P1_ISA_AVX2);
	REQUIRE(p1_ctx_init(&b, s) == TAW_OK);
	p1_isa_force(-1);
	CHECK(!a.avx2 && b.avx2);
	CHECK(a.ncm == (s->all ? P1_NCM_ALL : P1_NCM) && b.ncm == a.ncm);
	const size_t nx = 2 * a.ncm + 1;
	long bad = 0;
	for (size_t i = 0; i < N; i++) {
		if (i == N / 3) {
			p1_ctx_reset(&a);
			p1_ctx_reset(&b);
		}
		for (int k = 7; k >= 0; k--) {
			float xa[P1_XALL], xb[P1_XALL];
			p1_sel_ctx sa, sb;
			p1_ctx_inputs(&a, xa, &sa);
			p1_ctx_inputs(&b, xb, &sb);
			bad += memcmp(xa, xb, nx * sizeof(float)) != 0 ||
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

/* The decoder's ten lanes, the encoder's feature pass with every candidate
 * (16 models), and lanes without a chunk context (no chunk stage). */
static void ctx_parity(void)
{
	p1_shape s = geo;
	table_route(&s.route);
	ctx_parity_shape(&s);
	s.all = true;
	ctx_parity_shape(&s);
	s.all = false;
	for (unsigned k = 0; k < P1_NHASH; k++)
		s.lane[k] = (uint8_t)k;
	ctx_parity_shape(&s);
}

/* A model's inputs depend only on its own context: lane k of a decoder
 * shape reads what the candidate lane[k] reads in the feature pass. The
 * encoder ranks and fits on that. */
static void lanes_match_candidates(void)
{
	enum { N = 60000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	fill_table(src, N, 9);
	p1_shape pick = geo, all = geo;
	table_route(&pick.route);
	all.route = pick.route;
	all.all = true;
	p1_ctx a, b;
	REQUIRE(p1_ctx_init(&a, &pick) == TAW_OK);
	REQUIRE(p1_ctx_init(&b, &all) == TAW_OK);
	long bad = 0;
	for (size_t i = 0; i < N; i++) {
		for (int k = 7; k >= 0; k--) {
			float xa[P1_XALL], xb[P1_XALL];
			p1_sel_ctx sa, sb;
			p1_ctx_inputs(&a, xa, &sa);
			p1_ctx_inputs(&b, xb, &sb);
			for (unsigned t = 0; t < P1_NHASH; t++) {
				const unsigned id = pick.lane[t];
				bad += xa[2 + t] != xb[2 + id] ||
				    xa[P1_NCM + 2 + t] != xb[P1_NCM_ALL + 2 + id];
			}
			bad += xa[0] != xb[0] || xa[1] != xb[1] ||
			    xa[P1_NCM] != xb[P1_NCM_ALL] ||
			    xa[P1_IN_MATCH] != xb[2 * P1_NCM_ALL] ||
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

static taw_status load(const taw_buf *mb)
{
	taw_pred pr;
	taw_status st = taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_CHUNK,
	    TAW_QUANT_F16, mb->p, mb->n);
	taw_pred_free(&pr);
	return st;
}

static taw_status load_patched(const taw_buf *mb, size_t off, uint8_t v)
{
	uint8_t saved = mb->p[off];
	mb->p[off] = v;
	taw_status st = load(mb);
	mb->p[off] = saved;
	return st;
}

static void validation(void)
{
	p1_shape s = geo;
	table_route(&s.route);
	taw_buf mb = build_model(&s, 9, 1.0f);
	const size_t lanes = P1_MODEL_HEAD, route = P1_MODEL_HEAD + P1_NHASH;
	CHECK(mb.n == P1_MODEL_HEAD + P1_NHASH + P2_ROUTE_BYTES +
	    p1_weights_size(&s, TAW_QUANT_F16));
	CHECK(load(&mb) == TAW_OK);
	/* the head is profile 1's, with the same limits */
	CHECK(load_patched(&mb, 0, P1_LINES_MAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, 5, P1_HMAX + 1) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, 6, 1) == TAW_E_FORMAT);
	/* lanes: known ids, increasing */
	CHECK(load_patched(&mb, lanes + 9, P2_NCAND) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, lanes + 1, P2_CX_O2) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, lanes + 4, P2_CX_O6) == TAW_E_FORMAT);
	/* no chunk context in the lanes: the routing table must not be
	 * there, so the size is wrong */
	uint8_t saved[P1_NHASH];
	memcpy(saved, mb.p + lanes, sizeof saved);
	for (unsigned k = 0; k < P1_NHASH; k++)
		mb.p[lanes + k] = (uint8_t)k;
	CHECK(load(&mb) == TAW_E_FORMAT);
	memcpy(mb.p + lanes, saved, sizeof saved);
	/* a line row bit without its chunk row bit */
	CHECK(mb.p[route + 128 + 2 * 4] == 0xEF && mb.p[route + 160 + 2 * 4] == 0xEF);
	CHECK(load_patched(&mb, route + 160 + 2 * 4, 0xFF) == TAW_E_FORMAT);
	CHECK(load_patched(&mb, route + 160, 0x00) == TAW_OK);
	/* every class and chunk row value is valid */
	CHECK(load_patched(&mb, route, 0xFF) == TAW_OK);
	CHECK(load_patched(&mb, route + 128, 0x00) == TAW_OK);
	/* truncated, and the profile 1 loader refuses the section */
	taw_pred pr;
	CHECK(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_CHUNK, TAW_QUANT_F16,
	    mb.p, mb.n - 1) == TAW_E_FORMAT);
	CHECK(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_CHUNK, TAW_QUANT_F16,
	    mb.p, P1_MODEL_HEAD + 3) == TAW_E_FORMAT);
	CHECK(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_MIX, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_E_FORMAT);
	CHECK(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_CHUNK, TAW_QUANT_BF16,
	    mb.p, mb.n) == TAW_E_UNSUPPORTED);
	taw_buf_free(&mb);

	/* profile 1's lanes: the section has no routing table */
	for (unsigned k = 0; k < P1_NHASH; k++)
		s.lane[k] = (uint8_t)k;
	mb = build_model(&s, 9, 1.0f);
	CHECK(mb.n == P1_MODEL_HEAD + P1_NHASH +
	    p1_weights_size(&s, TAW_QUANT_F16));
	CHECK(load(&mb) == TAW_OK);
	taw_buf_free(&mb);
}

/* ---- golden vector ------------------------------------------------------------------
 * A fixed hand-built model over fixed text. Any change to profile 2's
 * arithmetic (the routing rule, the chunk and line state, the chunk
 * contexts, the lanes, and all of profile 1's) changes these CRCs: that is
 * a format break and needs a new profile id. */

static uint32_t golden_crc(bool plain, uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N), *dst = malloc(N);
	REQUIRE(src && dst);
	fill_table(src, N / 2, 77);
	td_fill(src + N / 2, N / 2, TD_TEXT, 1234);
	for (size_t i = N / 2; i < N; i += 4096) /* some match-model work */
		memcpy(src + i, src + N / 2, 64);
	p1_shape s = geo;
	table_route(&s.route);
	if (plain)
		for (unsigned k = 0; k < P1_NHASH; k++)
			s.lane[k] = (uint8_t)k;
	taw_buf mb = build_model(&s, 42, 0.4f);
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_CHUNK, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_OK);
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

/* Profile 2 with profile 1's lanes computes what profile 1 computes: the
 * same weights give the same coded bytes. */
static uint32_t p1_crc(uint32_t *coded_len)
{
	enum { N = 1 << 16 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	fill_table(src, N / 2, 77);
	td_fill(src + N / 2, N / 2, TD_TEXT, 1234);
	for (size_t i = N / 2; i < N; i += 4096)
		memcpy(src + i, src + N / 2, 64);
	p1_shape s = geo;
	for (unsigned k = 0; k < P1_NHASH; k++)
		s.lane[k] = (uint8_t)k;
	taw_buf mb = build_model(&s, 42, 0.4f);
	/* the same section without the ten lane bytes */
	memmove(mb.p + P1_MODEL_HEAD, mb.p + P1_MODEL_HEAD + P1_NHASH,
	    mb.n - P1_MODEL_HEAD - P1_NHASH);
	mb.n -= P1_NHASH;
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_F32_STRICT_MIX, TAW_QUANT_F16,
	    mb.p, mb.n) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, N);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);
	uint32_t crc = taw_crc32c(0, out.p, out.n);
	*coded_len = (uint32_t)out.n;
	taw_pred_free(&pr);
	taw_buf_free(&out);
	taw_buf_free(&mb);
	free(src);
	return crc;
}

static void golden(void)
{
	static const struct {
		bool plain;
		uint32_t len, crc;
	} want[] = {
		{false, GOLDEN_LEN_CHUNK, GOLDEN_CRC_CHUNK},
		{true, GOLDEN_LEN_PLAIN, GOLDEN_CRC_PLAIN},
	};
	for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
		for (int isa = 0; isa < P1_ISA_COUNT; isa++) {
			if (!p1_isa_available((p1_isa)isa))
				continue;
			p1_isa_force(isa);
			uint32_t len, crc = golden_crc(want[i].plain, &len);
			if (len != want[i].len || crc != want[i].crc) {
				fprintf(stderr, "golden plain=%d isa=%d: len %u crc 0x%08X\n",
				    want[i].plain, isa, len, crc);
				check_failures++;
			}
		}
	}
	p1_isa_force(-1);
	uint32_t len, crc = p1_crc(&len);
	CHECK(len == want[1].len && crc == want[1].crc);
}

/* ---- archives ------------------------------------------------------------------- */

/* Whole archives: every ISA encodes the same bytes and decodes them. */
static void archive_parity(void)
{
	enum { N = 60000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	fill_table(src, N, 21);
	const taw_params p = {.level = 2, .predictor = TAW_PREDICTOR_CHUNK};
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
	taw_info info;
	REQUIRE(taw_read_info(ref, ref_n, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_F32_STRICT_CHUNK);
	CHECK(info.model_size <= N / 32);
	free(ref);
	free(src);
}

static size_t packed(const uint8_t *src, size_t n, taw_predictor pred,
    int level, unsigned threads, uint8_t *lanes)
{
	const taw_params p = {.level = level, .predictor = pred,
		.threads = threads};
	void *arc, *dec;
	size_t arc_n, dec_n;
	REQUIRE(taw_compress(src, n, &p, &arc, &arc_n) == TAW_OK);
	REQUIRE(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == n && memcmp(dec, src, n) == 0);
	if (lanes)
		memcpy(lanes, (uint8_t *)arc + TAW_HEADER_SIZE + P1_MODEL_HEAD,
		    P1_NHASH);
	free(dec);
	free(arc);
	return arc_n;
}

/* What the profile is for: on a table the line contexts are chosen and
 * the archive is well under profile 1's; on bytes without structure it
 * falls back to profile 1's ten models and costs at most their ten lane
 * bytes more. The thread count changes nothing. */
static void archive_gain(void)
{
	enum { N = 200000 };
	uint8_t *src = malloc(N), lanes[P1_NHASH], lanes4[P1_NHASH];
	REQUIRE(src);
	fill_table(src, N, 3);
	const size_t mix = packed(src, N, TAW_PREDICTOR_MIX, 2, 1, nullptr);
	const size_t chunk = packed(src, N, TAW_PREDICTOR_CHUNK, 2, 1, lanes);
	CHECK(chunk * 100 < mix * 90);
	CHECK(lanes[P1_NHASH - 1] == P2_CX_ABOVE);
	CHECK(packed(src, N, TAW_PREDICTOR_CHUNK, 2, 4, lanes4) == chunk);
	CHECK(memcmp(lanes, lanes4, sizeof lanes) == 0);

	td_fill(src, N, TD_SKEWED, 17);
	const size_t mix_s = packed(src, N, TAW_PREDICTOR_MIX, 2, 1, nullptr);
	const size_t chunk_s = packed(src, N, TAW_PREDICTOR_CHUNK, 2, 1, lanes);
	CHECK(chunk_s < mix_s + mix_s / 100 + P1_NHASH + P2_ROUTE_BYTES);
	free(src);
}

/* The seekable variant: blocks reset the chunk stage, ranges decode. */
static void archive_seek(void)
{
	enum { N = 150000, BS = 1 << 14 };
	uint8_t *src = malloc(N), *got = malloc(N);
	REQUIRE(src && got);
	fill_table(src, N, 11);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_CHUNK,
		.seekable = true, .block_size = BS, .threads = 3};
	void *arc;
	size_t arc_n;
	REQUIRE(taw_compress(src, N, &p, &arc, &arc_n) == TAW_OK);
	taw_info info;
	REQUIRE(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_F32_STRICT_CHUNK);
	CHECK(info.kind == TAW_KIND_SEEKABLE);
	taw_source in;
	taw_source_memory(&in, arc, arc_n);
	taw_seek *s;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	CHECK(taw_seek_size(s) == N);
	static const size_t offs[] = {0, BS - 7, 3 * BS + 100, N - 5000};
	for (size_t i = 0; i < sizeof offs / sizeof *offs; i++) {
		const size_t len = offs[i] + 5000 <= N ? 5000 : N - offs[i];
		CHECK(taw_seek_read(s, offs[i], got, len) == TAW_OK);
		CHECK(memcmp(got, src + offs[i], len) == 0);
	}
	taw_seek_close(s);
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress_mt(arc, arc_n, 4, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == N && memcmp(dec, src, N) == 0);
	free(dec);
	free(arc);
	free(src);
	free(got);
}

/* Too small for the largest head inside the n/32 budget: refused, with
 * nothing written. */
static void archive_small(void)
{
	uint8_t src[4000];
	fill_table(src, sizeof src, 1);
	const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_CHUNK};
	void *arc = nullptr;
	size_t arc_n = 0;
	CHECK(taw_compress(src, sizeof src, &p, &arc, &arc_n) == TAW_E_ARG);
	CHECK(arc == nullptr);
	CHECK(taw_compress(src, 0, &p, &arc, &arc_n) == TAW_E_ARG);
}

int main(void)
{
	p1_tables_init();
	chunk_stage();
	ctx_parity();
	lanes_match_candidates();
	validation();
	golden();
	archive_parity();
	archive_gain();
	archive_seek();
	archive_small();
	return CHECK_DONE();
}
