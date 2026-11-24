/* SPDX-License-Identifier: GPL-3.0-or-later */
/* .awseek: roundtrip, random access, I/O locality, corruption isolation. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "taw_internal.h"
#include "testdata.h"

/* Memory source that counts the bytes it serves. */
typedef struct {
	const uint8_t *p;
	uint64_t bytes;
} counting;

static taw_status count_read(void *ctx, uint64_t off, void *dst, size_t n)
{
	counting *c = ctx;
	memcpy(dst, c->p + off, n);
	c->bytes += n;
	return TAW_OK;
}

static void *compress_seek_p(const uint8_t *src, size_t n, uint32_t bs,
    taw_predictor pred, size_t *arc_n)
{
	taw_params p = {.seekable = true, .block_size = bs, .predictor = pred,
		.level = pred == TAW_PREDICTOR_MIX || pred == TAW_PREDICTOR_GATE
			? 1 : 0};
	void *arc;
	REQUIRE(taw_compress(src, n, &p, &arc, arc_n) == TAW_OK);
	return arc;
}

/* Container tests use the fast order-0 predictor. */
static void *compress_seek(const uint8_t *src, size_t n, uint32_t bs,
    size_t *arc_n)
{
	return compress_seek_p(src, n, bs, TAW_PREDICTOR_ORDER0, arc_n);
}

static void roundtrip_p(const uint8_t *src, size_t n, uint32_t bs,
    taw_predictor pred)
{
	size_t arc_n;
	void *arc = compress_seek_p(src, n, bs, pred, &arc_n);
	taw_info info;
	CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.kind == TAW_KIND_SEEKABLE);
	CHECK(info.block_size == (bs ? bs : TAW_BLOCK_DEFAULT));
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == n && (n == 0 || memcmp(src, dec, n) == 0));
	free(dec);
	free(arc);
}

static void roundtrip(const uint8_t *src, size_t n, uint32_t bs)
{
	roundtrip_p(src, n, bs, TAW_PREDICTOR_ORDER0);
}

/* Opt-in dynamic blocks; target 0 uses block_size, threshold 0 is ratio mode. */
static void *compress_seek_dyn(const uint8_t *src, size_t n, uint32_t bs,
    uint32_t target, uint32_t threshold, taw_predictor pred, size_t *arc_n)
{
	taw_params p = {.seekable = true, .block_size = bs, .predictor = pred,
		.level = pred == TAW_PREDICTOR_MIX ? 1 : 0,
		.dyn_blocks = true, .chunk_target = target,
		.change_threshold = threshold};
	void *arc;
	REQUIRE(taw_compress(src, n, &p, &arc, arc_n) == TAW_OK);
	return arc;
}

/* Index invariants of a dynamic archive (20-byte entries). */
static void dyn_index_stats(const uint8_t *arc, size_t arc_n,
    uint32_t *nblocks, uint64_t *sum, uint32_t *minlen, uint32_t *maxlen)
{
	const uint64_t ix = taw_rd64(arc + arc_n - 16);
	const uint32_t nb = taw_rd32(arc + ix + 4);
	*nblocks = nb;
	*sum = 0;
	*minlen = UINT32_MAX;
	*maxlen = 0;
	for (uint32_t i = 0; i < nb; i++) {
		const uint8_t *e = arc + ix + 16 + 20 * (size_t)i;
		const uint32_t rl = taw_rd32(e + 16);
		CHECK(rl != 0 && taw_rd32(e + 8) <= rl);
		*sum += rl;
		if (rl < *minlen)
			*minlen = rl;
		if (rl > *maxlen)
			*maxlen = rl;
	}
}

/* Profile 1: one shared model, history reset per block. */
static void random_access_mix(void)
{
	enum { N = 300000, BS = 1 << 15 };
	uint8_t *src = malloc(N), *buf = malloc(2 * BS);
	REQUIRE(src && buf);
	td_fill(src, N, TD_TEXT, 12);
	size_t arc_n;
	void *arc = compress_seek_p(src, N, BS, TAW_PREDICTOR_MIX, &arc_n);
	taw_info info;
	CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_F32_STRICT_MIX);
	taw_source mem;
	taw_source_memory(&mem, arc, arc_n);
	taw_seek *s;
	REQUIRE(taw_seek_open(&mem, &s) == TAW_OK);
	uint64_t r = 5;
	for (int t = 0; t < 60; t++) {
		/* serial, then parallel reads (several blocks, partial edges) */
		taw_seek_set_threads(s, t < 20 ? 1 : t < 40 ? 3 : 8);
		uint64_t off = td_next(&r) % N;
		size_t len = (size_t)(td_next(&r) % (2 * BS));
		if (len > N - off)
			len = (size_t)(N - off);
		REQUIRE(taw_seek_read(s, off, buf, len) == TAW_OK);
		CHECK(memcmp(buf, src + off, len) == 0);
	}
	taw_seek_close(s);
	/* The whole archive on several threads. */
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress_mt(arc, arc_n, 4, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == N && memcmp(dec, src, N) == 0);
	free(dec);
	free(arc);
	free(buf);
	free(src);
}

static taw_status buf_write(void *ctx, const void *p, size_t n)
{
	taw_buf *b = ctx;
	taw_buf_put(b, p, n);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

static taw_status encode_files(const uint8_t *src, size_t n,
    const taw_params *p, const taw_file *f, size_t nf, taw_buf *out)
{
	taw_source in;
	taw_source_memory(&in, src, n);
	*out = (taw_buf){0};
	const taw_sink sink = {.write = buf_write, .ctx = out};
	return taw_seek_encode(&in, p, f, nf, &sink);
}

/* Directory, stored blocks, and archives independent of the thread count. */
static void files_and_threads(void)
{
	enum { N = 200000, BS = 1 << 14 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N / 2, TD_TEXT, 7);
	td_fill(src + N / 2, N - N / 2, TD_RANDOM, 8); /* stored blocks */
	const taw_file f[] = {
		{"/text.txt", 0, N / 2},
		{"/noise.bin", N / 2, N - N / 2 - 100},
		{"/.meta", N - 50, 50},
	};
	taw_params p = {.block_size = BS, .level = 1,
		.predictor = TAW_PREDICTOR_MIX, .threads = 1};
	taw_buf a, b;
	REQUIRE(encode_files(src, N, &p, f, 3, &a) == TAW_OK);
	p.threads = 4;
	REQUIRE(encode_files(src, N, &p, f, 3, &b) == TAW_OK);
	CHECK(a.n == b.n && memcmp(a.p, b.p, a.n) == 0);
	taw_buf_free(&b);

	taw_info info;
	CHECK(taw_read_info(a.p, a.n, &info) == TAW_OK && info.has_files);
	taw_source in;
	taw_source_memory(&in, a.p, a.n);
	taw_seek *s;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	const taw_file *got;
	CHECK(taw_seek_files(s, &got) == 3);
	CHECK(!strcmp(got[2].name, "/.meta") && got[2].offset == N - 50);
	CHECK(taw_seek_find(s, "/noise.bin") == 1);
	CHECK(taw_seek_find(s, "/nope") == -1);
	uint8_t buf[50];
	CHECK(taw_seek_read(s, got[2].offset, buf, 50) == TAW_OK);
	CHECK(memcmp(buf, src + N - 50, 50) == 0);
	taw_seek_close(s);

	/* the random half is stored: some index entry has coded_len == raw */
	{
		uint64_t ix = taw_rd64(a.p + a.n - 16);
		uint32_t nb = taw_rd32(a.p + ix + 4), stored = 0;
		for (uint32_t i = 0; i < nb; i++)
			stored += taw_rd32(a.p + ix + 16 + 16 * i + 8) == BS;
		CHECK(stored >= (N / 2) / BS - 1);
		CHECK(a.n < N + 4096); /* random data does not grow */
	}
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress(a.p, a.n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == N && memcmp(dec, src, N) == 0);
	free(dec);

	/* a corrupt directory (CRC fixed up) is refused */
	{
		uint64_t ix = taw_rd64(a.p + a.n - 16);
		uint32_t nb = taw_rd32(a.p + ix + 4);
		uint8_t *d = a.p + ix + 16 + 16 * (uint64_t)nb;
		uint8_t save = d[4 + 18]; /* first name's '/' */
		d[4 + 18] = 'x';
		taw_wr32(a.p + a.n - 8, taw_crc32c(0, a.p + ix, a.n - 16 - ix));
		taw_source_memory(&in, a.p, a.n);
		CHECK(taw_seek_open(&in, &s) == TAW_E_FORMAT);
		d[4 + 18] = save;
	}
	taw_buf_free(&a);

	/* invalid directories */
	const taw_file unsorted[] = {{"/b", 100, 10}, {"/a", 0, 10}};
	const taw_file overlap[] = {{"/a", 0, 20}, {"/b", 10, 10}};
	const taw_file noslash[] = {{"a", 0, 10}};
	const taw_file dup[] = {{"/a", 0, 10}, {"/a", 10, 10}};
	const taw_file past[] = {{"/a", N - 5, 10}};
	p.threads = 1;
	CHECK(encode_files(src, N, &p, unsorted, 2, &a) == TAW_E_ARG);
	CHECK(encode_files(src, N, &p, overlap, 2, &a) == TAW_E_ARG);
	CHECK(encode_files(src, N, &p, noslash, 1, &a) == TAW_E_ARG);
	CHECK(encode_files(src, N, &p, dup, 2, &a) == TAW_E_ARG);
	CHECK(encode_files(src, N, &p, past, 1, &a) == TAW_E_ARG);
	free(src);
}

static void random_access(void)
{
	enum { N = (1 << 20) + 12345, BS = 4096 };
	uint8_t *src = malloc(N), *buf = malloc(3 * BS);
	REQUIRE(src && buf);
	td_fill(src, N, TD_TEXT, 11);
	size_t arc_n;
	void *arc = compress_seek(src, N, BS, &arc_n);

	counting cnt = {.p = arc};
	taw_source in = {.read_at = count_read, .ctx = &cnt, .size = arc_n};
	taw_seek *s;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	CHECK(taw_seek_size(s) == N);
	uint64_t open_bytes = cnt.bytes;

	/* Random ranges, including block-straddling ones. */
	uint64_t r = 99;
	for (int t = 0; t < 2000; t++) {
		uint64_t off = td_next(&r) % N;
		size_t len = (size_t)(td_next(&r) % (3 * BS));
		if (len > N - off)
			len = (size_t)(N - off);
		REQUIRE(taw_seek_read(s, off, buf, len) == TAW_OK);
		CHECK(memcmp(buf, src + off, len) == 0);
	}

	/* One byte from the middle reads at most one coded block. */
	taw_seek_close(s);
	cnt.bytes = 0;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	open_bytes = cnt.bytes;
	CHECK(taw_seek_read(s, N / 2, buf, 1) == TAW_OK);
	CHECK(buf[0] == src[N / 2]);
	CHECK(cnt.bytes - open_bytes <= BS);
	CHECK(cnt.bytes < arc_n / 4);

	/* Bounds. */
	CHECK(taw_seek_read(s, N, buf, 0) == TAW_OK);
	CHECK(taw_seek_read(s, N, buf, 1) == TAW_E_RANGE);
	CHECK(taw_seek_read(s, N - 1, buf, 2) == TAW_E_RANGE);
	CHECK(taw_seek_read(s, UINT64_MAX, buf, 1) == TAW_E_RANGE);
	taw_seek_close(s);

	/* Corrupting block 3 only breaks reads that touch block 3. */
	uint8_t *a = arc;
	uint64_t ix = taw_rd64(a + arc_n - 16);
	uint64_t b3 = taw_rd64(a + ix + 16 + 3 * 16);
	a[b3 + 2] ^= 0x10;
	taw_source mem;
	taw_source_memory(&mem, arc, arc_n);
	REQUIRE(taw_seek_open(&mem, &s) == TAW_OK);
	CHECK(taw_seek_read(s, 3 * BS + 100, buf, 10) == TAW_E_CHECKSUM);
	CHECK(taw_seek_read(s, 5 * BS, buf, BS) == TAW_OK);
	CHECK(memcmp(buf, src + 5 * BS, BS) == 0);
	CHECK(taw_seek_read(s, 2 * BS, buf, 2 * BS) == TAW_E_CHECKSUM);
	taw_seek_close(s);
	a[b3 + 2] ^= 0x10;

	/* Index, footer and truncation damage is caught at open. */
	a[ix + 20] ^= 1;
	CHECK(taw_seek_open(&mem, &s) == TAW_E_CHECKSUM);
	a[ix + 20] ^= 1;
	a[arc_n - 1] ^= 1;
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);
	a[arc_n - 1] ^= 1;
	taw_source_memory(&mem, arc, arc_n - 1);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);
	taw_source_memory(&mem, arc, 20);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);

	/* A stream archive is not seekable. */
	void *aw;
	size_t aw_n;
	REQUIRE(taw_compress(src, 1000, &(taw_params){.predictor =
	    TAW_PREDICTOR_ORDER0}, &aw, &aw_n) == TAW_OK);
	taw_source_memory(&mem, aw, aw_n);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_ARG);

	free(aw);
	free(arc);
	free(buf);
	free(src);
}

/* The default (flag off) path must be byte-identical to origin/main. */
static void default_unchanged(void)
{
	static const uint32_t crc[TD_KINDS] = {0x2f4c13cc, 0x3f6480cc,
		0x525c32a5, 0xa35c2865};
	static const size_t sizes[TD_KINDS] = {914, 100464, 54968, 26977};
	for (int kind = 0; kind < TD_KINDS; kind++) {
		uint8_t src[100000];
		td_fill(src, sizeof src, kind, 1234 + (uint64_t)kind);
		size_t n;
		void *arc = compress_seek_p(src, sizeof src, 4096,
		    TAW_PREDICTOR_ORDER0, &n);
		CHECK(n == sizes[kind]);
		CHECK(taw_crc32c(0, arc, n) == crc[kind]);
		taw_info info;
		CHECK(taw_read_info(arc, n, &info) == TAW_OK &&
		    !info.has_dyn_blocks);
		free(arc);
	}
}

/* ---- corruption golden vector for the widened index (§7.2) ---------------- */

/* Patch a u32 in a copy of the archive, recompute the index CRC so the
 * decoder must reject the *meaning* of the field, not a broken checksum. */
static taw_status open_index_patched(const uint8_t *arc, size_t an,
    uint64_t ix, size_t off, uint32_t v)
{
	uint8_t *c = malloc(an);
	REQUIRE(c);
	memcpy(c, arc, an);
	taw_wr32(c + off, v);
	taw_wr32(c + an - 8, taw_crc32c(0, c + (size_t)ix,
	    an - TAW_FOOTER_SIZE - (size_t)ix));
	taw_source in;
	taw_seek *s = nullptr;
	taw_source_memory(&in, c, an);
	const taw_status st = taw_seek_open(&in, &s);
	if (st == TAW_OK)
		taw_seek_close(s);
	free(c);
	return st;
}

/* Patch the header flag byte, with the header CRC fixed up. */
static taw_status open_flag_patched(const uint8_t *arc, size_t an,
    uint8_t flags)
{
	uint8_t *c = malloc(an);
	REQUIRE(c);
	memcpy(c, arc, an);
	c[9] = flags;
	taw_wr32(c + 28, taw_crc32c(0, c, 28));
	taw_source in;
	taw_seek *s = nullptr;
	taw_source_memory(&in, c, an);
	const taw_status st = taw_seek_open(&in, &s);
	if (st == TAW_OK)
		taw_seek_close(s);
	free(c);
	return st;
}

/* A fixed golden archive with every §7.2 refusal pinned to TAW_E_FORMAT:
 * raw_len == 0, sum(raw_len) != total_raw_len, coded_len > raw_len,
 * coded_offset out of range, raw_len > 2^26, and entry-size disagreement
 * (flag bit 1 with 16-byte entries, flag bit 0 with 20-byte entries).
 * An ordinary index byte flip without CRC repair must still be a checksum
 * error, and the unpatched golden must decode byte-for-byte. */
static void dynamic_corruption(void)
{
	enum { N = 100000, BS = 4096 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 17);
	size_t an;
	uint8_t *a = compress_seek_dyn(src, N, BS, BS, 0,
	    TAW_PREDICTOR_ORDER0, &an);
	const uint64_t ix = taw_rd64(a + an - 16);
	const uint32_t nb = taw_rd32(a + (size_t)ix + 4);
	const size_t e0 = (size_t)ix + TAW_INDEX_HEAD;
	const size_t el = e0 + TAW_INDEX_ENTRY_DYN * (size_t)(nb - 1);

	/* Golden: td_fill(TD_TEXT, 100000, seed 17), order-0, BS = target =
	 * 4096, 1 thread -> 55086 bytes, 25 entries, index_crc 2BC6ECA7. */
	CHECK(an == 55086);
	CHECK(ix == 54554);
	CHECK(nb == 25);
	CHECK(taw_rd32(a + an - 8) == 0x2BC6ECA7u);
	CHECK(taw_crc32c(0, a, an) == 0x8D6620DAu);
	CHECK(taw_rd32(a + e0 + 16) == 6020);   /* first raw_len  */
	CHECK(taw_rd32(a + el + 16) == 3277);   /* last raw_len   */
	uint64_t sum = 0;
	for (uint32_t i = 0; i < nb; i++)
		sum += taw_rd32(a + e0 + TAW_INDEX_ENTRY_DYN * (size_t)i + 16);
	CHECK(sum == N);

	taw_source in;
	taw_seek *s;
	taw_source_memory(&in, a, an);
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	CHECK(taw_seek_size(s) == N);
	taw_seek_close(s);
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress(a, an, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == N && memcmp(dec, src, N) == 0);
	free(dec);

	/* Index damage with no CRC repair is still caught as a checksum. */
	{
		uint8_t *c = malloc(an);
		REQUIRE(c);
		memcpy(c, a, an);
		c[e0 + 19] ^= 1;
		taw_source_memory(&in, c, an);
		CHECK(taw_seek_open(&in, &s) == TAW_E_CHECKSUM);
		free(c);
	}

	/* Semantic refusals, each with a valid index CRC. */
	CHECK(open_index_patched(a, an, ix, e0 + 16, 0) == TAW_E_FORMAT);
	CHECK(open_index_patched(a, an, ix, el + 16, 3278) == TAW_E_FORMAT);
	CHECK(open_index_patched(a, an, ix, e0 + 8, 6021) == TAW_E_FORMAT);
	CHECK(open_index_patched(a, an, ix, e0, (uint32_t)an) == TAW_E_FORMAT);
	CHECK(open_index_patched(a, an, ix, e0 + 16, TAW_CHUNK_MAX + 1u) ==
	    TAW_E_FORMAT);

	/* Flag and entry size must agree; the header CRC is repaired too. */
	CHECK(open_flag_patched(a, an, a[9] & (uint8_t)~TAW_FLAG_DYN_BLOCKS) ==
	    TAW_E_FORMAT);
	size_t fn;
	uint8_t *f = compress_seek(src, N, BS, &fn);
	CHECK(open_flag_patched(f, fn, f[9] | TAW_FLAG_DYN_BLOCKS) ==
	    TAW_E_FORMAT);
	free(f);
	free(a);
	free(src);
}

static void dynamic_blocks(void)
{
	/* Roundtrips and index sums for every data kind and edge size. */
	static const size_t sizes[] = {0, 1, 4095, 4096, 4097, 100000,
		(1u << 20) + 123};
	for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
		uint8_t *src = malloc(sizes[k] ? sizes[k] : 1);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, sizes[k], kind, 17 + (size_t)kind);
			size_t arc_n;
			void *arc = compress_seek_dyn(src, sizes[k], 4096, 4096, 0,
			    TAW_PREDICTOR_ORDER0, &arc_n);
			taw_info info;
			CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
			CHECK(info.kind == TAW_KIND_SEEKABLE &&
			    info.has_dyn_blocks && info.block_size == 4096);
			uint32_t nb, mn, mx;
			uint64_t sum;
			dyn_index_stats(arc, arc_n, &nb, &sum, &mn, &mx);
			CHECK(sum == sizes[k]);
			/* variable, but never below min_len = target/4 nor
			 * above max_len = 4*target */
			CHECK(nb <= (sizes[k] ? (uint32_t)((sizes[k] + 1023) / 1024) : 0));
			CHECK(nb >= (sizes[k] ? (uint32_t)((sizes[k] + 16383) / 16384) : 0));
			void *dec;
			size_t dec_n;
			CHECK(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
			CHECK(dec_n == sizes[k] &&
			    memcmp(dec, src, sizes[k]) == 0);
			free(dec);
			free(arc);
		}
		free(src);
	}

	/* Boundaries grow over homogeneous content and shrink at changes. */
	{
		enum { SEG = 32768, N = 4 * SEG };
		uint8_t *src = malloc(N);
		REQUIRE(src);
		td_fill(src, SEG, TD_TEXT, 1);
		td_fill(src + SEG, 2 * SEG, TD_ZERO, 2);
		td_fill(src + 3 * SEG, SEG, TD_RANDOM, 3);
		size_t arc_n;
		void *arc = compress_seek_dyn(src, N, 1u << 16, 32768, 0,
		    TAW_PREDICTOR_ORDER0, &arc_n);
		uint32_t nb, mn, mx;
		uint64_t sum;
		dyn_index_stats(arc, arc_n, &nb, &sum, &mn, &mx);
		CHECK(sum == N);
		CHECK(nb < (uint32_t)(N / 16384)); /* fewer than 8 fixed blocks */
		CHECK(mx >= 2 * SEG);              /* the zero run is one block */
		void *dec;
		size_t dec_n;
		CHECK(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
		CHECK(dec_n == N && memcmp(dec, src, N) == 0);
		free(dec);
		free(arc);
		free(src);
	}

	/* Profile 1: random access, serial and parallel, over variable blocks. */
	{
		enum { N = 600000, BS = 1 << 16 };
		uint8_t *src = malloc(N), *buf = malloc(3 * BS);
		REQUIRE(src && buf);
		td_fill(src, N / 2, TD_TEXT, 21);
		td_fill(src + N / 2, N - N / 2, TD_SKEWED, 22);
		size_t arc_n;
		void *arc = compress_seek_dyn(src, N, BS, 16384, 0,
		    TAW_PREDICTOR_MIX, &arc_n);
		taw_source mem;
		taw_source_memory(&mem, arc, arc_n);
		taw_seek *s;
		REQUIRE(taw_seek_open(&mem, &s) == TAW_OK);
		uint64_t r = 1234;
		for (int t = 0; t < 120; t++) {
			taw_seek_set_threads(s, t < 40 ? 1 : t < 80 ? 3 : 8);
			uint64_t off = td_next(&r) % N;
			size_t len = (size_t)(td_next(&r) % (2 * BS));
			if (len > N - off)
				len = N - off;
			REQUIRE(taw_seek_read(s, off, buf, len) == TAW_OK);
			CHECK(memcmp(buf, src + off, len) == 0);
		}
		taw_seek_close(s);
		void *dec;
		size_t dec_n;
		CHECK(taw_decompress_mt(arc, arc_n, 4, &dec, &dec_n) == TAW_OK);
		CHECK(dec_n == N && memcmp(dec, src, N) == 0);
		free(dec);
		free(arc);
		free(buf);
		free(src);
	}

	/* Directory, threshold mode, thread independence and index damage. */
	{
		enum { N = 200000, BS = 1 << 15 };
		uint8_t *src = malloc(N);
		REQUIRE(src);
		td_fill(src, N / 2, TD_TEXT, 31);
		td_fill(src + N / 2, N - N / 2, TD_SKEWED, 32);
		const taw_file f[] = {{"/head", 0, N / 2},
		    {"/tail", N / 2, N / 2 - 10}, {"/.meta", N - 10, 10}};
		taw_params p = {.block_size = BS, .level = 1,
			.predictor = TAW_PREDICTOR_MIX, .dyn_blocks = true,
			.chunk_target = 8192, .threads = 1};
		taw_buf a, b;
		REQUIRE(encode_files(src, N, &p, f, 3, &a) == TAW_OK);
		p.threads = 6;
		REQUIRE(encode_files(src, N, &p, f, 3, &b) == TAW_OK);
		CHECK(a.n == b.n && memcmp(a.p, b.p, a.n) == 0);
		taw_buf_free(&b);

		taw_source in;
		taw_source_memory(&in, a.p, a.n);
		taw_seek *s;
		REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
		const taw_file *got;
		CHECK(taw_seek_files(s, &got) == 3);
		CHECK(taw_seek_find(s, "/.meta") == 2);
		for (size_t i = 0; i < 3; i++) {
			uint8_t *out = malloc((size_t)got[i].size);
			REQUIRE(out);
			CHECK(taw_seek_read(s, got[i].offset, out,
			    (size_t)got[i].size) == TAW_OK);
			CHECK(memcmp(out, src + got[i].offset, got[i].size) == 0);
			free(out);
		}
		taw_seek_close(s);

		/* Change-spike mode is deterministic and roundtrips too. */
		p.threads = 1;
		p.change_threshold = 250;
		REQUIRE(encode_files(src, N, &p, f, 3, &b) == TAW_OK);
		void *dec;
		size_t dec_n;
		CHECK(taw_decompress(b.p, b.n, &dec, &dec_n) == TAW_OK);
		CHECK(dec_n == N && memcmp(dec, src, N) == 0);
		free(dec);
		taw_buf_free(&b);

		/* A bogus raw_len (with the index CRC fixed up) is refused. */
		const uint64_t ix = taw_rd64(a.p + a.n - 16);
		uint8_t *e = a.p + ix + 16 + 20 * 3;
		const uint32_t save = taw_rd32(e + 16);
		taw_wr32(e + 16, save + 1);
		taw_wr32(a.p + a.n - 8,
		    taw_crc32c(0, a.p + ix, a.n - 16 - ix));
		taw_source_memory(&in, a.p, a.n);
		CHECK(taw_seek_open(&in, &s) == TAW_E_FORMAT);
		taw_wr32(e + 16, save);
		taw_wr32(a.p + a.n - 8,
		    taw_crc32c(0, a.p + ix, a.n - 16 - ix));
		CHECK(taw_seek_open(&in, &s) == TAW_OK);
		taw_seek_close(s);
		taw_buf_free(&a);

		/* Invalid opt-in combinations. */
		void *o;
		size_t on;
		taw_params bad = {.dyn_blocks = true};
		CHECK(taw_compress(src, 100, &bad, &o, &on) == TAW_E_ARG);
		bad = (taw_params){.seekable = true, .dyn_blocks = true,
			.change_threshold = 257};
		CHECK(taw_compress(src, 100, &bad, &o, &on) == TAW_E_ARG);
		bad = (taw_params){.seekable = true, .dyn_blocks = true,
			.chunk_target = 1u << 27};
		CHECK(taw_compress(src, 100, &bad, &o, &on) == TAW_E_ARG);
		free(src);
	}
}

int main(void)
{
	default_unchanged();
	dynamic_corruption();
	dynamic_blocks();
	roundtrip(nullptr, 0, 4096);
	uint8_t one = 7;
	roundtrip(&one, 1, 4096);

	static const size_t sizes[] = {4095, 4096, 4097, 3 * 4096, 100000};
	for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
		uint8_t *src = malloc(sizes[i]);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, sizes[i], kind, i + 17 * (size_t)kind);
			roundtrip(src, sizes[i], 4096);
			roundtrip(src, sizes[i], 1u << 16);
		}
		free(src);
	}
	uint8_t big[5000];
	td_fill(big, sizeof big, TD_SKEWED, 5);
	roundtrip(big, sizeof big, 0); /* default block size */

	taw_params bad = {.seekable = true, .block_size = 5000};
	void *o;
	size_t on;
	CHECK(taw_compress(big, 10, &bad, &o, &on) == TAW_E_ARG);
	bad.block_size = 2048;
	CHECK(taw_compress(big, 10, &bad, &o, &on) == TAW_E_ARG);

	/* Profile 1 roundtrips, including a partial last block. */
	{
		enum { N = 100000 };
		uint8_t *src = malloc(N);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, N, kind, 40 + (size_t)kind);
			roundtrip_p(src, N, 1u << 14, TAW_PREDICTOR_MIX);
		}
		td_fill(src, N, TD_TEXT, 3);
		roundtrip_p(src, N, 4096, TAW_PREDICTOR_AUTO);
		roundtrip_p(src, N, 1u << 14, TAW_PREDICTOR_GATE);
		free(src);
	}

	random_access();
	random_access_mix();
	files_and_threads();
	return CHECK_DONE();
}
