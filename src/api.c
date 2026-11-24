/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Public entry points: dispatch on variant, own the output buffers. */
#include <stdlib.h>

#include "aw.h"
#include "awseek.h"
#include "format.h"
#include "header.h"

[[maybe_unused]] static taw_status finish(taw_status st, taw_buf *b, void **out,
    size_t *out_n)
{
	if (st != TAW_OK) {
		taw_buf_free(b);
		return st;
	}
	if (b->oom) {
		taw_buf_free(b);
		return TAW_E_NOMEM;
	}
	*out = b->p ? b->p : malloc(1);
	*out_n = b->n;
	return *out ? TAW_OK : TAW_E_NOMEM;
}

#if TAW_HAVE_ENCODER
static taw_status encode_with(const uint8_t *src, size_t n,
    const taw_params *p, const taw_model_choice *m, taw_buf *b)
{
	return taw_aw_encode(src, n, p, m, b);
}

static taw_status buf_sink(void *ctx, const void *p, size_t n)
{
	taw_buf *b = ctx;
	taw_buf_put(b, p, n);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

taw_status taw_compress(const void *src, size_t n, const taw_params *params,
    void **out, size_t *out_n)
{
	if ((!src && n) || !out || !out_n)
		return TAW_E_ARG;
	taw_params p;
	taw_status st = taw_params_resolve(params, &p);
	if (st != TAW_OK)
		return st;
	if (p.seekable) {
		taw_source in;
		taw_source_memory(&in, src, n);
		taw_buf b = {0};
		const taw_sink sink = {.write = buf_sink, .ctx = &b};
		st = taw_seek_encode(&in, &p, nullptr, 0, &sink);
		return finish(st, &b, out, out_n);
	}
	const size_t unit = n;
	taw_model_choice m;
	st = taw_model_fit(src, n, unit, &p, &m);
	if (st != TAW_OK)
		return st;
	taw_buf b = {0};
	st = encode_with(src, n, &p, &m, &b);
	if (st == TAW_OK && m.profile != TAW_PROFILE_INT_ORDER0 &&
	    p.predictor == TAW_PREDICTOR_AUTO) {
		/* Keep the mixer only if it beats the placeholder. */
		const taw_model_choice z = {.profile = TAW_PROFILE_INT_ORDER0,
			.quant = TAW_QUANT_NONE};
		taw_buf b0 = {0};
		if (encode_with(src, n, &p, &z, &b0) == TAW_OK && !b0.oom &&
		    b0.n <= b.n) {
			taw_buf_free(&b);
			b = b0;
		} else {
			taw_buf_free(&b0);
		}
	}
	free(m.model);
	return finish(st, &b, out, out_n);
}

#endif

#if TAW_HAVE_DECODER
static taw_status decompress_seekable(const void *src, size_t n,
    unsigned threads, void **out, size_t *out_n)
{
	taw_source in;
	taw_source_memory(&in, src, n);
	taw_seek *s;
	taw_status st = taw_seek_open(&in, &s);
	if (st != TAW_OK)
		return st;
	taw_seek_set_threads(s, threads);
	uint64_t total = taw_seek_size(s);
	uint8_t *dst = total < SIZE_MAX ? malloc(total ? (size_t)total : 1)
					: nullptr;
	if (!dst) {
		taw_seek_close(s);
		return TAW_E_NOMEM;
	}
	st = taw_seek_read(s, 0, dst, (size_t)total);
	taw_seek_close(s);
	if (st != TAW_OK) {
		free(dst);
		return st;
	}
	*out = dst;
	*out_n = (size_t)total;
	return TAW_OK;
}

taw_status taw_decompress(const void *src, size_t n, void **out,
    size_t *out_n)
{
	return taw_decompress_mt(src, n, 1, out, out_n);
}

taw_status taw_decompress_mt(const void *src, size_t n, unsigned threads,
    void **out, size_t *out_n)
{
	if (!src || !out || !out_n)
		return TAW_E_ARG;
	taw_info info;
	taw_status st = taw_read_info(src, n, &info);
	if (st != TAW_OK)
		return st;
	if (info.kind == TAW_KIND_SEEKABLE)
		return decompress_seekable(src, n, threads, out, out_n);
	taw_buf b = {0};
	st = taw_aw_decode(src, n, &b);
	return finish(st, &b, out, out_n);
}

#endif
taw_status taw_read_info(const void *src, size_t n, taw_info *info)
{
	if (!src || !info)
		return TAW_E_ARG;
	taw_header h;
	taw_status st = taw_header_parse(src, n, &h);
	if (st != TAW_OK)
		return st;
	*info = (taw_info){
		.kind = (taw_kind)h.kind,
		.has_files = (h.flags & TAW_FLAG_FILES) != 0,
		.has_dyn_blocks = (h.flags & TAW_FLAG_DYN_BLOCKS) != 0,
		.arith_profile = h.profile,
		.quant = h.quant,
		.level = h.level,
		.block_size = h.block_size,
		.model_size = h.model_size,
	};
	return TAW_OK;
}
