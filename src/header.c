/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <string.h>

#include "header.h"

void taw_header_write(taw_buf *b, const taw_header *h)
{
	uint8_t p[TAW_HEADER_SIZE] = {0};
	memcpy(p, TAW_MAGIC, 4);
	p[4] = TAW_FORMAT_VERSION;
	p[5] = h->kind;
	p[6] = h->profile;
	p[7] = h->quant;
	p[8] = h->level;
	p[9] = h->flags;
	/* p[10..11] reserved: zero */
	taw_wr32(p + 12, h->block_size);
	taw_wr64(p + 16, h->model_size);
	taw_wr32(p + 24, h->model_crc);
	taw_wr32(p + 28, taw_crc32c(0, p, 28));
	taw_buf_put(b, p, sizeof p);
}

static bool is_pow2(uint32_t v)
{
	return v && !(v & (v - 1));
}

taw_status taw_header_parse(const uint8_t *p, size_t n, taw_header *h)
{
	if (n < TAW_HEADER_SIZE || memcmp(p, TAW_MAGIC, 4) != 0)
		return TAW_E_FORMAT;
	if (taw_rd32(p + 28) != taw_crc32c(0, p, 28))
		return TAW_E_CHECKSUM;
	if (p[4] != TAW_FORMAT_VERSION)
		return TAW_E_UNSUPPORTED;
	if ((p[9] & ~TAW_FLAG_ALL) || taw_rd16(p + 10) != 0)
		return TAW_E_UNSUPPORTED;
	if ((p[9] & TAW_FLAG_ALL) && p[5] != TAW_KIND_AWSEEK)
		return TAW_E_UNSUPPORTED;
	*h = (taw_header){
		.kind = p[5],
		.profile = p[6],
		.quant = p[7],
		.level = p[8],
		.flags = p[9],
		.block_size = taw_rd32(p + 12),
		.model_size = taw_rd64(p + 16),
		.model_crc = taw_rd32(p + 24),
	};
	switch (h->kind) {
	case TAW_KIND_AW:
		if (h->block_size != 0)
			return TAW_E_FORMAT;
		break;
	case TAW_KIND_AWSEEK:
		if (!is_pow2(h->block_size) || h->block_size < TAW_BLOCK_MIN ||
		    h->block_size > TAW_BLOCK_MAX)
			return TAW_E_FORMAT;
		break;
	default:
		return TAW_E_UNSUPPORTED;
	}
	if (!taw_pred_supported(h->profile, h->quant))
		return TAW_E_UNSUPPORTED;
	return TAW_OK;
}

taw_status taw_params_resolve(const taw_params *in, taw_params *out)
{
	*out = in ? *in : (taw_params){0};
	if (out->level == 0)
		out->level = TAW_LEVEL_DEFAULT;
	if (out->level < TAW_LEVEL_MIN || out->level > TAW_LEVEL_MAX)
		return TAW_E_ARG;
	/* R-4 predictor policy: value 3 is void (bad argument); 6/11/14 are
	 * allocated to sibling profiles that are not implemented here, so they
	 * report TAW_E_UNSUPPORTED as on #11/#13. Implemented 0/1/2 pass.
	 * This replaces the old implicit "> MIX" upper bound. */
	switch (out->predictor) {
	case TAW_PREDICTOR_AUTO:
	case TAW_PREDICTOR_ORDER0:
	case TAW_PREDICTOR_MIX:
		break;
	case TAW_PREDICTOR_CALIBRATED:
	case TAW_PREDICTOR_GATE:
	case TAW_PREDICTOR_HNET:
		return TAW_E_UNSUPPORTED;
	default:
		return TAW_E_ARG;
	}
	if ((unsigned)out->weights > TAW_WEIGHTS_IQ2)
		return TAW_E_ARG;
	if ((unsigned)out->gpu > TAW_GPU_ON)
		return TAW_E_ARG;
	if (out->block_size == 0)
		out->block_size = TAW_BLOCK_DEFAULT;
	if (out->seekable &&
	    (!is_pow2(out->block_size) || out->block_size < TAW_BLOCK_MIN ||
		out->block_size > TAW_BLOCK_MAX))
		return TAW_E_ARG;
	if (out->dyn_blocks && (!out->seekable || out->chunk_target > TAW_CHUNK_MAX ||
	    out->change_threshold > 256))
		return TAW_E_ARG;
	return TAW_OK;
}

taw_status taw_open_common(const uint8_t *src, size_t n, uint8_t want_kind,
    taw_header *h, const uint8_t **model)
{
	taw_status st = taw_header_parse(src, n, h);
	if (st != TAW_OK)
		return st;
	if (h->kind != want_kind)
		return TAW_E_ARG;
	if (h->model_size > n - TAW_HEADER_SIZE)
		return TAW_E_FORMAT;
	*model = src + TAW_HEADER_SIZE;
	if (taw_crc32c(0, *model, (size_t)h->model_size) != h->model_crc)
		return TAW_E_CHECKSUM;
	return TAW_OK;
}
