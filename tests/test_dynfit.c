/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arm `dynfit`: fitted-model + container golden for dynamic .awseek blocks.
 * test_p1's golden pins a hand-built model and test_awseek's golden pins an
 * order-0 container; neither calls the fitter for the seek path, which uses
 * taw_model_fit_source() directly. These vectors pin both the fitted model
 * (size/CRC) and the dynamic container (archive size, index CRC, block count
 * and the first/last variable raw_len) at levels 1 and 5 for F16 and Q8,
 * plus byte-exact decode and thread independence. The canonical fitgolden.h
 * supplies fgold_env_scrub(); this arm keeps its own dfgold table and
 * dyn_fit_check() because the seek container needs more than fgold's fields.
 * No local fitgolden.h/test_fitgolden.c or duplicate target is defined. */
#include "fitgolden.h" /* canonical harness first: defines _POSIX_C_SOURCE */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int level;
	taw_weights w;
	size_t arc_n, model_size;
	uint32_t model_crc, index_crc;
	uint32_t nblocks, first_raw, last_raw;
} dfgold;

/* source: td_fill(TD_TEXT, 1 MiB, seed 1234), MIX, GPU off, dynamic blocks
 * with block_size = target = 64 KiB, 1 thread. Re-baseline only if the fitter
 * arithmetic or the §7.2 container changes. */
static const dfgold goldens[] = {
	{1, TAW_WEIGHTS_F16, 275927, 1112, 0x95BA8FD5u, 0x6C1F8D5Bu, 16, 87870, 60778},
	{1, TAW_WEIGHTS_Q8,  275425,  608, 0xB3CB1319u, 0x98CB82B2u, 16, 87870, 60778},
	{5, TAW_WEIGHTS_F16, 272968,  568, 0xF814D9B2u, 0x6C5474E0u, 16, 87870, 60778},
	{5, TAW_WEIGHTS_Q8,  273913, 1912, 0xB7954B2Fu, 0x817D7DDFu, 16, 87870, 60778},
};

static void dyn_fit_check(const char *arm, const dfgold *g, size_t ng)
{
	enum { N = 1 << 20, BS = 1 << 16 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < ng; i++) {
		taw_params p = {.seekable = true, .level = g[i].level,
			.predictor = TAW_PREDICTOR_MIX, .weights = g[i].w,
			.gpu = TAW_GPU_OFF, .block_size = BS, .dyn_blocks = true,
			.chunk_target = BS, .threads = 1};
		void *arc_v = nullptr;
		size_t an = 0;
		REQUIRE(taw_compress(src, N, &p, &arc_v, &an) == TAW_OK);
		uint8_t *a = arc_v;
		taw_info info;
		REQUIRE(taw_read_info(a, an, &info) == TAW_OK);
		const uint32_t mcrc = taw_crc32c(0, a + TAW_HEADER_SIZE,
		    (size_t)info.model_size);
		const uint32_t hcrc = taw_rd32(a + 24);
		const uint64_t ix = taw_rd64(a + an - TAW_FOOTER_SIZE);
		const size_t e0 = (size_t)ix + TAW_INDEX_HEAD;
		const uint32_t nb = taw_rd32(a + (size_t)ix + 4);
		const uint32_t icrc = taw_rd32(a + an - 8);
		uint64_t sum = 0;
		for (uint32_t b = 0; b < nb; b++)
			sum += taw_rd32(a + e0 + TAW_INDEX_ENTRY_DYN * (size_t)b + 16);
		const bool ok = an == g[i].arc_n &&
		    info.model_size == g[i].model_size &&
		    mcrc == g[i].model_crc && hcrc == mcrc &&
		    icrc == g[i].index_crc && nb == g[i].nblocks &&
		    sum == N &&
		    taw_rd32(a + e0 + 16) == g[i].first_raw &&
		    taw_rd32(a + e0 + TAW_INDEX_ENTRY_DYN * (size_t)(nb - 1) + 16) ==
		        g[i].last_raw &&
		    info.kind == TAW_KIND_SEEKABLE && info.has_dyn_blocks &&
		    info.block_size == BS;
		if (!ok) {
			fprintf(stderr,
			    "%s L%d %s: arc %zu/%zu model %llu/%llu mcrc %08X/%08X "
			    "icrc %08X/%08X blocks %u/%u\n", arm, g[i].level,
			    g[i].w == TAW_WEIGHTS_F16 ? "F16" : "Q8",
			    an, g[i].arc_n, (unsigned long long)info.model_size,
			    (unsigned long long)g[i].model_size, mcrc,
			    g[i].model_crc, icrc, g[i].index_crc, nb,
			    g[i].nblocks);
			check_failures++;
		}
		void *dec = nullptr;
		size_t dn = 0;
		CHECK(taw_decompress(a, an, &dec, &dn) == TAW_OK);
		CHECK(dn == N && memcmp(dec, src, N) == 0);
		free(dec);

		/* The seek path's fit and boundary pass are sequential, so the
		 * archive must not depend on the thread count. */
		if (g[i].level == 1) {
			p.threads = 6;
			void *b = nullptr;
			size_t bn = 0;
			REQUIRE(taw_compress(src, N, &p, &b, &bn) == TAW_OK);
			CHECK(bn == an && memcmp(b, a, an) == 0);
			free(b);
		}
		free(arc_v);
	}
	free(src);
}

int main(void)
{
	fgold_env_scrub(); /* inherit no TAW_* knob from the environment */
	dyn_fit_check("dyn", goldens, sizeof goldens / sizeof *goldens);
	return CHECK_DONE();
}
