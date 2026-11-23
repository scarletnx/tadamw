/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Reusable fitted-model golden helper.
 *
 * test_p1.c:golden() pins a *hand-built* model, so it is blind to the fitter.
 * fit_check() instead runs taw_compress() (MIX, GPU off) on a deterministic
 * in-repo 1 MiB source and pins the archive length, the model section size,
 * and the model CRC (both computed from the bytes and read from the header),
 * then verifies a full decode round trip.
 *
 * Arms that set environment knobs should define _POSIX_C_SOURCE before any
 * include and use setenv/unsetenv in their own main(), then call fit_check()
 * with that arm's table. This header deliberately does not touch the
 * environment, so one process can check several arms in sequence.
 */
#ifndef TAW_TESTS_FITGOLDEN_H
#define TAW_TESTS_FITGOLDEN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "check.h"
#include "format.h"        /* TAW_HEADER_SIZE */
#include "taw_internal.h"  /* taw_crc32c, and tadamw/tadamw.h */
#include "testdata.h"

typedef struct {
	int level;
	taw_weights w;
	size_t arc_n;
	uint64_t model_size;
	uint32_t model_crc;
} fgold;

static void fit_check(const char *arm, const fgold *g, size_t ng)
{
	enum { N = 1 << 20 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 1234);
	for (size_t i = 0; i < ng; i++) {
		taw_params p = {.level = g[i].level,
			.predictor = TAW_PREDICTOR_MIX,
			.weights = g[i].w, .gpu = TAW_GPU_OFF};
		void *arc = NULL;
		size_t an = 0;
		REQUIRE(taw_compress(src, N, &p, &arc, &an) == TAW_OK);
		taw_info info;
		REQUIRE(taw_read_info(arc, an, &info) == TAW_OK);
		const uint8_t *a = arc;
		uint32_t crc = taw_crc32c(0, a + TAW_HEADER_SIZE,
		    (size_t)info.model_size);
		uint32_t hcrc = (uint32_t)a[24] | (uint32_t)a[25] << 8 |
		    (uint32_t)a[26] << 16 | (uint32_t)a[27] << 24;
		if (an != g[i].arc_n || info.model_size != g[i].model_size ||
		    crc != g[i].model_crc || crc != hcrc) {
			fprintf(stderr,
			    "%s lv%d %s: arc %zu/%zu model %llu/%llu "
			    "crc %08X/%08X header %08X\n", arm, g[i].level,
			    g[i].w == TAW_WEIGHTS_F16 ? "F16" : "Q8",
			    an, g[i].arc_n,
			    (unsigned long long)info.model_size,
			    (unsigned long long)g[i].model_size, crc,
			    g[i].model_crc, hcrc);
			check_failures++;
		}
		/* A wrong optimizer must not ship a model that fails to decode. */
		void *dec = NULL;
		size_t dn = 0;
		REQUIRE(taw_decompress(arc, an, &dec, &dn) == TAW_OK);
		CHECK(dn == N && memcmp(dec, src, N) == 0);
		free(dec);
		free(arc);
	}
	free(src);
}

#endif
