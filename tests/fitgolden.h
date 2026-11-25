/* SPDX-License-Identifier: GPL-3.0-or-later */
/* CANONICAL fitted-model golden harness (integration/golden-canonical).
 *
 * test_p1.c:golden() pins a *hand-built* model, so it cannot see the fitter.
 * This header pins the output of the real encoder instead: archive length,
 * model section size and model CRC, plus a byte-exact decode round trip.
 *
 * Single source of truth. Every staged arm test includes THIS header; no PR
 * may carry its own fitgolden.h, baseline table, main(), or local fit_check.
 * Include this header FIRST in every test translation unit: it defines
 * _POSIX_C_SOURCE before any system header so unsetenv() is declared.
 */
#ifndef TAW_TESTS_FITGOLDEN_H
#define TAW_TESTS_FITGOLDEN_H

/* A translation unit may use only one of the helpers below (a base-only
 * test uses fit_check, an EMA-style arm uses fit_probe). Mark the shared
 * static helpers as intentionally unused where needed so -Werror builds do
 * not force each arm to fork this header or pass -Wno-unused-function.
 * Portable fallback for compilers without __attribute__((unused)). */
#if defined(__GNUC__) || defined(__clang__)
#define TAW_TEST_UNUSED __attribute__((unused))
#else
#define TAW_TEST_UNUSED
#endif

#define _POSIX_C_SOURCE 200809L /* C_EXTENSIONS OFF, C23: setenv/unsetenv */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"        /* TAW_HEADER_SIZE */
#include "taw_internal.h"  /* taw_crc32c, and tadamw/tadamw.h */
#include "testdata.h"

/* Fixed layout: all tables are positional, so field order must never change.
 * model_size stays uint64_t (header field width); PRs that used size_t can
 * copy their values verbatim. Predictor/profile live in fgold_opts below. */
typedef struct {
	int level;
	taw_weights w;
	size_t arc_n;
	uint64_t model_size;
	uint32_t model_crc;
} fgold;

/* For predictor/profile arms. NULL opts == plain MIX, no profile assertion. */
typedef struct {
	taw_predictor pred;     /* e.g. TAW_PREDICTOR_GATE / _HNET */
	uint8_t profile;        /* expected arith_profile; 0 = do not check */
} fgold_opts;

/* Remove every inherited TAW_* knob (including ones a future PR will add).
 * Collect first: unsetenv() may compact environ while we walk it. Every
 * baseline or arm main() must call this before setting its own knobs. */
static TAW_TEST_UNUSED void fgold_env_scrub(void)
{
	extern char **environ;
	char names[128][128];
	size_t nn = 0;
	if (!environ)
		return;
	for (char **e = environ; *e && nn < 128; e++) {
		const char *eq = strchr(*e, '=');
		if (!eq || (size_t)(eq - *e) < 4 || strncmp(*e, "TAW_", 4))
			continue;
		size_t n = (size_t)(eq - *e);
		if (n >= sizeof names[0])
			continue;
		memcpy(names[nn], *e, n);
		names[nn][n] = '\0';
		nn++;
	}
	for (size_t i = 0; i < nn; i++)
		unsetenv(names[i]);
}

/* One deterministic 1 MiB text source, shared by every probe. */
static TAW_TEST_UNUSED const uint8_t *fit_source(void)
{
	enum { N = 1 << 20 };
	static uint8_t *src;
	if (!src) {
		src = malloc(N);
		REQUIRE(src);
		td_fill(src, N, TD_TEXT, 1234);
	}
	return src;
}

/* Measured signature of one real fit. Kept separate from assertion so an
 * arm can measure and compare (e.g. #19's MALTER A/B) without a golden table. */
typedef struct {
	fgold g;
	uint8_t profile;
} fgold_seen;

static TAW_TEST_UNUSED void fgold_fit(int level, taw_weights w, taw_predictor pred,
    fgold_seen *seen, void **arc_out, size_t *an_out)
{
	enum { N = 1 << 20 };
	const uint8_t *src = fit_source();
	taw_params p = {.level = level, .predictor = pred,
		.weights = w, .gpu = TAW_GPU_OFF};
	void *arc = NULL;
	size_t an = 0;
	REQUIRE(taw_compress(src, N, &p, &arc, &an) == TAW_OK);
	taw_info info;
	REQUIRE(taw_read_info(arc, an, &info) == TAW_OK);
	const uint8_t *a = arc;
	uint32_t crc = taw_crc32c(0, a + TAW_HEADER_SIZE,
	    (size_t)info.model_size);
	uint32_t hcrc = taw_rd32(a + 24);
	REQUIRE(crc == hcrc); /* header must agree with the model bytes */
	*seen = (fgold_seen){{level, w, an, (uint64_t)info.model_size, crc},
	    info.arith_profile};
	*arc_out = arc;
	*an_out = an;
}

/* #19-compatible helper names: an arm that used the old header compiles
 * unchanged when it includes this one. */
static TAW_TEST_UNUSED void fit_probe(int level, taw_weights w, fgold *g, void **arc_out,
    size_t *an_out)
{
	fgold_seen seen;
	fgold_fit(level, w, TAW_PREDICTOR_MIX, &seen, arc_out, an_out);
	*g = seen.g;
}

static TAW_TEST_UNUSED void fit_roundtrip(const void *arc, size_t an)
{
	enum { N = 1 << 20 };
	const uint8_t *src = fit_source();
	void *dec = NULL;
	size_t dn = 0;
	REQUIRE(taw_decompress(arc, an, &dec, &dn) == TAW_OK);
	CHECK(dn == N && memcmp(dec, src, N) == 0);
	free(dec);
}

/* Assert every row's signature, with optional predictor/profile checks, and
 * require the archive to round-trip. */
static TAW_TEST_UNUSED void fit_check_ex(const char *arm, const fgold *g, size_t ng,
    const fgold_opts *o)
{
	taw_predictor pred = o ? o->pred : TAW_PREDICTOR_MIX;
	uint8_t want_profile = o ? o->profile : 0;
	for (size_t i = 0; i < ng; i++) {
		fgold_seen seen;
		void *arc = NULL;
		size_t an = 0;
		fgold_fit(g[i].level, g[i].w, pred, &seen, &arc, &an);
		if (seen.g.arc_n != g[i].arc_n ||
		    seen.g.model_size != g[i].model_size ||
		    seen.g.model_crc != g[i].model_crc ||
		    (want_profile && seen.profile != want_profile)) {
			fprintf(stderr,
			    "%s lv%d %s pred=%d: arc %zu/%zu model %llu/%llu "
			    "crc %08X/%08X profile %u/%u\n", arm, g[i].level,
			    g[i].w == TAW_WEIGHTS_F16 ? "F16" : "Q8",
			    (int)pred, seen.g.arc_n, g[i].arc_n,
			    (unsigned long long)seen.g.model_size,
			    (unsigned long long)g[i].model_size,
			    seen.g.model_crc, g[i].model_crc, seen.profile,
			    want_profile);
			check_failures++;
		}
		fit_roundtrip(arc, an);
		free(arc);
	}
}

static TAW_TEST_UNUSED void fit_check(const char *arm, const fgold *g, size_t ng)
{
	fit_check_ex(arm, g, ng, NULL);
}

/* The one canonical baseline table. Levels 1/5 from council/e; the two L3
 * rows were re-measured on main during reconciliation (0x1388A8EA /
 * 0x1A6E3799) and cover the >=6-epoch gate. */
static const fgold fitgolden_base[] = {
	{1, TAW_WEIGHTS_F16, 262938, 1112, 0x197291E6u},
	{1, TAW_WEIGHTS_Q8,  262437,  608, 0x47F69FFBu},
	{3, TAW_WEIGHTS_F16, 261701,  568, 0x1388A8EAu},
	{3, TAW_WEIGHTS_Q8,  263030, 1888, 0x1A6E3799u},
	{5, TAW_WEIGHTS_F16, 262157,  568, 0xC9DED846u},
	{5, TAW_WEIGHTS_Q8,  262175,  608, 0xB0F10114u},
};
#define FITGOLDEN_BASE_N (sizeof fitgolden_base / sizeof *fitgolden_base)

/* #6 NAMO's staged test uses this name; keep it as an alias, not a second
 * table. sizeof / sizeof * work through the macro exactly as before. */
#define base_golden fitgolden_base

/* No TAW_GOLDEN_PRINT: an env var must never turn goldens into no-ops. */

#endif
