/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Canonical harness: tests/fitgolden.h (no private copies).
 * Pins default-off, explicit off, EMA(0.999)/SWA/EMA(0.9), the `late` gate,
 * MALTER default-off, MALTER+EMA, and GPU averaging exclusion. */
#include "fitgolden.h"

static const fgold ema999_gold[] = {
	{1, TAW_WEIGHTS_F16, 262978, 1112, 0xE9694BFB},
	{3, TAW_WEIGHTS_F16, 261598,  568, 0xF0FF6B8A},
	{3, TAW_WEIGHTS_Q8,  262954, 1888, 0xB47C1123},
	{5, TAW_WEIGHTS_F16, 262749, 1752, 0x08782D54},
	{5, TAW_WEIGHTS_Q8,  262089,  608, 0x0AA9FB0F},
};
#define EMA999_N (sizeof ema999_gold / sizeof *ema999_gold)

static const fgold swa_gold[] = {
	{5, TAW_WEIGHTS_F16, 262978, 1752, 0x5F98BE8F},
};
static const fgold ema9_gold[] = {
	{5, TAW_WEIGHTS_Q8, 262182, 608, 0x5D2B02B4},
};
/* `late` = EMA(0.999) for >= 6 epochs: L1 equals baseline, L3/L5 equal
 * the explicit ema999 rows. This pins the gate, not just the math. */
static const fgold late_gold[] = {
	{1, TAW_WEIGHTS_F16, 262938, 1112, 0x197291E6},
	{3, TAW_WEIGHTS_F16, 261598,  568, 0xF0FF6B8A},
	{5, TAW_WEIGHTS_F16, 262749, 1752, 0x08782D54},
};
#define LATE_N (sizeof late_gold / sizeof *late_gold)

/* Absolute MALTER-path rows (L5/Q8, GPU off, no tuned LRs).
 * Inequality checks alone let any byte-changing averaging bug
 * pass; these pin the exact archive/model signature. */
static const fgold malter_off[] = {
	{5, TAW_WEIGHTS_Q8, 262754, 1912, 0xD7CD2BC4u},
};
static const fgold malter_ema999[] = {
	{5, TAW_WEIGHTS_Q8, 262700, 1912, 0x8AAD93D5u},
};

static bool same_sig(const fgold *a, const fgold *b)
{
	return a->arc_n == b->arc_n && a->model_size == b->model_size &&
	       a->model_crc == b->model_crc;
}

static void expect_diff(const char *what, const fgold *a, const fgold *b)
{
	if (same_sig(a, b)) {
		fprintf(stderr, "%s: model did not change (crc %08X)\n", what,
		    a->model_crc);
		check_failures++;
	}
}

/* MALTER must be off by default (unset AND "off"), and averaging must
 * still wrap the MALTER optimizer when asked. Previously only crc != crc
 * was asserted, so a MALTER default flip would have been invisible. */
static void malt_ab(void)
{
	fgold off, avged;
	void *arc;
	size_t an;

	fgold_env_scrub();
	fit_check("malter-default-off", &fitgolden_base[5], 1); /* L5 Q8 base */
	setenv("TAW_MALT", "off", 1);
	fit_check("malter-explicit-off", &fitgolden_base[5], 1);
	setenv("TAW_MALT", "malter", 1);
	setenv("TAW_FIT_AVG", "off", 1);
	fit_check("malter-on-golden", malter_off, 1); /* absolute */
	fit_probe(5, TAW_WEIGHTS_Q8, &off, &arc, &an);
	fit_roundtrip(arc, an);
	free(arc);
	expect_diff("malter-on", &off, &fitgolden_base[5]); /* supplement */
	setenv("TAW_FIT_AVG", "ema999", 1);
	fit_check("malter-plus-ema-golden", malter_ema999, 1); /* absolute */
	fit_probe(5, TAW_WEIGHTS_Q8, &avged, &arc, &an);
	fit_roundtrip(arc, an);
	free(arc);
	expect_diff("malter-plus-ema", &avged, &off); /* supplement */
	fgold_env_scrub();
}

/* Like fit_probe in fitgolden.h, but keeps the requested GPU mode so the
 * averaging-exclusion rule on GPU-trained candidates can be asserted. */
static taw_status gpu_probe(int level, taw_gpu_mode gm, fgold *g,
    void **arc_out, size_t *an_out)
{
	enum { N = 1 << 20 };
	const uint8_t *src = fit_source();
	taw_params p = {.level = level, .predictor = TAW_PREDICTOR_MIX,
		.weights = TAW_WEIGHTS_Q8, .gpu = gm};
	void *arc = nullptr;
	size_t an = 0;
	taw_status st = taw_compress(src, N, &p, &arc, &an);
	if (st != TAW_OK)
		return st;
	taw_info info;
	REQUIRE(taw_read_info(arc, an, &info) == TAW_OK);
	const uint8_t *a = arc;
	uint32_t crc = taw_crc32c(0, a + TAW_HEADER_SIZE,
	    (size_t)info.model_size);
	REQUIRE(crc == taw_rd32(a + 24));
	*g = (fgold){level, TAW_WEIGHTS_Q8, an, (uint64_t)info.model_size, crc};
	*arc_out = arc;
	*an_out = an;
	return TAW_OK;
}

/* GPU-trained candidates have no CPU trajectory, so TAW_FIT_AVG must be
 * ignored for them. On a GPU-less host assert the complementary rule: AUTO
 * falls back to CPU and averaging applies (matches the CPU ema999 golden). */
static void gpu_exclusion(void)
{
	fgold a = {0}, b = {0};
	void *arc1 = nullptr, *arc2 = nullptr;
	size_t an1 = 0, an2 = 0;

	fgold_env_scrub();
	setenv("TAW_FIT_AVG", "off", 1);
	taw_status st = gpu_probe(5, TAW_GPU_ON, &a, &arc1, &an1);
	if (st == TAW_E_GPU) {
		setenv("TAW_FIT_AVG", "ema999", 1);
		REQUIRE(gpu_probe(5, TAW_GPU_AUTO, &b, &arc2, &an2) == TAW_OK);
		fit_roundtrip(arc2, an2);
		if (!same_sig(&b, &ema999_gold[4])) {
			fprintf(stderr, "gpu fallback: AUTO+ema999 != cpu ema999 "
			    "(%zu/%llu/%08X)\n", b.arc_n,
			    (unsigned long long)b.model_size, b.model_crc);
			check_failures++;
		}
		fprintf(stderr, "gpu-exclusion: no GPU; AUTO fallback verified\n");
	} else {
		REQUIRE(st == TAW_OK);
		fit_roundtrip(arc1, an1);
		setenv("TAW_FIT_AVG", "ema999", 1);
		REQUIRE(gpu_probe(5, TAW_GPU_ON, &b, &arc2, &an2) == TAW_OK);
		fit_roundtrip(arc2, an2);
		if (!same_sig(&a, &b)) {
			fprintf(stderr, "gpu exclusion violated: EMA changed "
			    "GPU-trained model\n");
			check_failures++;
		}
	}
	free(arc1);
	free(arc2);
	fgold_env_scrub();
}

int main(void)
{
	fgold_env_scrub();
	/* Default must be upstream on all six rows, including level 3. */
	fit_check("ema-default", fitgolden_base, FITGOLDEN_BASE_N);
	setenv("TAW_FIT_AVG", "off", 1);
	fit_check("ema-off-explicit", fitgolden_base, FITGOLDEN_BASE_N);
	setenv("TAW_FIT_AVG", "ema999", 1);
	fit_check("ema-999", ema999_gold, EMA999_N);
	setenv("TAW_FIT_AVG", "swa", 1);
	fit_check("ema-swa", swa_gold, sizeof swa_gold / sizeof *swa_gold);
	setenv("TAW_FIT_AVG", "ema9", 1);
	fit_check("ema-9", ema9_gold, sizeof ema9_gold / sizeof *ema9_gold);
	setenv("TAW_FIT_AVG", "late", 1);
	fit_check("ema-late", late_gold, LATE_N);
	malt_ab();
	gpu_exclusion();
	fgold_env_scrub();
	return CHECK_DONE();
}
