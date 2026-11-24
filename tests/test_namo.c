/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Feature goldens for NAMO / NAMO-D and the shared shrink-gate alias.
 * Level 5 has s1 > 1: those cells exercise the v2s/v2c/sig untie mirror. */
#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>

#include "fitgolden.h"

static const char *const knobs[] = {"TAW_HYBRID", "TAW_MALT", "TAW_MALTER",
    "TAW_EPS_MODE", "TAW_EPS_SIGN_BETA", "TAW_EPS_GATE_P",
    "TAW_NAMO_D_AXIS", "TAW_NAMO_WD", "TAW_NAMO_BETA", "TAW_NAMO_CLAMP",
    "TAW_NAMO_SCALE", "TAW_NAMO_NEST", "TAW_MUON_LR0", "TAW_MUON_LR1",
    "TAW_MUON_NEST", "TAW_VEC_LR0", "TAW_VEC_LR1"};

static void clear_knobs(void)
{
	for (size_t i = 0; i < sizeof knobs / sizeof *knobs; i++)
		unsetenv(knobs[i]);
}

static void put(const char *k, const char *v)
{
	REQUIRE(setenv(k, v, 1) == 0);
}

/* Re-baselined after the X3 shared-NS / shared-gate refactor. */
static const fgold namo[] = {
	{1, TAW_WEIGHTS_F16, 262491, 1112, 0xBDD1E363u},
	{1, TAW_WEIGHTS_Q8,  261988,  608, 0x8479754Cu},
	{5, TAW_WEIGHTS_F16, 263050, 1752, 0x8D502120u},
	{5, TAW_WEIGHTS_Q8,  262830, 1912, 0xB6AD329Du},
};
static const fgold namod[] = {
	{1, TAW_WEIGHTS_F16, 262450, 1112, 0x7E870997u},
	{1, TAW_WEIGHTS_Q8,  261941,  608, 0x4C6F6736u},
	{5, TAW_WEIGHTS_F16, 261973,  568, 0xA331EA04u},
	{5, TAW_WEIGHTS_Q8,  262830, 1912, 0x5744DC67u},
};
/* TAW_EPS_MODE=c3 aliases into the shared shrink gate with the same output order. */
static const fgold namo_gate[] = {
	{1, TAW_WEIGHTS_F16, 263090, 1112, 0xE08B6854u},
	{1, TAW_WEIGHTS_Q8,  262593,  608, 0xB4BA0916u},
	{5, TAW_WEIGHTS_F16, 261504,  568, 0xCFD4F585u},
	{5, TAW_WEIGHTS_Q8,  261703,  608, 0xFFC26241u},
};
static const fgold namod_row[] = {
	{5, TAW_WEIGHTS_Q8, 262803, 1912, 0xD8AF8BFFu},
};
static const fgold namo_nest[] = {
	{5, TAW_WEIGHTS_Q8, 262837, 1912, 0x3AB6AFBCu},
};
/* On-NAMO scale arm: TAW_NAMO_SCALE must actually change NAMO's fit. */
static const fgold namo_scale2[] = {
	{5, TAW_WEIGHTS_Q8, 262832, 1912, 0x93C6A25Au},
};

int main(void)
{
	fgold_env_scrub(); /* canonical scrub before any arm is configured */
	clear_knobs();
	fit_check("off", base_golden,
	    sizeof base_golden / sizeof *base_golden);

	/* The shared gate and every NAMO knob are inert off-NAMO. */
	clear_knobs();
	put("TAW_EPS_MODE", "c3");
	put("TAW_EPS_GATE_P", "2");
	put("TAW_NAMO_D_AXIS", "row");
	put("TAW_NAMO_WD", "0.01");
	put("TAW_NAMO_BETA", "0.999");
	put("TAW_NAMO_CLAMP", "0.42");
	put("TAW_NAMO_SCALE", "7");
	put("TAW_NAMO_NEST", "1");
	fit_check("gate-leak", base_golden,
	    sizeof base_golden / sizeof *base_golden);

	clear_knobs();
	put("TAW_HYBRID", "namo");
	fit_check("namo", namo, sizeof namo / sizeof *namo);

	clear_knobs();
	put("TAW_HYBRID", "namod");
	fit_check("namod", namod, sizeof namod / sizeof *namod);

	clear_knobs();
	put("TAW_HYBRID", "namo");
	put("TAW_EPS_MODE", "c3");
	fit_check("namo+c3", namo_gate, sizeof namo_gate / sizeof *namo_gate);

	clear_knobs();
	put("TAW_HYBRID", "namod");
	put("TAW_NAMO_D_AXIS", "row");
	fit_check("namod-row", namod_row, sizeof namod_row / sizeof *namod_row);

	clear_knobs();
	put("TAW_HYBRID", "namo");
	put("TAW_NAMO_NEST", "1");
	fit_check("namo-nest", namo_nest, sizeof namo_nest / sizeof *namo_nest);

	clear_knobs();
	put("TAW_HYBRID", "namo");
	put("TAW_NAMO_SCALE", "2");
	fit_check("namo-scale2", namo_scale2,
	    sizeof namo_scale2 / sizeof *namo_scale2);

	clear_knobs();
	return CHECK_DONE();
}
