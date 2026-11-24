/* SPDX-License-Identifier: GPL-3.0-or-later */
/* PR #8: C3 sign gate. Direct gate math (foundation's single implementation)
 * plus fitted-model goldens. Knob-off must equal the CANONICAL base table.
 * Include the canonical harness FIRST: it defines _POSIX_C_SOURCE. */
#include "fitgolden.h"
#include "fit_opt.h"

static float absf_(float x) { return x < 0.0f ? -x : x; }

/* Measured on clang Release, x86-64, td_fill(TD_TEXT, 1234), 1 MiB. Tables
 * are assertions only; the canonical header has no print/bypass mode. */
/* Tuned AdamW + shrink gate: the PR's winning arm (external bench golden
 * 180047 / 178687 / 372296 / 368712 is pinned in RESULTS.md). */
static const fgold SHRINK[] = {
	{1, TAW_WEIGHTS_F16, 262835, 1112, 0xE67FE8A4u},
	{1, TAW_WEIGHTS_Q8, 262327, 608, 0xD5E7CF15u},
	{5, TAW_WEIGHTS_F16, 261709, 568, 0xA1CA8224u},
	{5, TAW_WEIGHTS_Q8, 261673, 608, 0x84D8C2B4u},
};
/* Plain (untuned) arms, cross-checked against council/e's measured table. */
static const fgold SHRINK_PLAIN[] = {
	{1, TAW_WEIGHTS_F16, 262756, 1112, 0xC2451483u},
	{5, TAW_WEIGHTS_Q8, 261609, 608, 0x76602E67u},
};
static const fgold MEAN_PLAIN[] = {
	{1, TAW_WEIGHTS_F16, 262870, 1112, 0x6428140Bu},
	{5, TAW_WEIGHTS_Q8, 262041, 608, 0xE46A8206u},
};
static const fgold MEAN[] = {
	{1, TAW_WEIGHTS_Q8, 261983, 608, 0x4457DB52u},
	{5, TAW_WEIGHTS_Q8, 262146, 608, 0xFF294E39u},
};
static const fgold LAYERS1[] = {
	{1, TAW_WEIGHTS_Q8, 261849, 608, 0xD8236A2Eu},
	{5, TAW_WEIGHTS_Q8, 262039, 608, 0xA1A5D534u},
};
static const fgold SCOPE_COLD[] = {{5, TAW_WEIGHTS_Q8, 262150, 608, 0xF457EEAFu}};
static const fgold BETA95[] = {{5, TAW_WEIGHTS_Q8, 261868, 608, 0x789E99C6u}};
static const fgold GATE_P2[] = {{5, TAW_WEIGHTS_Q8, 261648, 608, 0xCD905AC3u}};

static void tuned(void)
{
	setenv("TAW_LR0", "0.055", 1);
	setenv("TAW_LR1", "0.00055", 1);
	setenv("TAW_B1", "0.95", 1);
	setenv("TAW_B2", "0.999", 1);
	setenv("TAW_WD", "1e-3", 1);
}

static void gate_direct(void)
{
	float s = 0.0f;
	CHECK(absf_(fit_gate_step(&s, 1.0f, 0.9f, 1.0f) - 0.1f) < 1e-6f);
	CHECK(absf_(fit_gate_step(&s, 1.0f, 0.9f, 1.0f) - 0.19f) < 1e-6f);
	float sn = 0.0f;
	CHECK(absf_(fit_gate_step(&sn, -1.0f, 0.9f, 1.0f) - 0.1f) < 1e-6f);
	/* alternating signs shrink the EMA and hence the gate */
	CHECK(fit_gate_step(&sn, 1.0f, 0.9f, 1.0f) < 0.1f);
	float s2 = 0.0f;
	CHECK(absf_(fit_gate_step(&s2, 1.0f, 0.9f, 2.0f) - 0.01f) < 1e-6f);
}

int main(void)
{
	gate_direct();

	/* Every mask/knob must be observably different from the winning arm;
	 * otherwise its golden would be vacuous. All use the same L5/Q8 cell. */
	CHECK(SHRINK[3].model_crc != MEAN[1].model_crc);
	CHECK(SHRINK[3].model_crc != LAYERS1[1].model_crc);
	CHECK(SHRINK[3].model_crc != SCOPE_COLD[0].model_crc);
	CHECK(SHRINK[3].model_crc != BETA95[0].model_crc);
	CHECK(SHRINK[3].model_crc != GATE_P2[0].model_crc);
	CHECK(fitgolden_base[5].model_crc != SHRINK[3].model_crc);

	fgold_env_scrub();
	fit_check("base", fitgolden_base, FITGOLDEN_BASE_N);
	setenv("TAW_EPS_MODE", "none", 1);
	fit_check("off", fitgolden_base, FITGOLDEN_BASE_N);

	fgold_env_scrub();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	fit_check("shrink_plain", SHRINK_PLAIN,
	    sizeof SHRINK_PLAIN / sizeof *SHRINK_PLAIN);

	fgold_env_scrub();
	setenv("TAW_EPS_MODE", "c3", 1);
	fit_check("mean_plain", MEAN_PLAIN,
	    sizeof MEAN_PLAIN / sizeof *MEAN_PLAIN);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	fit_check("tuned_shrink", SHRINK, sizeof SHRINK / sizeof *SHRINK);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	fit_check("mean", MEAN, sizeof MEAN / sizeof *MEAN);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	setenv("TAW_EPS_LAYERS", "1", 1);
	fit_check("layers1", LAYERS1, sizeof LAYERS1 / sizeof *LAYERS1);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	setenv("TAW_EPS_SCOPE", "cold", 1);
	fit_check("scope_cold", SCOPE_COLD, 1);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	setenv("TAW_EPS_SIGN_BETA", "0.95", 1);
	fit_check("beta95", BETA95, 1);

	fgold_env_scrub();
	tuned();
	setenv("TAW_EPS_MODE", "c3", 1);
	setenv("TAW_EPS_GATE_MODE", "shrink", 1);
	setenv("TAW_EPS_GATE_P", "2", 1);
	fit_check("gate_p2", GATE_P2, 1);

	fgold_env_scrub();
	return CHECK_DONE();
}
