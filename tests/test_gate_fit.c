/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 11 arm of the canonical fitted-model golden harness. Tables are
 * 5-field fgold rows; predictor/profile live in fgold_opts (no private
 * fitgolden.h, no baseline table copy). */
#include "fitgolden.h"

static const fgold gate[] = {
	{1, TAW_WEIGHTS_F16, 262892, 1116, 0xF6B8493Fu},
	{1, TAW_WEIGHTS_Q8,  262391,  612, 0xEE0B36ACu},
	{5, TAW_WEIGHTS_F16, 262124,  572, 0x4D5E3E3Eu},
	{5, TAW_WEIGHTS_Q8,  262181,  612, 0x1C43247Fu},
};

static const fgold_opts gate_opts = {TAW_PREDICTOR_GATE, 11};

int main(void)
{
	fgold_env_scrub();
	fit_check_ex("gate", gate, sizeof gate / sizeof *gate, &gate_opts);
	return CHECK_DONE();
}
