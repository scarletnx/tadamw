/* SPDX-License-Identifier: GPL-3.0-or-later */
/* hnet_fit arm of the CANONICAL fitgolden harness. Includes the shared
 * header only: no private table, helper or main for the baseline profile.
 * Pins four real profile-14 fits (levels 1/5, F16/Q8). */
#include "fitgolden.h"

int main(void)
{
	static const fgold hnet[] = {
		{1, TAW_WEIGHTS_F16, 263054, 1144, 0x9596B02Eu},
		{1, TAW_WEIGHTS_Q8, 262529, 624, 0xF049F05Fu},
		{5, TAW_WEIGHTS_F16, 262243, 584, 0x392BACDFu},
		{5, TAW_WEIGHTS_Q8, 262283, 624, 0xFC301D97u},
	};
	static const fgold_opts o = {
		.pred = TAW_PREDICTOR_HNET, .profile = 14
	};
	fgold_env_scrub();
	fit_check_ex("hnet", hnet, sizeof hnet / sizeof *hnet, &o);
	return CHECK_DONE();
}
