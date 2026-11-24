/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Tuned AdamW control. Includes the CANONICAL fitgolden.h and owns no
 * baseline table or helpers. fitgolden.h must be included first: it defines
 * _POSIX_C_SOURCE. */
#include "fitgolden.h"

/* TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 */
static const fgold tuned[] = {
	{1, TAW_WEIGHTS_F16, 262932, 1112, 0x406C2AD3u},
	{1, TAW_WEIGHTS_Q8,  262428,  608, 0x8294F2E3u},
	{5, TAW_WEIGHTS_F16, 262106,  568, 0x3117A208u},
	{5, TAW_WEIGHTS_Q8,  262178,  608, 0x52123DC9u},
};
#define TUNED_N (sizeof tuned / sizeof *tuned)

int main(void)
{
	fgold_env_scrub(); /* drop inherited TAW_* before setting this arm's knobs */
	setenv("TAW_LR0", "0.055", 1);
	setenv("TAW_LR1", "0.00055", 1);
	setenv("TAW_B1", "0.95", 1);
	setenv("TAW_B2", "0.999", 1);
	setenv("TAW_WD", "1e-3", 1);
	fit_check("tuned_lr", tuned, TUNED_N);
	return CHECK_DONE();
}
