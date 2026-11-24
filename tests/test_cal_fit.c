/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 6 fitted-model arm. Includes the canonical harness; this file owns
 * no baseline table and no helper. TAW_CALIBRATED=6 is the allocated
 * predictor slot, TAW_CALIBRATED=1 remains an env alias, and the ablation
 * knobs pin the three isolated variants of the calibration head. */
#include "fitgolden.h"

/* TAW_PREDICTOR_CALIBRATED, all calibration parameters fitted. */
static const fgold cal[] = {
	{1, TAW_WEIGHTS_F16, 262359, 1120, 0x6D9232C4u},
	{1, TAW_WEIGHTS_Q8,  261857,  616, 0x02220DDBu},
	{5, TAW_WEIGHTS_F16, 261594,  576, 0x8008F788u},
	{5, TAW_WEIGHTS_Q8,  261640,  616, 0x35C818DCu},
};

/* Ablations on the two Q8 cells: identity head, fitted scalars only,
 * fitted APM blend only. */
static const fgold ident[] = {
	{1, TAW_WEIGHTS_Q8, 262438, 616, 0xF66E35E7u},
	{5, TAW_WEIGHTS_Q8, 262213, 616, 0x9917CFC5u},
};
static const fgold calonly[] = {
	{1, TAW_WEIGHTS_Q8, 262501, 616, 0xAEFDDAC2u},
	{5, TAW_WEIGHTS_Q8, 262288, 616, 0xC18420E0u},
};
static const fgold apmonly[] = {
	{1, TAW_WEIGHTS_Q8, 261839, 616, 0x6C98FDC6u},
	{5, TAW_WEIGHTS_Q8, 261648, 616, 0x03E107E4u},
};

static const fgold_opts p6 = {
	.pred = TAW_PREDICTOR_CALIBRATED, .profile = 6,
};

int main(void)
{
	fgold_env_scrub();
	_Static_assert(TAW_PREDICTOR_CALIBRATED == 6, "predictor slot");

	fit_check_ex("cal-on", cal, sizeof cal / sizeof *cal, &p6);

	/* Identity calibration and identity APM blend: isolates the extra
	 * 8 header bytes from the fitted values. */
	setenv("TAW_CAL_PARAMS", "256,0,2,1,1", 1);
	setenv("TAW_CAL_NO_APM", "1", 1);
	fit_check_ex("cal-identity", ident, sizeof ident / sizeof *ident, &p6);

	/* Fitted scalars from the winning L1/L5 candidates, identity blend. */
	setenv("TAW_CAL_PARAMS", "252,6,2,1,1", 1);
	fit_check_ex("cal-scalars-only", calonly,
	    sizeof calonly / sizeof *calonly, &p6);

	/* Identity scalars, APM weights still fitted on the probe. */
	setenv("TAW_CAL_PARAMS", "256,0,2,1,1", 1);
	unsetenv("TAW_CAL_NO_APM");
	fit_check_ex("cal-blend-only", apmonly,
	    sizeof apmonly / sizeof *apmonly, &p6);

	fgold_env_scrub();
	return CHECK_DONE();
}
