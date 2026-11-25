/* SPDX-License-Identifier: GPL-3.0-or-later */
/* The ONLY test_fitgolden.c and the ONLY taw_test(fitgolden) registration.
 *
 * Pins the weights chosen by the real fitter, not the hand-built model in
 * test_p1.c. Every encoder-side knob must reproduce fitgolden_base[] with
 * its setting off. Source: td_fill(TD_TEXT, seed 1234), 1 MiB, MIX, GPU off;
 * levels 1/3/5 are CPU-only and deterministic. Re-baseline only if fitter
 * arithmetic or build flags change.
 */
#include "fitgolden.h"

int main(void)
{
	fgold_env_scrub();
	fit_check("base", fitgolden_base, FITGOLDEN_BASE_N);
	return CHECK_DONE();
}
