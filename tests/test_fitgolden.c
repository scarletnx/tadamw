/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Fitted-model goldens on the current main fitter.
 *
 * Source: td_fill(TD_TEXT, seed 1234), 1 MiB. MIX, GPU off. Levels 1 and 5
 * stay under fit_all's GPU threshold, so the fit is CPU-only and
 * deterministic. Values measured on origin/main @ 1aa3dd7 (GCC Release,
 * -ffp-contract=off -fno-fast-math) and recorded in council/e
 * REVIEW_ROLEPLAY.md. Re-baseline only if the fitter arithmetic or the build
 * flags change -- a silent golden update is a format regression.
 */
#include "fitgolden.h"

static const fgold base[] = {
	{1, TAW_WEIGHTS_F16, 262938, 1112, 0x197291E6u},
	{1, TAW_WEIGHTS_Q8,  262437,  608, 0x47F69FFBu},
	{5, TAW_WEIGHTS_F16, 262157,  568, 0xC9DED846u},
	{5, TAW_WEIGHTS_Q8,  262175,  608, 0xB0F10114u},
};

int main(void)
{
	fit_check("base", base, sizeof base / sizeof *base);
	return CHECK_DONE();
}
