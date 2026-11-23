/* SPDX-License-Identifier: GPL-3.0-or-later */
/* GPU fitting (taw_params.gpu): AUTO always works; with a usable GPU, ON
 * gives archives that decode, are reproducible, do not depend on the
 * thread count, and compress about as well as the CPU fitter. Without a
 * GPU (or without TADAMW_GPU) the GPU cases are skipped (exit 77). */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "taw_internal.h"
#include "testdata.h"

static size_t pack(const uint8_t *src, size_t n, const taw_params *p,
    void **arc)
{
	size_t arc_n = 0;
	REQUIRE(taw_compress(src, n, p, arc, &arc_n) == TAW_OK);
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress(*arc, arc_n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == n && memcmp(src, dec, n) == 0);
	free(dec);
	return arc_n;
}

static bool same(const void *a, size_t an, const void *b, size_t bn)
{
	return an == bn && memcmp(a, b, an) == 0;
}

int main(void)
{
	enum { N = 1 << 18 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N / 2, TD_TEXT, 1);
	td_fill(src + N / 2, N / 2, TD_SKEWED, 2);

	taw_params p = {.level = 3, .predictor = TAW_PREDICTOR_MIX,
		.gpu = (taw_gpu_mode)7};
	void *arc;
	size_t arc_n;
	CHECK(taw_compress(src, N, &p, &arc, &arc_n) == TAW_E_ARG);

	/* AUTO: the GPU if there is one, else the CPU */
	p.gpu = TAW_GPU_AUTO;
	void *a_auto;
	pack(src, N, &p, &a_auto);
	free(a_auto);

	p.gpu = TAW_GPU_ON;
	taw_status st = taw_compress(src, N, &p, &arc, &arc_n);
	if (st == TAW_E_GPU) {
		fprintf(stderr, "no usable GPU (TAW_FIT_DEBUG=1 says why): "
				"GPU cases skipped\n");
		free(src);
		return check_failures ? 1 : 77;
	}
	REQUIRE(st == TAW_OK);
	free(arc);

	void *a_gpu, *a_gpu2, *a_cpu;
	const size_t n_gpu = pack(src, N, &p, &a_gpu);
	const size_t n_gpu2 = pack(src, N, &p, &a_gpu2);
	CHECK(same(a_gpu, n_gpu, a_gpu2, n_gpu2)); /* reproducible */
	p.gpu = TAW_GPU_OFF;
	const size_t n_cpu = pack(src, N, &p, &a_cpu);
	fprintf(stderr, "level 3: gpu %zu bytes, cpu %zu bytes\n", n_gpu, n_cpu);
	CHECK(n_gpu < n_cpu + n_cpu / 50); /* within 2% */
	free(a_gpu);
	free(a_gpu2);
	free(a_cpu);

	/* .awseek: the archive does not depend on the thread count */
	p = (taw_params){.level = 4, .seekable = true, .block_size = 1 << 16,
		.threads = 1, .gpu = TAW_GPU_ON};
	void *s1, *s3;
	const size_t n1 = pack(src, N, &p, &s1);
	p.threads = 3;
	const size_t n3 = pack(src, N, &p, &s3);
	CHECK(same(s1, n1, s3, n3));
	free(s1);
	free(s3);

	/* profile 2: the nets of both lane sets are trained on the GPU */
	p = (taw_params){.level = 3, .predictor = TAW_PREDICTOR_CHUNK,
		.gpu = TAW_GPU_ON};
	void *c_gpu, *c_cpu;
	const size_t nc_gpu = pack(src, N, &p, &c_gpu);
	p.gpu = TAW_GPU_OFF;
	const size_t nc_cpu = pack(src, N, &p, &c_cpu);
	fprintf(stderr, "profile 2, level 3: gpu %zu bytes, cpu %zu bytes\n",
	    nc_gpu, nc_cpu);
	CHECK(nc_gpu < nc_cpu + nc_cpu / 50);
	free(c_gpu);
	free(c_cpu);

	free(src);
	return CHECK_DONE();
}
