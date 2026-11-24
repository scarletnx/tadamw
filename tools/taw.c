/* SPDX-License-Identifier: GPL-3.0-or-later */
/* taw: command-line front end for the tar_aw archive format. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tadamw/tadamw.h"

static const char *prog = "taw";

static void usage(FILE *f)
{
	fprintf(f,
#if TAW_HAVE_ENCODER
	    "usage: taw c [-l level] [-s] [-B block_size] [-p predictor] [-w weights]\n"
	    "             [-j threads] [-g gpu] [-D [-T target] [-C change]] <in> <out>\n"
#endif
#if TAW_HAVE_DECODER
	    "       taw d [-j threads] <in> <out>\n"
	    "       taw x [-j threads] <in.awseek> <offset> <length> <out>\n"
	    "       taw x [-j threads] <in.awseek> </file/name> <out>\n"
#endif
	    "       taw i <in>\n"
	    "       taw --version\n"
	    "\n"
	    "  c   compress (.awseek output or -s gives the seekable variant)\n"
	    "      -l 1..9          fitting effort (default 5)\n"
	    "      -p auto|order0|mix|cal|gate|hnet\n"
	    "                       predictor (default auto: the fitted mixer\n"
	    "                       if it beats order0). cal/gate/hnet are the\n"
	    "                       allocated predictor slots 6/11/14; they are\n"
	    "                       TAW_E_ARG until the owning profile lands.\n"
	    "                       Values 3..5, 7..10 and 12..13 are void.\n"
	    "      -w auto|f16|q8|q4|q2|fp8|nvfp4|iq8|iq4|iq2\n"
	    "                       mixer weight format (default auto)\n"
	    "      -j N             worker threads (default 1)\n"
	    "      -B N             .awseek nominal block size (power of two)\n"
	    "      -D               dynamic blocks: boundaries at content changes\n"
	    "                       (opt-in; needs -s or a .awseek output name)\n"
	    "      -T N             target block bytes (default: -B)\n"
	    "      -C N             change spike threshold 0..256 (0: ratio mode)\n"
	    "      -g off|auto|on   fit on a GPU (default off; auto falls back to\n"
	    "                       the CPU, on fails without a usable GPU)\n"
	    "  d   decompress a .aw or .awseek archive\n"
	    "      -j N             decode .awseek blocks on N threads (default 1;\n"
	    "                       each holds its own decoder tables)\n"
	    "  x   extract a byte range, or a named file, from a .awseek archive\n"
	    "  i   print header information\n"
	    "  '-' as <in>/<out> means stdin/stdout (not for x's <in>).\n");
}

static int fail(const char *what, const char *detail)
{
	fprintf(stderr, "%s: %s: %s\n", prog, what, detail);
	return 1;
}

static bool read_all(const char *path, uint8_t **out, size_t *out_n)
{
	FILE *f = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb");
	if (!f)
		return false;
	size_t cap = 1 << 16, n = 0;
	uint8_t *p = malloc(cap);
	bool ok = p != nullptr;
	while (ok) {
		if (n == cap) {
			uint8_t *q = cap <= SIZE_MAX / 2 ? realloc(p, cap * 2)
							 : nullptr;
			if (!q) {
				ok = false;
				break;
			}
			p = q;
			cap *= 2;
		}
		size_t k = fread(p + n, 1, cap - n, f);
		n += k;
		if (k == 0) {
			ok = !ferror(f);
			break;
		}
	}
	if (f != stdin)
		fclose(f);
	if (!ok) {
		free(p);
		return false;
	}
	*out = p;
	*out_n = n;
	return true;
}

static bool write_all(const char *path, const void *p, size_t n)
{
	FILE *f = strcmp(path, "-") == 0 ? stdout : fopen(path, "wb");
	if (!f)
		return false;
	bool ok = fwrite(p, 1, n, f) == n;
	if (f == stdout)
		ok = fflush(f) == 0 && ok;
	else
		ok = fclose(f) == 0 && ok;
	return ok;
}

[[maybe_unused]] static bool ends_with(const char *s, const char *suffix)
{
	size_t a = strlen(s), b = strlen(suffix);
	return a >= b && strcmp(s + a - b, suffix) == 0;
}

static bool parse_u64(const char *s, uint64_t *v)
{
	char *end;
	errno = 0;
	unsigned long long x = strtoull(s, &end, 0);
	if (errno || end == s || *end || *s == '-')
		return false;
	*v = x;
	return true;
}

#if TAW_HAVE_ENCODER
static const char *const weights[] = {"auto", "f16", "q8", "q4", "q2", "fp8",
	"nvfp4", "iq8", "iq4", "iq2", nullptr};
static const char *const gpus[] = {"off", "auto", "on", nullptr};

static bool parse_choice(const char *s, const char *const *names,
    uint64_t *v)
{
	for (uint64_t i = 0; names[i]; i++)
		if (strcmp(s, names[i]) == 0) {
			*v = i;
			return true;
		}
	return false;
}

/* Predictor values are NOT array indices: R-4 allocates them to equal the
 * arithmetic profile ID. A sparse name table keeps the CLI and the public
 * enum in agreement; value 3 has no name. */
static const struct {
	const char *name;
	taw_predictor value;
} predictor_names[] = {
	{"auto", TAW_PREDICTOR_AUTO},
	{"order0", TAW_PREDICTOR_ORDER0},
	{"mix", TAW_PREDICTOR_MIX},
	{"cal", TAW_PREDICTOR_CALIBRATED},
	{"gate", TAW_PREDICTOR_GATE},
	{"hnet", TAW_PREDICTOR_HNET},
};

static bool parse_predictor(const char *s, taw_predictor *out)
{
	for (size_t i = 0; i < sizeof predictor_names / sizeof *predictor_names;
	     i++)
		if (strcmp(s, predictor_names[i].name) == 0) {
			*out = predictor_names[i].value;
			return true;
		}
	return false;
}

#endif

static const char *profile_name(unsigned p)
{
	switch (p) {
	case 0: return "INT_ORDER0";
	case 1: return "F32_STRICT_MIX";
	case 6: return "CALIBRATED_MIX";
	case 11: return "ENTROPY_GATE";
	case 14: return "HNET_BOUNDARY";
	default: return "?";
	}
}

static const char *quant_name(unsigned q)
{
	static const char *const n[] = {"none", "f16", "bf16", "f32", "q8",
		"q4", "q2", "fp8", "nvfp4", "iq8", "iq4", "iq2"};
	return q < sizeof n / sizeof *n ? n[q] : "?";
}

#if TAW_HAVE_ENCODER
static int cmd_compress(int argc, char **argv)
{
	taw_params p = {0};
	int i = 0;
	for (; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
		uint64_t v;
		if (strcmp(argv[i], "-s") == 0) {
			p.seekable = true;
		} else if (strcmp(argv[i], "-D") == 0) {
			p.dyn_blocks = true;
		} else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= (1u << 26)) {
			p.chunk_target = (uint32_t)v;
			i++;
		} else if (strcmp(argv[i], "-C") == 0 && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v <= 256) {
			p.change_threshold = (uint32_t)v;
			i++;
		} else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v <= 9) {
			p.level = (int)v;
			i++;
		} else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc &&
		    parse_predictor(argv[i + 1], &p.predictor)) {
			i++;
		} else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc &&
		    parse_choice(argv[i + 1], weights, &v)) {
			p.weights = (taw_weights)v;
			i++;
		} else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc &&
		    parse_choice(argv[i + 1], gpus, &v)) {
			p.gpu = (taw_gpu_mode)v;
			i++;
		} else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= 256) {
			p.threads = (unsigned)v;
			i++;
		} else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v <= UINT32_MAX) {
			p.block_size = (uint32_t)v;
			i++;
		} else {
			usage(stderr);
			return 2;
		}
	}
	if (argc - i != 2) {
		usage(stderr);
		return 2;
	}
	const char *in = argv[i], *out = argv[i + 1];
	if (ends_with(out, ".awseek"))
		p.seekable = true;

	uint8_t *src;
	size_t n;
	if (!read_all(in, &src, &n))
		return fail(in, strerror(errno));
	void *arc;
	size_t arc_n;
	taw_status st = taw_compress(src, n, &p, &arc, &arc_n);
	free(src);
	if (st != TAW_OK)
		return fail("compress", taw_strerror(st));
	bool ok = write_all(out, arc, arc_n);
	free(arc);
	return ok ? 0 : fail(out, strerror(errno));
}

#endif

#if TAW_HAVE_DECODER
/* A leading "-j N"; returns the arguments consumed (0 or 2), -1 on error. */
static int parse_threads(int argc, char **argv, unsigned *threads)
{
	*threads = 1;
	if (argc >= 1 && strcmp(argv[0], "-j") == 0) {
		uint64_t v;
		if (argc < 2 || !parse_u64(argv[1], &v) || v < 1 || v > 256)
			return -1;
		*threads = (unsigned)v;
		return 2;
	}
	return 0;
}

static int cmd_decompress(int argc, char **argv)
{
	unsigned threads;
	const int used = parse_threads(argc, argv, &threads);
	if (used < 0) {
		usage(stderr);
		return 2;
	}
	argc -= used;
	argv += used;
	if (argc != 2) {
		usage(stderr);
		return 2;
	}
	uint8_t *arc;
	size_t n;
	if (!read_all(argv[0], &arc, &n))
		return fail(argv[0], strerror(errno));
	void *dst;
	size_t dst_n;
	taw_status st = taw_decompress_mt(arc, n, threads, &dst, &dst_n);
	free(arc);
	if (st != TAW_OK)
		return fail(argv[0], taw_strerror(st));
	bool ok = write_all(argv[1], dst, dst_n);
	free(dst);
	return ok ? 0 : fail(argv[1], strerror(errno));
}

static taw_status fd_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	int fd = *(int *)ctx;
	uint8_t *d = dst;
	while (n) {
		ssize_t k = pread(fd, d, n, (off_t)off);
		if (k <= 0) {
			if (k < 0 && errno == EINTR)
				continue;
			return TAW_E_IO;
		}
		d += k;
		off += (uint64_t)k;
		n -= (size_t)k;
	}
	return TAW_OK;
}

static int cmd_extract(int argc, char **argv)
{
	unsigned threads;
	const int used = parse_threads(argc, argv, &threads);
	if (used < 0) {
		usage(stderr);
		return 2;
	}
	argc -= used;
	argv += used;
	uint64_t off = 0, len = 0;
	const char *name = nullptr, *out;
	if (argc == 3 && argv[1][0] == '/') {
		name = argv[1];
		out = argv[2];
	} else if (argc == 4 && parse_u64(argv[1], &off) &&
	    parse_u64(argv[2], &len) && len <= SIZE_MAX) {
		out = argv[3];
	} else {
		usage(stderr);
		return 2;
	}
	int fd = open(argv[0], O_RDONLY);
	struct stat sb;
	if (fd < 0 || fstat(fd, &sb) != 0) {
		int e = errno;
		if (fd >= 0)
			close(fd);
		return fail(argv[0], strerror(e));
	}
	taw_source src = {.read_at = fd_read_at, .ctx = &fd,
		.size = (uint64_t)sb.st_size};
	taw_seek *s;
	taw_status st = taw_seek_open(&src, &s);
	if (st != TAW_OK) {
		close(fd);
		return fail(argv[0], taw_strerror(st));
	}
	taw_seek_set_threads(s, threads); /* pread is thread safe */
	if (name) {
		const taw_file *f;
		long k = taw_seek_find(s, name);
		if (k < 0 || taw_seek_files(s, &f) == 0 || f[k].size > SIZE_MAX) {
			taw_seek_close(s);
			close(fd);
			return fail(name, "no such file in the archive");
		}
		off = f[k].offset;
		len = f[k].size;
	}
	uint8_t *buf = malloc(len ? (size_t)len : 1);
	if (!buf)
		st = TAW_E_NOMEM;
	else
		st = taw_seek_read(s, off, buf, (size_t)len);
	taw_seek_close(s);
	close(fd);
	if (st != TAW_OK) {
		free(buf);
		return fail(argv[0], taw_strerror(st));
	}
	bool ok = write_all(out, buf, (size_t)len);
	free(buf);
	return ok ? 0 : fail(out, strerror(errno));
}

#endif

static int cmd_info(int argc, char **argv)
{
	if (argc != 1) {
		usage(stderr);
		return 2;
	}
	uint8_t hdr[32];
	FILE *f = strcmp(argv[0], "-") == 0 ? stdin : fopen(argv[0], "rb");
	if (!f)
		return fail(argv[0], strerror(errno));
	size_t k = fread(hdr, 1, sizeof hdr, f);
	if (f != stdin)
		fclose(f);
	taw_info info;
	taw_status st = taw_read_info(hdr, k, &info);
	if (st != TAW_OK)
		return fail(argv[0], taw_strerror(st));
	printf("kind:       %s\n",
	    info.kind == TAW_KIND_SEEKABLE ? "awseek" : "aw");
	printf("profile:    %u (%s)\n", info.arith_profile,
	    profile_name(info.arith_profile));
	printf("quant:      %u (%s)\n", info.quant, quant_name(info.quant));
	printf("level:      %u\n", info.level);
	if (info.kind == TAW_KIND_SEEKABLE) {
		printf("block_size: %" PRIu32 "\n", info.block_size);
		printf("dynamic:    %s\n", info.has_dyn_blocks ? "yes" : "no");
	}
	printf("model_size: %" PRIu64 "\n", info.model_size);
#if TAW_HAVE_DECODER
	if (info.has_files && strcmp(argv[0], "-") != 0) {
		int fd = open(argv[0], O_RDONLY);
		struct stat sb;
		taw_seek *s;
		if (fd >= 0 && fstat(fd, &sb) == 0) {
			taw_source src = {.read_at = fd_read_at, .ctx = &fd,
				.size = (uint64_t)sb.st_size};
			if (taw_seek_open(&src, &s) == TAW_OK) {
				const taw_file *fl;
				size_t nf = taw_seek_files(s, &fl);
				printf("files:      %zu\n", nf);
				for (size_t j = 0; j < nf; j++)
					printf("  %12" PRIu64 "  %s\n", fl[j].size,
					    fl[j].name);
				taw_seek_close(s);
			}
		}
		if (fd >= 0)
			close(fd);
	}
#endif
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 2 && strcmp(argv[1], "--version") == 0) {
		printf("taw %s\n", taw_version());
		return 0;
	}
	if (argc >= 2 && (strcmp(argv[1], "-h") == 0 ||
			     strcmp(argv[1], "--help") == 0)) {
		usage(stdout);
		return 0;
	}
	if (argc < 2) {
		usage(stderr);
		return 2;
	}
	const char *cmd = argv[1];
	argc -= 2;
	argv += 2;
#if TAW_HAVE_ENCODER
	if (strcmp(cmd, "c") == 0)
		return cmd_compress(argc, argv);
#endif
#if TAW_HAVE_DECODER
	if (strcmp(cmd, "d") == 0)
		return cmd_decompress(argc, argv);
	if (strcmp(cmd, "x") == 0)
		return cmd_extract(argc, argv);
#endif
	if (strcmp(cmd, "i") == 0)
		return cmd_info(argc, argv);
	usage(stderr);
	return 2;
}
