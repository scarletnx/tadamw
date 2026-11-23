/* SPDX-License-Identifier: GPL-3.0-or-later */
/* libtadamw public API (functions and types keep the taw_ prefix). The
 * library is C23 inside; this header also works from C99, C11 and C++.
 *
 * Which halves are built is in <tadamw/config.h>: TAW_HAVE_ENCODER and
 * TAW_HAVE_DECODER (CMake options TADAMW_ENCODER / TADAMW_DECODER). Only the
 * functions of the halves built are declared. */
#ifndef TADAMW_TADAMW_H
#define TADAMW_TADAMW_H

#include <stddef.h>
#include <stdint.h>
#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L)
#include <stdbool.h>
#endif

#include <tadamw/config.h>

/* The library's exported symbols; everything else stays hidden. */
#if defined(__GNUC__) || defined(__clang__)
#define TAW_API __attribute__((visibility("default")))
#else
#define TAW_API
#endif

/* A status that must be checked. */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || \
    (defined(__cplusplus) && __cplusplus >= 201703L)
#define TAW_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#define TAW_NODISCARD __attribute__((warn_unused_result))
#else
#define TAW_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TAW_VERSION_MAJOR 0
#define TAW_VERSION_MINOR 1
#define TAW_VERSION_PATCH 0

typedef enum {
	TAW_OK = 0,
	TAW_E_ARG,         /* invalid argument */
	TAW_E_NOMEM,       /* allocation failed */
	TAW_E_FORMAT,      /* not a tar_aw archive, or structurally corrupt */
	TAW_E_UNSUPPORTED, /* valid but uses a version/profile/quant we refuse */
	TAW_E_CHECKSUM,    /* CRC mismatch */
	TAW_E_RANGE,       /* requested range outside the archive's data */
	TAW_E_IO,          /* read/write callback failed */
	TAW_E_GPU,         /* TAW_GPU_ON, but no usable GPU */
} taw_status;

/* Library version string, e.g. "0.1.0". */
TAW_API const char *taw_version(void);

/* Human-readable text for a status code. */
TAW_API const char *taw_strerror(taw_status s);

/* ---- compression -------------------------------------------------------- */

#define TAW_LEVEL_MIN 1
#define TAW_LEVEL_MAX 9
#define TAW_LEVEL_DEFAULT 5
#define TAW_BLOCK_DEFAULT (UINT32_C(1) << 20)

/* Model fitting on a GPU (Vulkan, loaded at run time; TAW_HAVE_GPU says
 * whether this build has it). OFF fits on the CPU. AUTO uses a GPU when one
 * works and the CPU otherwise. ON fails with TAW_E_GPU without one. A GPU
 * fit may give a different (equally valid) archive than a CPU fit. */
typedef enum {
	TAW_GPU_OFF = 0,
	TAW_GPU_AUTO,
	TAW_GPU_ON,
} taw_gpu_mode;

/* Predictor choice. AUTO fits the mixer and keeps it only if the archive
 * comes out smaller than with the order-0 placeholder. The predictor is an
 * encoder-side API value, not an archive field (see docs/format.md §1).
 *
 * Integration ruling R-4: a new predictor's value equals its arithmetic
 * profile ID. AUTO/ORDER0/MIX are grandfathered at 0/1/2. Values 3..5,
 * 7..10, 12..13 and 15..255 are refused (TAW_E_ARG); value 3 from the early
 * prototypes is void — never allocated, never shipped in a merged build. */
typedef enum {
	TAW_PREDICTOR_AUTO = 0,
	TAW_PREDICTOR_ORDER0 = 1, /* legacy value; arith profile 0 */
	TAW_PREDICTOR_MIX = 2,    /* legacy value; arith profile 1; TAW_E_ARG
	                             if the input is too small for any model */
	TAW_PREDICTOR_CALIBRATED = 6, /* arith profile 6, PR #10 */
	TAW_PREDICTOR_GATE = 11,      /* arith profile 11, PR #11 */
	TAW_PREDICTOR_HNET = 14,      /* arith profile 14, PR #13 */
} taw_predictor;

/* Weight format for the mixer (docs/format.md §3). AUTO lets the encoder
 * choose among all of them. */
typedef enum {
	TAW_WEIGHTS_AUTO = 0,
	TAW_WEIGHTS_F16,
	TAW_WEIGHTS_Q8,
	TAW_WEIGHTS_Q4,    /* 4-bit integers, row scale */
	TAW_WEIGHTS_Q2,    /* 2-bit integers, row scale and offset */
	TAW_WEIGHTS_FP8,   /* E4M3, row scale */
	TAW_WEIGHTS_NVFP4, /* E2M1 in 16-weight blocks with E4M3 scales */
	TAW_WEIGHTS_IQ8,   /* 8-bit non-linear table, row scale */
	TAW_WEIGHTS_IQ4,   /* 4-bit non-linear table (IQ4_NL), row scale */
	TAW_WEIGHTS_IQ2,   /* 2-bit non-linear table, row scale */
} taw_weights;

typedef struct {
	int level;           /* fitting effort 1..9; 0 = TAW_LEVEL_DEFAULT */
	bool seekable;       /* produce .awseek instead of .aw */
	uint32_t block_size; /* .awseek block size (power of two, 4 KiB..1 GiB);
	                        0 = TAW_BLOCK_DEFAULT */
	taw_predictor predictor;
	taw_weights weights;
	unsigned threads;    /* worker threads for .awseek (fitting and blocks);
	                        0 = 1. The archive does not depend on it. */
	taw_gpu_mode gpu;    /* fit on a GPU; default TAW_GPU_OFF */
} taw_params;

/* ---- shared types --------------------------------------------------------- */

typedef enum { TAW_KIND_STREAM = 0, TAW_KIND_SEEKABLE = 1 } taw_kind;

typedef struct {
	taw_kind kind;
	bool has_files;      /* .awseek with a file directory */
	uint8_t arith_profile;
	uint8_t quant;
	uint8_t level;
	uint32_t block_size; /* 0 for streams */
	uint64_t model_size;
} taw_info;

/* Parse and validate the 32-byte header at the start of an archive. */
TAW_NODISCARD TAW_API taw_status taw_read_info(const void *src, size_t n,
    taw_info *info);

/* A random-access byte source. read_at must fill exactly n bytes at off;
 * the library never asks for bytes past size. */
typedef struct {
	taw_status (*read_at)(void *ctx, uint64_t off, void *dst, size_t n);
	void *ctx;
	uint64_t size;
} taw_source;

/* Source over a memory buffer (which must outlive the source). */
TAW_API void taw_source_memory(taw_source *s, const void *p, size_t n);

/* A sequential byte sink. */
typedef struct {
	taw_status (*write)(void *ctx, const void *p, size_t n);
	void *ctx;
} taw_sink;

/* A named byte range of a .awseek archive's data (docs/format.md §7.1). */
typedef struct {
	const char *name; /* UTF-8 path starting with '/' */
	uint64_t offset;  /* in the decoded data */
	uint64_t size;
} taw_file;

#if TAW_HAVE_ENCODER
/* ---- encoder --------------------------------------------------------------- */

/* Compress src[0..n) into a new malloc'd buffer *out of *out_n bytes.
 * params may be NULL for defaults. */
TAW_NODISCARD TAW_API taw_status taw_compress(const void *src, size_t n,
    const taw_params *params, void **out, size_t *out_n);

/* Build a .awseek from src (src->size bytes, read with read_at, which must be
 * safe to call from several threads) and write it to out in order. files
 * (nfiles, may be 0) become the directory: sorted, non-overlapping ranges
 * inside the data, names as in docs/format.md §7.1. Memory stays bounded by
 * a few blocks per thread, so the input may be far larger than memory.
 * params->seekable is implied; params->threads sets the parallelism, and the
 * archive does not depend on it. */
TAW_NODISCARD TAW_API taw_status taw_seek_encode(const taw_source *src,
    const taw_params *params, const taw_file *files, size_t nfiles,
    const taw_sink *out);
#endif

#if TAW_HAVE_DECODER
/* ---- decoder --------------------------------------------------------------- */

/* Decompress a whole .aw or .awseek archive into a new malloc'd buffer. */
TAW_NODISCARD TAW_API taw_status taw_decompress(const void *src, size_t n,
    void **out, size_t *out_n);

/* The same, decoding .awseek blocks on up to `threads` threads (0 or 1 =
 * this thread only; .aw streams always decode on one). Each thread holds
 * its own decoder tables (docs/format.md §5.1 gives their size). */
TAW_NODISCARD TAW_API taw_status taw_decompress_mt(const void *src, size_t n,
    unsigned threads, void **out, size_t *out_n);

typedef struct taw_seek taw_seek;

/* Open a .awseek archive: reads and validates header, model and index. The
 * source is copied; its ctx must stay valid until taw_seek_close. A taw_seek
 * is not thread safe; open one per thread (they may share a source whose
 * read_at is thread safe). */
TAW_NODISCARD TAW_API taw_status taw_seek_open(const taw_source *src,
    taw_seek **out);

/* Let taw_seek_read decode the blocks of one request on up to `threads`
 * threads (0 or 1: serial, the default). The source's read_at must then be
 * safe to call from several threads. Each thread holds its own decoder
 * tables. */
TAW_API void taw_seek_set_threads(taw_seek *s, unsigned threads);

/* Uncompressed size of the archive's data. */
TAW_API uint64_t taw_seek_size(const taw_seek *s);

/* Read uncompressed bytes [off, off+n), decoding only overlapping blocks. */
TAW_NODISCARD TAW_API taw_status taw_seek_read(taw_seek *s, uint64_t off,
    void *dst, size_t n);

/* The archive's files, in offset order (none without a directory). The array
 * lives as long as the taw_seek. */
TAW_API size_t taw_seek_files(const taw_seek *s, const taw_file **files);

/* Index of the file called name, or -1. */
TAW_API long taw_seek_find(const taw_seek *s, const char *name);

TAW_API void taw_seek_close(taw_seek *s);
#endif

#ifdef __cplusplus
}
#endif

#endif /* TADAMW_TADAMW_H */
