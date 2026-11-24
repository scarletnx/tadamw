/* SPDX-License-Identifier: GPL-3.0-or-later */
/* On-disk constants (docs/format.md). */
#ifndef TAW_FORMAT_H
#define TAW_FORMAT_H

#include <stdint.h>

constexpr uint8_t TAW_FORMAT_VERSION = 1;

constexpr uint8_t TAW_KIND_AW = 0;
constexpr uint8_t TAW_KIND_AWSEEK = 1;

constexpr uint8_t TAW_PROFILE_INT_ORDER0 = 0;
constexpr uint8_t TAW_PROFILE_F32_STRICT_MIX = 1; /* reserved */

constexpr uint8_t TAW_QUANT_NONE = 0;
constexpr uint8_t TAW_QUANT_F16 = 1;
constexpr uint8_t TAW_QUANT_BF16 = 2;
constexpr uint8_t TAW_QUANT_F32 = 3;
constexpr uint8_t TAW_QUANT_Q8 = 4;
constexpr uint8_t TAW_QUANT_Q4 = 5;
constexpr uint8_t TAW_QUANT_Q2 = 6;
constexpr uint8_t TAW_QUANT_FP8 = 7;   /* E4M3 */
constexpr uint8_t TAW_QUANT_NVFP4 = 8; /* E2M1, E4M3 block scales */
constexpr uint8_t TAW_QUANT_IQ8 = 9;
constexpr uint8_t TAW_QUANT_IQ4 = 10;
constexpr uint8_t TAW_QUANT_IQ2 = 11;
constexpr uint8_t TAW_QUANT_COUNT = 12;

constexpr uint8_t TAW_FLAG_FILES = 1;     /* awseek: the index has a directory */
constexpr uint8_t TAW_FLAG_DYN_BLOCKS = 2; /* awseek: variable-length blocks;
                                              index entries carry raw_len */
constexpr uint8_t TAW_FLAG_ALL = TAW_FLAG_FILES | TAW_FLAG_DYN_BLOCKS;
constexpr size_t TAW_NAME_MAX = 4096;

constexpr size_t TAW_HEADER_SIZE = 32;
constexpr uint32_t TAW_CHUNK_MAX = 1u << 26;
constexpr uint32_t TAW_BLOCK_MIN = 1u << 12;
constexpr uint32_t TAW_BLOCK_MAX = 1u << 30;
constexpr size_t TAW_FOOTER_SIZE = 16;
constexpr size_t TAW_INDEX_HEAD = 16;
constexpr size_t TAW_INDEX_ENTRY = 16;      /* fixed blocks */
constexpr size_t TAW_INDEX_ENTRY_DYN = 20;  /* + u32 raw_len (flag bit 1) */

#define TAW_MAGIC "TAW\x1A"
#define TAW_INDEX_MAGIC "TAWI"
#define TAW_FOOTER_MAGIC "TAWE"

#endif
