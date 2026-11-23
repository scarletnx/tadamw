#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Fixed golden corpus set from COMPAT.md. Every encode/decode runs with a
# scrubbed environment (no inherited TAW_* knobs, TAW_FIT_AVG=off), then each
# archive's byte length and SHA-256 must match the recorded pair.
#
# Configurable (env):
#   GOLDEN_DATA   directory holding the corpora (default /tmp/opencode/tadamw-bench)
#   GOLDEN_TEXT   text1m corpus path  (default $GOLDEN_DATA/text1m)
#   GOLDEN_LIBC   libc1m corpus path  (default $GOLDEN_DATA/libc1m)
#   TAW_BIN       taw executable      (default <repo>/build/tools/taw)
#   GOLDEN_WORK   scratch output dir  (default <repo>/build/compat-golden)
# The corpora are checked against the SHAs pinned in COMPAT.md; point the
# variables at a copy at a different path, not at regenerated data.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TAW_BIN=${TAW_BIN:-"$ROOT/build/tools/taw"}
DATA=${GOLDEN_DATA:-/tmp/opencode/tadamw-bench}
TEXT=${GOLDEN_TEXT:-"$DATA/text1m"}
LIBC=${GOLDEN_LIBC:-"$DATA/libc1m"}
WORK=${GOLDEN_WORK:-"$ROOT/build/compat-golden"}

scrub() {
  env -i PATH=/usr/bin:/bin HOME=/tmp LC_ALL=C TZ=UTC \
      TAW_FIT_AVG=off "$@"
}

sha() { sha256sum "$1" | cut -d' ' -f1; }

check_corpus() {
  if [ ! -f "$1" ]; then
    echo "FAIL: corpus missing: $1" >&2
    exit 1
  fi
  got=$(sha "$1")
  if [ "$got" != "$2" ]; then
    echo "FAIL: $3 sha256 $got, want $2" >&2
    echo "      re-baseline COMPAT.md before using a different corpus" >&2
    exit 1
  fi
}

if [ ! -x "$TAW_BIN" ]; then
  echo "FAIL: taw executable not found: $TAW_BIN" >&2
  exit 1
fi
check_corpus "$TEXT" 2364f861d5e7f418095fad55dfc1a2d7c53eccfe30c1f2d1578b3b1fe7422129 text1m
check_corpus "$LIBC" 49bfd4c2dc804f7a35a87093b794a1d9c9d73d6f7e0c4626eec6d201b67c508d libc1m

mkdir -p "$WORK"
for pair in "text1m:$TEXT" "libc1m:$LIBC"; do
  c=${pair%%:*}
  src=${pair#*:}
  for l in 1 5; do
    for p in mix auto; do
      arc="$WORK/${c}_L${l}_${p}.aw"
      scrub "$TAW_BIN" c -l "$l" -p "$p" "$src" "$arc"
      scrub "$TAW_BIN" d "$arc" "$WORK/${c}_L${l}_${p}.dec"
      cmp "$src" "$WORK/${c}_L${l}_${p}.dec"
    done
  done
done
scrub "$TAW_BIN" c -l 1 -p mix -w q8    "$TEXT" "$WORK/text1m_L1_mix_q8.aw"
scrub "$TAW_BIN" c -l 1 -p mix -j 8     "$LIBC" "$WORK/libc1m_L1_mix_j8.aw"
scrub "$TAW_BIN" c -l 1 -p mix -B 4096  "$TEXT" "$WORK/text1m_L1_mix_B4k.awseek"
scrub "$TAW_BIN" x "$WORK/text1m_L1_mix_B4k.awseek" 4000 5000 "$WORK/range.bin"
scrub "$TAW_BIN" d -j 8 "$WORK/text1m_L1_mix_B4k.awseek" "$WORK/seek_j8.dec"
cmp "$TEXT" "$WORK/seek_j8.dec"

failures=0
check_pair() {
  name=$1 want_size=$2 want_sha=$3
  f="$WORK/$name"
  if [ ! -f "$f" ]; then
    echo "FAIL: $name missing" >&2
    failures=$((failures + 1))
    return
  fi
  got_size=$(stat -c %s "$f")
  got_sha=$(sha "$f")
  if [ "$got_size" != "$want_size" ] || [ "$got_sha" != "$want_sha" ]; then
    echo "FAIL: $name size $got_size (want $want_size) sha $got_sha (want $want_sha)" >&2
    failures=$((failures + 1))
  else
    echo "ok: $name"
  fi
}

# name, bytes, sha256 -- the COMPAT.md golden table.
while read -r name size hash; do
  [ -n "$name" ] || continue
  check_pair "$name" "$size" "$hash"
done <<'EOF'
text1m_L1_mix.aw 181557 085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d
text1m_L1_auto.aw 181557 085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d
text1m_L1_mix_q8.aw 181557 085f2c7326e287798e9b5e7349d2939abe0e05fe372c432a8715d01d2f5d862d
text1m_L5_mix.aw 179166 ca621be0b340b7e04fa65d4a781d9a25766aa6312be7d0be174963b36eae187a
text1m_L5_auto.aw 179166 ca621be0b340b7e04fa65d4a781d9a25766aa6312be7d0be174963b36eae187a
libc1m_L1_mix.aw 379197 6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b
libc1m_L1_auto.aw 379197 6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b
libc1m_L1_mix_j8.aw 379197 6ba18cf3bd779c8d786acce5b9c183b6dcbcbafcc4476d1caf2a02449761be3b
libc1m_L5_mix.aw 375291 89863914c8deff6b43522f58f711757abda94d5d6eb44f6c369905c628faccf2
libc1m_L5_auto.aw 375291 89863914c8deff6b43522f58f711757abda94d5d6eb44f6c369905c628faccf2
text1m_L1_mix_B4k.awseek 334373 82f38fe571e5206630b61942defdc8169da48666fceedb9d9717f4288aaa98df
range.bin 5000 e20185c652f152d0dc36b0f6a6903ea09576cd892461571c42bcb913887a21f0
EOF

if [ "$failures" -ne 0 ]; then
  echo "FAIL: $failures golden pair(s) mismatched" >&2
  exit 1
fi
echo "golden corpus set OK (12 pairs)"
