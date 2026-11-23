# tadamw

A general-purpose compressed archive format whose byte predictor is a small
model fitted to the input with AdamW and embedded in the archive, as a C
library (`libtadamw`) plus command-line tools. Two variants:

- `.aw`: a single compressed stream
- `.awseek`: fixed-size blocks plus a seek table for random access, and
  optionally a directory of named files

Status: working draft. The container, range coder, seeking and the
arithmetic profiles work: profile 0 is an integer order-0 placeholder, and
profile 1 (`F32_STRICT_MIX`) is the fitted mixer. Adaptive integer
context models feed a small MLP that the encoder fits with AdamW and
stores as F16 or Q8. Profile 2 (`F32_STRICT_CHUNK`, opt-in) adds a chunk
stage after arXiv 2507.07955: a routing table fitted to the input cuts the
bytes into chunks and lines for four more context models, and the model
names the ten that run. Archives are 1.5% smaller on enwik8 and Silesia
(20% on a table-like file); decoding is up to 19% slower on text. Every SIMD path (AVX2/F16C, AVX-512, SSE4.2 CRC) has a
generic one beside it that produces identical bits. See `docs/`.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /usr/local     # optional
```

Requires a C23 compiler (clang >= 18 or gcc >= 14) and CMake >= 3.25.
Sanitizer build: `-DTADAMW_SANITIZE=ON -DTADAMW_WERROR=ON`.

| option | default | |
|--------|---------|-|
| `TADAMW_ENCODER` | ON | compression and model fitting |
| `TADAMW_DECODER` | ON | decompression and random access |
| `BUILD_SHARED_LIBS` | OFF | `libtadamw.so` instead of `libtadamw.a` |
| `TADAMW_TOOLS` | ON | `taw` (and `taw-nx`) |
| `TADAMW_NX` | ON | `taw-nx`, when OpenSSL 3 and LZ4 are found |
| `TADAMW_TESTS` | ON | the tests (need both halves) |
| `TADAMW_INSTALL` | ON | install rules |
| `TADAMW_GPU` | ON if `vulkan/vulkan.h` and `glslangValidator` are found | GPU model fitting; Vulkan is opened at run time, so the library never links it |

## Library

```cmake
find_package(tadamw 0.1 REQUIRED)
target_link_libraries(app PRIVATE tadamw::tadamw)
```

or `pkg-config --cflags --libs tadamw`. The header is
`<tadamw/tadamw.h>` (the API keeps the `taw_` prefix) and works from C99,
C11, C23 and C++. `<tadamw/config.h>` says which halves the installed library
has (`TAW_HAVE_ENCODER`, `TAW_HAVE_DECODER`), and only their functions are
declared. A shared library exports the `taw_*` API alone.

- `taw_compress` / `taw_decompress`: whole buffers, either variant.
- `taw_seek_encode`: a `.awseek` from any byte source to any sink, in
  parallel and with bounded memory, optionally with named files.
- `taw_seek_open` / `taw_seek_read` / `taw_seek_files`: random access.
- `taw_params.gpu` (`TAW_GPU_OFF`/`AUTO`/`ON`): fit the model on a GPU
  through Vulkan, opened at run time; about 4x faster encoding at level 9
  on a GT 1030 (docs/design.md, "GPU fitting"). `<tadamw/config.h>` says
  whether the library has it (`TAW_HAVE_GPU`).

## Usage

```sh
taw c [-l 1..9] in out.aw              # stream
taw c -p mix -w q8 in out.aw           # force the mixer, Q8 weights
taw c -p chunk in out.aw               # the mixer with the chunk stage (profile 2)
taw c -l 9 -g on in out.aw             # fit on a GPU (-g auto: if there is one)
taw c [-B block_size] [-j N] in out.awseek   # seekable (or -s), N threads
taw d archive out
taw x archive.awseek OFFSET LENGTH out # random-access range
taw x archive.awseek /name out         # a named file
taw i archive                          # header info and files
```

## taw-nx: Switch dumps

`tools/taw-nx` turns XCI/NSP dumps into `.awseek` archives (the dump plus a
`/.meta` file) and back, byte for byte. It stores NCAs and the gamecard info
decrypted, since ciphertext does not compress, and encrypts them again on the
way out. Keys come from `~/.switch/prod.keys`. See [docs/nx.md](docs/nx.md).

```sh
taw-nx enc [-l 1..9] [-j threads] game.nsp game.awseek
taw-nx dec game.awseek game.nsp
```

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). `tools/hactool` is a copy of SciresM's
hactool (ISC) as ScarletNX carries it, with ScarletNX's and tadamw's changes
under GPL-3.0-or-later; see `tools/hactool/README.md`.
