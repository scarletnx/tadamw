/* SPDX-License-Identifier: GPL-3.0-or-later */
/* taw-nx: Switch XCI/NSP dumps to and from .awseek archives (docs/nx.md).
 *
 * Encrypted data does not compress, so the archive stores the dump with
 * every NCA it can decrypt in plain form (hactool's snx_hac_plain, as
 * ScarletNX uses it): header and sections decrypted where they lie, every
 * other byte as the dump has it. An XCI's encrypted gamecard info goes into
 * /.meta decrypted, its place in the image zeroed. Decoding encrypts all of
 * it again (snx_hac_encrypt, snx_hac_xci_info) and gives back the original
 * byte for byte; /.meta's SHA-256 of the original checks it.
 *
 * The archive is an ordinary .awseek with two files: /<dump name> (the plain
 * image) and /.meta (text, see docs/nx.md). `taw i` lists them and
 * `taw x archive /.meta -` prints the metadata.
 *
 * Keys come from prod.keys in both directions (and title keys from the
 * dump's own common tickets); no key is written or printed. The encoder
 * relies on nothing it has not checked: every NCA (and the gamecard info)
 * must encrypt back to the dump's exact bytes, or it stays as it was. */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include <openssl/evp.h>

#include "snx_hac.h"
#include "tadamw/tadamw.h"

constexpr size_t COPY_CHUNK = 4u << 20;
constexpr uint64_t CARD_INFO = 0x190;
constexpr uint64_t CARD_IV = 0x120;
constexpr size_t CARD_INFO_LEN = 0x70;
#define META_NAME "/.meta"
#define META_MAGIC "tadamw-nx 1"

static const char *prog = "taw-nx";
static int verbose = 1;

/* ---- small helpers ------------------------------------------------------ */

static int fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int fail(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "%s: ", prog);
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
	return 1;
}

static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...)
{
	if (!verbose)
		return;
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

static int pread_all(int fd, void *buf, size_t n, uint64_t off)
{
	uint8_t *p = buf;
	while (n) {
		ssize_t k = pread(fd, p, n, (off_t)off);
		if (k <= 0) {
			if (k < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p += k;
		n -= (size_t)k;
		off += (uint64_t)k;
	}
	return 0;
}

static int pwrite_all(int fd, const void *buf, size_t n, uint64_t off)
{
	const uint8_t *p = buf;
	while (n) {
		ssize_t k = pwrite(fd, p, n, (off_t)off);
		if (k <= 0) {
			if (k < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p += k;
		n -= (size_t)k;
		off += (uint64_t)k;
	}
	return 0;
}

static bool ends_with_ci(const char *s, const char *suffix)
{
	size_t a = strlen(s), b = strlen(suffix);
	return a >= b && strcasecmp(s + a - b, suffix) == 0;
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

static unsigned cpu_count(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	return n > 0 ? (unsigned)n : 1;
}

[[maybe_unused]] static void hex_put(char *out, const uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		sprintf(out + 2 * i, "%02x", p[i]);
}

[[maybe_unused]] static int hex_get(const char *s, uint8_t *out, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned v;
		if (sscanf(s + 2 * i, "%2x", &v) != 1)
			return -1;
		out[i] = (uint8_t)v;
	}
	return s[2 * n] == 0 || s[2 * n] == '\n' ? 0 : -1;
}

static int sha256_fd(int fd, uint64_t n, uint8_t out[32], uint8_t *buf)
{
	EVP_MD_CTX *md = EVP_MD_CTX_new();
	if (!md || !EVP_DigestInit_ex(md, EVP_sha256(), nullptr)) {
		EVP_MD_CTX_free(md);
		return -1;
	}
	for (uint64_t off = 0; off < n;) {
		size_t k = n - off < COPY_CHUNK ? (size_t)(n - off) : COPY_CHUNK;
		if (pread_all(fd, buf, k, off) || !EVP_DigestUpdate(md, buf, k)) {
			EVP_MD_CTX_free(md);
			return -1;
		}
		off += k;
	}
	unsigned len;
	int ok = EVP_DigestFinal_ex(md, out, &len);
	EVP_MD_CTX_free(md);
	return ok ? 0 : -1;
}

/* ---- keys and tickets ----------------------------------------------------- */

static void secret_free(void *p, size_t n)
{
	if (p) {
		explicit_bzero(p, n);
		free(p);
	}
}

/* prod.keys into hactool's keyset: "name = hex" lines; values of a length no
 * key has are skipped, as ScarletNX does. Never prints a value. */
static int load_keys(SnxHac *hac, const char *path)
{
	int fd = open(path, O_RDONLY);
	struct stat sb;
	if (fd < 0 || fstat(fd, &sb) != 0) {
		int e = errno;
		if (fd >= 0)
			close(fd);
		return fail("%s: %s", path, strerror(e));
	}
	size_t n = (size_t)sb.st_size;
	char *data = malloc(n + 1);
	if (!data || pread_all(fd, data, n, 0) != 0) {
		close(fd);
		secret_free(data, n + 1);
		return fail("%s: cannot read", path);
	}
	close(fd);
	data[n] = 0;
	unsigned known = 0;
	for (char *line = data; line && *line;) {
		char *next = strchr(line, '\n');
		if (next)
			*next++ = 0;
		char *eq = strchr(line, '=');
		if (eq) {
			char *name = line, *value = eq + 1, *e = eq;
			*eq = 0;
			while (*name == ' ' || *name == '\t')
				name++;
			while (e > name && (e[-1] == ' ' || e[-1] == '\t'))
				*--e = 0;
			while (*value == ' ' || *value == '\t')
				value++;
			size_t vl = strcspn(value, " \t\r;#");
			value[vl] = 0;
			for (char *c = name; *c; c++)
				if (*c >= 'A' && *c <= 'Z')
					*c = (char)(*c - 'A' + 'a');
			if (*name && vl && vl % 32 == 0)
				known += (unsigned)snx_hac_set_key(hac, name, value);
		}
		line = next;
	}
	secret_free(data, n + 1);
	snx_hac_derive_keys(hac);
	say("keys: %s (%u known names)\n", path, known);
	return 0;
}

static int keys_open(SnxHac **hac, const char *keys)
{
	char *dflt = nullptr;
	if (!keys) {
		const char *home = getenv("HOME");
		size_t n = (home ? strlen(home) : 0) + sizeof "/.switch/prod.keys";
		if (!home || !(dflt = malloc(n)))
			return fail("no HOME for ~/.switch/prod.keys: use -k");
		snprintf(dflt, n, "%s/.switch/prod.keys", home);
	}
	*hac = snx_hac_new();
	int rc = !*hac || load_keys(*hac, keys ? keys : dflt);
	free(dflt);
	return rc;
}

/* The dump's tickets' title keys into the keyset. */
static void read_tickets(SnxHac *hac, int container)
{
	const SnxHacEntry *e;
	size_t count = snx_hac_entries(hac, container, &e);
	unsigned common = 0, personal = 0;
	for (size_t i = 0; i < count; i++) {
		if (!ends_with_ci(e[i].name, ".tik"))
			continue;
		unsigned char t[1024];
		size_t n = e[i].size < sizeof t ? (size_t)e[i].size : sizeof t;
		if (snx_hac_read(hac, container, i, t, n) == 0) {
			int r = snx_hac_ticket_key(hac, t, n);
			common += r == 1;
			personal += r == 0;
		}
		explicit_bzero(t, sizeof t);
	}
	if (common || personal)
		say("tickets: %u common, %u personalized (not usable)\n", common,
		    personal);
}

/* hactool ends the process on some dumps it cannot read (snx_hac_on_exit):
 * say why and remove what this run was writing. */
static const char *cleanup_paths[2];

static void on_hactool_exit(void *data)
{
	(void)data;
	const char *why = snx_hac_message();
	fprintf(stderr, "%s: hactool stopped: %s\n", prog,
	    *why ? why : "(no message)");
	for (int i = 0; i < 2; i++)
		if (cleanup_paths[i])
			unlink(cleanup_paths[i]);
}

/* ---- callbacks for snx_hac_plain / snx_hac_encrypt ------------------------ */

typedef struct {
	int fd;
	uint64_t at;
	bool failed;
} fd_sink;

static int sink_write(const void *bytes, size_t n, void *data)
{
	fd_sink *s = data;
	if (pwrite_all(s->fd, bytes, n, s->at) != 0) {
		s->failed = true;
		return 1;
	}
	s->at += n;
	return 0;
}

/* .meta: what decoding needs besides the image */
typedef struct {
	int kind; /* 0 NSP, 1 XCI */
	uint64_t size;
	uint8_t sha[32];
	uint64_t *nca; /* offset, size pairs */
	size_t nnca;
	bool card;     /* card info decrypted, zeroed in the image */
	uint8_t card_info[CARD_INFO_LEN];
} meta;

static void meta_free(meta *m)
{
	free(m->nca);
	explicit_bzero(m, sizeof *m);
}

#if TAW_HAVE_ENCODER
/* ---- encode ---------------------------------------------------------------- */

typedef struct {
	int fd; /* the original dump */
	uint64_t at;
	uint8_t *buf;
	bool differs;
} fd_compare;

static int compare_write(const void *bytes, size_t n, void *data)
{
	fd_compare *c = data;
	const uint8_t *p = bytes;
	while (n) {
		size_t k = n < COPY_CHUNK ? n : COPY_CHUNK;
		if (pread_all(c->fd, c->buf, k, c->at) != 0 ||
		    memcmp(c->buf, p, k) != 0) {
			c->differs = true;
			return 1;
		}
		p += k;
		n -= k;
		c->at += k;
	}
	return 0;
}

static int copy_range(int from, int to, uint64_t off, uint64_t n, uint8_t *buf)
{
	while (n) {
		size_t k = n < COPY_CHUNK ? (size_t)n : COPY_CHUNK;
		if (pread_all(from, buf, k, off) || pwrite_all(to, buf, k, off))
			return -1;
		off += k;
		n -= k;
	}
	return 0;
}

/* 0 NSP, 1 XCI, -1 neither (by content: PFS0 magic, or HEAD at 0x100). */
static int dump_kind(int fd)
{
	uint8_t b[0x104];
	if (pread_all(fd, b, sizeof b, 0) != 0)
		return -1;
	if (!memcmp(b, "PFS0", 4))
		return 0;
	if (!memcmp(b + 0x100, "HEAD", 4))
		return 1;
	return -1;
}

/* The plain image of the dump into tmp_fd; what decoding needs into *m. */
static int make_plain(SnxHac *hac, const char *in_path, int in_fd,
    int tmp_fd, meta *m, uint8_t *buf)
{
	if (copy_range(in_fd, tmp_fd, 0, m->size, buf))
		return fail("cannot copy the dump: %s", strerror(errno));
	int c = snx_hac_open(hac, in_path, m->kind);
	if (c < 0)
		return fail("%s: cannot read the %s", in_path,
		    m->kind ? "XCI's secure partition" : "NSP");
	read_tickets(hac, c);
	const SnxHacEntry *e;
	size_t count = snx_hac_entries(hac, c, &e);
	FILE *in = fopen(in_path, "rb");
	FILE *tmp = fdopen(dup(tmp_fd), "rb");
	if (!in || !tmp) {
		if (in)
			fclose(in);
		if (tmp)
			fclose(tmp);
		return fail("cannot open the dump twice");
	}
	setvbuf(tmp, nullptr, _IONBF, 0);
	m->nca = calloc(2 * (count ? count : 1), sizeof *m->nca);
	uint64_t plain_bytes = 0;
	unsigned kept = 0;
	int rc = 0;
	for (size_t i = 0; m->nca && i < count; i++) {
		if (!ends_with_ci(e[i].name, ".nca") || e[i].size < 0xC00 ||
		    e[i].offset + e[i].size > m->size)
			continue;
		int already = 0;
		fd_sink sink = {.fd = tmp_fd, .at = e[i].offset};
		bool ok = snx_hac_plain(hac, in, e[i].offset, e[i].size, nullptr,
		    sink_write, &sink, &already) == 0 && !sink.failed;
		if (ok && already) { /* plain in the dump already */
			if (copy_range(in_fd, tmp_fd, e[i].offset, e[i].size, buf))
				rc = -1;
			continue;
		}
		if (ok) { /* it must come back exactly */
			fd_compare cmp = {.fd = in_fd, .at = e[i].offset, .buf = buf};
			ok = snx_hac_encrypt(hac, tmp, e[i].offset, e[i].size,
			    compare_write, &cmp) == 0 && !cmp.differs &&
			    cmp.at == e[i].offset + e[i].size;
		}
		if (!ok) {
			say("  %s: kept encrypted (%s)\n", e[i].name,
			    *snx_hac_message() ? snx_hac_message()
					       : "does not encrypt back exactly");
			kept++;
			if (copy_range(in_fd, tmp_fd, e[i].offset, e[i].size, buf))
				rc = -1;
			continue;
		}
		m->nca[2 * m->nnca] = e[i].offset;
		m->nca[2 * m->nnca + 1] = e[i].size;
		m->nnca++;
		plain_bytes += e[i].size;
	}
	fclose(in);
	fclose(tmp);
	if (!m->nca)
		return fail("out of memory");
	if (rc)
		return fail("cannot write the plain image: %s", strerror(errno));
	say("NCAs: %zu decrypted (%" PRIu64 " bytes), %u kept as they are\n",
	    m->nnca, plain_bytes, kept);

	/* an XCI's gamecard info: decrypted into .meta, zeroed in the image */
	if (m->kind == 1) {
		uint8_t rev[16], iv[16], enc[CARD_INFO_LEN], back[CARD_INFO_LEN];
		if (pread_all(in_fd, rev, 16, CARD_IV) ||
		    pread_all(in_fd, enc, CARD_INFO_LEN, CARD_INFO))
			return fail("cannot read the XCI header");
		for (int k = 0; k < 16; k++)
			iv[k] = rev[15 - k];
		if (snx_hac_xci_info_decrypt(hac, iv, enc, m->card_info) == 0 &&
		    snx_hac_xci_info(hac, iv, m->card_info, back) == 0 &&
		    !memcmp(back, enc, CARD_INFO_LEN)) {
			static const uint8_t zero[CARD_INFO_LEN];
			if (pwrite_all(tmp_fd, zero, CARD_INFO_LEN, CARD_INFO))
				return fail("cannot write the plain image");
			m->card = true;
			say("gamecard info: decrypted into " META_NAME "\n");
		} else {
			say("gamecard info: kept encrypted (no xci_header_key)\n");
		}
	}
	return 0;
}

/* .meta as text (docs/nx.md). */
static char *meta_text(const meta *m, const char *name, size_t *len)
{
	size_t cap = 512 + m->nnca * 48 + strlen(name);
	char *t = malloc(cap), *p = t;
	if (!t)
		return nullptr;
	char sha[65], card[2 * CARD_INFO_LEN + 1];
	hex_put(sha, m->sha, 32);
	p += sprintf(p, META_MAGIC "\nkind %s\nname %s\nsize %" PRIu64
	    "\nsha256 %s\n", m->kind ? "xci" : "nsp", name, m->size, sha);
	for (size_t k = 0; k < m->nnca; k++)
		p += sprintf(p, "nca %" PRIu64 " %" PRIu64 "\n", m->nca[2 * k],
		    m->nca[2 * k + 1]);
	if (m->card) {
		hex_put(card, m->card_info, CARD_INFO_LEN);
		p += sprintf(p, "cardinfo %s\n", card);
	}
	*len = (size_t)(p - t);
	return t;
}

/* The archive's data: the plain image file, then .meta. */
typedef struct {
	int fd;
	uint64_t image;
	const char *meta;
	size_t meta_len;
} nx_source;

static taw_status nx_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	nx_source *s = ctx;
	uint8_t *d = dst;
	if (off < s->image) {
		size_t k = s->image - off < n ? (size_t)(s->image - off) : n;
		if (pread_all(s->fd, d, k, off))
			return TAW_E_IO;
		d += k;
		off += k;
		n -= k;
	}
	if (n)
		memcpy(d, s->meta + (off - s->image), n);
	return TAW_OK;
}

typedef struct {
	int fd;
	uint64_t at;
} out_sink;

static taw_status out_write(void *ctx, const void *p, size_t n)
{
	out_sink *o = ctx;
	if (pwrite_all(o->fd, p, n, o->at))
		return TAW_E_IO;
	o->at += n;
	return TAW_OK;
}

static int cmd_encode(int argc, char **argv)
{
	taw_params p = {.level = 1, .block_size = 64u << 20};
	const char *keys = nullptr;
	int i = 0;
	for (; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
		uint64_t v;
		if (!strcmp(argv[i], "-l") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= 9) {
			p.level = (int)v;
			i++;
		} else if (!strcmp(argv[i], "-j") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= 256) {
			p.threads = (unsigned)v;
			i++;
		} else if (!strcmp(argv[i], "-B") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= 1024 &&
		    !(v & (v - 1))) {
			p.block_size = (uint32_t)(v << 20);
			i++;
		} else if (!strcmp(argv[i], "-D")) {
			p.dyn_blocks = true;
		} else if (!strcmp(argv[i], "-T") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= (1u << 26)) {
			p.chunk_target = (uint32_t)v;
			i++;
		} else if (!strcmp(argv[i], "-C") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v <= 256) {
			p.change_threshold = (uint32_t)v;
			i++;
		} else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
			keys = argv[++i];
		} else if (!strcmp(argv[i], "-q")) {
			verbose = 0;
		} else {
			return 2;
		}
	}
	if (argc - i != 2)
		return 2;
	const char *in_path = argv[i], *out_path = argv[i + 1];
	if (!p.threads)
		p.threads = cpu_count();

	int in_fd = open(in_path, O_RDONLY);
	struct stat sb;
	if (in_fd < 0 || fstat(in_fd, &sb) != 0)
		return fail("%s: %s", in_path, strerror(errno));
	meta m = {.size = (uint64_t)sb.st_size, .kind = dump_kind(in_fd)};
	if (m.kind < 0) {
		close(in_fd);
		return fail("%s: neither an NSP (PFS0) nor an XCI", in_path);
	}
	const char *base = strrchr(in_path, '/');
	base = base ? base + 1 : in_path;
	size_t nl = strlen(base) + 2;
	char *name = malloc(nl);
	SnxHac *hac = nullptr;
	uint8_t *buf = malloc(COPY_CHUNK);
	size_t tl = strlen(out_path) + 8;
	char *tmp_path = malloc(tl);
	int rc = 1, tmp_fd = -1, out_fd = -1;
	char *mt = nullptr;
	if (!name || !buf || !tmp_path) {
		fail("out of memory");
		goto out;
	}
	snprintf(name, nl, "/%s", base);
	if (keys_open(&hac, keys))
		goto out;
	snprintf(tmp_path, tl, "%s.plain", out_path);
	tmp_fd = open(tmp_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
	if (tmp_fd < 0) {
		fail("%s: %s", tmp_path, strerror(errno));
		goto out;
	}
	cleanup_paths[0] = tmp_path;
	cleanup_paths[1] = out_path;
	snx_hac_on_exit(on_hactool_exit, nullptr);
	if (sha256_fd(in_fd, m.size, m.sha, buf)) {
		fail("%s: cannot read", in_path);
		goto out;
	}
	say("%s: %s, %" PRIu64 " bytes\n", in_path, m.kind ? "XCI" : "NSP",
	    m.size);
	if (make_plain(hac, in_path, in_fd, tmp_fd, &m, buf))
		goto out;

	size_t ml;
	if (!(mt = meta_text(&m, name + 1, &ml))) {
		fail("out of memory");
		goto out;
	}
	nx_source src = {.fd = tmp_fd, .image = m.size, .meta = mt,
		.meta_len = ml};
	const taw_source in = {.read_at = nx_read_at, .ctx = &src,
		.size = m.size + ml};
	const taw_file files[2] = {{name, 0, m.size}, {META_NAME, m.size, ml}};
	out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (out_fd < 0) {
		fail("%s: %s", out_path, strerror(errno));
		goto out;
	}
	out_sink os = {.fd = out_fd};
	const taw_sink sink = {.write = out_write, .ctx = &os};
	say("compressing: level %d, %" PRIu32 " MiB blocks, %u thread(s)\n",
	    p.level, p.block_size >> 20, p.threads);
	taw_status st = taw_seek_encode(&in, &p, files, 2, &sink);
	if (st != TAW_OK) {
		fail("%s: %s", out_path, taw_strerror(st));
		goto out;
	}
	if (fsync(out_fd) != 0) {
		fail("%s: %s", out_path, strerror(errno));
		goto out;
	}
	say("%s: %" PRIu64 " bytes (%.2f%% of the dump)\n", out_path, os.at,
	    m.size ? 100.0 * (double)os.at / (double)m.size : 0.0);
	rc = 0;
out:
	if (out_fd >= 0)
		close(out_fd);
	if (rc && out_fd >= 0)
		unlink(out_path);
	if (tmp_fd >= 0) {
		close(tmp_fd);
		unlink(tmp_path);
	}
	cleanup_paths[0] = cleanup_paths[1] = nullptr;
	close(in_fd);
	free(tmp_path);
	free(name);
	free(mt);
	free(buf);
	meta_free(&m);
	snx_hac_free(hac);
	return rc;
}
#endif

#if TAW_HAVE_DECODER
/* ---- decode ---------------------------------------------------------------- */

static taw_status fd_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	return pread_all(*(int *)ctx, dst, n, off) ? TAW_E_IO : TAW_OK;
}

static int meta_parse(char *t, meta *m)
{
	char *line = strtok(t, "\n");
	if (!line || strcmp(line, META_MAGIC))
		return fail(META_NAME ": not taw-nx metadata (or a newer version)");
	bool have_size = false, have_sha = false, have_kind = false;
	size_t cap = 0;
	while ((line = strtok(nullptr, "\n"))) {
		char *v = strchr(line, ' ');
		if (!v)
			return fail(META_NAME ": bad line");
		*v++ = 0;
		if (!strcmp(line, "kind")) {
			m->kind = !strcmp(v, "xci") ? 1 : !strcmp(v, "nsp") ? 0 : -1;
			if (m->kind < 0)
				return fail(META_NAME ": unknown kind");
			have_kind = true;
		} else if (!strcmp(line, "size")) {
			have_size = parse_u64(v, &m->size);
		} else if (!strcmp(line, "sha256")) {
			have_sha = hex_get(v, m->sha, 32) == 0;
		} else if (!strcmp(line, "nca")) {
			uint64_t off, n;
			char *sp = strchr(v, ' ');
			if (!sp)
				return fail(META_NAME ": bad nca line");
			*sp = 0;
			if (!parse_u64(v, &off) || !parse_u64(sp + 1, &n))
				return fail(META_NAME ": bad nca line");
			if (m->nnca == cap) {
				cap = cap ? 2 * cap : 16;
				uint64_t *q = realloc(m->nca, 2 * cap * sizeof *q);
				if (!q)
					return fail("out of memory");
				m->nca = q;
			}
			m->nca[2 * m->nnca] = off;
			m->nca[2 * m->nnca + 1] = n;
			m->nnca++;
		} else if (!strcmp(line, "cardinfo")) {
			if (hex_get(v, m->card_info, CARD_INFO_LEN))
				return fail(META_NAME ": bad cardinfo");
			m->card = true;
		} /* "name" and unknown keys: informational */
	}
	if (!have_kind || !have_size || !have_sha)
		return fail(META_NAME ": incomplete");
	for (size_t k = 0; k < m->nnca; k++)
		if (m->nca[2 * k + 1] < 0xC00 || m->nca[2 * k] > m->size ||
		    m->nca[2 * k + 1] > m->size - m->nca[2 * k])
			return fail(META_NAME ": NCA outside the dump");
	if (m->card && (m->kind != 1 || m->size < CARD_INFO + CARD_INFO_LEN))
		return fail(META_NAME ": cardinfo on a dump without one");
	return 0;
}

/* Parallel expansion: each worker has its own taw_seek over the archive. */
typedef struct {
	int in_fd, out_fd;
	uint64_t arc_size, image_off, image_size, chunk;
	pthread_mutex_t mu;
	uint64_t next;
	bool failed;
	taw_status st;
} expand_job;

static void *expand_worker(void *arg)
{
	expand_job *J = arg;
	taw_source src = {.read_at = fd_read_at, .ctx = &J->in_fd,
		.size = J->arc_size};
	taw_seek *s = nullptr;
	uint8_t *buf = malloc((size_t)J->chunk);
	taw_status st = buf ? taw_seek_open(&src, &s) : TAW_E_NOMEM;
	if (st != TAW_OK)
		s = nullptr;
	/* work items are blocks of the archive's data, clipped to the image */
	const uint64_t first = J->image_off / J->chunk;
	const uint64_t end = J->image_off + J->image_size;
	for (;;) {
		pthread_mutex_lock(&J->mu);
		if (st != TAW_OK && !J->failed) {
			J->failed = true;
			J->st = st;
		}
		uint64_t blk = first + J->next;
		bool stop = J->failed || blk * J->chunk >= end;
		if (!stop)
			J->next++;
		pthread_mutex_unlock(&J->mu);
		if (stop)
			break;
		uint64_t lo = blk * J->chunk, hi = lo + J->chunk;
		if (lo < J->image_off)
			lo = J->image_off;
		if (hi > end)
			hi = end;
		st = taw_seek_read(s, lo, buf, (size_t)(hi - lo));
		if (st == TAW_OK &&
		    pwrite_all(J->out_fd, buf, (size_t)(hi - lo), lo - J->image_off))
			st = TAW_E_IO;
	}
	if (s)
		taw_seek_close(s);
	free(buf);
	return nullptr;
}

static int cmd_decode(int argc, char **argv)
{
	unsigned jobs = 0;
	const char *keys = nullptr;
	int i = 0;
	for (; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
		uint64_t v;
		if (!strcmp(argv[i], "-j") && i + 1 < argc &&
		    parse_u64(argv[i + 1], &v) && v >= 1 && v <= 256) {
			jobs = (unsigned)v;
			i++;
		} else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
			keys = argv[++i];
		} else if (!strcmp(argv[i], "-q")) {
			verbose = 0;
		} else {
			return 2;
		}
	}
	if (argc - i != 2)
		return 2;
	const char *in_path = argv[i], *out_path = argv[i + 1];
	if (!jobs)
		jobs = cpu_count();

	int in_fd = open(in_path, O_RDONLY);
	struct stat sb;
	if (in_fd < 0 || fstat(in_fd, &sb) != 0)
		return fail("%s: %s", in_path, strerror(errno));
	int rc = 1, out_fd = -1;
	meta m = {0};
	SnxHac *hac = nullptr;
	char *mt = nullptr;
	uint8_t *buf = malloc(COPY_CHUNK);
	taw_source src = {.read_at = fd_read_at, .ctx = &in_fd,
		.size = (uint64_t)sb.st_size};
	taw_seek *s = nullptr;
	taw_status st = buf ? taw_seek_open(&src, &s) : TAW_E_NOMEM;
	if (st != TAW_OK) {
		fail("%s: %s", in_path, taw_strerror(st));
		goto out;
	}
	const taw_file *f;
	size_t nf = taw_seek_files(s, &f);
	long mk = taw_seek_find(s, META_NAME);
	long ik = -1;
	for (size_t k = 0; k < nf; k++)
		if ((long)k != mk) {
			ik = (long)k;
			break;
		}
	if (mk < 0 || ik < 0 || f[mk].size > (1u << 26)) {
		fail("%s: not a taw-nx archive (no " META_NAME " and dump)", in_path);
		goto out;
	}
	mt = malloc((size_t)f[mk].size + 1);
	if (!mt || (st = taw_seek_read(s, f[mk].offset, mt, (size_t)f[mk].size)) !=
	    TAW_OK) {
		fail(META_NAME ": %s", mt ? taw_strerror(st) : "out of memory");
		goto out;
	}
	mt[f[mk].size] = 0;
	if (meta_parse(mt, &m))
		goto out;
	if (f[ik].size != m.size) {
		fail("%s: %s is %" PRIu64 " bytes, " META_NAME " says %" PRIu64,
		    in_path, f[ik].name, f[ik].size, m.size);
		goto out;
	}
	const uint64_t image_off = f[ik].offset;
	say("expanding %s, %u thread(s)\n", f[ik].name, jobs);
	taw_seek_close(s);
	s = nullptr;
	/* one block per work item: blocks decode independently */
	uint8_t head[32];
	taw_info info;
	if (pread_all(in_fd, head, sizeof head, 0) ||
	    taw_read_info(head, sizeof head, &info) != TAW_OK) {
		fail("%s: cannot read the header", in_path);
		goto out;
	}
	if ((m.nnca || m.card) && keys_open(&hac, keys))
		goto out;

	out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (out_fd < 0 || ftruncate(out_fd, (off_t)m.size) != 0) {
		fail("%s: %s", out_path, strerror(errno));
		goto out;
	}
	cleanup_paths[0] = out_path;
	snx_hac_on_exit(on_hactool_exit, nullptr);

	expand_job J = {.in_fd = in_fd, .out_fd = out_fd,
		.arc_size = (uint64_t)sb.st_size, .image_off = image_off,
		.image_size = m.size, .chunk = info.block_size};
	pthread_mutex_init(&J.mu, nullptr);
	pthread_t *th = calloc(jobs, sizeof *th);
	unsigned started = 0;
	for (; th && started < jobs; started++)
		if (pthread_create(&th[started], nullptr, expand_worker, &J))
			break;
	if (!started)
		expand_worker(&J);
	for (unsigned k = 0; k < started; k++)
		pthread_join(th[k], nullptr);
	free(th);
	pthread_mutex_destroy(&J.mu);
	if (J.failed) {
		fail("%s: %s", in_path, taw_strerror(J.st));
		goto out;
	}

	if (m.nnca) { /* the plain NCAs encrypted again, in place */
		int c = snx_hac_open(hac, out_path, m.kind);
		if (c < 0) {
			fail("cannot read the expanded %s", m.kind ? "XCI" : "NSP");
			goto out;
		}
		read_tickets(hac, c);
		FILE *fp = fopen(out_path, "rb");
		if (!fp) {
			fail("%s: %s", out_path, strerror(errno));
			goto out;
		}
		setvbuf(fp, nullptr, _IONBF, 0);
		say("encrypting %zu NCA(s)\n", m.nnca);
		bool bad = false;
		for (size_t k = 0; k < m.nnca && !bad; k++) {
			uint64_t off = m.nca[2 * k], n = m.nca[2 * k + 1];
			fd_sink sink = {.fd = out_fd, .at = off};
			if (snx_hac_encrypt(hac, fp, off, n, sink_write, &sink) ||
			    sink.failed || sink.at != off + n) {
				fail("NCA at 0x%" PRIx64 ": %s", off,
				    sink.failed ? strerror(errno) : snx_hac_message());
				bad = true;
			}
		}
		fclose(fp);
		if (bad)
			goto out;
	}
	if (m.card) { /* the gamecard info encrypted again */
		uint8_t rev[16], iv[16], enc[CARD_INFO_LEN];
		if (pread_all(out_fd, rev, 16, CARD_IV)) {
			fail("%s: cannot read the XCI header", out_path);
			goto out;
		}
		for (int k = 0; k < 16; k++)
			iv[k] = rev[15 - k];
		if (snx_hac_xci_info(hac, iv, m.card_info, enc) ||
		    pwrite_all(out_fd, enc, CARD_INFO_LEN, CARD_INFO)) {
			fail("gamecard info: prod.keys has no xci_header_key");
			goto out;
		}
	}
	uint8_t sha[32];
	if (sha256_fd(out_fd, m.size, sha, buf) || memcmp(sha, m.sha, 32)) {
		fail("%s does not match the original's SHA-256 (wrong prod.keys?)",
		    out_path);
		goto out;
	}
	if (fsync(out_fd) != 0) {
		fail("%s: %s", out_path, strerror(errno));
		goto out;
	}
	say("%s: %" PRIu64 " bytes, SHA-256 matches the original\n", out_path,
	    m.size);
	rc = 0;
out:
	if (s)
		taw_seek_close(s);
	if (out_fd >= 0)
		close(out_fd);
	if (rc && out_fd >= 0)
		unlink(out_path);
	cleanup_paths[0] = nullptr;
	close(in_fd);
	free(mt);
	free(buf);
	meta_free(&m);
	snx_hac_free(hac);
	return rc;
}
#endif

static void usage(FILE *f)
{
	fprintf(f,
#if TAW_HAVE_ENCODER
	    "usage: taw-nx enc [-l level] [-j threads] [-B block_MiB] [-D [-T target] [-C change]]\n"
	    "                  [-k prod.keys] [-q] <in.xci|in.nsp> <out.awseek>\n"
#endif
#if TAW_HAVE_DECODER
	    "       taw-nx dec [-j threads] [-k prod.keys] [-q] <in.awseek> <out.xci|out.nsp>\n"
#endif
	    "\n"
	    "  enc  decrypt every NCA (and an XCI's gamecard info) that encrypts back\n"
	    "       exactly, and store the dump as a .awseek with two files: the dump\n"
	    "       and " META_NAME ". Level 1..9 (default 1), blocks default 64 MiB,\n"
	    "       -D enables content-aligned variable blocks (default: fixed).\n"
	    "       threads default one per CPU. A temporary <out>.plain the size of\n"
	    "       the dump is written next to the output.\n"
	    "  dec  expand, encrypt again, and check the SHA-256: the result is the\n"
	    "       original dump byte for byte.\n"
	    "  -k   keys file (default ~/.switch/prod.keys); title keys come from the\n"
	    "       dump's own common tickets.\n"
	    "  `taw i` lists the archive's files; `taw x <a.awseek> " META_NAME " -`\n"
	    "  prints the metadata.\n");
}

int main(int argc, char **argv)
{
	if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
		usage(stdout);
		return 0;
	}
	if (argc == 2 && !strcmp(argv[1], "--version")) {
		printf("taw-nx %s\n", taw_version());
		return 0;
	}
	int rc = 2;
#if TAW_HAVE_ENCODER
	if (argc >= 2 && !strcmp(argv[1], "enc"))
		rc = cmd_encode(argc - 2, argv + 2);
#endif
#if TAW_HAVE_DECODER
	if (argc >= 2 && !strcmp(argv[1], "dec"))
		rc = cmd_decode(argc - 2, argv + 2);
#endif
	if (rc == 2)
		usage(stderr);
	return rc;
}
