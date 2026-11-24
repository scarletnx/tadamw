/* SPDX-License-Identifier: GPL-3.0-or-later */
/* .awseek seekable variant (docs/format.md §7). */
#include <stdlib.h>
#include <string.h>

#include "awseek.h"
#include "header.h"

/* ---- sources --------------------------------------------------------------- */

static taw_status mem_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	memcpy(dst, (const uint8_t *)ctx + off, n);
	return TAW_OK;
}

void taw_source_memory(taw_source *s, const void *p, size_t n)
{
	*s = (taw_source){.read_at = mem_read_at, .ctx = (void *)p, .size = n};
}

const uint8_t *taw_source_view(const taw_source *s)
{
	return s->read_at == mem_read_at ? s->ctx : nullptr;
}

/* ---- directory ------------------------------------------------------------- */

/* A directory as the encoder takes it or the reader parsed it: 0, or
 * TAW_E_ARG (encoder) / TAW_E_FORMAT (reader) via the caller. */
static bool files_valid(const taw_file *f, size_t n, uint64_t total)
{
	uint64_t end = 0;
	for (size_t i = 0; i < n; i++) {
		size_t len = f[i].name ? strlen(f[i].name) : 0;
		if (len == 0 || len > TAW_NAME_MAX || f[i].name[0] != '/')
			return false;
		if (f[i].offset < end || f[i].offset > total ||
		    f[i].size > total - f[i].offset)
			return false;
		end = f[i].offset + f[i].size;
		for (size_t j = 0; j < i; j++)
			if (!strcmp(f[i].name, f[j].name))
				return false;
	}
	return true;
}

#if TAW_HAVE_ENCODER
static void files_put(taw_buf *b, const taw_file *f, size_t n)
{
	taw_buf_put_u32(b, (uint32_t)n);
	for (size_t i = 0; i < n; i++) {
		size_t len = strlen(f[i].name);
		taw_buf_put_u64(b, f[i].offset);
		taw_buf_put_u64(b, f[i].size);
		taw_buf_put_u8(b, (uint8_t)len);
		taw_buf_put_u8(b, (uint8_t)(len >> 8));
		taw_buf_put(b, f[i].name, len);
	}
}

/* ---- dynamic block boundaries (docs/format.md §7.2) ------------------------ */

/* The "representation" is the order-1 context of a byte; the dissimilarity
 * score is its surprise under a saturating (prev,byte) histogram. All state
 * is carried strictly in input order, so the boundaries are identical for any
 * thread count and for memory or streaming sources. */
typedef struct {
	uint16_t *cnt; /* [256][256], saturating */
	uint16_t tot[256];
	uint8_t prev;
	bool have_prev;
	uint32_t mean;      /* EWMA of surprise, 8.8 fixed point */
	unsigned last_dev;  /* deviation of the last byte from mean, floored */
} change_ctx;

static taw_status change_init(change_ctx *c)
{
	c->cnt = calloc(256u * 256u, sizeof *c->cnt);
	if (!c->cnt)
		return TAW_E_NOMEM;
	memset(c->tot, 0, sizeof c->tot);
	c->prev = 0;
	c->have_prev = false;
	c->mean = 0;
	c->last_dev = 0;
	return TAW_OK;
}

static void change_free(change_ctx *c)
{
	free(c->cnt);
	c->cnt = nullptr;
}

/* Byte surprise on [0,256]: 0 for the most likely successor of the previous
 * byte, 256 for an unlikely one. Add-one smoothing keeps a first occurrence
 * surprising. */
static unsigned change_byte(change_ctx *c, uint8_t x)
{
	unsigned s = 128; /* no context for the first byte */
	if (c->have_prev) {
		const size_t k = (size_t)c->prev * 256u + x;
		const unsigned tot = c->tot[c->prev];
		const unsigned est = (unsigned)(((uint64_t)(c->cnt[k] + 1u) << 16) /
		    (tot + 256u)); /* 0..65536 */
		s = 256u - (est >> 8);
		if (c->cnt[k] < UINT16_MAX)
			c->cnt[k]++;
		if (tot < UINT16_MAX)
			c->tot[c->prev] = (uint16_t)(tot + 1);
	}
	c->prev = x;
	c->have_prev = true;

	/* A boundary should mark a *change* of context, not merely a byte that
	 * is surprising in a uniformly surprising region. Compare surprise to
	 * its EWMA and keep only sizeable deviations. */
	if (!c->mean) {
		c->mean = s << 8;
		c->last_dev = 0;
	} else {
		const unsigned m = c->mean >> 8;
		const unsigned dev = s > m ? s - m : m - s;
		c->last_dev = dev < 8 ? 0 : dev;
		const int delta = (int)(s << 8) - (int)c->mean;
		c->mean = (uint32_t)((int)c->mean + delta / 256);
	}
	return s;
}

/* Sequential reader over any source, with a bounded scratch for non-memory
 * sources. Used twice: score total, then place the boundaries. */
typedef struct {
	const taw_source *src;
	const uint8_t *view;
	uint64_t at, total;
	uint8_t *buf;
	size_t cap;
} seq_reader;

static void seq_init(seq_reader *r, const taw_source *src, uint8_t *buf,
    size_t cap)
{
	r->src = src;
	r->view = taw_source_view(src);
	r->at = 0;
	r->total = src->size;
	r->buf = buf;
	r->cap = cap;
}

static taw_status seq_next(seq_reader *r, const uint8_t **p, size_t *n)
{
	*n = 0;
	*p = nullptr;
	if (r->at >= r->total)
		return TAW_OK;
	const uint64_t left = r->total - r->at;
	const size_t k = left < r->cap ? (size_t)left : r->cap;
	if (r->view) {
		*p = r->view + r->at;
	} else {
		const taw_status st = r->src->read_at(r->src->ctx, r->at, r->buf, k);
		if (st != TAW_OK)
			return st;
		*p = r->buf;
	}
	r->at += k;
	*n = k;
	return TAW_OK;
}

typedef struct {
	uint32_t *p;
	size_t n, cap;
} len_list;

static bool lens_push(len_list *l, uint32_t v)
{
	if (l->n == l->cap) {
		const size_t cap = l->cap ? l->cap * 2 : 64;
		uint32_t *q = realloc(l->p, cap * sizeof *q);
		if (!q)
			return false;
		l->p = q;
		l->cap = cap;
	}
	l->p[l->n++] = v;
	return true;
}

/* Pick raw_len for each block. Target controls min/max length and the number
 * of equal-change shares; a nonzero change_threshold instead cuts at a
 * single surprising byte. */
static taw_status choose_blocks(const taw_source *src, const taw_params *p,
    uint64_t *nblocks_out, uint64_t **offs_out, uint32_t **lens_out)
{
	const uint64_t total = src->size;
	const uint64_t target = p->chunk_target ? p->chunk_target : p->block_size;
	const uint64_t min_len = target / 4 ? target / 4 : 1;
	const uint64_t max4 = target > TAW_CHUNK_MAX / 4 ? TAW_CHUNK_MAX : target * 4;
	const uint64_t max_len = max4 > target ? max4 : target;
	const uint64_t ntarget = total ? (total + target - 1) / target : 0;
	const size_t cap = (size_t)(total < 65536 ? (total ? total : 1) : 65536);
	uint8_t *scratch = nullptr;
	change_ctx c = {0};
	len_list lens = {0};
	uint64_t *offs = nullptr;
	taw_status st = TAW_E_NOMEM;

	if (!total) {
		*nblocks_out = 0;
		*offs_out = nullptr;
		*lens_out = nullptr;
		return TAW_OK;
	}
	if (change_init(&c) != TAW_OK)
		goto out;
	if (!(scratch = malloc(cap)))
		goto out;

	/* pass 1: how much change is there in total? */
	uint64_t total_change = 0;
	seq_reader r;
	seq_init(&r, src, scratch, cap);
	for (;;) {
		const uint8_t *b;
		size_t n;
		if ((st = seq_next(&r, &b, &n)) != TAW_OK)
			goto out;
		if (!n)
			break;
		for (size_t i = 0; i < n; i++) {
			change_byte(&c, b[i]);
			total_change += c.last_dev;
		}
	}
	uint64_t share = ntarget ? total_change / ntarget : 1;
	if (!share)
		share = 1;

	/* pass 2: cut where accumulated surprise reaches its share (or at a
	 * single surprising byte in threshold mode), between min and max. */
	change_free(&c);
	if (change_init(&c) != TAW_OK)
		goto out;
	seq_init(&r, src, scratch, cap);
	uint64_t len = 0, acc = 0;
	for (;;) {
		const uint8_t *b;
		size_t n;
		if ((st = seq_next(&r, &b, &n)) != TAW_OK)
			goto out;
		if (!n)
			break;
		for (size_t i = 0; i < n; i++) {
			const unsigned s = change_byte(&c, b[i]);
			len++;
			acc += c.last_dev;
			const bool hit = p->change_threshold
			    ? s >= p->change_threshold : acc >= share;
			if (len >= max_len || (len >= min_len && hit)) {
				if (!lens_push(&lens, (uint32_t)len))
					goto out;
				len = acc = 0;
			}
		}
	}
	if (len && !lens_push(&lens, (uint32_t)len))
		goto out;
	if (lens.n > UINT32_MAX)
		goto out;

	offs = malloc((lens.n + 1) * sizeof *offs);
	if (!offs)
		goto out;
	offs[0] = 0;
	for (size_t i = 0; i < lens.n; i++)
		offs[i + 1] = offs[i] + lens.p[i];
	if (offs[lens.n] != total) {
		st = TAW_E_FORMAT;
		goto out;
	}
	*nblocks_out = lens.n;
	*offs_out = offs;
	*lens_out = lens.p;
	offs = nullptr;
	lens.p = nullptr;
	st = TAW_OK;
out:
	free(offs);
	free(lens.p);
	free(scratch);
	change_free(&c);
	return st;
}

/* ---- encoder --------------------------------------------------------------- */

typedef struct {
	taw_buf out;      /* coded bytes, or a copy of a stored block */
	const uint8_t *raw_view; /* stored block of a memory source */
	size_t raw_len;
	uint32_t crc;
	taw_status st;
} block_result;

typedef struct {
	const taw_source *in;
	const uint8_t *view;
	uint64_t total, bs;
	bool dyn;
	const uint64_t *offs; /* [nblocks+1], dynamic blocks only */
	uint64_t max_raw;
	const taw_model_choice *m;
	taw_pred *pred;   /* [threads] */
	bool *pred_ok;
	uint8_t **buf;    /* [threads] */
	block_result *res; /* [blocks] */
	/* writer */
	const taw_sink *sink;
	uint64_t at;
	taw_buf index;
} enc_job;

static void encode_block(void *arg, size_t i, unsigned w)
{
	enc_job *J = arg;
	block_result *r = &J->res[i];
	const uint64_t off = J->dyn ? J->offs[i] : (uint64_t)i * J->bs;
	const size_t len = J->dyn ? (size_t)(J->offs[i + 1] - off) :
	    (size_t)(J->total - off < J->bs ? J->total - off : J->bs);
	const uint8_t *src = J->view ? J->view + off : nullptr;
	if (!src) {
		if (!J->buf[w] && !(J->buf[w] = malloc((size_t)J->max_raw))) {
			r->st = TAW_E_NOMEM;
			return;
		}
		if ((r->st = J->in->read_at(J->in->ctx, off, J->buf[w], len)) !=
		    TAW_OK)
			return;
		src = J->buf[w];
	}
	if (!J->pred_ok[w]) {
		r->st = taw_pred_init(&J->pred[w], J->m->profile, J->m->quant,
		    J->m->model, J->m->model_size);
		if (r->st != TAW_OK)
			return;
		J->pred_ok[w] = true;
	} else {
		taw_pred_reset(&J->pred[w]);
	}
	r->raw_len = len;
	r->crc = taw_crc32c(0, src, len);
	taw_rc_enc e;
	taw_rc_enc_init(&e, &r->out);
	taw_pred_encode(&J->pred[w], &e, src, len);
	taw_rc_enc_flush(&e);
	if (r->out.oom) {
		r->st = TAW_E_NOMEM;
		return;
	}
	if (r->out.n >= len) { /* stored: coded_len == raw_len */
		r->out.n = 0;
		if (J->view) {
			r->raw_view = src;
		} else {
			taw_buf_put(&r->out, src, len);
			if (r->out.oom)
				r->st = TAW_E_NOMEM;
		}
	}
}

static taw_status write_block(void *arg, size_t i)
{
	enc_job *J = arg;
	block_result *r = &J->res[i];
	taw_status st = r->st;
	const uint8_t *p = r->raw_view ? r->raw_view : r->out.p;
	const size_t n = r->raw_view ? r->raw_len : r->out.n;
	if (st == TAW_OK && n > UINT32_MAX)
		st = TAW_E_ARG;
	if (st == TAW_OK && n)
		st = J->sink->write(J->sink->ctx, p, n);
	if (st == TAW_OK) {
		taw_buf_put_u64(&J->index, J->at);
		taw_buf_put_u32(&J->index, (uint32_t)n);
		taw_buf_put_u32(&J->index, r->crc);
		if (J->dyn)
			taw_buf_put_u32(&J->index, (uint32_t)r->raw_len);
		J->at += n;
	}
	taw_buf_free(&r->out);
	return st;
}

static taw_status sink_put(const taw_sink *s, const taw_buf *b)
{
	if (b->oom)
		return TAW_E_NOMEM;
	return b->n ? s->write(s->ctx, b->p, b->n) : TAW_OK;
}

taw_status taw_seek_encode(const taw_source *src, const taw_params *params,
    const taw_file *files, size_t nfiles, const taw_sink *out)
{
	if (!src || !src->read_at || !out || !out->write || (nfiles && !files))
		return TAW_E_ARG;
	taw_params p;
	taw_params in = params ? *params : (taw_params){0};
	in.seekable = true;
	taw_status st = taw_params_resolve(&in, &p);
	if (st != TAW_OK)
		return st;
	const uint64_t total = src->size, bs = p.block_size;
	uint64_t nblocks = (total + bs - 1) / bs;
	uint64_t *offs = nullptr;
	uint32_t *lens = nullptr;
	if (p.dyn_blocks &&
	    (st = choose_blocks(src, &p, &nblocks, &offs, &lens)) != TAW_OK)
		return st;
	if (nblocks > UINT32_MAX || nfiles > UINT32_MAX ||
	    !files_valid(files, nfiles, total)) {
		free(offs);
		free(lens);
		return TAW_E_ARG;
	}
	const unsigned threads = p.threads ? p.threads : 1;
	uint64_t max_raw = bs;
	if (p.dyn_blocks)
		for (uint64_t i = 0; i < nblocks; i++)
			if (offs[i + 1] - offs[i] > max_raw)
				max_raw = offs[i + 1] - offs[i];

	taw_model_choice m;
	if ((st = taw_model_fit_source(src, bs, &p, &m)) != TAW_OK) {
		free(offs);
		free(lens);
		return st;
	}

	taw_buf head = {0};
	taw_header h = {
		.kind = TAW_KIND_AWSEEK,
		.profile = m.profile,
		.quant = m.quant,
		.level = (uint8_t)p.level,
		.flags = (nfiles ? TAW_FLAG_FILES : 0) |
		    (p.dyn_blocks ? TAW_FLAG_DYN_BLOCKS : 0),
		.block_size = p.block_size,
		.model_size = m.model_size,
		.model_crc = taw_crc32c(0, m.model, m.model_size),
	};
	taw_header_write(&head, &h);
	taw_buf_put(&head, m.model, m.model_size);

	enc_job J = {.in = src, .view = taw_source_view(src), .total = total,
		.bs = bs, .dyn = p.dyn_blocks, .offs = offs, .max_raw = max_raw,
		.m = &m, .sink = out, .at = head.n};
	unsigned tw = threads < nblocks ? threads : (unsigned)(nblocks ? nblocks : 1);
	J.pred = calloc(tw, sizeof *J.pred);
	J.pred_ok = calloc(tw, sizeof *J.pred_ok);
	J.buf = calloc(tw, sizeof *J.buf);
	J.res = calloc(nblocks ? (size_t)nblocks : 1, sizeof *J.res);
	taw_buf_put(&J.index, TAW_INDEX_MAGIC, 4);
	taw_buf_put_u32(&J.index, (uint32_t)nblocks);
	taw_buf_put_u64(&J.index, total);
	if (!J.pred || !J.pred_ok || !J.buf || !J.res)
		st = TAW_E_NOMEM;
	if (st == TAW_OK)
		st = sink_put(out, &head);
	if (st == TAW_OK)
		st = taw_parallel(tw, (size_t)nblocks, 2 * (size_t)tw, encode_block,
		    write_block, &J);
	if (st == TAW_OK) {
		if (nfiles)
			files_put(&J.index, files, nfiles);
		/* the index starts where the blocks end; the footer follows */
		const uint32_t crc =
		    J.index.oom ? 0 : taw_crc32c(0, J.index.p, J.index.n);
		taw_buf_put_u64(&J.index, J.at);
		taw_buf_put_u32(&J.index, crc);
		taw_buf_put(&J.index, TAW_FOOTER_MAGIC, 4);
		st = sink_put(out, &J.index);
	}
	for (uint64_t i = 0; J.res && i < nblocks; i++)
		taw_buf_free(&J.res[i].out);
	for (unsigned w = 0; w < tw && J.pred; w++) {
		if (J.pred_ok[w])
			taw_pred_free(&J.pred[w]);
		free(J.buf[w]);
	}
	free(J.pred);
	free(J.pred_ok);
	free(J.buf);
	free(J.res);
	taw_buf_free(&J.index);
	taw_buf_free(&head);
	free(m.model);
	free(offs);
	free(lens);
	return st;
}

#endif

#if TAW_HAVE_DECODER
/* ---- random-access reader ------------------------------------------------- */

typedef struct {
	uint64_t offset;
	uint32_t coded_len, crc;
	uint32_t raw_len; /* dynamic blocks only */
} seek_entry;

struct taw_seek {
	taw_source src;
	taw_header h;
	uint8_t *model;
	taw_pred pr;
	uint64_t total;
	bool dyn;
	uint32_t nblocks;
	uint64_t *raw_off; /* [nblocks+1], decoded offsets, dynamic blocks only */
	uint64_t max_raw;
	seek_entry *ent;
	taw_file *files;  /* names point into names */
	size_t nfiles;
	char *names;
	uint8_t *coded;   /* scratch for one coded block */
	size_t coded_cap;
	uint8_t *block;   /* last decoded block */
	int64_t cached;   /* its index, or -1 */
	/* parallel reads (taw_seek_set_threads): per-worker state */
	unsigned threads;
	struct seek_worker *wk;
};

typedef struct seek_worker {
	taw_pred pr;
	bool pr_ok;
	uint8_t *coded, *block;
	size_t coded_cap;
} seek_worker;

static taw_status src_read(const taw_source *s, uint64_t off, void *dst,
    uint64_t n)
{
	if (off > s->size || n > s->size - off || n > SIZE_MAX)
		return TAW_E_FORMAT;
	if (n == 0)
		return TAW_OK;
	return s->read_at(s->ctx, off, dst, (size_t)n);
}

/* The directory at d (n bytes, exactly), after the block entries. */
static taw_status parse_files(taw_seek *s, const uint8_t *d, size_t n)
{
	if (n < 4)
		return TAW_E_FORMAT;
	uint32_t count = taw_rd32(d);
	size_t at = 4, names = 0;
	/* first pass: sizes */
	for (uint32_t i = 0; i < count; i++) {
		if (n - at < 18)
			return TAW_E_FORMAT;
		size_t len = taw_rd16(d + at + 16);
		if (len > n - at - 18)
			return TAW_E_FORMAT;
		names += len + 1;
		at += 18 + len;
	}
	if (at != n)
		return TAW_E_FORMAT;
	s->files = calloc(count ? count : 1, sizeof *s->files);
	s->names = malloc(names ? names : 1);
	if (!s->files || !s->names)
		return TAW_E_NOMEM;
	at = 4;
	char *np = s->names;
	for (uint32_t i = 0; i < count; i++) {
		size_t len = taw_rd16(d + at + 16);
		if (memchr(d + at + 18, 0, len))
			return TAW_E_FORMAT;
		memcpy(np, d + at + 18, len);
		np[len] = 0;
		s->files[i] = (taw_file){.name = np, .offset = taw_rd64(d + at),
			.size = taw_rd64(d + at + 8)};
		np += len + 1;
		at += 18 + len;
	}
	s->nfiles = count;
	return files_valid(s->files, count, s->total) ? TAW_OK : TAW_E_FORMAT;
}

static taw_status seek_load(taw_seek *s)
{
	const uint64_t size = s->src.size;
	uint8_t hb[TAW_HEADER_SIZE];
	taw_status st;

	if (size < TAW_HEADER_SIZE + TAW_FOOTER_SIZE)
		return TAW_E_FORMAT;
	if ((st = src_read(&s->src, 0, hb, sizeof hb)) != TAW_OK)
		return st;
	if ((st = taw_header_parse(hb, sizeof hb, &s->h)) != TAW_OK)
		return st;
	if (s->h.kind != TAW_KIND_AWSEEK)
		return TAW_E_ARG;

	const uint64_t data_start = TAW_HEADER_SIZE + s->h.model_size;
	if (s->h.model_size > size - TAW_HEADER_SIZE - TAW_FOOTER_SIZE)
		return TAW_E_FORMAT;
	s->model = malloc(s->h.model_size ? (size_t)s->h.model_size : 1);
	if (!s->model)
		return TAW_E_NOMEM;
	if ((st = src_read(&s->src, TAW_HEADER_SIZE, s->model,
		 s->h.model_size)) != TAW_OK)
		return st;
	if (taw_crc32c(0, s->model, (size_t)s->h.model_size) != s->h.model_crc)
		return TAW_E_CHECKSUM;
	if ((st = taw_pred_init(&s->pr, s->h.profile, s->h.quant, s->model,
		 (size_t)s->h.model_size)) != TAW_OK)
		return st;

	uint8_t ft[TAW_FOOTER_SIZE];
	const uint64_t footer_off = size - TAW_FOOTER_SIZE;
	if ((st = src_read(&s->src, footer_off, ft, sizeof ft)) != TAW_OK)
		return st;
	if (memcmp(ft + 12, TAW_FOOTER_MAGIC, 4) != 0)
		return TAW_E_FORMAT;
	const uint64_t index_off = taw_rd64(ft);
	if (index_off < data_start || index_off > footer_off ||
	    footer_off - index_off < TAW_INDEX_HEAD)
		return TAW_E_FORMAT;
	const uint64_t index_len = footer_off - index_off;
	if (index_len > SIZE_MAX)
		return TAW_E_NOMEM;
	uint8_t *ix = malloc((size_t)index_len);
	if (!ix)
		return TAW_E_NOMEM;
	if ((st = src_read(&s->src, index_off, ix, index_len)) != TAW_OK)
		goto out;
	st = TAW_E_CHECKSUM;
	if (taw_crc32c(0, ix, (size_t)index_len) != taw_rd32(ft + 8))
		goto out;
	st = TAW_E_FORMAT;
	if (memcmp(ix, TAW_INDEX_MAGIC, 4) != 0)
		goto out;
	s->nblocks = taw_rd32(ix + 4);
	s->total = taw_rd64(ix + 8);
	s->dyn = (s->h.flags & TAW_FLAG_DYN_BLOCKS) != 0;
	const uint64_t bs = s->h.block_size;
	const size_t esz = s->dyn ? TAW_INDEX_ENTRY_DYN : TAW_INDEX_ENTRY;
	if (!s->dyn && s->total / bs + (s->total % bs != 0) != s->nblocks)
		goto out;
	const uint64_t entries_end = TAW_INDEX_HEAD + (uint64_t)s->nblocks * esz;
	const bool has_files = s->h.flags & TAW_FLAG_FILES;
	if (has_files ? index_len < entries_end : index_len != entries_end)
		goto out;
	st = TAW_E_NOMEM;
	s->ent = calloc(s->nblocks ? s->nblocks : 1, sizeof *s->ent);
	if (!s->ent)
		goto out;
	if (s->dyn && !(s->raw_off = malloc(((size_t)s->nblocks + 1) *
	    sizeof *s->raw_off)))
		goto out;
	st = TAW_E_FORMAT;
	uint64_t raw_end = 0;
	for (uint32_t i = 0; i < s->nblocks; i++) {
		const uint8_t *e = ix + TAW_INDEX_HEAD + (size_t)i * esz;
		seek_entry *en = &s->ent[i];
		en->offset = taw_rd64(e);
		en->coded_len = taw_rd32(e + 8);
		en->crc = taw_rd32(e + 12);
		if (en->offset < data_start || en->offset > index_off ||
		    en->coded_len > index_off - en->offset)
			goto out;
		if (s->dyn) {
			en->raw_len = taw_rd32(e + 16);
			if (en->raw_len == 0 || en->raw_len > TAW_CHUNK_MAX ||
			    en->coded_len > en->raw_len ||
			    en->raw_len > s->total - raw_end)
				goto out;
			s->raw_off[i] = raw_end;
			raw_end += en->raw_len;
			if (en->raw_len > s->max_raw)
				s->max_raw = en->raw_len;
		}
	}
	if (s->dyn) {
		if (raw_end != s->total)
			goto out;
		s->raw_off[s->nblocks] = raw_end;
	} else {
		s->max_raw = bs;
	}
	st = TAW_OK;
	if (has_files)
		st = parse_files(s, ix + entries_end,
		    (size_t)(index_len - entries_end));
out:
	free(ix);
	return st;
}

taw_status taw_seek_open(const taw_source *src, taw_seek **out)
{
	if (!src || !src->read_at || !out)
		return TAW_E_ARG;
	taw_seek *s = calloc(1, sizeof *s);
	if (!s)
		return TAW_E_NOMEM;
	s->src = *src;
	s->cached = -1;
	taw_status st = seek_load(s);
	if (st != TAW_OK) {
		taw_seek_close(s);
		return st;
	}
	*out = s;
	return TAW_OK;
}

uint64_t taw_seek_size(const taw_seek *s)
{
	return s->total;
}

size_t taw_seek_files(const taw_seek *s, const taw_file **files)
{
	*files = s->files;
	return s->nfiles;
}

long taw_seek_find(const taw_seek *s, const char *name)
{
	for (size_t i = 0; i < s->nfiles; i++)
		if (!strcmp(s->files[i].name, name))
			return (long)i;
	return -1;
}

static uint64_t block_start(const taw_seek *s, uint32_t i)
{
	return s->dyn ? s->raw_off[i] : (uint64_t)i * s->h.block_size;
}

static size_t block_raw_len(const taw_seek *s, uint32_t i)
{
	if (s->dyn)
		return s->ent[i].raw_len;
	uint64_t start = block_start(s, i);
	uint64_t left = s->total - start;
	return left < s->h.block_size ? (size_t)left : s->h.block_size;
}

/* Block containing byte off (off < total). */
static uint32_t block_at(const taw_seek *s, uint64_t off)
{
	if (!s->dyn)
		return (uint32_t)(off / s->h.block_size);
	uint32_t lo = 1, hi = s->nblocks;
	while (lo < hi) {
		const uint32_t mid = lo + (hi - lo) / 2;
		if (s->raw_off[mid] <= off)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo - 1;
}

/* Decode block i into out (block_raw_len bytes) with pr and the coded
 * scratch *coded; checks the CRC. */
static taw_status decode_block_into(const taw_seek *s, uint32_t i,
    taw_pred *pr, uint8_t **coded, size_t *cap, uint8_t *out)
{
	const seek_entry *en = &s->ent[i];
	const size_t len = block_raw_len(s, i);
	taw_status st;
	if (en->coded_len == len) { /* stored */
		if ((st = src_read(&s->src, en->offset, out, len)) != TAW_OK)
			return st;
	} else {
		if (en->coded_len > *cap) {
			uint8_t *c = realloc(*coded, en->coded_len);
			if (!c)
				return TAW_E_NOMEM;
			*coded = c;
			*cap = en->coded_len;
		}
		if ((st = src_read(&s->src, en->offset, *coded,
			 en->coded_len)) != TAW_OK)
			return st;
		taw_pred_reset(pr);
		taw_rc_dec d;
		taw_rc_dec_init(&d, *coded, en->coded_len);
		taw_pred_decode(pr, &d, out, len);
	}
	return taw_crc32c(0, out, len) == en->crc ? TAW_OK : TAW_E_CHECKSUM;
}

static taw_status load_block(taw_seek *s, uint32_t i)
{
	if (s->cached == i)
		return TAW_OK;
	s->cached = -1;
	if (!s->block) {
		s->block = malloc((size_t)s->max_raw);
		if (!s->block)
			return TAW_E_NOMEM;
	}
	taw_status st = decode_block_into(s, i, &s->pr, &s->coded,
	    &s->coded_cap, s->block);
	if (st != TAW_OK)
		return st;
	s->cached = i;
	return TAW_OK;
}

void taw_seek_set_threads(taw_seek *s, unsigned threads)
{
	if (!s)
		return;
	if (threads > 256)
		threads = 256;
	if (threads != s->threads) {
		for (unsigned w = 0; s->wk && w < s->threads; w++) {
			if (s->wk[w].pr_ok)
				taw_pred_free(&s->wk[w].pr);
			free(s->wk[w].coded);
			free(s->wk[w].block);
		}
		free(s->wk);
		s->wk = nullptr;
	}
	s->threads = threads;
}

typedef struct {
	taw_seek *s;
	uint64_t off;    /* of the request */
	size_t n;
	uint8_t *dst;
	uint32_t first;  /* block of job 0 */
	taw_status *st;  /* per job */
} par_read;

static void read_one(void *arg, size_t j, unsigned w)
{
	par_read *R = arg;
	taw_seek *s = R->s;
	seek_worker *k = &s->wk[w];
	const uint32_t i = R->first + (uint32_t)j;
	const uint64_t start = block_start(s, i);
	const size_t len = block_raw_len(s, i);
	taw_status st = TAW_OK;
	if (!k->pr_ok) {
		st = taw_pred_init(&k->pr, s->h.profile, s->h.quant, s->model,
		    (size_t)s->h.model_size);
		if (st != TAW_OK) {
			R->st[j] = st;
			return;
		}
		k->pr_ok = true;
	}
	/* Whole blocks go straight to dst; the edges through a buffer. */
	const bool whole = start >= R->off && start + len <= R->off + R->n;
	uint8_t *out;
	if (whole) {
		out = R->dst + (start - R->off);
	} else {
		if (!k->block && !(k->block = malloc((size_t)s->max_raw))) {
			R->st[j] = TAW_E_NOMEM;
			return;
		}
		out = k->block;
	}
	st = decode_block_into(s, i, &k->pr, &k->coded, &k->coded_cap, out);
	if (st == TAW_OK && !whole) {
		const uint64_t a = R->off > start ? R->off : start;
		const uint64_t b = R->off + R->n < start + len ? R->off + R->n :
		    start + len;
		memcpy(R->dst + (a - R->off), k->block + (a - start),
		    (size_t)(b - a));
	}
	R->st[j] = st;
}

static taw_status read_parallel(taw_seek *s, uint64_t off, void *dst,
    size_t n, uint32_t first, uint32_t last)
{
	const size_t jobs = (size_t)(last - first) + 1;
	const unsigned tw = s->threads < jobs ? s->threads : (unsigned)jobs;
	if (!s->wk && !(s->wk = calloc(s->threads, sizeof *s->wk)))
		return TAW_E_NOMEM;
	par_read R = {.s = s, .off = off, .n = n, .dst = dst, .first = first};
	R.st = calloc(jobs, sizeof *R.st);
	if (!R.st)
		return TAW_E_NOMEM;
	taw_status st = taw_parallel(tw, jobs, 0, read_one, nullptr, &R);
	for (size_t j = 0; j < jobs && st == TAW_OK; j++)
		st = R.st[j];
	free(R.st);
	return st;
}

taw_status taw_seek_read(taw_seek *s, uint64_t off, void *dst, size_t n)
{
	if (!s || (!dst && n))
		return TAW_E_ARG;
	if (off > s->total || n > s->total - off)
		return TAW_E_RANGE;
	if (n && s->threads > 1) {
		const uint32_t first = block_at(s, off);
		const uint32_t last = block_at(s, off + n - 1);
		if (last > first)
			return read_parallel(s, off, dst, n, first, last);
	}
	uint8_t *d = dst;
	while (n) {
		uint32_t i = block_at(s, off);
		size_t in = (size_t)(off - block_start(s, i));
		taw_status st = load_block(s, i);
		if (st != TAW_OK)
			return st;
		size_t k = block_raw_len(s, i) - in;
		if (k > n)
			k = n;
		memcpy(d, s->block + in, k);
		d += k;
		off += k;
		n -= k;
	}
	return TAW_OK;
}

void taw_seek_close(taw_seek *s)
{
	if (!s)
		return;
	taw_seek_set_threads(s, 0);
	taw_pred_free(&s->pr);
	free(s->model);
	free(s->ent);
	free(s->raw_off);
	free(s->files);
	free(s->names);
	free(s->coded);
	free(s->block);
	free(s);
}

#endif
