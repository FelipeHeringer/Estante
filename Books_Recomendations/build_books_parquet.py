"""One-off: goodreads_books.json (9.2 GB) -> books.parquet (~50 MB).

Streams line by line with orjson, keeping only the columns we need, so the
huge fields (popular_shelves, description, similar_books) are never
materialised into a DataFrame.
"""
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

SRC = './goodreads_books/goodreads_books.json'
DST = './books.parquet'
BATCH = 200_000

SCHEMA = pa.schema([
    ('book_id',            pa.int64()),
    ('work_id',            pa.int64()),
    ('title',              pa.string()),
    ('average_rating',     pa.float32()),
    ('ratings_count',      pa.int32()),
    ('text_reviews_count', pa.int32()),
    ('num_pages',          pa.int32()),
    ('publication_year',   pa.int16()),
    ('is_ebook',           pa.bool_()),
    ('authors',            pa.list_(pa.int64())),
    ('primary_author',     pa.int64()),
])


def as_int(v):
    return int(v) if v else None


def as_small(v, lo, hi):
    """Cast to int but drop values that would overflow a narrow column."""
    if not v:
        return None
    n = int(v)
    return n if lo <= n <= hi else None


def as_float(v):
    return float(v) if v else None


def main():
    cols = {name: [] for name in SCHEMA.names}
    writer = pq.ParquetWriter(DST, SCHEMA, compression='zstd')
    n = 0
    try:
        with open(SRC, 'rb') as f:
            for line in f:
                d = orjson.loads(line)
                cols['book_id'].append(as_int(d['book_id']))
                cols['work_id'].append(as_int(d['work_id']))
                cols['title'].append(d['title'] or None)
                cols['average_rating'].append(as_float(d['average_rating']))
                cols['ratings_count'].append(as_small(d['ratings_count'], -2**31, 2**31 - 1))
                cols['text_reviews_count'].append(as_small(d['text_reviews_count'], -2**31, 2**31 - 1))
                cols['num_pages'].append(as_small(d['num_pages'], -2**31, 2**31 - 1))
                cols['publication_year'].append(as_small(d['publication_year'], -32768, 32767))
                cols['is_ebook'].append(d['is_ebook'] == 'true')
                authors = d['authors']
                cols['authors'].append([as_int(a['author_id']) for a in authors])
                # Primary author = first one whose role is blank; None if the
                # book only credits editors/contributors/etc.
                cols['primary_author'].append(next(
                    (as_int(a['author_id']) for a in authors if a['role'] == ''),
                    None))
                n += 1
                if n % BATCH == 0:
                    writer.write_table(pa.table(cols, schema=SCHEMA))
                    cols = {name: [] for name in SCHEMA.names}
                    print(f'{n:,} rows', flush=True)
        if cols['book_id']:
            writer.write_table(pa.table(cols, schema=SCHEMA))
    finally:
        writer.close()
    print(f'done: {n:,} rows -> {DST}', flush=True)


if __name__ == '__main__':
    main()
