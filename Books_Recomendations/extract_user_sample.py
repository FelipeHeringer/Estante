"""Amostra de usuários do goodreads_interactions.csv (lido em chunks, 2 passadas)."""
import numpy as np
import pandas as pd

DTYPES = {"user_id": "int32", "book_id": "int32", "is_read": "int8", "rating": "int8", "is_reviewed": "int8"}


def _chunks(csv_path, chunksize):
    return pd.read_csv(csv_path, dtype=DTYPES, usecols=["user_id", "book_id", "rating"], chunksize=chunksize)


def sample_users(csv_path, book_id_map_path, n_users, min_rated, seed=0, chunksize=2_000_000):
    """Interações avaliadas (rating > 0) de até n_users usuários com >= min_rated avaliações."""
    counts = np.zeros(0, dtype=np.int64)
    for ch in _chunks(csv_path, chunksize):
        ch = ch[ch.rating > 0]
        c = np.bincount(ch.user_id.to_numpy())
        if len(c) > len(counts):
            counts = np.pad(counts, (0, len(c) - len(counts)))
        counts[: len(c)] += c
    eligible = np.flatnonzero(counts >= min_rated)
    chosen = np.random.default_rng(seed).choice(eligible, size=min(n_users, len(eligible)), replace=False)
    keep = np.zeros(len(counts), dtype=bool)
    keep[chosen] = True

    parts = []
    for ch in _chunks(csv_path, chunksize):
        ch = ch[ch.rating > 0]
        parts.append(ch[keep[ch.user_id.to_numpy()]])
    out = pd.concat(parts, ignore_index=True)
    bmap = pd.read_csv(book_id_map_path).rename(columns={"book_id_csv": "book_id", "book_id": "book_id_real"})
    out = out.merge(bmap, on="book_id", how="inner").drop(columns="book_id").rename(columns={"book_id_real": "book_id"})
    return out[["user_id", "book_id", "rating"]].astype({"user_id": "int32", "book_id": "int64", "rating": "int8"})


if __name__ == "__main__":
    df = sample_users("goodreads_interactions.csv", "book_id_map.csv", n_users=50_000, min_rated=20)
    df.to_parquet("user_interactions.parquet", index=False)
    print(f"{df.user_id.nunique()} usuários, {len(df)} interações")
