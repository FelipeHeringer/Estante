"""Funções de recomendação de livros (features, perfil, métricas, ranking)."""

import numpy as np
import pandas as pd


def hit_at_k(ranked, positives, k):
    if not positives:
        return None
    return float(any(i in positives for i in ranked[:k]))


def recall_at_k(ranked, positives, k):
    if not positives:
        return None
    return len(set(ranked[:k]) & positives) / len(positives)


def ndcg_at_k(ranked, positives, k):
    if not positives:
        return None
    dcg = sum(1 / np.log2(pos + 2) for pos, i in enumerate(ranked[:k]) if i in positives)
    idcg = sum(1 / np.log2(pos + 2) for pos in range(min(len(positives), k)))
    return dcg / idcg


def book_features(df, genre_cols):
    """Features numéricas limpas e sem NaN, mesmo índice de df."""
    year = df["publication_year"].where(df["publication_year"].between(1400, 2026))
    pages = df["num_pages"].where(df["num_pages"].between(1, 5000))
    out = pd.DataFrame(index=df.index)
    out["log_ratings"] = np.log1p(df["ratings_count"].fillna(0))
    out["average_rating"] = df["average_rating"].fillna(df["average_rating"].median())
    out["year_missing"] = year.isna().astype(int)
    out["pages_missing"] = pages.isna().astype(int)
    out["log_pages"] = np.log1p(pages.fillna(pages.median()))
    out["decade"] = (year.fillna(year.median()) // 10 * 10)
    g = df[genre_cols].clip(lower=0).fillna(0)
    tot = g.sum(axis=1).replace(0, np.nan)
    for c in genre_cols:
        out[f"g_{c}"] = (g[c] / tot).fillna(0)
    return out


def user_profile(book_probs, read_ids, ratings):
    """Média das probabilidades de cluster dos livros lidos, ponderada pela nota."""
    seen, ids, w = set(), [], []
    for i, r in zip(read_ids, ratings):
        if i in book_probs.index and i not in seen:
            seen.add(i)
            ids.append(i)
            w.append(r)
    if not ids:
        raise ValueError("nenhum livro informado existe na tabela de clusters")
    w = np.asarray(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(ids), 1 / len(ids))
    prof = w @ book_probs.loc[ids].to_numpy()
    return prof / prof.sum()
