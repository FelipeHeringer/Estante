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


def split_user(book_ids, ratings, seed, n_profile=5):
    """n_profile leituras aleatórias formam o perfil; o restante fica escondido."""
    book_ids, ratings = np.asarray(book_ids), np.asarray(ratings)
    if len(book_ids) <= n_profile:
        raise ValueError(f"são necessárias mais de {n_profile} leituras")
    perm = np.random.default_rng(seed).permutation(len(book_ids))
    p, h = perm[:n_profile], perm[n_profile:]
    return book_ids[p], ratings[p], book_ids[h], ratings[h]


def generate_candidates(profile, book_probs, popularity, exclude, n):
    """Candidatos por afinidade de cluster x log da popularidade, sem os livros em exclude."""
    pop = popularity.reindex(book_probs.index).fillna(0).to_numpy()
    score = (book_probs.to_numpy() @ profile) * np.log1p(pop)
    score[book_probs.index.isin(list(exclude))] = -np.inf
    order = np.argsort(-score, kind="stable")[:n]
    return [int(book_probs.index[i]) for i in order if np.isfinite(score[i])]


def pair_features(profile, segment_probs, book_vector):
    """Entrada da rede: perfil + segmento + vetor do livro (1D -> 1D; 2D -> uma linha por livro)."""
    bv = np.asarray(book_vector, dtype=float)
    b2 = np.atleast_2d(bv)
    head = np.concatenate([profile, segment_probs])
    X = np.hstack([np.tile(head, (len(b2), 1)), b2])
    return X[0] if bv.ndim == 1 else X


def recommend(read_ids, ratings, book_probs, popularity, user_gmm_pipeline, model, book_vectors,
              top_n=10, n_candidates=200):
    profile = user_profile(book_probs, read_ids, ratings)
    segment = user_gmm_pipeline.predict_proba(profile[None])[0]
    cands = generate_candidates(profile, book_probs, popularity, set(read_ids), n_candidates)
    if not cands:
        return []
    X = pair_features(profile, segment, book_vectors.loc[cands].to_numpy())
    score = model.predict_proba(X)[:, 1]
    return [cands[i] for i in np.argsort(-score, kind="stable")[:top_n]]
