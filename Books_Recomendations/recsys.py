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
