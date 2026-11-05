def test_import():
    import recsys  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from recsys import hit_at_k, ndcg_at_k, recall_at_k


def test_recall_perfeito():
    assert recall_at_k([1, 2, 3], {1, 2}, 2) == 1.0


def test_recall_parcial():
    assert recall_at_k([9, 1, 2], {1, 2}, 2) == 0.5


def test_hit():
    assert hit_at_k([9, 1, 2], {1, 2}, 2) == 1.0
    assert hit_at_k([9, 8, 1], {1}, 2) == 0.0


def test_ndcg_ordem():
    assert ndcg_at_k([1, 9], {1}, 2) == 1.0
    assert abs(ndcg_at_k([9, 1], {1}, 2) - 1 / np.log2(3)) < 1e-9


def test_sem_positivos():
    for f in (hit_at_k, recall_at_k, ndcg_at_k):
        assert f([1, 2], set(), 2) is None

from recsys import book_features, user_profile

GEN = ["fiction", "romance"]


def _df(rows):
    return pd.DataFrame(rows, columns=["ratings_count", "average_rating", "num_pages",
                                       "publication_year", "fiction", "romance"])


def test_features_sem_nan_e_limpeza():
    df = _df([[100, 4.0, 300, 2000, -5, 2],
              [10, np.nan, 0, 32014, 1, 0],
              [np.nan, 3.5, np.nan, np.nan, 0, 0]])
    f = book_features(df, GEN)
    assert not f.isna().any().any()
    assert f["year_missing"].tolist() == [0, 1, 1]
    assert f["pages_missing"].tolist() == [0, 1, 1]
    assert (f[["g_fiction", "g_romance"]] >= 0).all().all()
    assert f.index.equals(df.index)


def test_features_genero_proporcao():
    f = book_features(_df([[1, 4, 100, 2000, 3, 1], [1, 4, 100, 2000, 0, 0]]), GEN)
    assert f.loc[0, ["g_fiction", "g_romance"]].sum() == pytest.approx(1)
    assert f.loc[0, "g_fiction"] == pytest.approx(0.75)
    assert f.loc[1, ["g_fiction", "g_romance"]].sum() == 0


def _probs():
    return pd.DataFrame({"p_0": [1.0, 0.0], "p_1": [0.0, 1.0]}, index=pd.Index([10, 20], name="book_id"))


def test_perfil_pondera_por_nota():
    assert user_profile(_probs(), [10, 20], [4, 1]) == pytest.approx([0.8, 0.2])


def test_perfil_notas_zero():
    assert user_profile(_probs(), [10, 20], [0, 0]) == pytest.approx([0.5, 0.5])


def test_perfil_ignora_desconhecido_e_repetido():
    assert user_profile(_probs(), [10, 10, 99], [5, 5, 5]) == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError):
        user_profile(_probs(), [98, 99], [5, 5])
