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

from recsys import generate_candidates, pair_features, recommend, split_user


def test_split_user():
    ids, r = np.arange(12), np.arange(12) % 5 + 1
    p_ids, p_r, h_ids, h_r = split_user(ids, r, seed=1)
    assert len(p_ids) == 5 and len(h_ids) == 7
    assert not set(p_ids) & set(h_ids)
    assert (p_r == r[p_ids]).all() and (h_r == r[h_ids]).all()
    again = split_user(ids, r, seed=1)
    assert (again[0] == p_ids).all()


def test_split_poucas_leituras():
    with pytest.raises(ValueError):
        split_user(np.arange(5), np.ones(5), seed=0)


def _world():
    ids = [1, 2, 3, 4, 5, 6]
    probs = pd.DataFrame({"p_0": [1, 1, 1, 0, 0, 0], "p_1": [0, 0, 0, 1, 1, 1]}, index=pd.Index(ids, name="book_id"), dtype=float)
    pop = pd.Series([10, 100, 50, 1000, 5, 1], index=ids, dtype=float)
    vecs = pd.DataFrame({"v": np.array(ids) / 10}, index=probs.index)
    return probs, pop, vecs


def test_candidatos_excluem_lidos_e_limitam():
    probs, pop, _ = _world()
    c = generate_candidates(np.array([1.0, 0.0]), probs, pop, exclude={2}, n=10)
    assert 2 not in c and sorted(c) == [1, 3, 4, 5, 6] and len(c) == 5
    assert c[:2] == [3, 1]  # cluster 0 primeiro, mais popular primeiro


class _Seg:
    def predict_proba(self, X):
        return np.tile([1.0], (len(X), 1))


class _Fake:
    def predict_proba(self, X):
        s = X[:, -1]
        return np.column_stack([1 - s, s])


def test_pair_features_tamanho():
    X = pair_features(np.array([.5, .5]), np.array([1.0]), np.array([[1, 2, 3], [4, 5, 6]]))
    assert X.shape == (2, 6)
    assert pair_features(np.array([.5, .5]), np.array([1.0]), np.array([1, 2, 3])).shape == (6,)


def test_recommend_com_modelo_fake():
    probs, pop, vecs = _world()
    rec = recommend([1, 2], [5, 5], probs, pop, _Seg(), _Fake(), vecs, top_n=3, n_candidates=10)
    assert rec == [6, 5, 4]  # maior valor de v primeiro
    assert not {1, 2} & set(rec)

from recsys import clr


def test_clr_centra_e_aguenta_zero():
    x = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    z = clr(x)
    assert z.shape == x.shape and np.isfinite(z).all()
    assert z.sum(axis=1) == pytest.approx([0, 0], abs=1e-9)
    assert z[0, 0] == pytest.approx(z[0, 1])
