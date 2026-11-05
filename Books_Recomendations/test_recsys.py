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
