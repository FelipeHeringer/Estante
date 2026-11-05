import pandas as pd

from extract_user_sample import sample_users


def _files(tmp_path):
    rows = [  # user_id, book_id_csv, is_read, rating, is_reviewed
        (0, 0, 1, 5, 0), (0, 1, 1, 4, 0), (0, 2, 1, 3, 0), (0, 3, 1, 0, 0),  # A: 3 avaliadas + 1 sem nota
        (1, 0, 1, 5, 0), (1, 1, 1, 4, 0), (1, 2, 1, 0, 0),                    # B: 2 avaliadas
        (2, 1, 1, 2, 0), (2, 2, 1, 2, 0), (2, 3, 1, 2, 0),                    # C: 3 avaliadas
    ]
    csv = tmp_path / "inter.csv"
    pd.DataFrame(rows, columns=["user_id", "book_id", "is_read", "rating", "is_reviewed"]).to_csv(csv, index=False)
    m = tmp_path / "map.csv"
    pd.DataFrame({"book_id_csv": [0, 1, 2, 3], "book_id": [100, 101, 102, 103]}).to_csv(m, index=False)
    return str(csv), str(m)


def test_filtra_por_minimo_e_ignora_sem_nota(tmp_path):
    csv, m = _files(tmp_path)
    df = sample_users(csv, m, n_users=10, min_rated=3, chunksize=4)
    assert sorted(df.user_id.unique()) == [0, 2]
    assert (df.rating > 0).all()
    assert len(df[df.user_id == 0]) == 3


def test_converte_book_id_para_real(tmp_path):
    csv, m = _files(tmp_path)
    df = sample_users(csv, m, n_users=10, min_rated=3, chunksize=4)
    assert set(df.book_id) == {100, 101, 102, 103}


def test_limita_n_users_com_seed_fixa(tmp_path):
    csv, m = _files(tmp_path)
    a = sample_users(csv, m, n_users=1, min_rated=3, seed=0, chunksize=4)
    b = sample_users(csv, m, n_users=1, min_rated=3, seed=0, chunksize=4)
    assert a.user_id.nunique() == 1
    assert a.equals(b)


def test_usuario_sem_nota_com_maior_id(tmp_path):
    csv, m = _files(tmp_path)
    with open(csv, "a") as f:
        f.write("9,0,1,0,0\n")  # id maior que qualquer avaliador
    df = sample_users(csv, m, n_users=10, min_rated=3, chunksize=4)
    assert sorted(df.user_id.unique()) == [0, 2]
