# Books Recomendations

Clusterização/recomendação de livros com o dataset público **Goodreads (UCSD Book Graph)**.
Os datasets não estão no repositório (vários GB) — siga os passos abaixo para reproduzir.

## 1. Baixar o dataset

Página oficial: <https://mengtingwan.github.io/data/goodreads.html>
(os arquivos são servidos por links do Google Drive listados na página).

Baixe estes arquivos:

| Arquivo | Onde colocar |
|---|---|
| `goodreads_books.json.gz` | descompactar em `goodreads_books/goodreads_books.json` |
| `goodreads_book_authors.json.gz` | descompactar em `goodreads_book_authors/goodreads_book_authors.json` |
| `goodreads_book_genres_initial.json.gz` | descompactar em `goodreads_book_genres_initial/goodreads_book_genres_initial.json` |
| `goodreads_interactions.csv` | raiz do projeto |
| `book_id_map.csv` | raiz do projeto |

Estrutura final esperada:

```
Books_Recomendations/
├── goodreads_books/goodreads_books.json
├── goodreads_book_authors/goodreads_book_authors.json
├── goodreads_book_genres_initial/goodreads_book_genres_initial.json
├── goodreads_interactions.csv
├── book_id_map.csv
├── build_books_parquet.py
└── clustering_model.ipynb
```

> Espaço em disco: ~15 GB descompactado (`goodreads_books.json` ≈ 9 GB, interações ≈ 4 GB).

## 2. Instalar dependências

```bash
pip install pipenv
pipenv install
```

## 3. Gerar o `books.parquet` (uma vez)

O JSON de livros é grande demais para o pandas, então o script lê linha a linha e salva só as colunas necessárias (~100 MB):

```bash
pipenv run python build_books_parquet.py
```

## 4. Rodar o notebook

Abra `clustering_model.ipynb` (kernel do pipenv) e execute **Run All**.
Os DataFrames **são criados em memória pelo próprio notebook** — nada além do passo 3 precisa ser feito antes:

| DataFrame | Origem |
|---|---|
| `interactions_data` | `goodreads_interactions.csv` agregado por livro em chunks + `book_id_map.csv` |
| `books` | `books.parquet` |
| `primary` | autor primário de cada livro (derivado de `books`) |
| `genres` | `goodreads_book_genres_initial.json` |
| `authors` | `goodreads_book_authors.json` |
| **`books_df`** | merge de todos acima: 1 linha por `book_id`, ~2,36M linhas × 29 colunas (~600 MB em RAM) |

A agregação das interações (~228M linhas) é a etapa mais lenta; leva alguns minutos.
Descrição detalhada das colunas e da qualidade dos dados: [Data_Understanding.md](Data_Understanding.md).

## 5. Modelagem e avaliação

Depois do passo 4 (que grava `books_df.parquet` ao final), execute na ordem:

| # | Comando / notebook | Gera |
|---|---|---|
| 1 | `pipenv run python extract_user_sample.py` | `user_interactions.parquet` (50 mil usuários com ≥ 20 avaliações) |
| 2 | `02_gmm_livros.ipynb` | `book_probs.parquet`, `book_vectors.parquet`, `book_meta.parquet`, `gmm_books.joblib` |
| 3 | `03_gmm_usuarios.ipynb` | `user_profiles.parquet`, `user_segments.parquet`, `gmm_users.joblib` |
| 4 | `04_rede_neural_avaliacao.ipynb` | `mlp.joblib`, `results.csv`, `figures/` |

A lógica testável (features, perfil do usuário, métricas, candidatos, recomendação) fica em `recsys.py`:

```bash
pipenv run pytest -q
```

Resultados da avaliação offline (Hit/Recall/NDCG@10, pool de até 5 positivos + 99 negativos por usuário de teste) ficam em `results.csv` e nas saídas do notebook 04.
