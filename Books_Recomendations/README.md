# Books Recomendations

ClusterizaÃ§Ã£o/recomendaÃ§Ã£o de livros com o dataset pÃºblico **Goodreads (UCSD Book Graph)**.
Os datasets nÃ£o estÃ£o no repositÃ³rio (vÃ¡rios GB) â€” siga os passos abaixo para reproduzir.

## 1. Baixar o dataset

PÃ¡gina oficial: <https://mengtingwan.github.io/data/goodreads.html>
(os arquivos sÃ£o servidos por links do Google Drive listados na pÃ¡gina).

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
â”œâ”€â”€ goodreads_books/goodreads_books.json
â”œâ”€â”€ goodreads_book_authors/goodreads_book_authors.json
â”œâ”€â”€ goodreads_book_genres_initial/goodreads_book_genres_initial.json
â”œâ”€â”€ goodreads_interactions.csv
â”œâ”€â”€ book_id_map.csv
â”œâ”€â”€ build_books_parquet.py
â””â”€â”€ clustering_model.ipynb
```

> EspaÃ§o em disco: ~15 GB descompactado (`goodreads_books.json` â‰ˆ 9 GB, interaÃ§Ãµes â‰ˆ 4 GB).

## 2. Instalar dependÃªncias

```bash
pip install pipenv
pipenv install
```

## 3. Gerar o `books.parquet` (uma vez)

O JSON de livros Ã© grande demais para o pandas, entÃ£o o script lÃª linha a linha e salva sÃ³ as colunas necessÃ¡rias (~100 MB):

```bash
pipenv run python build_books_parquet.py
```

## 4. Rodar o notebook

Abra `clustering_model.ipynb` (kernel do pipenv) e execute **Run All**.
Os DataFrames **sÃ£o criados em memÃ³ria pelo prÃ³prio notebook** â€” nada alÃ©m do passo 3 precisa ser feito antes:

| DataFrame | Origem |
|---|---|
| `interactions_data` | `goodreads_interactions.csv` agregado por livro em chunks + `book_id_map.csv` |
| `books` | `books.parquet` |
| `primary` | autor primÃ¡rio de cada livro (derivado de `books`) |
| `genres` | `goodreads_book_genres_initial.json` |
| `authors` | `goodreads_book_authors.json` |
| **`books_df`** | merge de todos acima: 1 linha por `book_id`, ~2,36M linhas Ã— 29 colunas (~600 MB em RAM) |

A agregaÃ§Ã£o das interaÃ§Ãµes (~228M linhas) Ã© a etapa mais lenta; leva alguns minutos.
DescriÃ§Ã£o detalhada das colunas e da qualidade dos dados: [Data_Understanding.md](Data_Understanding.md).
