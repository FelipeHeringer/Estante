# Data Understanding — Dataset de Livros (CRISP-DM)

Relatório da fase **Data Understanding** do CRISP-DM, referente ao `books_df`
construído em [clustering_model.ipynb](clustering_model.ipynb) a partir do
cruzamento entre metadados de livros, autores, gêneros e interações de
usuários do Goodreads.

## 1. Coleta dos Dados Iniciais

| Fonte | Arquivo | Papel no cruzamento |
|---|---|---|
| Livros | `books.parquet` (derivado de `goodreads_books.json`, 9.2 GB originais) | Base: 1 linha por `book_id` |
| Autores | `goodreads_book_authors/goodreads_book_authors.json` | Join por `author_id` (autor primário do livro) |
| Gêneros | `goodreads_book_genres_initial/goodreads_book_genres_initial.json` | Join por `book_id` (contagem de shelves por gênero) |
| Interações | `goodreads_interactions.csv` (4.3 GB, ~228M linhas) | Agregado por livro (chunked) e mapeado via `book_id_map.csv` |

Todas as fontes são públicas (Goodreads dataset da UCSD Book Graph). Não há
dados próprios da empresa/usuário nem PII além de hashes de `user_id`, que
não entram no `books_df`.

## 2. Descrição dos Dados

`books_df` final: **2.360.655 linhas × 29 colunas**, 1 linha por `book_id`
(chave primária, sem duplicatas).

| Grupo | Colunas |
|---|---|
| Identificação | `book_id`, `work_id`, `title` |
| Metadados do livro | `average_rating`, `ratings_count`, `text_reviews_count_x`, `num_pages`, `publication_year`, `is_ebook` |
| Autor | `author_id`, `name`, `author_avg_rating`, `author_ratings_count`, `text_reviews_count_y` |
| Gênero (10 shelves) | `fiction`, `history, historical fiction, biography`, `fantasy, paranormal`, `mystery, thriller, crime`, `poetry`, `romance`, `non-fiction`, `children`, `young-adult`, `comics, graphic` |
| Interações agregadas | `n_interactions`, `n_read`, `n_reviewed`, `n_rated`, `user_avg_rating` |

Tipos: 1 `int64` (`book_id`), 1 `bool` (`is_ebook`), 2 `str`, 1 `float32`,
24 `float64` (a maioria por causa de `NaN` em colunas originalmente
inteiras). Memória em pandas: ~605 MB.

## 3. Exploração dos Dados

**Metadados do livro** — `average_rating` concentrado (média 3,87, desvio
0,54, quase sempre entre 3 e 4,5); `ratings_count` extremamente assimétrico
(mediana 20, média 406, máximo 4,9M — poucos best-sellers dominam);
`num_pages` média 264 (mediana 245).

**Interações** — mesmo padrão de cauda longa: `n_interactions` mediana 9,
média 97, máximo 314.685. `user_avg_rating` (média das notas dadas por
usuários no arquivo de interações) tem média 3,84, muito próxima de
`average_rating` do livro (3,87) — sinal de consistência entre as duas
fontes.

**Gêneros** — cobertura por shelf (nº de livros com contagem > 0):

| Gênero | Livros |
|---|---|
| fiction | 1.234.392 |
| history/historical fiction/biography | 659.729 |
| romance | 655.682 |
| fantasy, paranormal | 535.322 |
| non-fiction | 529.593 |
| mystery, thriller, crime | 520.783 |
| young-adult | 361.643 |
| children | 254.859 |
| comics, graphic | 170.086 |
| poetry | 87.100 |

**414.038 livros (17,5%) não têm nenhum shelf de gênero preenchido** (todas
as 10 colunas em zero) — relevante para qualquer feature baseada em gênero.

**Correlações esperadas e observadas** — `ratings_count` (do livro) e
`n_rated`/`n_interactions` (das interações) são fortemente correlacionados,
por virem da mesma atividade de usuários em fontes diferentes; isso indica
redundância a considerar na modelagem (ver seção 5).

## 4. Verificação da Qualidade dos Dados

| Verificação | Resultado |
|---|---|
| `book_id` duplicado | 0 — chave primária íntegra |
| `title` duplicado | 593.428 — esperado (edições/reimpressões diferentes do mesmo título), não é erro |
| `publication_year` > 2026 | 480 livros (ex.: valores como 32014, 22016 — erro de digitação/OCR na fonte) |
| `publication_year` == 0 | 12 livros |
| `num_pages` == 0 | 7.528 livros |
| `num_pages` > 5.000 | 125 livros (possíveis outliers/erros) |
| Contagens de gênero negativas | 16.368 livros com pelo menos 1 gênero negativo (mínimo observado: -87 em `fiction`) — anomalia da fonte (shelves), não deveria existir contagem negativa |

**Valores ausentes** (% de `books_df`):

| Coluna | % Nulo |
|---|---|
| `num_pages` | 32,37% |
| `publication_year` | 25,40% |
| `author_id` / `author_avg_rating` / `author_ratings_count` / `name` / `text_reviews_count_y` | 2,33% |
| `user_avg_rating` | 1,49% |
| `work_id`, `ratings_count`, `average_rating`, `text_reviews_count_x` | 0,02% |
| Demais (título, gêneros, colunas de interação, `book_id`) | 0,00% |

`user_avg_rating` nulo é **esperado** para os 1,49% de livros sem nenhuma
interação no arquivo amostral (`n_interactions = 0`), e não foi preenchido
propositalmente (ver `books_df` no notebook) — diferente de "sem dado por
erro".

## 5. Perguntas Principais Respondidas

**Quais dados temos e de onde vêm?**
Metadados de ~2,36M livros do Goodreads, autor primário de cada livro,
contagem de leitores por gênero (shelves) e estatísticas agregadas de
~228M interações de usuários (leituras, avaliações, resenhas), todas unidas
por `book_id`.

**Qual o volume e é viável processar?**
O arquivo de interações (4,3 GB) inviabiliza carga direta em memória; foi
processado em chunks e agregado por livro antes do merge. O `books_df`
final (605 MB em memória) é tratável normalmente.

**Os dados já sugerem estrutura útil para clustering/recomendação?**
Sim: distribuição de popularidade em cauda longa (poucos livros muito
populares) sugere transformação log antes de qualquer clustering baseado em
distância; os 10 gêneros formam um vetor esparso de "perfil temático" por
livro, útil como feature; `average_rating` do livro e `user_avg_rating` das
interações concordam entre si, reforçando a confiabilidade de ambos.

**Existem problemas de qualidade que exigem tratamento na próxima etapa
(Data Preparation)?**
Sim — quatro pontos concretos: (1) anos de publicação inválidos (0 ou
> 2026, 492 livros); (2) contagens de gênero negativas (16.368 livros);
(3) proporção relevante de nulos em `num_pages` (32%) e `publication_year`
(25%), exigindo decisão de imputação vs. exclusão; (4) redundância entre
`ratings_count`/`text_reviews_count_x` (do livro) e `n_rated`/`n_reviewed`
(das interações), que vieram de fontes diferentes e medem quase a mesma
coisa — avaliar antes de usar ambas como features para não duplicar sinal.

**Quais atributos parecem mais relevantes para o objetivo do modelo?**
`average_rating`/`user_avg_rating`, `n_interactions`/`n_read` (popularidade
e engajamento), gêneros (perfil temático) e `author_avg_rating` são os
candidatos mais diretos a features; `title`, `name` (autor) e IDs servem
para identificação/rastreabilidade, não como features numéricas diretas.

## 6. Recomendações para Data Preparation

- Tratar/filtrar `publication_year` fora do intervalo plausível (ex.: < 1400 ou > ano atual).
- Aplicar `clip(lower=0)` nas 10 colunas de gênero para remover contagens negativas.
- Definir estratégia de imputação para `num_pages` e `publication_year` (ex.: mediana por gênero/época, ou flag de ausência) em vez de descartar ~32%/25% das linhas.
- Escolher entre `ratings_count`/`text_reviews_count_x` (livro) e `n_rated`/`n_reviewed` (interações) ou combiná-las explicitamente, evitando colinearidade no modelo.
- Aplicar `log1p` em `ratings_count`, `n_interactions`, `n_read` e demais colunas de cauda longa antes de qualquer algoritmo sensível a escala (ex.: K-Means).
