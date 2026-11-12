# Cuisines through ingredients and words

Final project for DTU course **02805 Social Graphs and Interactions**.

We study how cuisines differ from and connect to each other in two ways. The first is their ingredients, using network analysis: recipe–ingredient bipartite networks, ingredient co-occurrence networks, communities and centrality. The second is the language of recipes and user reviews, using NLP: for example, how often reviewers call a dish "authentic" and what they substitute. We use the **Food.com Recipes and Interactions** dataset, which contains about 230k recipes and 1.1M user reviews posted on Food.com between 1999 and 2018. Each recipe has its ingredient list, steps, nutrition and user tags (including cuisine tags). The dataset was collected and released by Majumder et al. (2019) and is available on Kaggle as [`shuyangli94/food-com-recipes-and-user-interactions`](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

> Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni and Julian McAuley. 2019. *Generating Personalized Recipes from Historical User Preferences.* In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5976–5982. Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1613

## Setup (macOS / Linux)

You need `git`, Python 3 (we use Python 3.14.5) and `unzip`.

```bash
# 1. Clone the repository
git clone <repository-url> sgi-project
cd sgi-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Register the environment as a Jupyter kernel
python -m ipykernel install --user --name socialgraphs-food --display-name "socialgraphs-food"
```

**Windows:** create the environment with `python -m venv .venv`, then activate it with
`.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell).
The other commands are the same. Run `scripts/download_data.sh` from Git Bash or WSL, or download the files by hand (see below).

### Get the data

You can download the files from our one drive folder and put it in `data/raw/`.

### Run the notebooks

```bash
source .venv/bin/activate
jupyter lab        # or: jupyter notebook
```

Open a notebook in `notebooks/` and choose the **socialgraphs-food** kernel (*Kernel → Change Kernel*). Notebooks expect to be started from the `notebooks/` folder or the repository root.

To re-run a notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_initial_eda.ipynb
```

## Folder structure

```
.
├── data/
│   ├── raw/              # original Kaggle files, never modified (NOT in git)
│   └── processed/        # project datasets built by notebook 02 (in git, see below)
├── notebooks/            # Jupyter notebooks, numbered in order
│   ├── 01_initial_eda.ipynb
│   └── 02_build_datasets.ipynb
├── src/                  # reusable Python code shared by notebooks
├── scripts/
│   └── download_data.sh  # downloads the dataset from Kaggle into data/raw/
├── figures/              # PNG figures saved by the notebooks
├── requirements.txt      # pinned Python dependencies
└── README.md
```

### Files expected in `data/raw/`

| File | Size | Rows | Content |
|---|---|---|---|
| `RAW_recipes.csv` | 294.5 MB | 231,637 | One row per recipe: `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`. The columns `tags`, `nutrition`, `steps` and `ingredients` are strings of Python lists (parse them with `ast.literal_eval`). |
| `RAW_interactions.csv` | 349.4 MB | 1,132,367 | One row per review: `user_id`, `recipe_id`, `date`, `rating` (0–5, where 0 most likely means "no stars given"), `review`. |

`scripts/download_data.sh --all` also downloads the authors' preprocessed files (`PP_recipes.csv`, `PP_users.csv`, `interactions_{train,validation,test}.csv`) and `ingr_map.pkl`, which maps raw ingredient strings to canonical ingredients.

## Project datasets (`data/processed/`)

These four files are committed to the repository, so you get them with `git clone`; you don't need the raw files to use them. `notebooks/02_build_datasets.ipynb` rebuilds them from the raw files (about 1 minute).

| File | Rows | Size | Content |
|---|---|---|---|
| `recipes.csv` | 26,727 | 31 MB | All recipes in one of 9 cuisines: italian, mexican, indian, french, greek, middle-eastern, chinese, thai, japanese |
| `reviews.csv` | 132,744 | 43 MB | The cleaned reviews of exactly those recipes |
| `recipes_mini.csv` | 8,019 | 9 MB | A random 30% of the recipes of each cuisine (seed 42) |
| `reviews_mini.csv` | 38,178 | 12 MB | The reviews of the mini recipes |

**Recipes columns:** `recipe_id`, `name`, `cuisine` (exactly one per recipe, no sub-cuisines: a `szechuan` recipe is `chinese`), `ingredients` (JSON list of raw ingredient strings, **not normalised yet**), `n_ingredients`, `steps` (JSON list), `n_steps`, `description`, `minutes`, `submitted`, `contributor_id`, `n_reviews`.

**Reviews columns:** `review_id` (row number in `RAW_interactions.csv`), `recipe_id`, `cuisine`, `user_id`, `date`, `rating` (1–5, empty if no stars were given), `review`.

**Cleaning applied:**
- Recipes in two of the 9 cuisines are dropped.
- Recipes with fewer than 2 ingredients are dropped, and so are duplicate recipes.
- Implausible `minutes` values (0, or above about 19 hours) are set to empty; the recipe is kept.
- Reviews written for community games are dropped (Zaar World Tour, Pick A Chef, Recipe Swap, Tag games, …).
- Replies by the recipe's author and reviews without text are dropped.

Loading:

```python
import json
import pandas as pd

recipes = pd.read_csv("data/processed/recipes.csv", parse_dates=["submitted"])
recipes["ingredients"] = recipes["ingredients"].map(json.loads)
recipes["steps"] = recipes["steps"].map(json.loads)
reviews = pd.read_csv("data/processed/reviews.csv", parse_dates=["date"])
reviews["rating"] = reviews["rating"].astype("Int64")

# Only some cuisines, e.g. Mexican, Japanese and Italian
my_recipes = recipes[recipes["cuisine"].isin(["mexican", "japanese", "italian"])]
my_reviews = reviews[reviews["recipe_id"].isin(my_recipes["recipe_id"])]
```

Section 8 of `notebooks/02_build_datasets.ipynb` has the same code as a ready-to-copy `load()` and `keep_cuisines()`.

## Raw data is never committed

`data/raw/` and any other generated file in `data/` are excluded in `.gitignore`. The raw CSV files are too large for GitHub. Only the four project datasets above are committed.
