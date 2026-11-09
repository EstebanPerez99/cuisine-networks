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

1. **Get a Kaggle API token.** Log in at [kaggle.com](https://www.kaggle.com), go to **Settings → API → Create New Token**. This downloads a file called `kaggle.json`. (Newer versions of the page may call this button *Create Legacy API Key*. The script needs the `kaggle.json` file.)
2. **Save it as `~/.kaggle/kaggle.json`** and make it private:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Download the dataset** into `data/raw/`:
   ```bash
   scripts/download_data.sh          # RAW_recipes.csv + RAW_interactions.csv (~645 MB)
   scripts/download_data.sh --all    # the full Kaggle dataset, including ingr_map.pkl (~900 MB)
   ```
   The script stops with an explanation if `~/.kaggle/kaggle.json` is missing.

   Without the Kaggle CLI, you can download the files from the Kaggle page in a browser and put them in `data/raw/`.

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
├── data/                 # NOT in git (see .gitignore)
│   ├── raw/              # original Kaggle files, never modified
│   └── processed/        # files produced by the notebooks (e.g. recipe_cuisines.csv)
├── notebooks/            # Jupyter notebooks, numbered in order
│   └── 01_initial_eda.ipynb
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

## Data is never committed

`data/` is listed in `.gitignore`. The CSV files are too large for GitHub, and everyone can download them with the script above. Do not commit data files or your `kaggle.json`; `.gitignore` excludes `kaggle.json` too, but keep the token in `~/.kaggle/`, outside the repository.
