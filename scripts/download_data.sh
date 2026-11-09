#!/usr/bin/env bash
# Download the Food.com Recipes and Interactions dataset (Majumder et al., 2019)
# from Kaggle into data/raw/.
#
# Usage:
#   scripts/download_data.sh          # only RAW_recipes.csv and RAW_interactions.csv (~645 MB)
#   scripts/download_data.sh --all    # the full Kaggle dataset (~900 MB, adds PP_* files and ingr_map.pkl)
#
# Requires the Kaggle CLI (installed by `pip install -r requirements.txt`)
# and an API token in ~/.kaggle/kaggle.json.

set -euo pipefail

DATASET="shuyangli94/food-com-recipes-and-user-interactions"
FILES=("RAW_recipes.csv" "RAW_interactions.csv")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"

die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Credentials
if [[ ! -f "${KAGGLE_JSON}" ]]; then
    die "Kaggle API token not found at ${KAGGLE_JSON}.
  1. Log in at https://www.kaggle.com and open Settings -> API.
  2. Click 'Create New Token' (legacy API key); this downloads kaggle.json.
  3. Move it into place and restrict its permissions:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
       chmod 600 ~/.kaggle/kaggle.json
  Then run this script again."
fi
chmod 600 "${KAGGLE_JSON}" 2>/dev/null || true

# 2. Kaggle CLI (prefer the project venv)
if [[ -x "${REPO_ROOT}/.venv/bin/kaggle" ]]; then
    KAGGLE="${REPO_ROOT}/.venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
    KAGGLE="$(command -v kaggle)"
else
    die "Kaggle CLI not found. Activate the venv and run: pip install -r requirements.txt"
fi

command -v unzip >/dev/null 2>&1 || die "'unzip' is not installed."

mkdir -p "${RAW_DIR}"

# 3. Download
if [[ "${1:-}" == "--all" ]]; then
    echo "Downloading the full dataset ${DATASET} into ${RAW_DIR} ..."
    "${KAGGLE}" datasets download "${DATASET}" -p "${RAW_DIR}"
else
    for f in "${FILES[@]}"; do
        if [[ -f "${RAW_DIR}/${f}" ]]; then
            echo "${f} already exists, skipping."
            continue
        fi
        echo "Downloading ${f} ..."
        "${KAGGLE}" datasets download "${DATASET}" -f "${f}" -p "${RAW_DIR}"
    done
fi

# 4. Unzip whatever arrived as a zip (Kaggle zips large files)
shopt -s nullglob
for z in "${RAW_DIR}"/*.zip; do
    echo "Unzipping $(basename "${z}") ..."
    unzip -o -q "${z}" -d "${RAW_DIR}"
    rm -f "${z}"
done

# 5. Check
for f in "${FILES[@]}"; do
    [[ -f "${RAW_DIR}/${f}" ]] || die "${f} is missing from ${RAW_DIR} after download."
done
echo "Done. Files in ${RAW_DIR}:"
ls -lh "${RAW_DIR}"
