"""Helpers for perceived-authenticity classification and network comparison."""

from pathlib import Path
import html
import re

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from tokenizers import Tokenizer

ANCHOR_THEMES = [
    "connection",
    "conformity",
    "dish_target",
    "affirmation",
    "deviation",
    "self_connectedness",
]
FEATURE_WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)

_CUE = re.compile(r"\b(?:authentic\w*|traditional\w*|genuine|real thing|real deal)\b")
_NEG = re.compile(r"\b(?:not|never|no|hardly|lacks?|lacked|without|anything but|far from)\b|n.t\b")
_CHANGE = re.compile(r"\b(?:more authentic|instead of|substitut\w*|would|could|should|next time|to make it more)\b")
_RESEMBLANCE = re.compile(
    r"\b(?:just like|exactly like|reminds me of|tastes? like the original|close to the original)\b"
)

_GTE_CACHE: dict[Path, tuple[Tokenizer, ort.InferenceSession]] = {}


def _get_gte(model_dir: Path) -> tuple[Tokenizer, ort.InferenceSession]:
    model_dir = Path(model_dir).resolve()
    if model_dir not in _GTE_CACHE:
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=512)
        tok.enable_padding(pad_id=tok.token_to_id("[PAD]"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        _GTE_CACHE[model_dir] = (tok, sess)
    return _GTE_CACHE[model_dir]


def embed_gte(texts, model_dir: Path, batch_size: int = 32) -> np.ndarray:
    """Encode texts with gte-small ONNX (mean pooling + L2 normalization)."""
    tok, sess = _get_gte(model_dir)
    cleaned = [" ".join(html.unescape(str(t)).split()) for t in texts]
    fields = {"input_ids": "ids", "token_type_ids": "type_ids", "attention_mask": "attention_mask"}
    in_names = [x.name for x in sess.get_inputs()]
    vecs = []
    for start in range(0, len(cleaned), batch_size):
        enc = tok.encode_batch(cleaned[start : start + batch_size])
        inputs = {n: np.asarray([getattr(t, fields[n]) for t in enc], dtype=np.int64) for n in in_names}
        hidden = sess.run(None, inputs)[0]
        mask = inputs["attention_mask"][:, :, None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        vecs.append(normalize(pooled))
    return np.vstack(vecs).astype(np.float32)


def kwic_flags(texts) -> np.ndarray:
    """Extract 4 clause-level KWIC flags: affirmative, negated, modification, resemblance."""
    rows = []
    for text in texts:
        flags = np.zeros(4, dtype=np.float32)
        t = html.unescape(str(text)).lower().replace("’", "'")
        for clause in re.split(r"[.!?;]|\bbut\b", t):
            for cue in _CUE.finditer(clause):
                before = " ".join(clause[: cue.start()].split()[-7:])
                neg = bool(_NEG.search(before))
                change = bool(_CHANGE.search(clause))
                flags[:3] = np.maximum(flags[:3], [not (neg or change), neg, change])
            if _RESEMBLANCE.search(clause) and not _NEG.search(clause):
                flags[3] = 1.0
        rows.append(flags)
    return np.asarray(rows, dtype=np.float32)


def choose_cutoff(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    """Maximize recall subject to false-positive rate <= max_fpr."""
    candidates = np.r_[np.inf, np.unique(scores)]

    def rank(t):
        pred = scores >= t
        fpr = pred[~labels].mean()
        return fpr <= max_fpr, pred[labels].mean(), -fpr, t

    return float(max(candidates, key=rank))


def fit_cv_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray | None = None,
    C: float = 0.1,
    n_splits: int = 5,
    random_state: int = 251,
):
    """Run 5-fold stratified CV to pick the cutoff, then fit final scaler + LogisticRegression."""
    w = np.ones(X_train.shape[1], dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    oof_scores = np.zeros(len(y_train), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in cv.split(X_train, y_train):
        sc = StandardScaler().fit(X_train[tr_idx])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X_train[tr_idx]) * w, y_train[tr_idx])
        oof_scores[val_idx] = clf.decision_function(sc.transform(X_train[val_idx]) * w)

    cutoff = choose_cutoff(y_train, oof_scores)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(scaler.transform(X_train) * w, y_train)
    return scaler, clf, cutoff, oof_scores


def summarize_predictions(name: str, y_true, y_pred) -> dict:
    """Compute TP/FP/FN/TN, precision, recall, and F1 for a binary prediction vector."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "variant": name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "F1": round(f1, 3),
    }


def classify_title(title: str) -> str:
    """Conservative title classifier separating desserts/sweets and drinks from savory dishes."""
    t = str(title).lower()
    if re.search(r"\b(?:smoothie|milkshake|lassi)\b", re.sub(r"\b(?:shrimp|prawn|ceviche)\s+cocktail\b", "", t)):
        return "drink"
    dt = re.sub(r"\b(?:salmon|crab|fish|scallion|rice)\s+cakes?\b", "", t)
    if re.search(
        r"\b(?:desserts?|cakes?|cupcakes?|cheesecakes?|cookies?|brownies?|blondies?|puddings?|custards?|"
        r"ice\s+cream|sorbets?|gelato|fudge|mousse|tiramisu|baklava|panna\s+cotta|sweet\s+rolls?|parfaits?|"
        r"cobblers?|fruit\s+tarts?|chocolate\s+tarts?|apple\s+pies?|pumpkin\s+pies?|pecan\s+pies?)\b",
        dt,
    ):
        return "dessert/sweet"
    dr = re.sub(r"\b(?:shrimp|prawn|ceviche)\s+cocktail\b|\bmargarita\s+(?:fajitas?|chicken|grill)\b", "", t)
    if re.search(
        r"\b(?:cocktail|margarita|martini|mojito|sangria|lemonade|iced\s+(?:tea|coffee)|hot\s+chocolate|"
        r"punch|mocktail|cosmopolitan|kir\s+royal|liqueur|shooter)\b",
        dr,
    ):
        return "drink"
    return "other"


def shuffle_test(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    strata: list[str],
    n_perm: int = 2000,
    seed: int = 42,
    return_null: bool = False,
) -> dict:
    """Week 2 stratified label-shuffle null test for Spearman rank correlation."""
    sub = df.dropna(subset=[x_col, y_col] + strata).reset_index(drop=True)
    xr = rankdata(sub[x_col].to_numpy())
    xr = xr - xr.mean()
    x_norm = np.linalg.norm(xr)
    y = sub[y_col].to_numpy()

    def fast_rho(vals):
        yr = rankdata(vals)
        yr = yr - yr.mean()
        return float(np.dot(xr, yr) / (x_norm * np.linalg.norm(yr)))

    obs = fast_rho(y)
    groups = [idx.to_numpy() for idx in sub.groupby(strata, observed=False).groups.values()]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    perm = y.copy()
    for i in range(n_perm):
        for g in groups:
            perm[g] = rng.permutation(y[g])
        null[i] = fast_rho(perm)
    z = (obs - null.mean()) / null.std(ddof=1)
    emp_p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1)
    result = {
        "n": len(sub),
        "obs_rho": round(obs, 4),
        "null_mean": round(float(null.mean()), 4),
        "z_score": round(float(z), 2),
        "empirical_p": round(float(emp_p), 5),
    }
    if return_null:
        result["null_dist"] = null
    return result
