"""Data loading, per-user temporal Train/Val/Test splitting, sparse matrix utilities,
and vectorized batch loaders for Explicit (RMSE) and Implicit (Top-N + Negative Sampling) training."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

POS_THRESHOLD = 4  # rating >= 4 is treated as a positive implicit interaction (y = 1)

AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_MAP = {
    0: "Other / Not specified",
    1: "Academic / Educator",
    2: "Artist",
    3: "Clerical / Admin",
    4: "College / Grad student",
    5: "Customer service",
    6: "Doctor / Health care",
    7: "Executive / Managerial",
    8: "Farmer",
    9: "Homemaker",
    10: "K-12 student",
    11: "Lawyer",
    12: "Programmer",
    13: "Retired",
    14: "Sales / Marketing",
    15: "Scientist",
    16: "Self-employed",
    17: "Technician / Engineer",
    18: "Tradesman / Craftsman",
    19: "Unemployed",
    20: "Writer",
}


@dataclass
class Data:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame  # indexed 0..n_items-1 aligned with column 'i'
    users: pd.DataFrame   # indexed 0..n_users-1 aligned with column 'u'
    n_users: int
    n_items: int


class FastTensorDataLoader:
    """Vectorized in-memory batch iterator (10-15x faster than item-wise TensorDataset)."""

    def __init__(self, *tensors: torch.Tensor, batch_size: int = 4096, shuffle: bool = True):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.n = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n)
            tensors = tuple(t[perm] for t in self.tensors)
        else:
            tensors = self.tensors
        for s in range(0, self.n, self.batch_size):
            e = s + self.batch_size
            yield tuple(t[s:e] for t in tensors)


def load_movielens(
    data_dir: str | Path = "data/ml-1m",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Data:
    """Load MovieLens-1M and perform per-user chronological splitting:
    oldest 80% -> train, next 10% -> val, newest 10% -> test (prevents future leakage)."""
    d = Path(data_dir)
    raw_ratings = (d / "ratings.dat").read_text(encoding="latin-1").replace("::", "\t")
    r = pd.read_csv(
        io.StringIO(raw_ratings),
        sep="\t",
        names=["user_id", "movie_id", "rating", "ts"],
        dtype={"user_id": np.int32, "movie_id": np.int32, "rating": np.float32, "ts": np.int64},
    )
    m = pd.read_csv(
        d / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    u = pd.read_csv(
        d / "users.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )

    uids = np.sort(r.user_id.unique())
    iids = np.sort(m.movie_id.unique())
    umap = {int(x): k for k, x in enumerate(uids)}
    imap = {int(x): k for k, x in enumerate(iids)}

    r["u"] = r.user_id.map(umap).astype(np.int32)
    r["i"] = r.movie_id.map(imap).astype(np.int32)
    m["i"] = m.movie_id.map(imap).astype(np.int32)
    m = m.sort_values("i").reset_index(drop=True)

    u["u"] = u.user_id.map(umap).astype(np.int32)
    u["age_label"] = u["age"].map(AGE_MAP).fillna(u["age"].astype(str))
    u["occupation_label"] = u["occupation"].map(OCCUPATION_MAP).fillna("Other")
    u = u.sort_values("u").reset_index(drop=True)

    r = r.sort_values(["u", "ts"], kind="stable").reset_index(drop=True)
    pos = r.groupby("u").cumcount().values
    n = r.groupby("u")["u"].transform("size").values
    is_train = pos < np.floor(train_frac * n)
    is_val = (~is_train) & (pos < np.floor((train_frac + val_frac) * n))
    cols = ["u", "i", "rating", "ts"]
    return Data(
        train=r.loc[is_train, cols].reset_index(drop=True),
        val=r.loc[is_val, cols].reset_index(drop=True),
        test=r.loc[~is_train & ~is_val, cols].reset_index(drop=True),
        movies=m,
        users=u,
        n_users=len(uids),
        n_items=len(iids),
    )


def to_csr(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    positive_only: bool = False,
    use_ratings: bool = False,
) -> sp.csr_matrix:
    """Build a sparse (n_users x n_items) interaction matrix."""
    if positive_only:
        df = df[df.rating >= POS_THRESHOLD]
    vals = df.rating.values.astype(np.float32) if use_ratings else np.ones(len(df), dtype=np.float32)
    return sp.csr_matrix((vals, (df.u.values, df.i.values)), shape=(n_users, n_items))


def eda_summary(data: Data) -> dict:
    """Compute descriptive statistics for Exploratory Data Analysis (EDA)."""
    allr = pd.concat([data.train, data.val, data.test], ignore_index=True)
    item_cnt = allr.groupby("i").size().sort_values(ascending=False)
    user_cnt = allr.groupby("u").size()
    top10_share = item_cnt.iloc[: max(1, int(0.1 * len(item_cnt)))].sum() / item_cnt.sum()
    top20_share = item_cnt.iloc[: max(1, int(0.2 * len(item_cnt)))].sum() / item_cnt.sum()
    return {
        "n_ratings": int(len(allr)),
        "n_train": int(len(data.train)),
        "n_val": int(len(data.val)),
        "n_test": int(len(data.test)),
        "n_users": int(data.n_users),
        "n_items": int(data.n_items),
        "n_rated_items": int(len(item_cnt)),
        "density": float(len(allr) / (data.n_users * data.n_items)),
        "sparsity": float(1.0 - len(allr) / (data.n_users * data.n_items)),
        "mean_rating": float(allr.rating.mean()),
        "std_rating": float(allr.rating.std()),
        "positive_rate_ge4": float((allr.rating >= POS_THRESHOLD).mean()),
        "user_ratings_min": int(user_cnt.min()),
        "user_ratings_median": float(user_cnt.median()),
        "user_ratings_mean": float(user_cnt.mean()),
        "user_ratings_max": int(user_cnt.max()),
        "item_ratings_median": float(item_cnt.median()),
        "item_ratings_max": int(item_cnt.max()),
        "rating_dist": {int(k): int(v) for k, v in allr.rating.value_counts().sort_index().items()},
        "share_ratings_of_top10pct_items": float(top10_share),
        "share_ratings_of_top20pct_items": float(top20_share),
    }


def sample_negatives(
    pos_u: np.ndarray,
    seen_keys: np.ndarray,
    n_items: int,
    n_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """For each positive user interaction, uniformly sample n_neg unseen items from Train."""
    u = np.repeat(pos_u, n_neg).astype(np.int64)
    neg = rng.integers(0, n_items, size=len(u), dtype=np.int64)
    for _ in range(30):
        keys = u * n_items + neg
        idx = np.minimum(np.searchsorted(seen_keys, keys), len(seen_keys) - 1)
        bad = seen_keys[idx] == keys
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()), dtype=np.int64)
    return u, neg


def ncf_loader(
    train: pd.DataFrame,
    n_items: int,
    n_neg: int,
    batch: int,
    rng: np.random.Generator,
    seen_keys: np.ndarray | None = None,
) -> FastTensorDataLoader:
    """Create 1-epoch implicit DataLoader: positive interactions (y=1) + n_neg sampled negatives (y=0)."""
    if seen_keys is None:
        seen_keys = np.sort(train.u.values.astype(np.int64) * n_items + train.i.values.astype(np.int64))
    pos = train[train.rating >= POS_THRESHOLD]
    pu = pos.u.values.astype(np.int64)
    pi = pos.i.values.astype(np.int64)
    nu, ni = sample_negatives(pu, seen_keys, n_items, n_neg, rng)
    u = np.concatenate([pu, nu])
    i = np.concatenate([pi, ni])
    y = np.concatenate([np.ones(len(pu), dtype=np.float32), np.zeros(len(nu), dtype=np.float32)])
    return FastTensorDataLoader(
        torch.from_numpy(u),
        torch.from_numpy(i),
        torch.from_numpy(y),
        batch_size=batch,
        shuffle=True,
    )


def explicit_loader(df: pd.DataFrame, batch: int, shuffle: bool = True) -> FastTensorDataLoader:
    """Create explicit rating DataLoader yielding (u, i, rating)."""
    return FastTensorDataLoader(
        torch.from_numpy(df.u.values.astype(np.int64)),
        torch.from_numpy(df.i.values.astype(np.int64)),
        torch.from_numpy(df.rating.values.astype(np.float32)),
        batch_size=batch,
        shuffle=shuffle,
    )
