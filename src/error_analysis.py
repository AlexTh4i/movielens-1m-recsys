"""Comprehensive Error and Bias Analysis on the MovieLens-1M Test Set:
1) User Activity Stratification (Light Users / Cold-Start Proxy vs. Moderate vs. Heavy Users)
2) User Mainstreamness Stratification (Gray-Sheep / Niche Users vs. Mainstream Users)
3) Popularity Bias, Head-Item Concentration, Catalog Coverage, Gini Index, and Zero-Hit Rate

Outputs:
  - results/error_analysis.json
  - results/error_analysis.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import POS_THRESHOLD, load_movielens, to_csr
from .evaluate import load_all_models
from .metrics import evaluate_ranking


def gini_coefficient(counts: np.ndarray) -> float:
    """Compute Gini index over item recommendation frequencies (0 = uniform, 1 = complete monopoly)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1.0) / n)


def main() -> None:
    p = argparse.ArgumentParser(description="Run user-cohort error and popularity-bias analysis.")
    p.add_argument("--data_dir", default="data/ml-1m")
    p.add_argument("--model_dir", default="saved_models")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_movielens(a.data_dir)
    models = load_all_models(data, a.model_dir, device, demo_subset=False)

    seen = to_csr(pd.concat([data.train, data.val], ignore_index=True), data.n_users, data.n_items)
    rel = to_csr(data.test, data.n_users, data.n_items, positive_only=True)

    pop = np.bincount(data.train.i.values, minlength=data.n_items)
    head_items = set(np.argsort(-pop)[: int(0.10 * data.n_items)])  # Top 10% most popular movies

    n_train = data.train.groupby("u").size().reindex(range(data.n_users), fill_value=0)
    train_pos = data.train[data.train.rating >= POS_THRESHOLD]
    avg_log_pop = train_pos.assign(lp=np.log1p(pop[train_pos.i.values])).groupby("u").lp.mean()
    avg_log_pop = avg_log_pop.reindex(range(data.n_users), fill_value=float(avg_log_pop.mean()))

    niche_cutoff = float(avg_log_pop.quantile(0.20))

    def assign_cohorts(users: np.ndarray):
        act_cohort = pd.qcut(
            n_train.values[users],
            q=[0.0, 0.25, 0.75, 1.0],
            labels=[
                "Light Users (Bottom 25% - Cold-Start Proxy)",
                "Moderate Users (Middle 50%)",
                "Heavy Users (Top 25%)",
            ],
        )
        taste_cohort = np.where(
            avg_log_pop.values[users] <= niche_cutoff,
            "Gray-Sheep / Niche Users (Bottom 20% Pop)",
            "Mainstream Users (Top 80% Pop)",
        )
        return act_cohort, taste_cohort

    cohort_rows = []
    bias_rows = []

    for name, model in models.items():
        res = evaluate_ranking(model, seen, rel, k=a.k, device=device)
        users = res["users"]
        act_cohort, taste_cohort = assign_cohorts(users)

        df_u = pd.DataFrame(
            {
                "precision": res[f"precision@{a.k}"],
                "recall": res[f"recall@{a.k}"],
                "ndcg": res[f"ndcg@{a.k}"],
                "hitrate": res[f"hitrate@{a.k}"],
                "Activity": act_cohort,
                "Taste": taste_cohort,
            }
        )

        for dim_name in ("Activity", "Taste"):
            for grp_label, sub in df_u.groupby(dim_name, observed=True):
                cohort_rows.append(
                    {
                        "Model": name,
                        "Dimension": dim_name,
                        "Cohort": str(grp_label),
                        "Users": int(len(sub)),
                        f"Precision@{a.k}": float(sub.precision.mean()),
                        f"Recall@{a.k}": float(sub.recall.mean()),
                        f"NDCG@{a.k}": float(sub.ndcg.mean()),
                        f"HitRate@{a.k}": float(sub.hitrate.mean()),
                    }
                )

        top = res["top"]
        rec_counts = np.bincount(top.ravel(), minlength=data.n_items)
        head_mask = np.isin(top, list(head_items))
        bias_rows.append(
            {
                "Model": name,
                "Avg Rec Item Popularity": float(pop[top].mean()),
                "Top-10% Head Share": float(head_mask.mean()),
                "Catalog Coverage": float(np.count_nonzero(rec_counts) / data.n_items),
                "Unique Recommended Movies": int(np.count_nonzero(rec_counts)),
                "Gini Concentration": gini_coefficient(rec_counts),
                f"Zero-Hit User Rate (NDCG@{a.k}=0)": float((res[f"ndcg@{a.k}"] == 0).mean()),
            }
        )

    df_cohorts = pd.DataFrame(cohort_rows)
    df_bias = pd.DataFrame(bias_rows)

    Path("results").mkdir(parents=True, exist_ok=True)
    payload = {
        "cohorts": cohort_rows,
        "popularity_bias": bias_rows,
    }
    (Path("results") / "error_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = (
        "# Error and Popularity-Bias Analysis (MovieLens-1M Test Split)\n\n"
        "## 1. Ranking Performance Across User Cohorts\n\n"
        + df_cohorts.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## 2. Popularity Bias, Catalog Coverage, and Failure Rate\n\n"
        + df_bias.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    (Path("results") / "error_analysis.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
