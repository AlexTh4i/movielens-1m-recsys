"""Unified Test-set evaluation across all baseline, matrix factorization, and NCF models.
Masks out all items interacted with in Train + Validation; ground-truth positives are Test items with rating >= 4.
Outputs:
  - results/eda_summary.json
  - results/metrics.json
  - results/metrics.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import eda_summary, load_movielens, to_csr
from .metrics import evaluate_ranking, predict_ratings, rmse
from .models import Popularity, load_model

MODEL_SPECS = [
    ("mf", "Biased SVD (MF)", "Explicit"),
    ("als", "Implicit ALS (MF)", "Implicit"),
    ("gmf", "GMF", "Implicit"),
    ("mlp", "MLP", "Implicit"),
    ("neumf", "NeuMF (Scratch)", "Implicit"),
    ("neumf_pretrained", "NeuMF (Pre-trained)", "Implicit"),
    ("neumf_explicit", "NeuMF-Explicit (MSE)", "Explicit"),
]


def load_all_models(data, model_dir: str | Path, device: str = "cpu", demo_subset: bool = False) -> dict:
    """Load Popularity baseline plus all available trained model checkpoints."""
    models: dict = {"Popularity": Popularity(data.train, data.n_items)}
    for kind, label, _ in MODEL_SPECS:
        if demo_subset and kind in ("gmf", "mlp", "neumf", "neumf_explicit"):
            continue
        ckpt = Path(model_dir) / f"{kind}.pt"
        if ckpt.exists():
            models[label] = load_model(ckpt, data.n_users, data.n_items, device)
    # Fallback in demo if neumf_pretrained is absent
    if demo_subset and "NeuMF (Pre-trained)" not in models:
        ckpt = Path(model_dir) / "neumf.pt"
        if ckpt.exists():
            models["NeuMF"] = load_model(ckpt, data.n_users, data.n_items, device)
    return models


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate all trained models on the MovieLens-1M Test split.")
    p.add_argument("--data_dir", default="data/ml-1m")
    p.add_argument("--model_dir", default="saved_models")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_movielens(a.data_dir)
    Path("results").mkdir(parents=True, exist_ok=True)

    eda = eda_summary(data)
    (Path("results") / "eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    print("EDA Summary:", json.dumps(eda, indent=2))

    models = load_all_models(data, a.model_dir, device, demo_subset=False)
    seen = to_csr(pd.concat([data.train, data.val], ignore_index=True), data.n_users, data.n_items)
    rel = to_csr(data.test, data.n_users, data.n_items, positive_only=True)
    y_true = data.test.rating.values
    mu = float(data.train.rating.mean())

    explicit_labels = {"Popularity", "Biased SVD (MF)", "NeuMF-Explicit (MSE)"}
    paradigm_lookup = {label: paradigm for _, label, paradigm in MODEL_SPECS}
    paradigm_lookup["Popularity"] = "Heuristic"
    paradigm_lookup["Global Mean"] = "Baseline"

    rows = [
        {
            "Model": "Global Mean",
            "Paradigm": "Baseline",
            "RMSE": rmse(np.full(len(y_true), mu), y_true),
            f"Precision@{a.k}": np.nan,
            f"Recall@{a.k}": np.nan,
            f"NDCG@{a.k}": np.nan,
            f"HitRate@{a.k}": np.nan,
        }
    ]

    for name, model in models.items():
        res = evaluate_ranking(model, seen, rel, k=a.k, device=device)
        m = res["mean"]
        if name == "Popularity":
            model_rmse = rmse(model.predict_rating(data.test), y_true)
        elif name in explicit_labels:
            model_rmse = rmse(predict_ratings(model, data.test, device), y_true)
        else:
            model_rmse = np.nan  # Implicit BCE/ALS models output ranking scores/logits, not 1-5 stars

        rows.append(
            {
                "Model": name,
                "Paradigm": paradigm_lookup.get(name, "Implicit"),
                "RMSE": model_rmse,
                f"Precision@{a.k}": m[f"precision@{a.k}"],
                f"Recall@{a.k}": m[f"recall@{a.k}"],
                f"NDCG@{a.k}": m[f"ndcg@{a.k}"],
                f"HitRate@{a.k}": m[f"hitrate@{a.k}"],
            }
        )

    df = pd.DataFrame(rows)
    print("\n=== MovieLens-1M Held-out Test Set Evaluation ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_json("results/metrics.json", orient="records", indent=2)
    (Path("results") / "metrics.md").write_text(
        df.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    print("\nSaved -> results/metrics.json and results/metrics.md")


if __name__ == "__main__":
    main()
