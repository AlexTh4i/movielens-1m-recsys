"""Generate publication-quality 300-DPI figures for the project report, README, and slides.
Outputs 8 PNG figures into results/figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import POS_THRESHOLD, load_movielens

PALETTE = {
    "Global Mean": "#94a3b8",
    "Popularity": "#64748b",
    "Biased SVD (MF)": "#2563eb",
    "Implicit ALS (MF)": "#0d9488",
    "GMF": "#f59e0b",
    "MLP": "#8b5cf6",
    "NeuMF (Scratch)": "#ec4899",
    "NeuMF (Pre-trained)": "#dc2626",
    "NeuMF-Explicit (MSE)": "#0284c7",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def plot_fig1_eda(data, out_dir: Path) -> None:
    allr = pd.concat([data.train, data.val, data.test], ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Panel A: Rating distribution
    dist = allr.rating.value_counts().sort_index()
    stars = [int(k) for k in dist.index]
    counts = dist.values
    pcts = counts / counts.sum() * 100.0
    colors = ["#94a3b8" if s < POS_THRESHOLD else "#2563eb" for s in stars]
    bars = ax1.bar([str(s) for s in stars], counts / 1e3, color=colors, edgecolor="#1e293b", width=0.62)
    for bar, pct in zip(bars, pcts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    ax1.set_title("(a) Explicit Star Rating Distribution (N = 1,000,209)")
    ax1.set_xlabel("Star Rating (Blue = Implicit Positive >= 4)")
    ax1.set_ylabel("Number of Ratings (Thousands)")
    ax1.set_ylim(0, max(counts / 1e3) * 1.16)

    # Panel B: Long-tail item popularity Pareto curve
    item_cnt = allr.groupby("i").size().sort_values(ascending=False).values
    cum_items = np.arange(1, len(item_cnt) + 1) / len(item_cnt) * 100.0
    cum_ratings = np.cumsum(item_cnt) / item_cnt.sum() * 100.0

    ax2.plot(cum_items, cum_ratings, color="#dc2626", lw=2.4, label="Empirical Lorenz / Pareto Curve")
    ax2.plot([0, 100], [0, 100], color="#94a3b8", ls=":", lw=1.5, label="Uniform Popularity Line")
    ax2.fill_between(cum_items, cum_ratings, cum_items, color="#fca5a5", alpha=0.2)

    idx20 = int(0.20 * len(item_cnt)) - 1
    y20 = cum_ratings[idx20]
    ax2.scatter([20.0], [y20], color="#991b1b", zorder=5, s=45)
    ax2.annotate(
        f"Top 20% movies capture\n{y20:.1f}% of all ratings",
        xy=(20.0, y20),
        xytext=(36.0, 56.0),
        arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 1.3},
        fontsize=9,
        fontweight="bold",
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#fca5a5", "alpha": 0.9},
    )
    ax2.set_title("(b) Long-Tail Item Popularity Concentration")
    ax2.set_xlabel("Top % of Rated Movies (Sorted by Popularity)")
    ax2.set_ylabel("Cumulative % of Total Ratings")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 102)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_eda_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_architecture(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def draw_box(ax, x, y, w, h, text, fc="#eff6ff", ec="#2563eb", fontsize=9.2, bold=False):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.25",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#0f172a",
        )

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"},
        )

    # Left: Matrix Factorization (Biased SVD / ALS)
    ax1.text(5, 9.5, "(a) Matrix Factorization (Biased SVD / ALS)", ha="center", fontsize=11.5, fontweight="bold")
    draw_box(ax1, 1.0, 0.8, 3.2, 1.1, "User Index u\n(One-Hot / ID)", fc="#f1f5f9", ec="#475569")
    draw_box(ax1, 5.8, 0.8, 3.2, 1.1, "Movie Index i\n(One-Hot / ID)", fc="#f1f5f9", ec="#475569")
    draw_box(ax1, 0.8, 3.3, 3.6, 1.3, "User Embedding p_u in R^d\n+ User Bias b_u", fc="#dbeafe", ec="#2563eb")
    draw_box(ax1, 5.6, 3.3, 3.6, 1.3, "Item Embedding q_i in R^d\n+ Item Bias b_i", fc="#dbeafe", ec="#2563eb")
    draw_box(ax1, 2.2, 5.9, 5.6, 1.3, "Linear Inner Product\n<p_u, q_i> + mu + b_u + b_i", fc="#e0f2fe", ec="#0284c7", bold=True)
    draw_box(ax1, 2.5, 8.0, 5.0, 0.95, "Output: r_hat(u,i) [SVD] or s(u,i) [ALS]", fc="#dcfce7", ec="#16a34a", bold=True)

    arrow(ax1, 2.6, 2.05, 2.6, 3.15)
    arrow(ax1, 7.4, 2.05, 7.4, 3.15)
    arrow(ax1, 2.8, 4.75, 4.3, 5.75)
    arrow(ax1, 7.2, 4.75, 5.7, 5.75)
    arrow(ax1, 5.0, 7.35, 5.0, 7.88)

    # Right: Neural Collaborative Filtering (NeuMF)
    ax2.text(5, 9.5, "(b) Neural Collaborative Filtering (NeuMF)", ha="center", fontsize=11.5, fontweight="bold")
    draw_box(ax2, 1.0, 0.6, 3.2, 0.95, "User Index u", fc="#f1f5f9", ec="#475569")
    draw_box(ax2, 5.8, 0.6, 3.2, 0.95, "Movie Index i", fc="#f1f5f9", ec="#475569")

    draw_box(ax2, 0.5, 2.5, 4.0, 1.2, "GMF Embeddings\np_u^G, q_i^G in R^32", fc="#fef3c7", ec="#d97706")
    draw_box(ax2, 5.5, 2.5, 4.0, 1.2, "MLP Embeddings\np_u^M, q_i^M in R^32", fc="#ede9fe", ec="#7c3aed")

    draw_box(ax2, 0.5, 4.7, 4.0, 1.35, "GMF Layer\nElement-wise Product\nphi_GMF = p_u^G (x) q_i^G", fc="#fde68a", ec="#b45309")
    draw_box(ax2, 5.5, 4.7, 4.0, 1.35, "MLP Tower (64->32->16)\nLinear + BN + ReLU + Drop\nphi_MLP([p_u^M || q_i^M])", fc="#ddd6fe", ec="#6d28d9")

    draw_box(ax2, 1.8, 6.75, 6.4, 0.92, "NeuMF Fusion Layer: h^T [phi_GMF || phi_MLP]", fc="#fce7f3", ec="#db2777", bold=True)
    draw_box(ax2, 2.0, 8.32, 6.0, 0.85, "Output Logit -> BCE Loss + Negative Sampling", fc="#fee2e2", ec="#dc2626", bold=True)

    arrow(ax2, 2.3, 1.7, 2.3, 2.38)
    arrow(ax2, 3.0, 1.7, 6.7, 2.38)
    arrow(ax2, 7.0, 1.7, 3.3, 2.38)
    arrow(ax2, 7.7, 1.7, 7.7, 2.38)
    arrow(ax2, 2.5, 3.85, 2.5, 4.55)
    arrow(ax2, 7.5, 3.85, 7.5, 4.55)
    arrow(ax2, 2.5, 6.2, 4.2, 6.62)
    arrow(ax2, 7.5, 6.2, 5.8, 6.62)
    arrow(ax2, 5.0, 7.82, 5.0, 8.20)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_architecture_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_mf_grid(results_dir: Path, out_dir: Path) -> None:
    grid_path = results_dir / "mf_grid_search.json"
    if not grid_path.exists():
        return
    df = pd.DataFrame(json.loads(grid_path.read_text(encoding="utf-8")))
    df["setting"] = df.apply(lambda r: f"lr={r['lr']:.3f}, reg={r['reg']:.3f}", axis=1)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    dims = sorted(df["dim"].unique())
    settings = list(dict.fromkeys(df["setting"]))
    x = np.arange(len(dims))
    width = 0.8 / len(settings)
    sub_colors = ["#2563eb", "#0d9488", "#f59e0b", "#8b5cf6"]

    best_idx = df["best_val_rmse"].idxmin()
    best_row = df.loc[best_idx]

    for idx, st in enumerate(settings):
        sub = df[df["setting"] == st].sort_values("dim")
        vals = sub["best_val_rmse"].values
        offset = (idx - (len(settings) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=st, color=sub_colors[idx % len(sub_colors)])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.0018,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Embedding Dim d = {d}" for d in dims])
    ymin = float(df["best_val_rmse"].min()) - 0.018
    ymax = float(df["best_val_rmse"].max()) + 0.015
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Best Validation RMSE (Lower is Better)")
    ax.set_title(
        f"Biased SVD Hyperparameter Grid Search (Optimal: d={int(best_row['dim'])}, "
        f"lr={best_row['lr']}, reg={best_row['reg']} -> Val RMSE={best_row['best_val_rmse']:.4f})"
    )
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_mf_grid_search.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_training_curves(results_dir: Path, out_dir: Path) -> None:
    hist_path = results_dir / "training_history.json"
    if not hist_path.exists():
        return
    hist = json.loads(hist_path.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    # Left: Explicit Validation RMSE
    exp_map = [("mf", "Biased SVD (MF)"), ("neumf_explicit", "NeuMF-Explicit (MSE)")]
    for key, label in exp_map:
        if key in hist:
            ep = [e["epoch"] for e in hist[key]["epochs"]]
            vr = [e["val_rmse"] for e in hist[key]["epochs"]]
            ax1.plot(ep, vr, marker="o", markersize=4.5, lw=2.1, label=label, color=PALETTE[label])
    ax1.set_title("(a) Explicit Models: Validation RMSE vs. Epoch")
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Validation RMSE (Lower is Better)")
    ax1.legend()

    # Right: Implicit Validation NDCG@10
    imp_map = [
        ("als", "Implicit ALS (MF)"),
        ("gmf", "GMF"),
        ("mlp", "MLP"),
        ("neumf", "NeuMF (Scratch)"),
        ("neumf_pretrained", "NeuMF (Pre-trained)"),
    ]
    for key, label in imp_map:
        if key in hist:
            ep = [e["epoch"] for e in hist[key]["epochs"]]
            vn = [e["val_ndcg@10"] for e in hist[key]["epochs"]]
            ax2.plot(ep, vn, marker="o", markersize=4.5, lw=2.1, label=label, color=PALETTE[label])
    ax2.set_title("(b) Implicit Models: Validation NDCG@10 vs. Epoch/Iter")
    ax2.set_xlabel("Epoch / ALS Iteration")
    ax2.set_ylabel("Validation NDCG@10 (Higher is Better)")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_model_comparison(results_dir: Path, out_dir: Path) -> None:
    df = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 1.65]})

    # Panel A: RMSE comparison
    sub_r = df.dropna(subset=["RMSE"]).copy()
    c_r = [PALETTE.get(m, "#475569") for m in sub_r["Model"]]
    bars1 = ax1.bar(sub_r["Model"], sub_r["RMSE"], color=c_r, width=0.56, edgecolor="#1e293b")
    for b in bars1:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.012,
            f"{b.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )
    ax1.set_title("(a) Explicit Rating Prediction (Test RMSE ↓)")
    ax1.set_ylabel("Test RMSE")
    ax1.set_ylim(0.70, float(sub_r["RMSE"].max()) + 0.08)
    ax1.tick_params(axis="x", rotation=22)

    # Panel B: Top-10 Ranking comparison
    sub_k = df.dropna(subset=["NDCG@10"]).copy()
    x = np.arange(len(sub_k))
    w = 0.26
    b_p = ax2.bar(x - w, sub_k["Precision@10"], width=w, label="Precision@10", color="#3b82f6")
    b_r = ax2.bar(x, sub_k["Recall@10"], width=w, label="Recall@10", color="#10b981")
    b_n = ax2.bar(x + w, sub_k["NDCG@10"], width=w, label="NDCG@10", color="#dc2626")

    for b in b_n:
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.004,
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#991b1b",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_k["Model"], rotation=22, ha="right")
    ax2.set_ylabel("Top-10 Full-Ranking Metric Score")
    ax2.set_ylim(0, float(sub_k[["Precision@10", "Recall@10", "NDCG@10"]].to_numpy().max()) * 1.18)
    ax2.set_title("(b) Full-Catalog Top-10 Ranking on Held-out Test Set (↑)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig6_ablation_ncf(results_dir: Path, out_dir: Path) -> None:
    df = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    ncf_order = ["GMF", "MLP", "NeuMF (Scratch)", "NeuMF (Pre-trained)"]
    sub = df[df["Model"].isin(ncf_order)].set_index("Model").reindex(ncf_order).dropna(how="all").reset_index()
    if sub.empty:
        return

    metrics = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8))
    for ax, m in zip(axes, metrics):
        colors = [PALETTE.get(mdl, "#2563eb") for mdl in sub["Model"]]
        bars = ax.bar(sub["Model"], sub[m], color=colors, width=0.58, edgecolor="#1e293b")
        ymin = float(sub[m].min()) * 0.88
        ymax = float(sub[m].max()) * 1.07
        ax.set_ylim(ymin, ymax)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (ymax - ymin) * 0.02,
                f"{b.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )
        ax.set_title(f"Test {m}")
        ax.tick_params(axis="x", rotation=26)

    fig.suptitle("NCF Architectural & Pre-training Ablation Study", fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_ablation_ncf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_error_cohorts(results_dir: Path, out_dir: Path) -> None:
    err_path = results_dir / "error_analysis.json"
    if not err_path.exists():
        return
    cohorts = pd.DataFrame(json.loads(err_path.read_text(encoding="utf-8"))["cohorts"])
    focus_models = ["Popularity", "Biased SVD (MF)", "Implicit ALS (MF)", "GMF", "MLP", "NeuMF (Pre-trained)"]
    cohorts = cohorts[cohorts["Model"].isin(focus_models)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Panel A: Activity groups
    act = cohorts[cohorts["Dimension"] == "Activity"]
    act_labels = [
        "Light Users (Bottom 25% - Cold-Start Proxy)",
        "Moderate Users (Middle 50%)",
        "Heavy Users (Top 25%)",
    ]
    short_act = ["Light Users\n(Bottom 25%)", "Moderate Users\n(Middle 50%)", "Heavy Users\n(Top 25%)"]
    x1 = np.arange(len(act_labels))
    models_present = [m for m in focus_models if m in act["Model"].unique()]
    w1 = 0.82 / len(models_present)
    for idx, mdl in enumerate(models_present):
        vals = [
            float(act[(act["Model"] == mdl) & (act["Cohort"] == c)]["NDCG@10"].iloc[0])
            for c in act_labels
        ]
        offset = (idx - (len(models_present) - 1) / 2) * w1
        ax1.bar(x1 + offset, vals, width=w1 * 0.92, label=mdl, color=PALETTE.get(mdl, "#334155"))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(short_act)
    ax1.set_ylabel("Test NDCG@10")
    ax1.set_ylim(0, 0.145)
    ax1.set_title("(a) Cold-Start Proxy: Performance by User Activity")
    ax1.legend(fontsize=8.0, ncol=2, loc="upper left")

    # Panel B: Gray-sheep vs Mainstream
    taste = cohorts[cohorts["Dimension"] == "Taste"]
    taste_labels = ["Gray-Sheep / Niche Users (Bottom 20% Pop)", "Mainstream Users (Top 80% Pop)"]
    short_taste = ["Gray-Sheep / Niche Users\n(Bottom 20% Popularity)", "Mainstream Users\n(Top 80% Popularity)"]
    x2 = np.arange(len(taste_labels))
    for idx, mdl in enumerate(models_present):
        vals = [
            float(taste[(taste["Model"] == mdl) & (taste["Cohort"] == c)]["NDCG@10"].iloc[0])
            for c in taste_labels
        ]
        offset = (idx - (len(models_present) - 1) / 2) * w1
        ax2.bar(x2 + offset, vals, width=w1 * 0.92, label=mdl, color=PALETTE.get(mdl, "#334155"))
    ax2.set_xticks(x2)
    ax2.set_xticklabels(short_taste)
    ax2.set_ylabel("Test NDCG@10")
    ax2.set_ylim(0, 0.102)
    ax2.set_title("(b) Gray-Sheep Analysis: Niche vs. Mainstream Tastes")
    ax2.legend(fontsize=8.0, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "fig7_error_user_groups.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig8_popularity_bias(results_dir: Path, out_dir: Path) -> None:
    metrics = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    bias = pd.DataFrame(json.loads((results_dir / "error_analysis.json").read_text(encoding="utf-8"))["popularity_bias"])
    merged = pd.merge(metrics, bias, on="Model", how="inner")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # Panel A: Catalog Coverage vs NDCG@10 scatter
    label_offsets = {
        "Popularity": (7, 5),
        "Biased SVD (MF)": (-15, 11),
        "Implicit ALS (MF)": (-28, -17),
        "GMF": (-26, -15),
        "MLP": (7, 5),
        "NeuMF (Scratch)": (8, -9),
        "NeuMF (Pre-trained)": (-48, 9),
        "NeuMF-Explicit (MSE)": (7, 6),
    }
    for _, row in merged.iterrows():
        mdl = row["Model"]
        cov = row["Catalog Coverage"] * 100.0
        ndcg = row["NDCG@10"]
        head_pct = row["Top-10% Head Share"] * 100.0
        ax1.scatter(
            cov,
            ndcg,
            s=max(90.0, head_pct * 3.8),
            color=PALETTE.get(mdl, "#334155"),
            alpha=0.88,
            edgecolors="#0f172a",
            linewidth=1.2,
        )
        dx, dy = label_offsets.get(mdl, (6, 6))
        ax1.annotate(
            mdl,
            xy=(cov, ndcg),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
        )
    ax1.set_title("(a) Accuracy vs. Catalog Coverage Frontier\n(Bubble Size = Top-10% Head Concentration)")
    ax1.set_xlabel("Catalog Coverage (% of Total Movies Recommended)")
    ax1.set_ylabel("Test NDCG@10")
    ax1.set_xlim(-1.0, 44.0)
    ax1.set_ylim(0.036, 0.087)

    # Panel B: Head-10% share and Zero-Hit rate
    y = np.arange(len(merged))
    h = 0.36
    zero_col = [c for c in merged.columns if c.startswith("Zero-Hit")][0]
    ax2.barh(y - h / 2, merged["Top-10% Head Share"] * 100.0, height=h, label="Top-10% Head Rec Share (%)", color="#f59e0b")
    ax2.barh(y + h / 2, merged[zero_col] * 100.0, height=h, label="Zero-Hit Users (NDCG@10 = 0, %)", color="#64748b")
    ax2.set_yticks(y)
    ax2.set_yticklabels(merged["Model"])
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("(b) Popularity Bias & Complete-Miss Rate", pad=26)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "fig8_popularity_bias_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    results_dir = Path("results")
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_movielens("data/ml-1m")
    plot_fig1_eda(data, out_dir)
    plot_fig2_architecture(out_dir)
    plot_fig3_mf_grid(results_dir, out_dir)
    plot_fig4_training_curves(results_dir, out_dir)
    plot_fig5_model_comparison(results_dir, out_dir)
    plot_fig6_ablation_ncf(results_dir, out_dir)
    plot_fig7_error_cohorts(results_dir, out_dir)
    plot_fig8_popularity_bias(results_dir, out_dir)
    print(f"Generated 8 high-resolution figures in {out_dir}")


if __name__ == "__main__":
    main()
