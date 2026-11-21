"""Interactive Streamlit Demo: Movie Recommendation System on MovieLens-1M
Comparing Popularity Baseline, Matrix Factorization (Biased SVD & Implicit ALS), and Pre-trained NeuMF.

Run with:
    streamlit run demo/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data import POS_THRESHOLD, load_movielens, to_csr
from src.evaluate import load_all_models

st.set_page_config(page_title="MovieLens-1M Recommender Demo", page_icon="🎬", layout="wide")
DATA_DIR = ROOT / "data" / "ml-1m"
MODEL_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"


@st.cache_resource
def load_demo_resources():
    data = load_movielens(DATA_DIR)
    models = load_all_models(data, MODEL_DIR, device="cpu", demo_subset=True)
    tv = pd.concat([data.train, data.val], ignore_index=True)
    seen = to_csr(tv, data.n_users, data.n_items)

    # Identify representative sample users for live classroom demonstration
    pop = np.bincount(data.train.i.values, minlength=data.n_items)
    n_train = data.train.groupby("u").size()
    pos_tr = data.train[data.train.rating >= POS_THRESHOLD]
    avg_lp = pos_tr.assign(lp=np.log1p(pop[pos_tr.i.values])).groupby("u").lp.mean()

    presets = {
        "Custom User ID": None,
        "Active Mainstream Fan (User #0)": 0,
        f"Heavy Cinephile (User #{int(n_train.idxmax())})": int(n_train.idxmax()),
        f"Light / Cold-Start User (User #{int(n_train.idxmin())})": int(n_train.idxmin()),
        f"Niche / Gray-Sheep User (User #{int(avg_lp.idxmin())})": int(avg_lp.idxmin()),
    }
    return data, models, seen, tv, pop, presets


data, models, seen, tv, item_pop, presets = load_demo_resources()
movies = data.movies

st.title("🎬 Movie Recommendation System: Popularity vs. SVD/ALS vs. Neural CF")
st.markdown(
    "**Dataset:** MovieLens-1M (1,000,209 ratings · 6,040 users · 3,883 movies) · "
    "**Protocol:** Per-user chronological 80/10/10 split · Full-catalog Top-K ranking"
)

# ---------------- Sidebar Controls ----------------
st.sidebar.header("🔎 User & Ranking Controls")
preset_choice = st.sidebar.selectbox("Quick Sample Personas", list(presets.keys()), index=0)
default_uid = 0 if presets[preset_choice] is None else presets[preset_choice]

uid = int(
    st.sidebar.number_input(
        f"Select User ID (0 – {data.n_users - 1})",
        min_value=0,
        max_value=data.n_users - 1,
        value=int(default_uid),
        step=1,
    )
)
k = int(st.sidebar.slider("Top-K Recommendations", min_value=5, max_value=20, value=10, step=1))

available_model_names = list(models.keys())
selected_models = st.sidebar.multiselect(
    "Models to Compare Side-by-Side",
    options=available_model_names,
    default=available_model_names,
)

u_row = data.users.iloc[uid]
u_hist_all = tv[tv.u == uid]
u_hist_pos = u_hist_all[u_hist_all.rating >= POS_THRESHOLD]
test_df = data.test[data.test.u == uid]
test_pos = set(test_df[test_df.rating >= POS_THRESHOLD].i.tolist())

st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Profile Summary")
st.sidebar.markdown(
    f"- **Internal ID:** `{uid}` (Original ML ID: `{int(u_row.user_id)}`)\n"
    f"- **Gender:** `{'Female' if u_row.gender == 'F' else 'Male'}`\n"
    f"- **Age Group:** `{u_row.age_label}`\n"
    f"- **Occupation:** `{u_row.occupation_label}`\n"
    f"- **History (Train+Val):** `{len(u_hist_all)}` ratings (`{len(u_hist_pos)}` liked ≥ 4★)\n"
    f"- **Held-out Test Positives (≥ 4★):** `{len(test_pos)}` movies"
)

tab_rec, tab_bench = st.tabs(["🎯 Personalized Recommendations", "📊 Benchmark Metrics & Error Analysis"])

with tab_rec:
    col_h1, col_h2 = st.columns([1.35, 1.0])
    with col_h1:
        st.subheader(f"📚 User #{uid}'s Top-Rated Historical Movies (Train + Val)")
        hist_merged = (
            u_hist_pos.merge(movies[["i", "title", "genres"]], on="i")
            .sort_values(["rating", "ts"], ascending=[False, False])
            [["title", "genres", "rating"]]
            .rename(columns={"title": "Movie Title", "genres": "Genres", "rating": "Rating (★)"})
        )
        st.dataframe(hist_merged.head(15), use_container_width=True, hide_index=True)

    with col_h2:
        st.subheader(f"🧪 Held-Out Ground Truth (Test Split ≥ 4★)")
        if len(test_pos) > 0:
            test_merged = (
                test_df[test_df.rating >= POS_THRESHOLD]
                .merge(movies[["i", "title", "genres"]], on="i")
                .sort_values("rating", ascending=False)[["title", "genres", "rating"]]
                .rename(columns={"title": "Movie Title", "genres": "Genres", "rating": "Test Rating (★)"})
            )
            st.dataframe(test_merged, use_container_width=True, hide_index=True)
        else:
            st.info("This user has no held-out Test ratings >= 4 stars.")

    st.markdown("---")
    st.subheader(f"🏆 Side-by-Side Top-{k} Unseen Movie Recommendations (`✅` = Verified Hit in Held-Out Test Set)")

    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        cols = st.columns(len(selected_models))
        rec_sets: dict[str, set[int]] = {}
        for col, name in zip(cols, selected_models):
            mdl = models[name]
            scores = mdl.score_all(torch.tensor([uid], dtype=torch.long)).float().cpu().clone()[0]
            scores[seen[uid].indices] = -float("inf")
            top_vals, top_idx = scores.topk(k)
            top_items = top_idx.numpy()
            top_scores = top_vals.numpy()
            rec_sets[name] = set(int(x) for x in top_items)

            rec_df = movies.iloc[top_items][["title", "genres"]].copy()
            rec_df.insert(0, "Rank", np.arange(1, k + 1))
            rec_df.insert(1, "Hit", ["✅" if int(i) in test_pos else "" for i in top_items])
            if "Explicit" in name or "SVD" in name:
                rec_df["Score"] = np.clip(top_scores, 1.0, 5.0).round(2)
            elif name == "Popularity":
                rec_df["Score"] = top_scores.round(2)
            else:
                # Convert logits / inner-product scores for clean display
                rec_df["Score"] = top_scores.round(3)
            rec_df = rec_df.rename(columns={"title": "Movie Title", "genres": "Genres"})

            hits = sum(int(i) in test_pos for i in top_items)
            avg_pop = float(item_pop[top_items].mean())
            with col:
                st.markdown(f"#### {name}")
                st.caption(f"Test Hits: **{hits}/{k}** (`P@{k} = {hits/k:.2f}`) · Avg Train Views: **{avg_pop:.0f}**")
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

        if len(selected_models) > 1:
            overlaps = []
            for idx_a, a_name in enumerate(selected_models):
                for b_name in selected_models[idx_a + 1 :]:
                    inter = len(rec_sets[a_name] & rec_sets[b_name])
                    union = len(rec_sets[a_name] | rec_sets[b_name])
                    jacc = inter / union if union else 0.0
                    overlaps.append(f"**{a_name} ∩ {b_name}**: `{inter}/{k}` (`Jaccard = {jacc:.2f}`)")
            st.info("**List Overlap & Personalization Diversity:**  " + "  ·  ".join(overlaps))

with tab_bench:
    st.subheader("📈 Held-Out Test Set Performance (Full-Catalog Ranking)")
    metrics_file = RESULTS_DIR / "metrics.json"
    if metrics_file.exists():
        m_df = pd.DataFrame(json.loads(metrics_file.read_text(encoding="utf-8")))
        st.dataframe(m_df.style.format(precision=4, na_rep="—"), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.evaluate` to populate Test metrics.")

    fig_dir = RESULTS_DIR / "figures"
    fig_pairs = [
        ("fig5_model_comparison.png", "fig6_ablation_ncf.png"),
        ("fig7_error_user_groups.png", "fig8_popularity_bias_coverage.png"),
    ]
    for f_left, f_right in fig_pairs:
        c1, c2 = st.columns(2)
        if (fig_dir / f_left).exists():
            c1.image(str(fig_dir / f_left), use_container_width=True)
        if (fig_dir / f_right).exists():
            c2.image(str(fig_dir / f_right), use_container_width=True)
