"""Build the 5-slide academic PowerPoint presentation (docs/presentation_5_slides.pptx)
and companion Markdown summary (docs/presentation_5_slides.md) from experimental outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(15, 23, 42)
BLUE = RGBColor(37, 99, 235)
DARK_SLATE = RGBColor(30, 41, 59)
MUTED = RGBColor(100, 116, 139)
LIGHT_BG = RGBColor(248, 250, 252)
CARD_BORDER = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)
ACCENT_RED = RGBColor(185, 28, 28)
ACCENT_TEAL = RGBColor(13, 148, 136)


def _add_header(slide, title_text: str, subtitle_text: str, slide_num: int) -> None:
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tf = banner.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.45)
    tf.margin_top = Inches(0.14)
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.size = Pt(11.5)
    p2.font.color.rgb = RGBColor(191, 219, 254)
    p2.font.name = "Calibri"

    # Slide number badge in footer
    foot = slide.shapes.add_textbox(Inches(11.8), Inches(7.08), Inches(1.3), Inches(0.32))
    fp = foot.text_frame.paragraphs[0]
    fp.text = f"Slide {slide_num} / 5"
    fp.alignment = PP_ALIGN.RIGHT
    fp.font.size = Pt(10)
    fp.font.bold = True
    fp.font.color.rgb = MUTED
    fp.font.name = "Calibri"


def _add_card(slide, left: float, top: float, width: float, height: float, heading: str, bullets: list[str], accent=BLUE):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    box.adjustments[0] = 0.03
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_BG
    box.line.color.rgb = CARD_BORDER
    box.line.width = Pt(1.0)

    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.20)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.16)
    tf.margin_bottom = Inches(0.12)

    p0 = tf.paragraphs[0]
    p0.text = heading
    p0.font.size = Pt(13.5)
    p0.font.bold = True
    p0.font.color.rgb = accent
    p0.font.name = "Calibri"
    p0.space_after = Pt(5)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11.0)
        p.font.color.rgb = DARK_SLATE
        p.font.name = "Calibri"
        p.space_after = Pt(4)


def build_presentation(results_dir: Path = Path("results"), out_dir: Path = Path("docs")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eda = json.loads((results_dir / "eda_summary.json").read_text(encoding="utf-8"))
    metrics_df = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    err_data = json.loads((results_dir / "error_analysis.json").read_text(encoding="utf-8"))
    bias_df = pd.DataFrame(err_data["popularity_bias"]).set_index("Model")
    fig_dir = results_dir / "figures"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    #Lookup key numbers dynamically
    best_rmse_row = metrics_df.dropna(subset=["RMSE"]).sort_values("RMSE").iloc[0]
    best_ndcg_row = metrics_df.dropna(subset=["NDCG@10"]).sort_values("NDCG@10", ascending=False).iloc[0]
    pop_row = metrics_df[metrics_df["Model"] == "Popularity"].iloc[0]
    svd_row = metrics_df[metrics_df["Model"] == "Biased SVD (MF)"].iloc[0]

    # ==================== SLIDE 1: TITLE, PROBLEM & DATA ====================
    s1 = prs.slides.add_slide(blank_layout)
    _add_header(
        s1,
        "Movie Recommendation on MovieLens-1M: SVD/ALS vs. Neural CF",
        "Advanced Artificial Intelligence Project  |  PyTorch Full-Catalog Evaluation & Live Streamlit System",
        1,
    )
    _add_card(
        s1,
        0.4,
        1.28,
        5.55,
        2.85,
        "1. Problem Statement & Objectives",
        [
            "Compare three recommendation paradigms on MovieLens-1M: Heuristic Popularity, Matrix Factorization (Biased SVD & Implicit ALS), and Neural Collaborative Filtering (GMF, MLP, NeuMF).",
            "Unify evaluation across Explicit Rating Prediction (RMSE) and Implicit Full-Catalog Top-10 Ranking (Precision@10, Recall@10, NDCG@10).",
            "Investigate real-world failure modes: Cold-Start users, Gray-Sheep (niche) tastes, Popularity Bias, and the Explicit-vs-Implicit ranking gap.",
        ],
        accent=BLUE,
    )
    _add_card(
        s1,
        0.4,
        4.32,
        5.55,
        2.65,
        "2. MovieLens-1M & Leak-Free Temporal Split",
        [
            f"Scale: {eda['n_ratings']:,} explicit ratings (1-5 stars) by {eda['n_users']:,} users across {eda['n_items']:,} movies (sparsity = {eda['sparsity']*100:.2f}%).",
            f"Long-Tail Skew: Top 20% blockbusters capture {eda['share_ratings_of_top20pct_items']*100:.1f}% of all interactions; positive rate (>= 4 stars) = {eda['positive_rate_ge4']*100:.1f}%.",
            f"Chronological Split (80/10/10 per user): {eda['n_train']:,} Train | {eda['n_val']:,} Val | {eda['n_test']:,} Test — strictly prevents future leakage.",
        ],
        accent=ACCENT_TEAL,
    )
    if (fig_dir / "fig1_eda_distributions.png").exists():
        s1.shapes.add_picture(str(fig_dir / "fig1_eda_distributions.png"), Inches(6.15), Inches(1.45), width=Inches(6.85))

    # ==================== SLIDE 2: ARCHITECTURES & METHODS ====================
    s2 = prs.slides.add_slide(blank_layout)
    _add_header(
        s2,
        "Methodology: Popularity, Matrix Factorization (SVD/ALS) & Neural CF",
        "Contrasting Linear Inner-Product Spaces with Dual-Pathway Non-Linear Interaction Learning",
        2,
    )
    if (fig_dir / "fig2_architecture_overview.png").exists():
        s2.shapes.add_picture(str(fig_dir / "fig2_architecture_overview.png"), Inches(0.45), Inches(1.22), width=Inches(7.55))

    _add_card(
        s2,
        8.18,
        1.22,
        4.78,
        1.82,
        "Baseline & Explicit SVD",
        [
            "Popularity: IMDB Bayesian Weighted Rating (q = 0.80) for Top-10; item mean for RMSE.",
            "Biased SVD (MF): r_hat = mu + b_u + b_i + <p_u, q_i>; Adam + L2 penalty tuned via 12-point grid search.",
        ],
        accent=BLUE,
    )
    _add_card(
        s2,
        8.18,
        3.18,
        4.78,
        1.75,
        "Implicit ALS (Hu et al., 2008)",
        [
            "Optimizes confidence-weighted objective c_ui = 1 + alpha * r'_ui over ALL items (observed & unobserved).",
            "Closed-form alternating ridge solver — bridges linear MF and Top-N implicit ranking.",
        ],
        accent=ACCENT_TEAL,
    )
    _add_card(
        s2,
        8.18,
        5.07,
        4.78,
        1.95,
        "NCF Family (He et al., WWW 2017)",
        [
            "GMF (element-wise product) + MLP (64->32->16 tower with BatchNorm & Dropout).",
            "Pre-trained NeuMF fuses converged GMF & MLP weights (alpha = 0.5) with 4:1 dynamic negative sampling & BCE loss.",
        ],
        accent=ACCENT_RED,
    )

    # ==================== SLIDE 3: QUANTITATIVE RESULTS ====================
    s3 = prs.slides.add_slide(blank_layout)
    _add_header(
        s3,
        "Experimental Evaluation: Test Set Results & NCF Ablation",
        "Full-Catalog Ranking across all Unseen Movies (Masking Train + Val; Positives = Test Rating >= 4)",
        3,
    )

    # Add clean results table on left
    tbl_rows = len(metrics_df) + 1
    tbl_shape = s3.shapes.add_table(tbl_rows, 5, Inches(0.38), Inches(1.25), Inches(6.25), Inches(3.85))
    tbl = tbl_shape.table
    col_widths = [Inches(2.15), Inches(0.95), Inches(1.05), Inches(1.00), Inches(1.10)]
    for idx_c, cw in enumerate(col_widths):
        tbl.columns[idx_c].width = cw

    headers = ["Model", "RMSE ↓", "P@10 ↑", "R@10 ↑", "NDCG@10 ↑"]
    for idx_c, h_text in enumerate(headers):
        cell = tbl.cell(0, idx_c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER if idx_c > 0 else PP_ALIGN.LEFT

    for r_idx, (_, row) in enumerate(metrics_df.iterrows(), start=1):
        vals = [
            str(row["Model"]),
            "—" if pd.isna(row["RMSE"]) else f"{row['RMSE']:.4f}",
            "—" if pd.isna(row["Precision@10"]) else f"{row['Precision@10']:.4f}",
            "—" if pd.isna(row["Recall@10"]) else f"{row['Recall@10']:.4f}",
            "—" if pd.isna(row["NDCG@10"]) else f"{row['NDCG@10']:.4f}",
        ]
        is_highlight = row["Model"] in (best_rmse_row["Model"], best_ndcg_row["Model"])
        for c_idx, val_str in enumerate(vals):
            cell = tbl.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(239, 246, 255) if is_highlight else (WHITE if r_idx % 2 == 1 else LIGHT_BG)
            p = cell.text_frame.paragraphs[0]
            p.text = val_str
            p.font.size = Pt(9.8)
            p.font.bold = is_highlight
            p.font.color.rgb = DARK_SLATE
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    _add_card(
        s3,
        0.38,
        5.25,
        6.25,
        1.78,
        "Key Empirical Findings",
        [
            f"Best Rating Accuracy: {best_rmse_row['Model']} achieves Test RMSE = {best_rmse_row['RMSE']:.4f} (vs. Popularity {pop_row['RMSE']:.4f}).",
            f"Best Top-10 Ranking: {best_ndcg_row['Model']} achieves NDCG@10 = {best_ndcg_row['NDCG@10']:.4f} and Recall@10 = {best_ndcg_row['Recall@10']:.4f}.",
            f"Explicit-Implicit Paradox: Biased SVD has low RMSE ({svd_row['RMSE']:.4f}) but weak NDCG@10 ({svd_row['NDCG@10']:.4f}) due to missing-not-at-random exposure bias; Implicit ALS & NCF resolve this via negative feedback.",
        ],
        accent=ACCENT_RED,
    )

    if (fig_dir / "fig5_model_comparison.png").exists():
        s3.shapes.add_picture(str(fig_dir / "fig5_model_comparison.png"), Inches(6.80), Inches(1.22), width=Inches(6.20))
    if (fig_dir / "fig6_ablation_ncf.png").exists():
        s3.shapes.add_picture(str(fig_dir / "fig6_ablation_ncf.png"), Inches(6.80), Inches(4.25), width=Inches(6.20))

    # ==================== SLIDE 4: ERROR & BIAS ANALYSIS ====================
    s4 = prs.slides.add_slide(blank_layout)
    _add_header(
        s4,
        "Error Analysis: Cold-Start, Gray-Sheep Users & Popularity Bias",
        "Disaggregating Performance Across User Cohorts and Catalog Diversity Metrics",
        4,
    )
    if (fig_dir / "fig7_error_user_groups.png").exists():
        s4.shapes.add_picture(str(fig_dir / "fig7_error_user_groups.png"), Inches(0.38), Inches(1.22), width=Inches(6.30))
    if (fig_dir / "fig8_popularity_bias_coverage.png").exists():
        s4.shapes.add_picture(str(fig_dir / "fig8_popularity_bias_coverage.png"), Inches(6.78), Inches(1.22), width=Inches(6.20))

    pop_cov = bias_df.loc["Popularity", "Catalog Coverage"] * 100.0
    best_m_name = str(best_ndcg_row["Model"])
    best_cov = bias_df.loc[best_m_name, "Catalog Coverage"] * 100.0 if best_m_name in bias_df.index else 0.0
    pop_head = bias_df.loc["Popularity", "Top-10% Head Share"] * 100.0
    best_head = bias_df.loc[best_m_name, "Top-10% Head Share"] * 100.0 if best_m_name in bias_df.index else 0.0

    _add_card(
        s4,
        0.38,
        4.55,
        6.30,
        2.45,
        "1. Cohort Failure Modes (Cold-Start & Gray-Sheep)",
        [
            "Activity Stratification: On Light Users (bottom 25%), Implicit ALS leads at 0.0746 NDCG@10 (vs. NeuMF 0.0730, >2.3x Popularity 0.0315); on Heavy Users, MLP/NeuMF reach ~0.12.",
            "Gray-Sheep Cohort (bottom 20% item popularity): Pre-trained NeuMF reaches 0.0757 NDCG@10 (+71.7% over Popularity 0.0441), surfacing niche hits (e.g., Rank #1 on User #5333).",
        ],
        accent=BLUE,
    )
    _add_card(
        s4,
        6.78,
        4.55,
        6.20,
        2.45,
        "2. Popularity Bias vs. Catalog Coverage",
        [
            f"Popularity Trap: Popularity recommends {pop_head:.1f}% Top-10% head items and covers only {pop_cov:.2f}% of the movie catalog.",
            f"Personalization Gain: {best_m_name} expands catalog coverage to {best_cov:.1f}% while reducing head-item concentration to {best_head:.1f}%.",
            "Explicit SVD suffers from rare-item bias inflation (recommending obscure 5-star items with high variance).",
        ],
        accent=ACCENT_TEAL,
    )

    # ==================== SLIDE 5: DEMO, CONCLUSIONS & FUTURE WORK ====================
    s5 = prs.slides.add_slide(blank_layout)
    _add_header(
        s5,
        "Interactive Demo System, Conclusions & Future Directions",
        "End-to-End Reproducible Pipeline & Live Streamlit Multi-Model Comparison",
        5,
    )
    _add_card(
        s5,
        0.40,
        1.28,
        6.10,
        5.72,
        "Live Interactive Demo (streamlit run demo/app.py)",
        [
            "Persona Selector: Instantly switch between Mainstream Fans, Heavy Cinephiles, Light/Cold-Start Users, and Gray-Sheep Niche Users (or any User ID 0-6,039).",
            "Historical Transparency: Displays user demographics (gender, age bracket, occupation) alongside top-rated Train+Val history and held-out Test positives.",
            "Side-by-Side Real-Time Ranking (<20ms): Compares Popularity, Biased SVD, Implicit ALS, and Pre-trained NeuMF simultaneously across all 3,883 movies.",
            "Ground-Truth Hit Verification: Automatically marks held-out Test hits (rating >= 4) and computes pairwise Jaccard list overlap to visualize personalization diversity.",
            "Integrated Analytics Tab: Interactive access to Test benchmark tables and error-analysis charts during live defense.",
        ],
        accent=BLUE,
    )
    _add_card(
        s5,
        6.75,
        1.28,
        6.18,
        2.85,
        "Core Conclusions",
        [
            "Objective Alignment Matters: Optimize MSE (Biased SVD) for star-rating prediction, but optimize Implicit Ranking (ALS / NeuMF with negative sampling) for Top-N discovery.",
            "Pre-training Unlocks NeuMF: Initializing NeuMF from GMF + MLP combines linear multiplicative and deep non-linear representations effectively.",
            "Beyond Accuracy: Evaluating Coverage, Gini concentration, and Gray-Sheep cohorts exposes biases hidden by aggregate NDCG@10.",
        ],
        accent=ACCENT_RED,
    )
    _add_card(
        s5,
        6.75,
        4.32,
        6.18,
        2.68,
        "Future Directions",
        [
            "Sequential Dynamics: Incorporate temporal order via Transformer models (SASRec, BERT4Rec) rather than static bag-of-items.",
            "Graph Representation Learning: High-order user-item bipartite message passing via LightGCN.",
            "Debiased Negative Sampling & Side Information: Popularity-smoothed negative sampling plus genre/plot embeddings from Hugging Face Transformers for true zero-shot cold-start.",
        ],
        accent=ACCENT_TEAL,
    )

    pptx_path = out_dir / "presentation_5_slides.pptx"
    prs.save(str(pptx_path))

    # Also write companion Markdown slide summary
    md_content = f"""# 5-Slide Project Presentation Summary

File generated: `docs/presentation_5_slides.pptx` (5 widescreen 16:9 slides)

## Slide 1: Movie Recommendation on MovieLens-1M — Problem & Temporal Protocol
- **Objective**: Compare Popularity, Matrix Factorization (`Biased SVD` & `Implicit ALS`), and Neural Collaborative Filtering (`GMF`, `MLP`, `NeuMF`) across `RMSE`, `Precision@10`, `Recall@10`, and `NDCG@10`.
- **Dataset**: MovieLens-1M (`{eda['n_ratings']:,}` ratings, `{eda['n_users']:,}` users, `{eda['n_items']:,}` movies, `{eda['sparsity']*100:.2f}%` sparsity).
- **Split**: Strict per-user chronological `80/10/10` split preventing future leakage.

## Slide 2: Methodology — Linear Matrix Factorization vs. Dual-Pathway NCF
- **Popularity**: IMDB Weighted Rating (`q = 0.80`) + item mean baseline.
- **Matrix Factorization**: Explicit Biased SVD (MSE + L2) vs. Implicit ALS (Hu et al., 2008 confidence-weighted ridge regression).
- **Neural CF**: GMF + MLP (`64 -> 32 -> 16`) fused into **Pre-trained NeuMF** (`alpha = 0.5`) with 4:1 negative sampling and BCE loss.

## Slide 3: Quantitative Results & Ablation Study
- **Best Rating Predictor**: `{best_rmse_row['Model']}` (`RMSE = {best_rmse_row['RMSE']:.4f}`).
- **Best Top-10 Ranker**: `{best_ndcg_row['Model']}` (`NDCG@10 = {best_ndcg_row['NDCG@10']:.4f}`, `Recall@10 = {best_ndcg_row['Recall@10']:.4f}`).
- **Explicit-Implicit Paradox**: Demonstrates why low RMSE does not imply strong Top-N ranking without negative feedback.

## Slide 4: Error Analysis — Cold-Start, Gray-Sheep Users & Popularity Bias
- Disaggregates performance across **Light / Moderate / Heavy** activity groups and **Gray-Sheep (Niche) vs. Mainstream** tastes.
- Quantifies the **Accuracy vs. Catalog Coverage** frontier (`Popularity` coverage `{pop_cov:.2f}%` vs. `{best_m_name}` `{best_cov:.1f}%`).

## Slide 5: Interactive Streamlit Demo, Conclusions & Future Work
- Live real-time demonstration (`streamlit run demo/app.py`) with persona presets, Test-set hit verification, and pairwise Jaccard diversity.
- Extensions to Sequential RecSys (`SASRec`), Graph CF (`LightGCN`), and Transformer content embeddings.
"""
    (out_dir / "presentation_5_slides.md").write_text(md_content, encoding="utf-8")
    print(f"Saved 5-slide presentation -> {pptx_path}")
    return pptx_path


if __name__ == "__main__":
    build_presentation()
