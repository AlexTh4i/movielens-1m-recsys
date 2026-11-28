"""Build TWO distinct 5-slide presentation decks (+ dedicated projector-readable charts):
1. Academic Defense Deck (English, minimal text, KPI hero cards, 1 large visual per slide, Speaker Notes):
   -> docs/presentation_5_slides.pptx  &  docs/presentation_5_slides.md
2. Non-Technical Visual Explainer Deck (Everyday Netflix analogies, zero jargon, Q&A cheat sheet):
   -> docs/presentation_nontech_5_slides.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# Modern Presentation Color Palette
NAVY = RGBColor(15, 23, 42)        # #0F172A
SLATE_DARK = RGBColor(30, 41, 59)  # #1E293B
SLATE_MID = RGBColor(71, 85, 105)  # #475569
SLATE_LIGHT = RGBColor(248, 250, 252)
BORDER = RGBColor(226, 232, 240)
WHITE = RGBColor(255, 255, 255)
BLUE = RGBColor(37, 99, 235)       # #2563EB
BLUE_SOFT = RGBColor(239, 246, 255)
TEAL = RGBColor(13, 148, 136)      # #0D9488
TEAL_SOFT = RGBColor(240, 253, 250)
RED = RGBColor(220, 38, 38)        # #DC2626
RED_SOFT = RGBColor(254, 242, 242)
AMBER = RGBColor(217, 119, 6)      # #D97706
AMBER_SOFT = RGBColor(255, 251, 235)
PURPLE = RGBColor(124, 58, 237)    # #7C3AED
PURPLE_SOFT = RGBColor(245, 243, 255)


# =====================================================================
# 1. Dedicated Large-Font Presentation Charts (Readable from Back Row!)
# =====================================================================
def generate_slide_charts(metrics_df: pd.DataFrame, bias_df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )

    # Chart A: Clean Horizontal Bar of Top-10 Ranking (NDCG@10) for Academic Slide 3
    order = [
        "Biased SVD (MF)",
        "NeuMF-Explicit (MSE)",
        "Popularity",
        "Implicit ALS (MF)",
        "MLP",
        "GMF",
        "NeuMF (Scratch)",
        "NeuMF (Pre-trained)",
    ]
    sub = metrics_df.set_index("Model").reindex(order).dropna(subset=["NDCG@10"]).reset_index()
    colors = [
        "#94a3b8" if m in ("Biased SVD (MF)", "NeuMF-Explicit (MSE)")
        else "#64748b" if m == "Popularity"
        else "#0d9488" if m == "Implicit ALS (MF)"
        else "#f59e0b" if m in ("GMF", "MLP", "NeuMF (Scratch)")
        else "#dc2626"
        for m in sub["Model"]
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    y = np.arange(len(sub))
    bars = ax.barh(y, sub["NDCG@10"], color=colors, height=0.62, edgecolor="#0f172a", linewidth=0.9)
    for bar, val, name in zip(bars, sub["NDCG@10"], sub["Model"]):
        ax.text(
            val + 0.0018,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold" if name == "NeuMF (Pre-trained)" else "normal",
            color="#b91c1c" if name == "NeuMF (Pre-trained)" else "#0f172a",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["Model"], fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 0.095)
    ax.set_xlabel("Test NDCG@10 (Higher is Better ↑)", fontsize=12, fontweight="bold")
    ax.set_title("Full-Catalog Top-10 Ranking Quality", fontsize=14, fontweight="bold", pad=10, color="#0f172a")
    fig.tight_layout()
    p_ndcg = out_dir / "slide_fig_ndcg.png"
    fig.savefig(p_ndcg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Chart B: Single Clean Scatter — Accuracy vs. Catalog Coverage for Academic Slide 4
    merged = pd.merge(metrics_df, bias_df, on="Model", how="inner")
    keep = ["Popularity", "Biased SVD (MF)", "Implicit ALS (MF)", "GMF", "MLP", "NeuMF (Pre-trained)"]
    merged = merged[merged["Model"].isin(keep)]
    c_map = {
        "Popularity": "#64748b",
        "Biased SVD (MF)": "#2563eb",
        "Implicit ALS (MF)": "#0d9488",
        "GMF": "#f59e0b",
        "MLP": "#7c3aed",
        "NeuMF (Pre-trained)": "#dc2626",
    }
    offsets = {
        "Popularity": (10, 8),
        "Biased SVD (MF)": (-52, 14),
        "Implicit ALS (MF)": (-65, -22),
        "GMF": (-46, -18),
        "MLP": (10, 6),
        "NeuMF (Pre-trained)": (-132, 10),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    # Shaded "Ideal Zone" top-right
    ax.axvspan(27, 43, ymin=0.50, ymax=1.0, color="#ecfdf5", alpha=0.65, zorder=0)
    ax.text(35.0, 0.0925, "Sweet Spot:\nHigh Accuracy + Diversity", ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#047857")

    for _, r in merged.iterrows():
        m = r["Model"]
        cov = r["Catalog Coverage"] * 100.0
        ndcg = r["NDCG@10"]
        ax.scatter(cov, ndcg, s=340, color=c_map[m], edgecolors="#0f172a", linewidth=1.6, zorder=4)
        dx, dy = offsets[m]
        ax.annotate(m, xy=(cov, ndcg), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.set_xlim(-1, 43.5)
    ax.set_ylim(0.036, 0.095)
    ax.set_xlabel("Catalog Coverage (% of 3,883 Movies Recommended ↑)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test NDCG@10 (↑)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs. Catalog Diversity Frontier", fontsize=14, fontweight="bold", pad=10, color="#0f172a")
    fig.tight_layout()
    p_cov = out_dir / "slide_fig_coverage.png"
    fig.savefig(p_cov, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Chart C: Friendly Non-Tech Bar Chart (Vietnamese Labels, Simple 4-Model Story)
    nt_models = [
        ("Popularity", "Chỉ gợi ý\nPhim Hot"),
        ("Biased SVD (MF)", "Máy đoán\nSố Sao (SVD)"),
        ("Implicit ALS (MF)", "Máy đoán Gu\n(ALS)"),
        ("NeuMF (Pre-trained)", "AI Học Sâu\n(NeuMF)"),
    ]
    m_idx = metrics_df.set_index("Model")
    b_idx = bias_df.set_index("Model")
    labels = [lbl for _, lbl in nt_models]
    ndcgs = [float(m_idx.loc[k, "NDCG@10"]) / float(m_idx.loc["Popularity", "NDCG@10"]) * 100.0 for k, _ in nt_models]
    covs = [float(b_idx.loc[k, "Unique Recommended Movies"]) for k, _ in nt_models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.9))
    bar_colors = ["#64748b", "#3b82f6", "#0d9488", "#dc2626"]
    b1 = ax1.bar(labels, ndcgs, color=bar_colors, width=0.56, edgecolor="#0f172a", linewidth=1.0)
    ax1.axhline(100, color="#64748b", ls="--", lw=1.2)
    for bar, val in zip(b1, ndcgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=12.5, fontweight="bold", color="#0f172a")
    ax1.set_ylim(0, 210)
    ax1.set_ylabel("Độ chính xác Top-10 (Phim Hot = 100%)", fontsize=11, fontweight="bold")
    ax1.set_title("Độ chuẩn khi gợi ý 10 phim mới", fontsize=13.5, fontweight="bold", color="#0f172a", pad=10)

    b2 = ax2.bar(labels, covs, color=bar_colors, width=0.56, edgecolor="#0f172a", linewidth=1.0)
    for bar, val in zip(b2, covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 35, f"{val:,.0f}\nphim", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="#0f172a")
    ax2.set_ylim(0, 1780)
    ax2.set_ylabel("Số phim khác nhau được gợi ý (trên 3.883)", fontsize=11, fontweight="bold")
    ax2.set_title("Độ phong phú kho phim (tránh lặp lại)", fontsize=13.5, fontweight="bold", color="#0f172a", pad=10)

    fig.tight_layout(w_pad=3.0)
    p_nt = out_dir / "slide_fig_nontech.png"
    fig.savefig(p_nt, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Chart D: Friendly Non-Tech Bar Chart (Plain English Labels, Simple 4-Advisor Story)
    nt_en_models = [
        ("Popularity", "Popular\n(Baseline)"),
        ("Biased SVD (MF)", "Biased SVD\n(Stars)"),
        ("Implicit ALS (MF)", "Implicit ALS\n(Taste)"),
        ("NeuMF (Pre-trained)", "Deep AI\n(NeuMF)"),
    ]
    labels_en = [lbl for _, lbl in nt_en_models]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
    b1 = ax1.bar(labels_en, ndcgs, color=bar_colors, width=0.52, edgecolor="#0f172a", linewidth=1.0)
    ax1.axhline(100, color="#64748b", ls="--", lw=1.2)
    for bar, val in zip(b1, ndcgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=12.5, fontweight="bold", color="#0f172a")
    ax1.tick_params(axis="x", labelsize=10.8)
    ax1.set_ylim(0, 210)
    ax1.set_ylabel("Top-10 Accuracy (Popular = 100%)", fontsize=11, fontweight="bold")
    ax1.set_title("How Well Top-10 Picks Match Real Taste", fontsize=13.5, fontweight="bold", color="#0f172a", pad=10)

    b2 = ax2.bar(labels_en, covs, color=bar_colors, width=0.52, edgecolor="#0f172a", linewidth=1.0)
    for bar, val in zip(b2, covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 35, f"{val:,.0f}\nmovies", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="#0f172a")
    ax2.tick_params(axis="x", labelsize=10.8)
    ax2.set_ylim(0, 1780)
    ax2.set_ylabel("Distinct Movies Suggested (out of 3,883)", fontsize=11, fontweight="bold")
    ax2.set_title("Catalog Variety (Discovering Hidden Gems)", fontsize=13.5, fontweight="bold", color="#0f172a", pad=10)

    fig.tight_layout(w_pad=3.0)
    p_nt_en = out_dir / "slide_fig_nontech_en.png"
    fig.savefig(p_nt_en, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"ndcg": p_ndcg, "coverage": p_cov, "nontech": p_nt, "nontech_en": p_nt_en}


# =====================================================================
# Shared Slide Helper Primitives
# =====================================================================
def _set_bg(slide, color: RGBColor) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _set_notes(slide, text: str) -> None:
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = text


def _add_top_bar(slide, kicker: str, headline: str, slide_num: int, total: int = 5) -> None:
    # Subtle top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.10))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    # Kicker (uppercase category tag)
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.24), Inches(11.5), Inches(0.32))
    p0 = tb.text_frame.paragraphs[0]
    p0.text = kicker.upper()
    p0.font.name = "Calibri"
    p0.font.size = Pt(11.5)
    p0.font.bold = True
    p0.font.color.rgb = BLUE

    # Action Headline (large, punchy)
    tb2 = slide.shapes.add_textbox(Inches(0.55), Inches(0.52), Inches(12.2), Inches(0.65))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p1 = tf2.paragraphs[0]
    p1.text = headline
    p1.font.name = "Calibri"
    p1.font.size = Pt(25)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    # Thin divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.22), Inches(12.23), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    # Minimal footer
    ft = slide.shapes.add_textbox(Inches(11.6), Inches(7.08), Inches(1.2), Inches(0.3))
    fp = ft.text_frame.paragraphs[0]
    fp.text = f"{slide_num} / {total}"
    fp.alignment = PP_ALIGN.RIGHT
    fp.font.name = "Calibri"
    fp.font.size = Pt(11)
    fp.font.bold = True
    fp.font.color.rgb = SLATE_MID


def _add_kpi_tile(slide, left: float, top: float, width: float, height: float, number: str, label: str, sub: str, accent: RGBColor, bg: RGBColor):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.08
    card.fill.solid()
    card.fill.fore_color.rgb = bg
    card.line.color.rgb = accent
    card.line.width = Pt(1.5)

    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)

    p_num = tf.paragraphs[0]
    p_num.text = number
    p_num.alignment = PP_ALIGN.CENTER
    p_num.font.name = "Calibri"
    p_num.font.size = Pt(30)
    p_num.font.bold = True
    p_num.font.color.rgb = accent

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.alignment = PP_ALIGN.CENTER
    p_lbl.font.name = "Calibri"
    p_lbl.font.size = Pt(13)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = NAVY

    if sub:
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.alignment = PP_ALIGN.CENTER
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = SLATE_MID


def _add_accent_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    points: list[str],
    accent: RGBColor,
    bg: RGBColor = SLATE_LIGHT,
    title_size: float = 17,
    body_size: float = 14.5,
    badge: str | None = None,
):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.05
    card.fill.solid()
    card.fill.fore_color.rgb = bg
    card.line.color.rgb = BORDER
    card.line.width = Pt(1.0)

    # Colored left accent stripe
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.12), Inches(0.08), Inches(height - 0.24))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background()

    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.26)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.16)
    tf.margin_bottom = Inches(0.12)

    p0 = tf.paragraphs[0]
    if badge:
        p0.text = f"{badge}   {title}"
    else:
        p0.text = title
    p0.font.name = "Calibri"
    p0.font.size = Pt(title_size)
    p0.font.bold = True
    p0.font.color.rgb = accent
    p0.space_after = Pt(8)

    for pt in points:
        p = tf.add_paragraph()
        p.text = f"▸  {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(body_size)
        p.font.color.rgb = SLATE_DARK
        p.space_after = Pt(7)


# =====================================================================
# 2. DECK 1: Academic Presentation Style (5 Clean Slides + Speaker Notes)
# =====================================================================
def build_academic_deck(eda: dict, metrics_df: pd.DataFrame, bias_df: pd.DataFrame, charts: dict[str, Path], out_dir: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    m_idx = metrics_df.set_index("Model")
    b_idx = bias_df.set_index("Model")
    svd_rmse = float(m_idx.loc["Biased SVD (MF)", "RMSE"])
    pop_ndcg = float(m_idx.loc["Popularity", "NDCG@10"])
    npt_ndcg = float(m_idx.loc["NeuMF (Pre-trained)", "NDCG@10"])
    lift_pct = (npt_ndcg - pop_ndcg) / pop_ndcg * 100.0

    # ---------- SLIDE 1: HERO COVER ----------
    s1 = prs.slides.add_slide(blank)
    _set_bg(s1, NAVY)

    # Blue vertical accent bar on left
    acc = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.15), Inches(0.12), Inches(3.2))
    acc.fill.solid()
    acc.fill.fore_color.rgb = BLUE
    acc.line.fill.background()

    # Course Tag
    t_tag = s1.shapes.add_textbox(Inches(1.0), Inches(1.12), Inches(11.0), Inches(0.4))
    p = t_tag.text_frame.paragraphs[0]
    p.text = "ADVANCED ARTIFICIAL INTELLIGENCE  ·  COURSE PROJECT DEFENSE"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(96, 165, 250)

    # Big Hero Title
    t_main = s1.shapes.add_textbox(Inches(1.0), Inches(1.62), Inches(11.3), Inches(1.8))
    tf = t_main.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Movie Recommendations on MovieLens-1M"
    p1.font.name = "Calibri"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "Popularity Baseline vs. Matrix Factorization (SVD / ALS) vs. Neural Collaborative Filtering"
    p2.font.name = "Calibri"
    p2.font.size = Pt(21)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(12)

    # 3 Clean Stat Pills along bottom
    pills = [
        ("1,000,209", "Explicit Ratings (1–5★)", "6,040 Users  ·  3,883 Movies", BLUE),
        ("80 / 10 / 10", "Per-User Temporal Split", "Zero Future Data Leakage", TEAL),
        ("Full Ranking", "4 Unified Metrics", "RMSE · P@10 · R@10 · NDCG@10", AMBER),
    ]
    for idx, (num, lbl, sub, col) in enumerate(pills):
        x = 1.0 + idx * 3.88
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.85), Inches(3.55), Inches(1.65))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = SLATE_DARK
        box.line.color.rgb = col
        box.line.width = Pt(1.8)
        ptf = box.text_frame
        ptf.word_wrap = True
        ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
        q0 = ptf.paragraphs[0]
        q0.text = num
        q0.alignment = PP_ALIGN.CENTER
        q0.font.name = "Calibri"
        q0.font.size = Pt(28)
        q0.font.bold = True
        q0.font.color.rgb = WHITE
        q1 = ptf.add_paragraph()
        q1.text = lbl
        q1.alignment = PP_ALIGN.CENTER
        q1.font.name = "Calibri"
        q1.font.size = Pt(14)
        q1.font.bold = True
        q1.font.color.rgb = RGBColor(147, 197, 253)
        q2 = ptf.add_paragraph()
        q2.text = sub
        q2.alignment = PP_ALIGN.CENTER
        q2.font.name = "Calibri"
        q2.font.size = Pt(11.5)
        q2.font.color.rgb = RGBColor(148, 163, 184)

    foot = s1.shapes.add_textbox(Inches(1.0), Inches(6.85), Inches(11.3), Inches(0.4))
    fp = foot.text_frame.paragraphs[0]
    fp.text = "PyTorch Implementation  |  github.com/AlexTh4i/movielens-1m-recsys  |  Interactive Streamlit Demo"
    fp.font.name = "Calibri"
    fp.font.size = Pt(12)
    fp.font.color.rgb = RGBColor(148, 163, 184)

    _set_notes(
        s1,
        "SPEAKER SCRIPT (~45s):\n"
        "Good morning Professor. Today we present our comparative study of movie recommendation models on MovieLens-1M.\n"
        "We compare three core families: a Bayesian Popularity baseline, Matrix Factorization (both Explicit Biased SVD and Implicit ALS), "
        "and Deep Neural Collaborative Filtering (GMF, MLP, and Pre-trained NeuMF).\n"
        "To keep evaluation realistic and leak-free, we split every user's history chronologically—oldest 80% Train, 10% Validation, newest 10% Test—"
        "and rank across all 3,883 unseen movies rather than sampling 99 negatives.\n\n"
        "[Gợi ý tiếng Việt]: Giới thiệu nhanh tên đề tài, bộ dữ liệu 1 triệu lượt chấm, và nhấn mạnh cách chia theo thời gian 80/10/10 không rò rỉ tương lai.",
    )

    # ---------- SLIDE 2: 3 PARADIGMS AT A GLANCE ----------
    s2 = prs.slides.add_slide(blank)
    _set_bg(s2, WHITE)
    _add_top_bar(s2, "01 · Methodology", "Three Collaborative Paradigms: Heuristic → Linear MF → Deep NCF", 2)

    cols_data = [
        (
            "1. Popularity Baseline",
            "Heuristic · Non-Personalized",
            SLATE_MID,
            SLATE_LIGHT,
            [
                "IMDB Bayesian Weighted Rating",
                "Balances rating quality & vote volume",
                "Same top movies for all users",
                "Item-mean fallback for RMSE",
            ],
            "WR = v/(v+m)·R + m/(v+m)·C",
        ),
        (
            "2. Matrix Factorization",
            "Linear Latent Space · SVD & ALS",
            TEAL,
            TEAL_SOFT,
            [
                "Biased SVD: Explicit 1–5★ (MSE + L2)",
                "Implicit ALS (Hu et al., 2008): learns from observed + unseen pairs",
                "Fast closed-form ridge updates",
                "12-config grid search on Validation",
            ],
            "r̂ = μ + bᵤ + bᵢ + ⟨pᵤ, qᵢ⟩",
        ),
        (
            "3. Neural CF (NeuMF)",
            "Deep Non-Linear · He et al. (2017)",
            RED,
            RED_SOFT,
            [
                "GMF branch: element-wise product",
                "MLP tower: 64→32→16 + BN + Dropout",
                "Two-stage Pre-training (α = 0.5)",
                "4:1 dynamic negative sampling (BCE)",
            ],
            "ŷ = σ( hᵀ [ φ_GMF ∥ φ_MLP ] )",
        ),
    ]
    for idx, (title, tag, acc_col, bg_col, pts, formula) in enumerate(cols_data):
        x = 0.55 + idx * 4.18
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(4.55))
        card.adjustments[0] = 0.05
        card.fill.solid()
        card.fill.fore_color.rgb = bg_col
        card.line.color.rgb = acc_col
        card.line.width = Pt(1.5)

        # Header block inside column
        hdr = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(0.88))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = acc_col
        hdr.line.fill.background()
        htf = hdr.text_frame
        htf.margin_left = Inches(0.18)
        htf.vertical_anchor = MSO_ANCHOR.MIDDLE
        hp1 = htf.paragraphs[0]
        hp1.text = title
        hp1.font.name = "Calibri"
        hp1.font.size = Pt(18)
        hp1.font.bold = True
        hp1.font.color.rgb = WHITE
        hp2 = htf.add_paragraph()
        hp2.text = tag
        hp2.font.name = "Calibri"
        hp2.font.size = Pt(11.5)
        hp2.font.color.rgb = RGBColor(241, 245, 249)

        # Bullets
        bt = s2.shapes.add_textbox(Inches(x + 0.16), Inches(2.48), Inches(3.56), Inches(2.65))
        btf = bt.text_frame
        btf.word_wrap = True
        for b_i, line_txt in enumerate(pts):
            bp = btf.paragraphs[0] if b_i == 0 else btf.add_paragraph()
            bp.text = f"▸  {line_txt}"
            bp.font.name = "Calibri"
            bp.font.size = Pt(14.5)
            bp.font.color.rgb = SLATE_DARK
            bp.space_after = Pt(10)

        # Formula pill at bottom of card
        fpill = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.18), Inches(5.22), Inches(3.52), Inches(0.60))
        fpill.adjustments[0] = 0.18
        fpill.fill.solid()
        fpill.fill.fore_color.rgb = WHITE
        fpill.line.color.rgb = acc_col
        fpill.line.width = Pt(1.0)
        ftf = fpill.text_frame
        ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
        fp0 = ftf.paragraphs[0]
        fp0.text = formula
        fp0.alignment = PP_ALIGN.CENTER
        fp0.font.name = "Cambria Math"
        fp0.font.size = Pt(13)
        fp0.font.bold = True
        fp0.font.color.rgb = NAVY

    # Bottom Pipeline Strip
    strip = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(6.22), Inches(12.23), Inches(0.70))
    strip.adjustments[0] = 0.2
    strip.fill.solid()
    strip.fill.fore_color.rgb = NAVY
    strip.line.fill.background()
    stf = strip.text_frame
    stf.vertical_anchor = MSO_ANCHOR.MIDDLE
    sp0 = stf.paragraphs[0]
    sp0.text = "Unified PyTorch Pipeline:    Chronological 80/10/10 Split   →   Early Stopping on Val   →   Full-Catalog Test Ranking (3,883 movies)"
    sp0.alignment = PP_ALIGN.CENTER
    sp0.font.name = "Calibri"
    sp0.font.size = Pt(14)
    sp0.font.bold = True
    sp0.font.color.rgb = WHITE

    _set_notes(
        s2,
        "SPEAKER SCRIPT (~60s):\n"
        "Here are our three model families side by side.\n"
        "First, our Popularity baseline uses the IMDB weighted rating formula—a strong non-personalized sanity check.\n"
        "Second, Matrix Factorization embeds users and movies into a linear vector space. Crucially, we implement BOTH Explicit Biased SVD "
        "(trained on 1–5 star MSE) and Implicit ALS (Hu et al., 2008), which also penalizes unobserved movies.\n"
        "Third, Neural Collaborative Filtering (He et al., 2017) combines a multiplicative GMF branch with a 3-layer MLP tower, "
        "using two-stage pre-training (alpha = 0.5) and 4-to-1 negative sampling.\n"
        "We also train an Explicit MSE NeuMF variant so we can compare RMSE fairly across architectures.\n\n"
        "[Gợi ý tiếng Việt]: Nhấn mạnh sự khác nhau cốt lõi: Popularity không cá nhân hóa; SVD/ALS dùng tích vô hướng tuyến tính; NeuMF kết hợp 2 nhánh GMF + MLP phi tuyến.",
    )

    # ---------- SLIDE 3: HEADLINE RESULTS ----------
    s3 = prs.slides.add_slide(blank)
    _set_bg(s3, WHITE)
    _add_top_bar(s3, "02 · Test Set Results", f"Biased SVD Wins RMSE ({svd_rmse:.4f}), Pre-trained NeuMF Wins Top-10 (+{lift_pct:.1f}%)", 3)

    # Top 3 Hero KPI Tiles
    _add_kpi_tile(s3, 0.55, 1.42, 3.85, 1.25, f"{svd_rmse:.4f}", "Best Rating Error (Test RMSE ↓)", "Biased SVD · −23.2% vs Global Mean", BLUE, BLUE_SOFT)
    _add_kpi_tile(s3, 4.74, 1.42, 3.85, 1.25, f"{npt_ndcg:.4f}", "Best Top-10 Ranking (NDCG@10 ↑)", "Pre-trained NeuMF · HitRate@10 = 34.7%", RED, RED_SOFT)
    _add_kpi_tile(s3, 8.93, 1.42, 3.85, 1.25, f"+{lift_pct:.1f}%", "Ranking Lift Over Popularity", "Implicit feedback unlocks Top-N accuracy", TEAL, TEAL_SOFT)

    # Left: Compact 6-Row Table (Clean 13pt!)
    key_models = ["Popularity", "Biased SVD (MF)", "Implicit ALS (MF)", "GMF", "MLP", "NeuMF (Pre-trained)"]
    tbl_shape = s3.shapes.add_table(len(key_models) + 1, 4, Inches(0.55), Inches(2.90), Inches(5.75), Inches(3.05))
    tbl = tbl_shape.table
    for ci, w in enumerate([Inches(2.35), Inches(1.10), Inches(1.10), Inches(1.20)]):
        tbl.columns[ci].width = w
    for ci, h in enumerate(["Model", "RMSE ↓", "R@10 ↑", "NDCG@10 ↑"]):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
        p.font.name = "Calibri"
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = WHITE

    for ri, m_name in enumerate(key_models, start=1):
        row = m_idx.loc[m_name]
        vals = [
            m_name,
            "—" if pd.isna(row["RMSE"]) else f"{row['RMSE']:.4f}",
            f"{row['Recall@10']:.4f}",
            f"{row['NDCG@10']:.4f}",
        ]
        highlight = m_name in ("Biased SVD (MF)", "NeuMF (Pre-trained)")
        for ci, v in enumerate(vals):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE_SOFT if m_name == "Biased SVD (MF)" else (RED_SOFT if m_name == "NeuMF (Pre-trained)" else (WHITE if ri % 2 else SLATE_LIGHT))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = v
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            p.font.name = "Calibri"
            p.font.size = Pt(12.5)
            p.font.bold = highlight
            p.font.color.rgb = NAVY

    # Bottom-left Punchy Takeaway Box (Explicit-Implicit Paradox)
    paradox = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(6.12), Inches(5.75), Inches(0.86))
    paradox.adjustments[0] = 0.12
    paradox.fill.solid()
    paradox.fill.fore_color.rgb = AMBER_SOFT
    paradox.line.color.rgb = AMBER
    paradox.line.width = Pt(1.4)
    ptf = paradox.text_frame
    ptf.word_wrap = True
    ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
    ptf.margin_left = Inches(0.16)
    pp = ptf.paragraphs[0]
    pp.text = "Key Insight: Explicit MSE (SVD / NeuMF-Explicit) scores low RMSE but trails Popularity on Top-10 because unobserved items get zero penalty."
    pp.font.name = "Calibri"
    pp.font.size = Pt(12)
    pp.font.bold = True
    pp.font.color.rgb = RGBColor(146, 64, 14)

    # Right: Single Large Presentation Chart (NDCG@10)
    s3.shapes.add_picture(str(charts["ndcg"]), Inches(6.52), Inches(2.85), width=Inches(6.28))

    _set_notes(
        s3,
        "SPEAKER SCRIPT (~75s):\n"
        "Slide 3 shows our main held-out Test results across all 6,040 users.\n"
        "Look at the two headline winners: Biased SVD achieves the lowest rating error at 0.8935 RMSE—a 23% reduction over global mean—"
        "while Pre-trained NeuMF dominates Top-10 ranking at 0.0801 NDCG@10 and 34.7% HitRate@10, beating Popularity by +77.6%.\n"
        "Notice a crucial scientific finding: Biased SVD gets the best RMSE, yet ranks worse than Popularity on Top-10 (0.0410 vs 0.0451). "
        "Why? Because explicit MSE never trains on unseen movies—obscure movies rated 5/5 by a handful of fans get inflated item biases.\n"
        "Once we introduce negative feedback via Implicit ALS (0.0665) or NCF negative sampling (GMF 0.0777, Pre-trained NeuMF 0.0801), Top-10 quality nearly doubles.\n\n"
        "[Gợi ý tiếng Việt]: Chỉ 3 ô số to ở trên trước -> giải thích ngắn nghịch lý vì sao SVD có RMSE thấp nhất nhưng Top-10 lại thấp hơn NeuMF/ALS.",
    )

    # ---------- SLIDE 4: ERROR ANALYSIS & DIVERSITY ----------
    s4 = prs.slides.add_slide(blank)
    _set_bg(s4, WHITE)
    _add_top_bar(s4, "03 · Error & Bias Analysis", "Personalization Unlocks Niche Users & Expands Catalog Coverage 12×", 4)

    # Left: Single Clean Scatter Chart (Accuracy vs Coverage)
    s4.shapes.add_picture(str(charts["coverage"]), Inches(0.50), Inches(1.50), width=Inches(6.85))

    # Right: 3 Punchy Visual Stat Cards
    _add_accent_card(
        s4,
        7.60,
        1.50,
        5.18,
        1.62,
        "Cold-Start Proxy (Light 25% Users)",
        [
            "Implicit ALS leads: 0.0746 NDCG@10 (2.4× Popularity)",
            "Closed-form L2 ridge resists overfitting on sparse users",
        ],
        accent=TEAL,
        bg=TEAL_SOFT,
        title_size=16,
        body_size=13.5,
    )
    _add_accent_card(
        s4,
        7.60,
        3.32,
        5.18,
        1.70,
        "Gray-Sheep / Niche Tastes (Bottom 20%)",
        [
            "Pre-trained NeuMF leads: 0.0757 (+71.7% vs Popularity)",
            "Real example (User #5333): Rank #1 verified Test hit!",
        ],
        accent=RED,
        bg=RED_SOFT,
        title_size=16,
        body_size=13.5,
    )
    _add_accent_card(
        s4,
        7.60,
        5.22,
        5.18,
        1.68,
        "Popularity Bias vs. Diversity",
        [
            "Popularity covers only 3.0% (116 movies; 97.8% head)",
            "MLP mimics popularity (96% head); NeuMF/ALS hit 34–37%",
        ],
        accent=BLUE,
        bg=BLUE_SOFT,
        title_size=16,
        body_size=13.5,
    )

    _set_notes(
        s4,
        "SPEAKER SCRIPT (~60s):\n"
        "Average metrics only tell half the story—Slide 4 breaks down where each model succeeds or fails.\n"
        "On the left, plotting Test NDCG@10 against Catalog Coverage reveals a clear frontier: Popularity sits in the bottom-left trap, "
        "recommending only 116 movies (3% of the catalog). Standalone MLP gets high accuracy mostly by repeating popular head movies (96% head share).\n"
        "In contrast, Pre-trained NeuMF and Implicit ALS sit in the upper-right sweet spot—covering 34% to 37% of the catalog (12x more movies).\n"
        "Across user cohorts: Implicit ALS wins on sparse Light Users (0.0746), while Pre-trained NeuMF dominates Gray-Sheep niche cinephiles (+71.7% over Popularity)—"
        "for instance, placing User #5333's exact held-out movie at Rank #1.\n\n"
        "[Gợi ý tiếng Việt]: Chỉ vùng màu xanh góc trên phải của biểu đồ -> nêu 3 ý nhanh bên phải: Cold-Start (ALS thắng), Gray-Sheep (NeuMF thắng), Độ phủ (tăng gấp 12 lần).",
    )

    # ---------- SLIDE 5: LIVE DEMO & CONCLUSIONS ----------
    s5 = prs.slides.add_slide(blank)
    _set_bg(s5, WHITE)
    _add_top_bar(s5, "04 · Live System & Conclusions", "Interactive Streamlit Demo & Key Takeaways", 5)

    # Left: Sleek Dark Live Demo Panel
    demo_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.50), Inches(5.85), Inches(5.35))
    demo_box.adjustments[0] = 0.05
    demo_box.fill.solid()
    demo_box.fill.fore_color.rgb = NAVY
    demo_box.line.color.rgb = BLUE
    demo_box.line.width = Pt(2.0)

    dtf = demo_box.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.TOP
    dtf.margin_left = Inches(0.32)
    dtf.margin_right = Inches(0.28)
    dtf.margin_top = Inches(0.28)

    dp0 = dtf.paragraphs[0]
    dp0.text = "🎬  LIVE DEMO WALKTHROUGH"
    dp0.font.name = "Calibri"
    dp0.font.size = Pt(14)
    dp0.font.bold = True
    dp0.font.color.rgb = RGBColor(96, 165, 250)

    dp1 = dtf.add_paragraph()
    dp1.text = "streamlit run demo/app.py"
    dp1.font.name = "Consolas"
    dp1.font.size = Pt(18)
    dp1.font.bold = True
    dp1.font.color.rgb = RGBColor(52, 211, 153)
    dp1.space_before = Pt(8)
    dp1.space_after = Pt(18)

    demo_steps = [
        ("1. Pick Any User or Persona", "Switch instantly between Mainstream, Cold-Start, Heavy, and Gray-Sheep User #5333"),
        ("2. Real-Time Side-by-Side (<20 ms)", "Compare Popularity, Biased SVD, Implicit ALS, and Pre-trained NeuMF live"),
        ("3. Ground-Truth Hit Verification", "Automatic ✅ badges against unseen chronological Test positives + Jaccard overlap"),
        ("4. Integrated Analytics Tab", "Full benchmark table & 300-DPI error analysis plots built right into the app"),
    ]
    for head, desc in demo_steps:
        p_h = dtf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Calibri"
        p_h.font.size = Pt(15)
        p_h.font.bold = True
        p_h.font.color.rgb = WHITE
        p_h.space_before = Pt(6)

        p_d = dtf.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Calibri"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = RGBColor(203, 213, 225)
        p_d.space_after = Pt(8)

    # Right Top: 3 Crisp Takeaways
    _add_accent_card(
        s5,
        6.72,
        1.50,
        6.06,
        3.50,
        "Three Key Takeaways",
        [
            "Match loss to task: MSE (Biased SVD) for rating prediction; Implicit Ranking (ALS / NeuMF) for Top-N discovery",
            "Pre-training matters: Fusing GMF + MLP lifts NeuMF above every single baseline on NDCG@10 (0.0801)",
            "Evaluate beyond averages: Coverage & cohort splits reveal popularity traps hidden by aggregate accuracy",
        ],
        accent=RED,
        bg=SLATE_LIGHT,
        title_size=18,
        body_size=14.5,
    )

    # Right Bottom: Future Work Pills
    _add_accent_card(
        s5,
        6.72,
        5.22,
        6.06,
        1.63,
        "Future Extensions",
        [
            "Sequential Transformers (SASRec / BERT4Rec) for watch order",
            "Graph CF (LightGCN) + Hugging Face text embeddings for zero-shot cold-start",
        ],
        accent=PURPLE,
        bg=PURPLE_SOFT,
        title_size=16,
        body_size=13.5,
    )

    _set_notes(
        s5,
        "SPEAKER SCRIPT (~45s -> SWITCH TO LIVE DEMO):\n"
        "To summarize our three takeaways: first, training objective matters more than model complexity—use MSE for star ratings, "
        "but implicit negative feedback for Top-10 discovery. Second, two-stage pre-training unlocks NeuMF's best performance. "
        "Third, cohort and diversity metrics prevent hidden popularity bubbles.\n"
        "In the future, we can extend this static setup with sequential Transformers like SASRec and graph models like LightGCN.\n"
        "Now I'd like to switch over to our live Streamlit demo and show User #5333 in real time—thank you, Professor!\n\n"
        "[Gợi ý tiếng Việt]: Chốt 3 ý chính -> mời thầy xem màn hình Demo Streamlit đang chạy sẵn -> chọn User #5333 cho thầy thấy NeuMF trúng Rank #1.",
    )

    out_path = out_dir / "presentation_5_slides.pptx"
    prs.save(str(out_path))
    return out_path


# =====================================================================
# 3. DECK 2: Non-Tech Friendly 5-Slide Deck (Vietnamese + English Terms)
# =====================================================================
def build_nontech_deck(charts: dict[str, Path], out_dir: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------- SLIDE 1: BÀI TOÁN DỄ HIỂU (NETFLIX ANALOGY) ----------
    s1 = prs.slides.add_slide(blank)
    _set_bg(s1, NAVY)

    acc = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.10), Inches(0.12), Inches(3.25))
    acc.fill.solid()
    acc.fill.fore_color.rgb = TEAL
    acc.line.fill.background()

    t0 = s1.shapes.add_textbox(Inches(1.0), Inches(1.05), Inches(11.2), Inches(0.4))
    p = t0.text_frame.paragraphs[0]
    p.text = "BẢN GIẢI THÍCH DỄ HIỂU (NON-TECH)  ·  HỆ THỐNG GỢI Ý PHIM BẰNG AI"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(94, 234, 212)

    t1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(11.3), Inches(1.8))
    tf = t1.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Làm sao Netflix biết bạn sắp thích xem phim gì?"
    p1.font.name = "Calibri"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "So sánh 3 cách máy tính gợi ý phim trên 1 triệu lượt đánh giá thật (MovieLens-1M)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(12)

    pills = [
        ("1 Triệu", "Lượt chấm 1–5 sao", "Của 6.040 khán giả trên 3.883 phim", BLUE),
        ("Thi Công Bằng", "Không xem trước đáp án", "Học 80% phim cũ → Đoán 10% phim tương lai", TEAL),
        ("Mục Tiêu", "Chọn đúng 10 phim mới", "Phim khán giả CHƯA xem nhưng sẽ thích", AMBER),
    ]
    for idx, (num, lbl, sub, col) in enumerate(pills):
        x = 1.0 + idx * 3.88
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.85), Inches(3.55), Inches(1.65))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = SLATE_DARK
        box.line.color.rgb = col
        box.line.width = Pt(1.8)
        ptf = box.text_frame
        ptf.word_wrap = True
        ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
        q0 = ptf.paragraphs[0]
        q0.text = num
        q0.alignment = PP_ALIGN.CENTER
        q0.font.name = "Calibri"
        q0.font.size = Pt(27)
        q0.font.bold = True
        q0.font.color.rgb = WHITE
        q1 = ptf.add_paragraph()
        q1.text = lbl
        q1.alignment = PP_ALIGN.CENTER
        q1.font.name = "Calibri"
        q1.font.size = Pt(14)
        q1.font.bold = True
        q1.font.color.rgb = RGBColor(147, 197, 253)
        q2 = ptf.add_paragraph()
        q2.text = sub
        q2.alignment = PP_ALIGN.CENTER
        q2.font.name = "Calibri"
        q2.font.size = Pt(11.5)
        q2.font.color.rgb = RGBColor(203, 213, 225)

    _set_notes(
        s1,
        "BẠN CẦN NHỚ GÌ Ở SLIDE 1?\n"
        "- Đồ án làm giống hệt tính năng gợi ý phim của Netflix.\n"
        "- Dữ liệu dùng là MovieLens-1M: 1 triệu lượt chấm điểm từ 1 đến 5 sao của 6.040 người trên 3.883 phim.\n"
        "- Cách chấm thi công bằng (Temporal Split): sắp xếp lịch sử của từng người theo thời gian, lấy 80% phim xem trước để dạy máy, 10% giữa để chỉnh máy, và giấu 10% phim xem sau cùng làm đề thi.",
    )

    # ---------- SLIDE 2: 3 'CHUYÊN GIA' ĐẤU NHAU ----------
    s2 = prs.slides.add_slide(blank)
    _set_bg(s2, WHITE)
    _add_top_bar(s2, "01 · Cách Máy Tính Suy Nghĩ", "3 'Chuyên Gia' Gợi Ý Phim Hoạt Động Khác Nhau Ra Sao?", 2)

    experts = [
        (
            "🏆 Ông 'Phim Hot'",
            "Popularity Baseline",
            SLATE_MID,
            SLATE_LIGHT,
            [
                "Giống bảng xếp hạng Top 10 phòng vé",
                "Ai vào cũng nhận danh sách y hệt nhau",
                "Không quan tâm bạn thích thể loại gì",
                "Dùng làm mốc tối thiểu để so sánh",
            ],
            "Ví dụ: Luôn khuyên Shawshank, Bố Già...",
        ),
        (
            "🎯 Ông 'Tìm Gu Ẩn'",
            "Matrix Factorization (SVD & ALS)",
            TEAL,
            TEAL_SOFT,
            [
                "Nhìn lịch sử xem để gom nhóm sở thích ngầm",
                "SVD: Giỏi đoán chính xác bạn sẽ chấm mấy sao",
                "ALS: Biết cả phim bạn bỏ qua → chọn Top 10 tốt hơn",
                "Nhanh, gọn, dễ triển khai thực tế",
            ],
            "Ví dụ: Thích Toy Story → Gợi ý phim hoạt hình",
        ),
        (
            "🧠 Ông 'AI Học Sâu'",
            "Neural CF (GMF + MLP → NeuMF)",
            RED,
            RED_SOFT,
            [
                "Dùng mạng nơ-ron học các mối liên hệ phức tạp",
                "Kết hợp 2 bộ não: tuyến tính (GMF) + học sâu (MLP)",
                "Luyện riêng 2 nhánh rồi ghép lại (Pre-trained)",
                "Học đối chiếu 1 phim thích với 4 phim chưa xem",
            ],
            "Ví dụ: Bắt trúng cả sở thích phim ít người biết",
        ),
    ]
    for idx, (title, sub, acc_col, bg_col, pts, ex_txt) in enumerate(experts):
        x = 0.55 + idx * 4.18
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(5.30))
        card.adjustments[0] = 0.05
        card.fill.solid()
        card.fill.fore_color.rgb = bg_col
        card.line.color.rgb = acc_col
        card.line.width = Pt(1.5)

        hdr = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(0.92))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = acc_col
        hdr.line.fill.background()
        htf = hdr.text_frame
        htf.margin_left = Inches(0.18)
        htf.vertical_anchor = MSO_ANCHOR.MIDDLE
        hp1 = htf.paragraphs[0]
        hp1.text = title
        hp1.font.name = "Calibri"
        hp1.font.size = Pt(18.5)
        hp1.font.bold = True
        hp1.font.color.rgb = WHITE
        hp2 = htf.add_paragraph()
        hp2.text = sub
        hp2.font.name = "Calibri"
        hp2.font.size = Pt(12)
        hp2.font.color.rgb = RGBColor(241, 245, 249)

        bt = s2.shapes.add_textbox(Inches(x + 0.16), Inches(2.55), Inches(3.56), Inches(3.2))
        btf = bt.text_frame
        btf.word_wrap = True
        for bi, line_txt in enumerate(pts):
            bp = btf.paragraphs[0] if bi == 0 else btf.add_paragraph()
            bp.text = f"▸  {line_txt}"
            bp.font.name = "Calibri"
            bp.font.size = Pt(14.5)
            bp.font.color.rgb = SLATE_DARK
            bp.space_after = Pt(12)

        pill = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.16), Inches(5.95), Inches(3.56), Inches(0.65))
        pill.adjustments[0] = 0.18
        pill.fill.solid()
        pill.fill.fore_color.rgb = WHITE
        pill.line.color.rgb = acc_col
        pill.line.width = Pt(1.0)
        ftf = pill.text_frame
        ftf.word_wrap = True
        ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
        fp0 = ftf.paragraphs[0]
        fp0.text = ex_txt
        fp0.alignment = PP_ALIGN.CENTER
        fp0.font.name = "Calibri"
        fp0.font.size = Pt(12)
        fp0.font.bold = True
        fp0.font.color.rgb = NAVY

    _set_notes(
        s2,
        "BẠN CẦN NHỚ GÌ Ở SLIDE 2?\n"
        "- Chỉ cần nhớ 3 cấp độ thông minh tăng dần: (1) Phim Hot ai cũng giống ai -> (2) SVD/ALS tìm sở thích ẩn -> (3) NeuMF dùng mạng nơ-ron học sâu.\n"
        "- Từ khóa ăn điểm với thầy: Pre-trained NeuMF (luyện GMF và MLP trước rồi ghép lại) + Negative Sampling (cho máy học cả phim người dùng không xem).",
    )

    # ---------- SLIDE 3: AI THẮNG & NGHỊCH LÝ DỄ HIỂU ----------
    s3 = prs.slides.add_slide(blank)
    _set_bg(s3, WHITE)
    _add_top_bar(s3, "02 · Kết Quả So Đấu", "AI Học Sâu (NeuMF) Gợi Ý Chuẩn Hơn +77.5% & Phong Phú Gấp 11 Lần!", 3)

    s3.shapes.add_picture(str(charts["nontech"]), Inches(0.65), Inches(1.40), width=Inches(12.0))

    # Bottom Explanation Callout (The Restaurant Paradox!)
    box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.55), Inches(12.0), Inches(1.38))
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = AMBER_SOFT
    box.line.color.rgb = AMBER
    box.line.width = Pt(1.6)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    p0 = tf.paragraphs[0]
    p0.text = "💡  Nghịch lý thú vị (rất dễ nhớ nếu thầy hỏi!): Tại sao SVD đoán sao chuẩn nhất (sai số chỉ 0.89★) nhưng gợi ý Top-10 lại kém?"
    p0.font.name = "Calibri"
    p0.font.size = Pt(15)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(146, 64, 14)
    p1 = tf.add_paragraph()
    p1.text = (
        "Hãy tưởng tượng 1 bộ phim siêu kén người xem, chỉ 5 fan cứng xem và cả 5 đều chấm 5 sao. "
        "SVD chỉ nhìn phim đã chấm nên tưởng đó là siêu phẩm và khuyên cho mọi người! "
        "Còn ALS và NeuMF nhìn cả hàng nghìn người KHÔNG thèm xem → tránh bẫy này và gợi ý đúng gu hơn hẳn."
    )
    p1.font.name = "Calibri"
    p1.font.size = Pt(13.5)
    p1.font.color.rgb = SLATE_DARK
    p1.space_before = Pt(4)

    _set_notes(
        s3,
        "BẠN CẦN NHỚ GÌ Ở SLIDE 3?\n"
        "- Đoán điểm sao (RMSE): SVD thắng (0.8935, thấp nhất là tốt nhất).\n"
        "- Chọn Top 10 phim mới (NDCG@10): AI NeuMF thắng (178% so với mốc Phim Hot = 100%).\n"
        "- Kho phim gợi ý: Phim Hot chỉ quanh quẩn 116 phim, còn AI NeuMF và ALS mở rộng ra hơn 1.300–1.447 phim!",
    )

    # ---------- SLIDE 4: CÂU CHUYỆN THẬT — USER #5333 ----------
    s4 = prs.slides.add_slide(blank)
    _set_bg(s4, WHITE)
    _add_top_bar(s4, "03 · Minh Họa Người Thật Việc Thật", "Thử Thách Khó Nhất: Khán Giả Gu Lạ (User #5333) & Khách Mới", 4)

    # Left: User #5333 Real Case Story Card
    story = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.48), Inches(6.25), Inches(5.38))
    story.adjustments[0] = 0.05
    story.fill.solid()
    story.fill.fore_color.rgb = SLATE_LIGHT
    story.line.color.rgb = BLUE
    story.line.width = Pt(1.6)
    stf = story.text_frame
    stf.word_wrap = True
    stf.vertical_anchor = MSO_ANCHOR.TOP
    stf.margin_left = Inches(0.26)
    stf.margin_right = Inches(0.22)
    stf.margin_top = Inches(0.24)

    sp0 = stf.paragraphs[0]
    sp0.text = "👤  Hồ Sơ Thật: Khán Giả #5333 (Gu Rất Kén Phim)"
    sp0.font.name = "Calibri"
    sp0.font.size = Pt(18)
    sp0.font.bold = True
    sp0.font.color.rgb = BLUE
    sp0.space_after = Pt(10)

    story_lines = [
        ("Lịch sử đã xem:", "Chỉ thích 8 phim tâm lý ít người biết (The Funeral, Firelight, Erin Brockovich...). Đề thi giấu 1 phim thật họ sẽ xem tiếp theo: Return to Me (2000)."),
        ("❌ Ông Phim Hot khuyên:", "Shawshank, Bố Già, Schindler's List → Trượt 0/5!"),
        ("❌ Ông SVD khuyên:", "Các phim cổ/lạ điểm ảo (Der bewegte Mann, An Actor's Revenge) → Trượt 0/5!"),
        ("✅ AI NeuMF khuyên:", "Đoán trúng phóc 'Return to Me (2000)' ngay VỊ TRÍ SỐ 1!"),
    ]
    for h_txt, d_txt in story_lines:
        ph = stf.add_paragraph()
        ph.text = h_txt
        ph.font.name = "Calibri"
        ph.font.size = Pt(14.5)
        ph.font.bold = True
        ph.font.color.rgb = RED if "NeuMF" in h_txt else (NAVY if "Lịch sử" in h_txt else SLATE_MID)
        ph.space_before = Pt(6)

        pd_p = stf.add_paragraph()
        pd_p.text = d_txt
        pd_p.font.name = "Calibri"
        pd_p.font.size = Pt(13.5)
        pd_p.font.color.rgb = SLATE_DARK
        pd_p.space_after = Pt(6)

    # Right: 3 Simple Cohort Takeaways
    _add_accent_card(
        s4,
        7.12,
        1.48,
        5.66,
        1.65,
        "Khách Mới / Ít Xem (Cold-Start)",
        [
            "Ít dữ liệu nên mạng nơ-ron đơn lẻ (MLP) dễ đoán sai",
            "ALS ổn định nhất — vượt mốc Phim Hot tới 2.4 lần!",
        ],
        accent=TEAL,
        bg=TEAL_SOFT,
        title_size=16.5,
        body_size=14,
    )
    _add_accent_card(
        s4,
        7.12,
        3.33,
        5.66,
        1.65,
        "Khách Gu Lạ (Gray-Sheep)",
        [
            "Phim bom tấn hoàn toàn bó tay với nhóm khán giả này",
            "AI NeuMF vượt mốc Phim Hot tới +72%!",
        ],
        accent=RED,
        bg=RED_SOFT,
        title_size=16.5,
        body_size=14,
    )
    _add_accent_card(
        s4,
        7.12,
        5.18,
        5.66,
        1.68,
        "Tránh Bẫy 'Nhàm Chán' (Filter Bubble)",
        [
            "Phim Hot chỉ xoay vòng 3% kho phim (116/3.883 phim)",
            "ALS & NeuMF khám phá tới 34–37% kho phim cho khán giả",
        ],
        accent=BLUE,
        bg=BLUE_SOFT,
        title_size=16.5,
        body_size=14,
    )

    _set_notes(
        s4,
        "BẠN CẦN NHỚ GÌ Ở SLIDE 4?\n"
        "- Khi thầy hỏi 'Error analysis của nhóm em phát hiện gì?', hãy kể đúng 3 nhóm này:\n"
        "  1. Người ít xem (Cold-Start proxy): ALS thắng vì ít bị overfit.\n"
        "  2. Người gu lạ (Gray-Sheep): NeuMF thắng áp đảo (+72%) — mở Demo bấm User #5333 làm bằng chứng sống!\n"
        "  3. Thiên kiến phim hot (Popularity bias): Phim Hot chỉ lặp đi lặp lại 116 phim, còn AI mở rộng gấp 12 lần.",
    )

    # ---------- SLIDE 5: KỊCH BẢN DEMO 60 GIÂY & 3 CÂU HỎI TỦ ----------
    s5 = prs.slides.add_slide(blank)
    _set_bg(s5, WHITE)
    _add_top_bar(s5, "04 · Sẵn Sàng Bảo Vệ Trước Thầy", "Kịch Bản Demo 60 Giây & 3 Câu Trả Lời 'Tủ' Khi Vấn Đáp", 5)

    # Left: 60-second Demo Script
    _add_accent_card(
        s5,
        0.55,
        1.48,
        5.85,
        5.38,
        "🎬  4 Bước Bấm Demo Chắc Thắng",
        [
            "Bước 1: Mở terminal gõ  streamlit run demo/app.py",
            "Bước 2: Ở cột trái, chọn mẫu 'Niche / Gray-Sheep User (User #5333)' → chỉ cho thầy thấy ô ✅ ngay vị trí #1 của NeuMF!",
            "Bước 3: Chuyển sang mẫu 'Light / Cold-Start User' hoặc 'Heavy Cinephile' để chứng minh hệ thống phản hồi tức thì (<20 ms)",
            "Bước 4: Bấm sang Tab 2 ('Benchmark Metrics & Error Analysis') để khoe bảng điểm & 4 biểu đồ tổng hợp ngay trên web",
        ],
        accent=BLUE,
        bg=BLUE_SOFT,
        title_size=18,
        body_size=14.5,
    )

    # Right: Top 3 Professor Q&A Cheat Sheet
    _add_accent_card(
        s5,
        6.72,
        1.48,
        6.06,
        1.65,
        "Hỏi 1: Tại sao NCF không có điểm RMSE?",
        [
            "Đáp: NCF chuẩn (He et al., 2017) dự đoán xác suất thích (0–1), không dự đoán sao. Nhưng tụi em có làm thêm bản NeuMF-Explicit (RMSE 0.9017) để so trực tiếp với SVD!",
        ],
        accent=RED,
        bg=RED_SOFT,
        title_size=15.5,
        body_size=13,
    )
    _add_accent_card(
        s5,
        6.72,
        3.33,
        6.06,
        1.65,
        "Hỏi 2: Tại sao NDCG@10 chỉ ~0.08 (không phải 0.6)?",
        [
            "Đáp: Vì tụi em đánh giá Full Ranking trên cả 3.883 phim (chuẩn thực tế, không gian lận bằng cách chỉ lấy 99 phim mẫu như bài báo cũ).",
        ],
        accent=TEAL,
        bg=TEAL_SOFT,
        title_size=15.5,
        body_size=13,
    )
    _add_accent_card(
        s5,
        6.72,
        5.18,
        6.06,
        1.68,
        "Hỏi 3: Làm sao tránh lộ đáp án (Data Leakage)?",
        [
            "Đáp: Chia theo mốc thời gian của từng user (80% cũ → Train, 10% → Val, 10% mới nhất → Test) và có unit test pytest kiểm chứng 0% rò rỉ.",
        ],
        accent=PURPLE,
        bg=PURPLE_SOFT,
        title_size=15.5,
        body_size=13,
    )

    _set_notes(
        s5,
        "LỜI KHUYÊN KHI TRÌNH BÀY:\n"
        "Bạn chỉ cần thuộc 3 câu trả lời ở Slide 5 này + bấm đúng User #5333 trên Streamlit demo là đảm bảo thuyết phục thầy cực mạnh!\n"
        "File nộp cho thầy là bản tiếng Anh `docs/presentation_5_slides.pptx`, còn bản này để bạn đọc hiểu và ôn tập trước giờ lên lớp.",
    )

    out_path = out_dir / "presentation_nontech_5_slides.pptx"
    prs.save(str(out_path))
    return out_path


# =====================================================================
# 3B. DECK 3: Non-Tech Friendly 5-Slide Deck (Plain English Storytelling)
# =====================================================================
def build_nontech_en_deck(charts: dict[str, Path], out_dir: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------- SLIDE 1: EVERYDAY NETFLIX HOOK ----------
    s1 = prs.slides.add_slide(blank)
    _set_bg(s1, NAVY)

    acc = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.10), Inches(0.12), Inches(3.25))
    acc.fill.solid()
    acc.fill.fore_color.rgb = TEAL
    acc.line.fill.background()

    t0 = s1.shapes.add_textbox(Inches(1.0), Inches(1.05), Inches(11.2), Inches(0.4))
    p = t0.text_frame.paragraphs[0]
    p.text = "PLAIN-ENGLISH AI EXPLAINER  ·  NO TECH JARGON REQUIRED"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(94, 234, 212)

    t1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(11.3), Inches(1.8))
    tf = t1.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "How Does Netflix Know What Movie You'll Watch Next?"
    p1.font.name = "Calibri"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "Comparing 3 smart ways computers recommend movies using 1 million real viewer ratings (MovieLens-1M)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(19.5)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(12)

    pills = [
        ("1 Million", "Real 1–5★ Movie Ratings", "From 6,040 viewers across 3,883 movies", BLUE),
        ("Fair Test", "No Peeking Ahead", "Learn 80% past watches → Predict 10% future", TEAL),
        ("The Goal", "Pick 10 New Movies", "Unseen films each viewer will truly enjoy", AMBER),
    ]
    for idx, (num, lbl, sub, col) in enumerate(pills):
        x = 1.0 + idx * 3.88
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.85), Inches(3.55), Inches(1.65))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = SLATE_DARK
        box.line.color.rgb = col
        box.line.width = Pt(1.8)
        ptf = box.text_frame
        ptf.word_wrap = True
        ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
        q0 = ptf.paragraphs[0]
        q0.text = num
        q0.alignment = PP_ALIGN.CENTER
        q0.font.name = "Calibri"
        q0.font.size = Pt(27)
        q0.font.bold = True
        q0.font.color.rgb = WHITE
        q1 = ptf.add_paragraph()
        q1.text = lbl
        q1.alignment = PP_ALIGN.CENTER
        q1.font.name = "Calibri"
        q1.font.size = Pt(14)
        q1.font.bold = True
        q1.font.color.rgb = RGBColor(147, 197, 253)
        q2 = ptf.add_paragraph()
        q2.text = sub
        q2.alignment = PP_ALIGN.CENTER
        q2.font.name = "Calibri"
        q2.font.size = Pt(11.5)
        q2.font.color.rgb = RGBColor(203, 213, 225)

    _set_notes(
        s1,
        "PLAIN-ENGLISH SPEAKER SCRIPT (~40s):\n"
        "Every time you open Netflix, it suggests 10 movies you haven't seen yet. How does a computer figure that out?\n"
        "In this project, we tested 3 approaches on 1 million real ratings from 6,040 movie fans across nearly 3,900 films.\n"
        "To make the test completely fair, we sorted each person's history by time—teaching the computer on their oldest 80% of movies "
        "and hiding their newest 10% as a surprise future exam.",
    )

    # ---------- SLIDE 2: 3 MOVIE ADVISORS ----------
    s2 = prs.slides.add_slide(blank)
    _set_bg(s2, WHITE)
    _add_top_bar(s2, "01 · How the System Thinks", "Three Movie Advisors: From Blockbusters to Deep AI", 2)

    advisors = [
        (
            "🏆 Box-Office Advisor",
            "Popularity Baseline",
            SLATE_MID,
            SLATE_LIGHT,
            [
                "Works like a 'Top 10 Most Popular' chart",
                "Gives every viewer the exact same movie list",
                "Ignores individual genres or personal habits",
                "Serves as our minimum sanity benchmark",
            ],
            "Example: Always recommends Shawshank, Godfather...",
        ),
        (
            "🎯 Taste-Pattern Advisor",
            "Matrix Factorization (SVD & ALS)",
            TEAL,
            TEAL_SOFT,
            [
                "Discovers hidden taste groups from watch history",
                "SVD excels at predicting exact 1–5★ star scores",
                "ALS also notices skipped movies → stronger Top-10",
                "Fast, lightweight, and widely used in industry",
            ],
            "Example: Loved Toy Story? Suggests animated hits",
        ),
        (
            "🧠 Deep AI Advisor",
            "Neural CF (GMF + MLP → NeuMF)",
            RED,
            RED_SOFT,
            [
                "Uses deep neural networks for subtle taste patterns",
                "Merges 2 AI pathways: simple matching + deep layers",
                "Trains each brain first, then fuses them together",
                "Learns by comparing liked films vs. 4 unseen ones",
            ],
            "Example: Spots hidden gems for very specific tastes",
        ),
    ]
    for idx, (title, sub, acc_col, bg_col, pts, ex_txt) in enumerate(advisors):
        x = 0.55 + idx * 4.18
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(5.30))
        card.adjustments[0] = 0.05
        card.fill.solid()
        card.fill.fore_color.rgb = bg_col
        card.line.color.rgb = acc_col
        card.line.width = Pt(1.5)

        hdr = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.48), Inches(3.88), Inches(0.92))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = acc_col
        hdr.line.fill.background()
        htf = hdr.text_frame
        htf.margin_left = Inches(0.18)
        htf.vertical_anchor = MSO_ANCHOR.MIDDLE
        hp1 = htf.paragraphs[0]
        hp1.text = title
        hp1.font.name = "Calibri"
        hp1.font.size = Pt(18)
        hp1.font.bold = True
        hp1.font.color.rgb = WHITE
        hp2 = htf.add_paragraph()
        hp2.text = sub
        hp2.font.name = "Calibri"
        hp2.font.size = Pt(12)
        hp2.font.color.rgb = RGBColor(241, 245, 249)

        bt = s2.shapes.add_textbox(Inches(x + 0.16), Inches(2.55), Inches(3.56), Inches(3.2))
        btf = bt.text_frame
        btf.word_wrap = True
        for bi, line_txt in enumerate(pts):
            bp = btf.paragraphs[0] if bi == 0 else btf.add_paragraph()
            bp.text = f"▸  {line_txt}"
            bp.font.name = "Calibri"
            bp.font.size = Pt(14.5)
            bp.font.color.rgb = SLATE_DARK
            bp.space_after = Pt(12)

        pill = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.16), Inches(5.95), Inches(3.56), Inches(0.65))
        pill.adjustments[0] = 0.18
        pill.fill.solid()
        pill.fill.fore_color.rgb = WHITE
        pill.line.color.rgb = acc_col
        pill.line.width = Pt(1.0)
        ftf = pill.text_frame
        ftf.word_wrap = True
        ftf.vertical_anchor = MSO_ANCHOR.MIDDLE
        fp0 = ftf.paragraphs[0]
        fp0.text = ex_txt
        fp0.alignment = PP_ALIGN.CENTER
        fp0.font.name = "Calibri"
        fp0.font.size = Pt(11.8)
        fp0.font.bold = True
        fp0.font.color.rgb = NAVY

    _set_notes(
        s2,
        "PLAIN-ENGLISH SPEAKER SCRIPT (~50s):\n"
        "Think of our 3 methods as 3 movie advisors competing head-to-head.\n"
        "First, the Box-Office Advisor simply recommends the most popular classics to everyone—no personalization at all.\n"
        "Second, our Taste-Pattern Advisor (SVD and ALS) learns hidden taste dimensions from what people rate—SVD focuses on guessing exact stars, "
        "while ALS also learns from what you skipped.\n"
        "Third, our Deep AI Advisor (Pre-trained NeuMF) blends two neural pathways trained separately and fused together so it can capture subtle, non-linear preferences.",
    )

    # ---------- SLIDE 3: WHO WON & THE PARADOX ----------
    s3 = prs.slides.add_slide(blank)
    _set_bg(s3, WHITE)
    _add_top_bar(s3, "02 · Head-to-Head Results", "Deep AI (NeuMF) Ranks +77.5% Better & Unlocks 11× More Movies!", 3)

    s3.shapes.add_picture(str(charts["nontech_en"]), Inches(0.65), Inches(1.40), width=Inches(12.0))

    box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.55), Inches(12.0), Inches(1.38))
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = AMBER_SOFT
    box.line.color.rgb = AMBER
    box.line.width = Pt(1.6)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    p0 = tf.paragraphs[0]
    p0.text = "💡  Surprising Paradox: Why did the best Star-Rating Predictor (SVD, 0.89★ avg error) lose at Top-10 picks?"
    p0.font.name = "Calibri"
    p0.font.size = Pt(15)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(146, 64, 14)
    p1 = tf.add_paragraph()
    p1.text = (
        "Imagine an obscure film watched by only 5 hardcore fans who all rated it 5 stars. "
        "Because SVD only looks at rated films, it thinks that movie is a masterpiece for everyone! "
        "ALS and Deep AI (NeuMF) also notice the thousands of viewers who skipped it — avoiding the trap and picking far better movies."
    )
    p1.font.name = "Calibri"
    p1.font.size = Pt(13.5)
    p1.font.color.rgb = SLATE_DARK
    p1.space_before = Pt(4)

    _set_notes(
        s3,
        "PLAIN-ENGLISH SPEAKER SCRIPT (~60s):\n"
        "Here is what happened on the future exam. Setting 'Most Popular' as our 100% baseline, Deep AI (Pre-trained NeuMF) jumped to 177.5%—a +77.5% boost in Top-10 ranking quality—while expanding movie variety from just 116 movies to over 1,300!\n"
        "We also uncovered a fascinating insight: Biased SVD was the single best at predicting exact star ratings (0.8935 RMSE), yet trailed on Top-10 lists. "
        "The reason is simple: without seeing what viewers choose NOT to watch, an explicit model gets fooled by obscure movies with a few 5-star reviews.",
    )

    # ---------- SLIDE 4: REAL VIEWER STORY (USER #5333) ----------
    s4 = prs.slides.add_slide(blank)
    _set_bg(s4, WHITE)
    _add_top_bar(s4, "03 · Real-World Proof", "The Hardest Test: Niche Viewers (User #5333) & New Users", 4)

    story = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.48), Inches(6.25), Inches(5.38))
    story.adjustments[0] = 0.05
    story.fill.solid()
    story.fill.fore_color.rgb = SLATE_LIGHT
    story.line.color.rgb = BLUE
    story.line.width = Pt(1.6)
    stf = story.text_frame
    stf.word_wrap = True
    stf.vertical_anchor = MSO_ANCHOR.TOP
    stf.margin_left = Inches(0.26)
    stf.margin_right = Inches(0.22)
    stf.margin_top = Inches(0.24)

    sp0 = stf.paragraphs[0]
    sp0.text = "👤  Real Case Study: Viewer #5333 (Very Niche Taste)"
    sp0.font.name = "Calibri"
    sp0.font.size = Pt(17.5)
    sp0.font.bold = True
    sp0.font.color.rgb = BLUE
    sp0.space_after = Pt(10)

    story_lines = [
        ("Watch History:", "Only liked 8 lesser-known dramas (The Funeral, Firelight, Erin Brockovich...). Our hidden future test film: Return to Me (2000)."),
        ("❌ Box-Office Advisor:", "Suggested Shawshank, Godfather, Schindler's List → 0/5 hits!"),
        ("❌ Star-Rating SVD:", "Suggested obscure inflated-score films (Der bewegte Mann, An Actor's Revenge) → 0/5 hits!"),
        ("✅ Deep AI (NeuMF):", "Predicted 'Return to Me (2000)' right at RANK #1!"),
    ]
    for h_txt, d_txt in story_lines:
        ph = stf.add_paragraph()
        ph.text = h_txt
        ph.font.name = "Calibri"
        ph.font.size = Pt(14.5)
        ph.font.bold = True
        ph.font.color.rgb = RED if "NeuMF" in h_txt else (NAVY if "Watch" in h_txt else SLATE_MID)
        ph.space_before = Pt(6)

        pd_p = stf.add_paragraph()
        pd_p.text = d_txt
        pd_p.font.name = "Calibri"
        pd_p.font.size = Pt(13.5)
        pd_p.font.color.rgb = SLATE_DARK
        pd_p.space_after = Pt(6)

    _add_accent_card(
        s4,
        7.12,
        1.48,
        5.66,
        1.65,
        "Light / New Viewers (Cold-Start)",
        [
            "With little history, deep layers alone can overfit",
            "ALS stays steadiest — 2.4× better than Popularity!",
        ],
        accent=TEAL,
        bg=TEAL_SOFT,
        title_size=16.5,
        body_size=14,
    )
    _add_accent_card(
        s4,
        7.12,
        3.33,
        5.66,
        1.65,
        "Niche Taste Viewers (Gray-Sheep)",
        [
            "Blockbusters completely miss specialized tastes",
            "Deep AI (NeuMF) jumps +72% ahead of Popularity!",
        ],
        accent=RED,
        bg=RED_SOFT,
        title_size=16.5,
        body_size=14,
    )
    _add_accent_card(
        s4,
        7.12,
        5.18,
        5.66,
        1.68,
        "Breaking the 'Filter Bubble'",
        [
            "Popularity recycles only 3% of the library (116 movies)",
            "ALS & NeuMF help people discover 34–37% of the catalog",
        ],
        accent=BLUE,
        bg=BLUE_SOFT,
        title_size=16.5,
        body_size=14,
    )

    _set_notes(
        s4,
        "PLAIN-ENGLISH SPEAKER SCRIPT (~50s):\n"
        "To make these numbers concrete, meet Viewer #5333—a real person in MovieLens who prefers quiet, lesser-known dramas.\n"
        "We hid the one movie they rated highly in the future: 'Return to Me (2000)'. "
        "Both Popularity and Biased SVD missed completely, scoring 0 out of 5—while our Deep AI NeuMF nailed 'Return to Me' at Rank #1!\n"
        "Overall, ALS shines for light users with sparse histories, while Pre-trained NeuMF dominates niche viewers and breaks the popularity filter bubble.",
    )

    # ---------- SLIDE 5: LIVE DEMO & 3 SIMPLE LESSONS ----------
    s5 = prs.slides.add_slide(blank)
    _set_bg(s5, WHITE)
    _add_top_bar(s5, "04 · Live Demo & Key Takeaways", "Try It Live in 60 Seconds & Three Simple Lessons", 5)

    _add_accent_card(
        s5,
        0.55,
        1.48,
        5.85,
        5.38,
        "🎬  Live Web App Tour (streamlit run demo/app.py)",
        [
            "Step 1: Choose any viewer ID (0–6,039) or pick 'Niche Viewer #5333' in one click",
            "Step 2: Compare all 4 advisors side-by-side in real time (under 20 milliseconds)",
            "Step 3: Check the green ✅ icons showing verified matches against the viewer's real future",
            "Step 4: Switch to Tab 2 ('Benchmark Metrics') to explore accuracy & diversity charts interactively",
        ],
        accent=BLUE,
        bg=BLUE_SOFT,
        title_size=17,
        body_size=14.5,
    )

    _add_accent_card(
        s5,
        6.72,
        1.48,
        6.06,
        1.65,
        "1. Pick the Right Tool for the Goal",
        [
            "Use Biased SVD to predict star ratings (0.89★ avg error), but use Implicit ALS or Deep AI to build Top-10 lists.",
        ],
        accent=RED,
        bg=RED_SOFT,
        title_size=16,
        body_size=13.5,
    )
    _add_accent_card(
        s5,
        6.72,
        3.33,
        6.06,
        1.65,
        "2. Teamwork Inside AI Works Better",
        [
            "Training simple matching (GMF) and deep layers (MLP) first, then merging them (Pre-trained NeuMF), beats every single baseline.",
        ],
        accent=TEAL,
        bg=TEAL_SOFT,
        title_size=16,
        body_size=13.5,
    )
    _add_accent_card(
        s5,
        6.72,
        5.18,
        6.06,
        1.68,
        "3. Variety Matters as Much as Accuracy",
        [
            "Smart AI doesn't just repeat blockbusters — it surfaces 11–12× more hidden gems across the library.",
        ],
        accent=PURPLE,
        bg=PURPLE_SOFT,
        title_size=16,
        body_size=13.5,
    )

    _set_notes(
        s5,
        "PLAIN-ENGLISH SPEAKER SCRIPT (~40s -> LAUNCH LIVE DEMO):\n"
        "To wrap up, three simple lessons: match the training goal to the task, combine pre-trained pathways for the best AI accuracy, "
        "and always measure catalog variety so users don't get stuck watching the same 100 blockbusters.\n"
        "Now let's switch over to our live Streamlit web app and watch all four advisors recommend movies in real time—thank you!",
    )

    out_path = out_dir / "presentation_nontech_en_5_slides.pptx"
    prs.save(str(out_path))
    return out_path


# =====================================================================
# 4. Main Entrypoint
# =====================================================================
def build_presentation(results_dir: Path = Path("results"), out_dir: Path = Path("docs")) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eda = json.loads((results_dir / "eda_summary.json").read_text(encoding="utf-8"))
    metrics_df = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    err_data = json.loads((results_dir / "error_analysis.json").read_text(encoding="utf-8"))
    bias_df = pd.DataFrame(err_data["popularity_bias"])

    charts = generate_slide_charts(metrics_df, bias_df, results_dir / "figures")
    p_acad = build_academic_deck(eda, metrics_df, bias_df, charts, out_dir)
    p_nontech = build_nontech_deck(charts, out_dir)
    p_nontech_en = build_nontech_en_deck(charts, out_dir)

    md_summary = """# 5-Slide Presentation Decks

Three complementary 5-slide decks have been generated in `docs/`:

1. **Academic Defense Deck (English — Submit & Present to Instructor):** `docs/presentation_5_slides.pptx` (+ `.pdf`)
   - Clean, visual keynote design (14–40pt typography, KPI hero tiles, 1 dedicated chart per slide, built-in Speaker Notes).
2. **Non-Technical Explainer Deck (Plain English — General Audience / Easy Storytelling):** `docs/presentation_nontech_en_5_slides.pptx` (+ `.pdf`)
   - Everyday Netflix analogies, zero heavy equations, real User `#5333` story, live demo steps, and plain-English Speaker Notes.
3. **Non-Technical Study & Defense Cheat Sheet (Vietnamese + English Terms):** `docs/presentation_nontech_5_slides.pptx` (+ `.pdf`)
   - Quick 2-minute mastery guide with 4-step live demo walkthrough and Top 3 Instructor Q&A answers.
"""
    (out_dir / "presentation_5_slides.md").write_text(md_summary, encoding="utf-8")
    print(f"Saved Academic Deck        -> {p_acad}")
    print(f"Saved Non-Tech Deck (VI)   -> {p_nontech}")
    print(f"Saved Non-Tech Deck (EN)   -> {p_nontech_en}")
    return p_acad, p_nontech, p_nontech_en


if __name__ == "__main__":
    build_presentation()
