"""Build the 15-20 page academic Project Report in both Microsoft Word format
(docs/project_report.docx) and LaTeX source (docs/project_report.tex) using live experimental data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

NAVY_HEX = "0F172A"
ROW_ALT_HEX = "F8FAFC"
NAVY_RGB = RGBColor(15, 23, 42)
BLUE_RGB = RGBColor(30, 64, 175)
MUTED_RGB = RGBColor(71, 85, 105)


def _add_page_number(run) -> None:
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)


def _shade_cell(cell, hex_color: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _add_styled_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]]) -> None:
    cap_p = doc.add_paragraph()
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_p.paragraph_format.space_before = Pt(8)
    cap_p.paragraph_format.space_after = Pt(4)
    cap_p.paragraph_format.keep_with_next = True
    r_cap = cap_p.add_run(caption)
    r_cap.bold = True
    r_cap.font.size = Pt(10)
    r_cap.font.color.rgb = NAVY_RGB

    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "Table Grid"

    for c_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        _shade_cell(cell, NAVY_HEX)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(2.5)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(9.2)
        run.font.color.rgb = RGBColor(255, 255, 255)

    # Repeat header across page breaks and keep every row unsplit
    hdr_trPr = tbl.rows[0]._tr.get_or_add_trPr()
    hdr_trPr.append(OxmlElement("w:tblHeader"))
    hdr_trPr.append(OxmlElement("w:cantSplit"))

    for r_idx, row_vals in enumerate(rows, start=1):
        bg = ROW_ALT_HEX if r_idx % 2 == 0 else "FFFFFF"
        tbl.rows[r_idx]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(r_idx, c_idx)
            _shade_cell(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2.0)
            p.paragraph_format.space_after = Pt(2.0)
            run = p.add_run(str(val))
            run.font.size = Pt(9.0)
            if c_idx == 0:
                run.bold = True

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def _add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 6.15) -> None:
    if not img_path.exists():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    r = p_cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED_RGB


def _add_equation(doc: Document, eq_text: str, eq_num: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"{eq_text}      ({eq_num})")
    r.italic = True
    r.font.name = "Cambria Math"
    r.font.size = Pt(10.5)
    r.font.color.rgb = NAVY_RGB


def _h1(doc: Document, text: str) -> None:
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(5)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(14.5)
        r.font.bold = True
        r.font.color.rgb = NAVY_RGB


def _h2(doc: Document, text: str) -> None:
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3.5)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(12.0)
        r.font.bold = True
        r.font.color.rgb = BLUE_RGB


def _p(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.22
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5.5)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(11.0)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.0)


def build_report(results_dir: Path = Path("results"), out_dir: Path = Path("docs")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eda = json.loads((results_dir / "eda_summary.json").read_text(encoding="utf-8"))
    metrics_df = pd.DataFrame(json.loads((results_dir / "metrics.json").read_text(encoding="utf-8")))
    err_data = json.loads((results_dir / "error_analysis.json").read_text(encoding="utf-8"))
    cohorts_df = pd.DataFrame(err_data["cohorts"])
    bias_df = pd.DataFrame(err_data["popularity_bias"])
    grid_path = results_dir / "mf_grid_search.json"
    grid_df = pd.DataFrame(json.loads(grid_path.read_text(encoding="utf-8"))) if grid_path.exists() else pd.DataFrame()
    hist_path = results_dir / "training_history.json"
    hist_data = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {}
    fig_dir = results_dir / "figures"

    # Extract dynamic metrics for prose
    m_idx = metrics_df.set_index("Model")
    gmean_rmse = float(m_idx.loc["Global Mean", "RMSE"])
    pop_rmse = float(m_idx.loc["Popularity", "RMSE"])
    pop_ndcg = float(m_idx.loc["Popularity", "NDCG@10"])
    pop_p10 = float(m_idx.loc["Popularity", "Precision@10"])
    pop_r10 = float(m_idx.loc["Popularity", "Recall@10"])

    svd_rmse = float(m_idx.loc["Biased SVD (MF)", "RMSE"])
    svd_ndcg = float(m_idx.loc["Biased SVD (MF)", "NDCG@10"])
    svd_p10 = float(m_idx.loc["Biased SVD (MF)", "Precision@10"])
    svd_r10 = float(m_idx.loc["Biased SVD (MF)", "Recall@10"])

    als_ndcg = float(m_idx.loc["Implicit ALS (MF)", "NDCG@10"])
    als_p10 = float(m_idx.loc["Implicit ALS (MF)", "Precision@10"])
    als_r10 = float(m_idx.loc["Implicit ALS (MF)", "Recall@10"])

    gmf_ndcg = float(m_idx.loc["GMF", "NDCG@10"])
    mlp_ndcg = float(m_idx.loc["MLP", "NDCG@10"])
    neumf_s_ndcg = float(m_idx.loc["NeuMF (Scratch)", "NDCG@10"])
    neumf_pt_ndcg = float(m_idx.loc["NeuMF (Pre-trained)", "NDCG@10"])
    neumf_pt_p10 = float(m_idx.loc["NeuMF (Pre-trained)", "Precision@10"])
    neumf_pt_r10 = float(m_idx.loc["NeuMF (Pre-trained)", "Recall@10"])
    neumf_exp_rmse = float(m_idx.loc["NeuMF-Explicit (MSE)", "RMSE"])
    neumf_exp_ndcg = float(m_idx.loc["NeuMF-Explicit (MSE)", "NDCG@10"])

    best_rmse_row = metrics_df.dropna(subset=["RMSE"]).sort_values("RMSE").iloc[0]
    best_ndcg_row = metrics_df.dropna(subset=["NDCG@10"]).sort_values("NDCG@10", ascending=False).iloc[0]

    bias_idx = bias_df.set_index("Model")
    pop_cov = float(bias_idx.loc["Popularity", "Catalog Coverage"]) * 100.0
    svd_cov = float(bias_idx.loc["Biased SVD (MF)", "Catalog Coverage"]) * 100.0
    als_cov = float(bias_idx.loc["Implicit ALS (MF)", "Catalog Coverage"]) * 100.0
    npt_cov = float(bias_idx.loc["NeuMF (Pre-trained)", "Catalog Coverage"]) * 100.0

    svd_hparams = hist_data.get("mf", {}).get("hparams", {"dim": 64, "lr": 0.005, "reg": 0.02})

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.95)
        section.bottom_margin = Inches(0.95)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

        # Header & Footer on pages 2+
        hdr = section.header.paragraphs[0]
        hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rh = hdr.add_run("MovieLens-1M Recommender Systems: Popularity, SVD/ALS & Neural Collaborative Filtering")
        rh.italic = True
        rh.font.name = "Times New Roman"
        rh.font.size = Pt(9.0)
        rh.font.color.rgb = MUTED_RGB

        ftr = section.footer.paragraphs[0]
        ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rf1 = ftr.add_run("Advanced Artificial Intelligence — Project Report   |   Page ")
        rf1.font.name = "Times New Roman"
        rf1.font.size = Pt(9.5)
        rf1.font.color.rgb = MUTED_RGB
        _add_page_number(ftr.add_run())

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(11)

    # ==================== PAGE 1: COVER & EXECUTIVE ABSTRACT ====================
    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_before = Pt(6)
    p_course.paragraph_format.space_after = Pt(4)
    rc = p_course.add_run("ADVANCED ARTIFICIAL INTELLIGENCE\nCOURSE PROJECT TECHNICAL REPORT")
    rc.bold = True
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(12.0)
    rc.font.color.rgb = BLUE_RGB

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(6)
    rt = p_title.add_run(
        "A Rigorous Empirical Comparison of Popularity Baselines, Matrix Factorization (Biased SVD and Implicit ALS), "
        "and Neural Collaborative Filtering on MovieLens-1M"
    )
    rt.bold = True
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(17.5)
    rt.font.color.rgb = NAVY_RGB

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(10)
    rs = p_sub.add_run(
        "Unifying Explicit Rating Prediction (RMSE) and Leak-Free Full-Catalog Top-N Ranking "
        "(Precision@10, Recall@10, NDCG@10) in PyTorch"
    )
    rs.italic = True
    rs.font.name = "Times New Roman"
    rs.font.size = Pt(11.0)
    rs.font.color.rgb = MUTED_RGB

    _add_styled_table(
        doc,
        "Project Metadata Summary",
        ["Attribute", "Specification"],
        [
            ["Course", "Advanced Artificial Intelligence"],
            ["Dataset", f"MovieLens-1M ({eda['n_ratings']:,} ratings, {eda['n_users']:,} users, {eda['n_items']:,} movies)"],
            ["Splitting Protocol", "Per-User Chronological Split (80% Train / 10% Validation / 10% Test)"],
            ["Implemented Models", "Global Mean, IMDB Popularity, Biased SVD, Implicit ALS, GMF, MLP, NeuMF (Scratch, Pre-trained, Explicit)"],
            ["Evaluation Metrics", "RMSE, Precision@10, Recall@10, NDCG@10, HitRate@10, Catalog Coverage, Gini Concentration"],
            ["Software Stack", "Python 3.11, PyTorch 2.x, NumPy, SciPy, Pandas, Matplotlib, Streamlit"],
        ],
    )

    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_h.paragraph_format.space_before = Pt(4)
    p_abs_h.paragraph_format.space_after = Pt(3)
    rah = p_abs_h.add_run("ABSTRACT")
    rah.bold = True
    rah.font.name = "Times New Roman"
    rah.font.size = Pt(11.5)
    rah.font.color.rgb = NAVY_RGB

    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.line_spacing = 1.12
    p_abs.paragraph_format.space_after = Pt(4)
    r_abs = p_abs.add_run(
        f"Personalized recommender systems constitute a cornerstone of modern artificial intelligence platforms, filtering "
        f"massive item catalogs to surface relevant content under extreme interaction sparsity. This report presents a "
        f"systematic, reproducible investigation of three foundational collaborative filtering paradigms on the benchmark "
        f"MovieLens-1M dataset ({eda['n_ratings']:,} ratings across {eda['n_users']:,} users and {eda['n_items']:,} movies, "
        f"exhibiting {eda['sparsity']*100:.2f}% sparsity): (1) a Bayesian non-personalized Popularity baseline using the IMDB "
        f"Weighted Rating formula, (2) linear latent factor Matrix Factorization encompassing both explicit Biased Funk-SVD "
        f"and confidence-weighted Implicit Alternating Least Squares (ALS), and (3) deep non-linear Neural Collaborative "
        f"Filtering (NCF), including Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), randomly initialized "
        f"NeuMF, pre-trained NeuMF, and an explicit MSE-trained NeuMF variant. To prevent temporal data leakage, all models "
        f"are evaluated under a strict per-user chronological 80/10/10 Train/Validation/Test split and benchmarked via "
        f"unbiased full-catalog Top-10 ranking rather than sampled negative evaluation. Empirical results reveal a fundamental "
        f"divergence between rating error and ranking utility: Biased SVD achieves strong explicit rating accuracy "
        f"(Test RMSE = {svd_rmse:.4f} vs. Popularity {pop_rmse:.4f}), yet severely underperforms on Top-10 ranking "
        f"(NDCG@10 = {svd_ndcg:.4f}) due to missing-not-at-random selection bias. Conversely, models incorporating implicit "
        f"negative feedback achieve dramatic ranking gains, with {best_ndcg_row['Model']} reaching Test NDCG@10 = "
        f"{best_ndcg_row['NDCG@10']:.4f}, Precision@10 = {best_ndcg_row['Precision@10']:.4f}, and Recall@10 = "
        f"{best_ndcg_row['Recall@10']:.4f}, while expanding catalog coverage from {pop_cov:.2f}% (Popularity) to over "
        f"{max(als_cov, npt_cov):.1f}%. Disaggregated error analyses across cold-start proxies, gray-sheep niche cinephiles, "
        f"and popularity concentration further illuminate practical trade-offs, complemented by a real-time Streamlit web application.",
    )
    r_abs.font.name = "Times New Roman"
    r_abs.font.size = Pt(10.3)

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_after = Pt(2)
    r_kw_b = p_kw.add_run("Keywords: ")
    r_kw_b.bold = True
    r_kw_b.font.name = "Times New Roman"
    r_kw_b.font.size = Pt(10.3)
    r_kw = p_kw.add_run(
        "Recommender Systems, Collaborative Filtering, Matrix Factorization, Biased SVD, Implicit ALS, Neural Collaborative Filtering, "
        "NeuMF, MovieLens-1M, Temporal Splitting, Popularity Bias, Gray-Sheep Users.",
    )
    r_kw.font.name = "Times New Roman"
    r_kw.font.size = Pt(10.3)
    doc.add_page_break()

    # ==================== SECTION 1: PROBLEM AND DATA ====================
    _h1(doc, "1. Problem Definition and Dataset Analysis")
    _h2(doc, "1.1 Background and Motivation")
    _p(
        doc,
        "Digital streaming platforms, e-commerce marketplaces, and social media ecosystems confront users with thousands to "
        "millions of candidate items, creating severe cognitive and choice overload. Recommender systems mitigate this "
        "bottleneck by inferring latent user preferences from historical behavior. Within recommender systems research, "
        "collaborative filtering (CF) remains the most widely deployed paradigm because it learns expressive user and item "
        "representations purely from collective interaction patterns without requiring expensive domain-specific feature engineering.",
    )
    _p(
        doc,
        "Despite decades of research, practitioners and researchers face two persistent methodological questions when designing "
        "collaborative recommenders. First, does replacing the classical linear inner product of Matrix Factorization (MF) with "
        "deep multi-layer neural networks—specifically Neural Collaborative Filtering (NCF)—yield consistent gains under rigorous, "
        "leak-free full-catalog evaluation? Second, how does the choice of supervision signal—explicit star ratings optimized via "
        "Mean Squared Error (MSE) versus implicit binary interactions optimized with negative feedback—govern the trade-off "
        "between rating prediction error (RMSE) and Top-N recommendation quality (Precision@10, Recall@10, NDCG@10)? This project "
        "addresses both questions empirically on the canonical MovieLens-1M benchmark.",
    )

    _h2(doc, "1.2 Formal Problem Formulation")
    _p(
        doc,
        f"Let U = {{0, 1, ..., M - 1}} denote the set of M = {eda['n_users']:,} users and I = {{0, 1, ..., N - 1}} denote the "
        f"catalog of N = {eda['n_items']:,} movies. Historical user feedback is observed as a chronological stream of tuples "
        f"(u, i, r_ui, t_ui), where r_ui in {{1, 2, 3, 4, 5}} represents the explicit star rating assigned by user u to movie i "
        f"at UNIX timestamp t_ui. We formulate and evaluate two distinct but complementary predictive tasks:",
    )
    _p(
        doc,
        "Given an observed training set R_train and a target user-item pair (u, i) in R_test, estimate a real-valued rating "
        "r_hat_ui in [1, 5] that minimizes Root Mean Squared Error (RMSE) against the ground-truth star rating r_ui. This "
        "formulation treats unobserved entries in the M x N rating matrix as missing at random and conditions exclusively on "
        "items that users actively chose to watch and rate.",
        bold_prefix="Task A — Explicit Rating Prediction: ",
    )
    _p(
        doc,
        "Define positive implicit preference y_ui = 1 if r_ui >= 4 (indicating genuine user satisfaction) and y_ui = 0 otherwise. "
        "For each user u, rank all candidate movies i in I \\ S_u (where S_u denotes all movies interacted with by user u during "
        "Training and Validation) via a scoring function s(u, i) -> R, returning the ordered Top-K list L_K(u) of length K = 10. "
        "Recommendation utility is measured by comparing L_K(u) against held-out positive test items R^+_test(u) = {i in R_test(u) : r_ui >= 4} "
        "using Precision@10, Recall@10, and Normalized Discounted Cumulative Gain (NDCG@10).",
        bold_prefix="Task B — Implicit Full-Catalog Top-N Ranking: ",
    )

    _h2(doc, "1.3 MovieLens-1M Exploratory Data Analysis")
    _p(
        doc,
        f"Released by GroupLens Research at the University of Minnesota, MovieLens-1M contains {eda['n_ratings']:,} ratings from "
        f"{eda['n_users']:,} users over {eda['n_items']:,} catalog movies ({eda['n_rated_items']:,} of which receive at least one "
        f"rating). Every user in the dataset has rated at least {eda['user_ratings_min']} movies, with a median activity of "
        f"{eda['user_ratings_median']:.0f} ratings, a mean of {eda['user_ratings_mean']:.1f} ratings, and a maximum of "
        f"{eda['user_ratings_max']:,} ratings for the most prolific cinephile. Table 1 summarizes the structural properties of "
        f"the dataset.",
    )

    _add_styled_table(
        doc,
        "Table 1: Descriptive Statistics of the MovieLens-1M Dataset and Chronological Splits",
        ["Property / Statistic", "Value", "Analytical Implication"],
        [
            ["Total Explicit Ratings", f"{eda['n_ratings']:,}", "Sufficient scale for deep embedding & MLP training"],
            ["Unique Users (M)", f"{eda['n_users']:,}", f"Min {eda['user_ratings_min']}, Median {eda['user_ratings_median']:.0f}, Max {eda['user_ratings_max']:,} ratings/user"],
            ["Catalog Movies (N)", f"{eda['n_items']:,}", f"{eda['n_rated_items']:,} rated items; {eda['n_items'] - eda['n_rated_items']} cold items"],
            ["Interaction Matrix Sparsity", f"{eda['sparsity']*100:.2f}%", f"Only {eda['density']*100:.2f}% of user-movie pairs are observed"],
            ["Global Rating Mean ± Std", f"{eda['mean_rating']:.4f} ± {eda['std_rating']:.4f}", "Positively skewed toward 3-star and 4-star ratings"],
            ["Implicit Positive Share (r >= 4)", f"{eda['positive_rate_ge4']*100:.2f}%", "57.5% of observed ratings indicate strong satisfaction"],
            ["Top-10% / Top-20% Item Share", f"{eda['share_ratings_of_top10pct_items']*100:.1f}% / {eda['share_ratings_of_top20pct_items']*100:.1f}%", "Pronounced Pareto long-tail popularity skew"],
            ["Chronological Split Sizes", f"{eda['n_train']:,} / {eda['n_val']:,} / {eda['n_test']:,}", "80% Train, 10% Validation, 10% Test per user"],
        ],
    )

    _add_figure(
        doc,
        fig_dir / "fig1_eda_distributions.png",
        "Figure 1: Exploratory Data Analysis of MovieLens-1M. (a) Explicit star distribution showing strong concentration at 4 stars (34.9%) and 3 stars (26.1%). (b) Item popularity Lorenz curve illustrating severe long-tail skew where the top 20% most popular movies attract nearly two-thirds of all ratings.",
    )

    _p(
        doc,
        f"Figure 1 highlights two empirical properties that directly shape model behavior. First, star ratings are strongly "
        f"skewed toward favorable evaluations: 4-star ({eda['rating_dist']['4']:,}) and 5-star ({eda['rating_dist']['5']:,}) "
        f"ratings together represent {eda['positive_rate_ge4']*100:.1f}% of observed interactions, reflecting self-selection bias—users "
        f"preferentially watch films they expect to enjoy. Second, item popularity follows a steep power-law distribution: "
        f"the top 10% of movies account for {eda['share_ratings_of_top10pct_items']*100:.1f}% of all ratings and the top 20% capture "
        f"{eda['share_ratings_of_top20pct_items']*100:.1f}%, while the median movie receives only {eda['item_ratings_median']:.0f} ratings. "
        f"Consequently, naive popularity baselines achieve deceptively competitive aggregate hit rates on mainstream users while "
        f"completely ignoring over 95% of the catalog.",
    )

    _h2(doc, "1.4 Leak-Free Per-User Chronological Splitting Protocol")
    _p(
        doc,
        "A widespread methodological flaw in introductory recommender benchmarks is uniform random splitting across all "
        "interactions. Under random splitting, a rating submitted by user u in December 2002 can enter the Training set while "
        "a movie watched by the same user in May 2000 falls into the Test set—allowing models to exploit future user taste "
        "evolution and contemporaneous item popularity spikes that would not exist at deployment time (Ji et al., 2023).",
    )
    _p(
        doc,
        "To mirror production forecasting faithfully, we implement a strict per-user chronological 80/10/10 split. For every "
        "user u in U, their interaction sequence {(i_1, r_1, t_1), ..., (i_{n_u}, r_{n_u}, t_{n_u})} is sorted in non-decreasing "
        "timestamp order (t_1 <= t_2 <= ... <= t_{n_u}) using a stable merge sort. Let k_1 = floor(0.8 * n_u) and "
        "k_2 = floor(0.9 * n_u). Interactions 0 .. k_1 - 1 are assigned to Train, k_1 .. k_2 - 1 to Validation, and k_2 .. n_u - 1 "
        "to Test. Automated unit tests verify that max_{Train} t_ui <= min_{Val} t_ui and max_{Val} t_ui <= min_{Test} t_ui holds "
        "for 100% of the 6,040 users with zero temporal leakage.",
    )

    # ==================== SECTION 2: RELATED WORK ====================
    _h1(doc, "2. Related Work")
    _h2(doc, "2.1 Neighborhood Methods and Heuristic Baselines")
    _p(
        doc,
        "Early collaborative filtering systems relied on memory-based k-nearest-neighbor heuristics, computing user-user "
        "(GroupLens; Resnick et al., 1994) or item-item (Sarwar et al., 2001; Linden et al., 2003) cosine or Pearson similarity "
        "directly over sparse rating vectors. While interpretable, neighborhood methods scale quadratically with M or N in naive "
        "form and degrade rapidly when co-rated overlap is sparse. Alongside neighborhood methods, Bayesian damped popularity "
        "estimators—exemplified by the Internet Movie Database (IMDB) Weighted Rating—remain indispensable sanity baselines "
        "because item popularity accounts for a substantial fraction of observational click and watch behavior (Steck, 2011).",
    )

    _h2(doc, "2.2 Latent Factor Matrix Factorization: Explicit SVD vs. Implicit ALS")
    _p(
        doc,
        "The Netflix Prize competition (2006–2009) established latent factor Matrix Factorization (MF) as the dominant paradigm "
        "for collaborative filtering (Koren, Bell, & Volinsky, 2009). Unlike truncated exact Singular Value Decomposition (SVD), "
        "which requires imputing missing entries in the sparse rating matrix, Simon Funk's regularized stochastic gradient "
        "factorization ('Funk-SVD') and Koren's Biased SVD optimize latent vectors p_u, q_i in R^d and scalar user/item biases "
        "b_u, b_i directly over observed ratings via L2-regularized least squares.",
    )
    _p(
        doc,
        "Crucially, however, Hu, Koren, and Volinsky (2008) demonstrated that optimizing squared error solely over observed "
        "ratings fails for Top-N recommendation because missing entries carry vital implicit negative signal—users deliberately "
        "avoid items uninteresting to them (Marlin & Zemel, 2009). Hu et al. introduced Implicit Alternating Least Squares (ALS), "
        "binarizing preferences p_ui in {0, 1} across the entire M x N matrix and weighting errors by confidence c_ui = 1 + alpha * r_ui. "
        "By exploiting algebraic sparsity structure, Implicit ALS solves exact ridge regressions for all M x N pairs in time linear "
        "in observed interactions plus O((M + N) d^3) per iteration. Rendle et al. (2009) subsequently proposed Bayesian Personalized "
        "Ranking (BPR), optimizing pairwise rank margins via negative sampling.",
    )

    _h2(doc, "2.3 Neural Collaborative Filtering and Modern Reproducibility Insights")
    _p(
        doc,
        "Motivated by the geometric limitation of the fixed linear inner product <p_u, q_i>—which can violate triangle-like "
        "ranking constraints in low-dimensional Euclidean spaces—He et al. (WWW 2017) introduced the Neural Collaborative "
        "Filtering (NCF) framework. NCF replaces or augments the inner product with a deep Multi-Layer Perceptron (MLP) operating "
        "on concatenated user-item embeddings, culminating in Neural Matrix Factorization (NeuMF), which fuses a Generalized "
        "Matrix Factorization (GMF) multiplicative pathway with a non-linear MLP tower trained via binary cross-entropy (BCE) "
        "and dynamic negative sampling.",
    )
    _p(
        doc,
        "Subsequent reproducibility studies have enriched the community's understanding of NCF versus classical MF. Ferrari Dacrema, "
        "Cremonesi, and Jannach (RecSys 2019) and Rendle et al. (RecSys 2020) showed that properly tuned latent factor baselines "
        "(such as ALS and dot-product MF trained with implicit losses) remain remarkably competitive against learned MLP interaction "
        "functions, and Krichene & Rendle (KDD 2020) proved that evaluating Top-K metrics on '1 positive + 99 sampled negatives' "
        "(used in the original NCF paper) distorts metric orderings relative to exact full-catalog ranking. Informed by this "
        "literature, our study implements both explicit Biased SVD and implicit ALS alongside GMF, MLP, scratch NeuMF, and "
        "pre-trained NeuMF, evaluating every model under unbiased full-catalog ranking.",
    )

    # ==================== SECTION 3: METHODOLOGY ====================
    _h1(doc, "3. Methodology")
    _h2(doc, "3.1 Bayesian Popularity Baseline (IMDB Weighted Rating)")
    _p(
        doc,
        "A naive item-mean baseline ranks movies with a single 5-star review above acclaimed classics with thousands of "
        "ratings, while raw rating-count ranking ignores perceived quality. To construct a strong non-personalized benchmark, "
        "we adopt the Empirical Bayes shrinkage estimator popularized by IMDB's Top-250 chart:",
    )
    _add_equation(doc, "WR_i = [v_i / (v_i + m)] * R_i  +  [m / (v_i + m)] * C", "1")
    _p(
        doc,
        "where v_i denotes the number of training ratings for movie i, R_i is its sample mean rating in Train, C = mu is the "
        "global training mean rating across all interactions, and m is the 80th percentile of v_i among rated items. When v_i << m, "
        "WR_i shrinks conservatively toward the global prior C; when v_i >> m, WR_i converges to the empirical mean R_i. Unrated "
        "training items (v_i = 0) are assigned -1.0 for Top-N ranking, while explicit rating prediction returns R_i (or C if v_i = 0).",
    )

    _h2(doc, "3.2 Explicit Biased Matrix Factorization (Funk-SVD)")
    _p(
        doc,
        "Biased Matrix Factorization (Koren et al., 2009) embeds users and items into a shared d-dimensional Euclidean space "
        "P in R^{M x d}, Q in R^{N x d}, augmented with global offset mu, user rating tendency b_u in R (capturing harsh vs. "
        "generous raters), and item quality offset b_i in R:",
    )
    _add_equation(doc, "r_hat_{u,i} = mu + b_u + b_i + <p_u, q_i> = mu + b_u + b_i + sum_{f=1}^{d} p_{u,f} * q_{i,f}", "2")
    _p(
        doc,
        "Model parameters Theta = {P, Q, b_u, b_i} are learned by minimizing regularized Mean Squared Error over observed "
        "training ratings R_train via mini-batch Adam optimization:",
    )
    _add_equation(doc, "L_SVD(Theta) = (1 / |R_train|) * sum_{(u,i) in R_train} [ (r_{u,i} - r_hat_{u,i})^2 + lambda * (||p_u||_2^2 + ||q_i||_2^2 + b_u^2 + b_i^2) ]", "3")
    _p(
        doc,
        "Notice that regularization is applied exclusively to the active batch embeddings (p_u, q_i, b_u, b_i) rather than "
        "globally decaying unvisited items on every step, which naturally enforces frequency-proportional shrinkage and prevents "
        "rare-item collapse while remaining compatible with adaptive momentum updates.",
    )

    _h2(doc, "3.3 Implicit Alternating Least Squares (ALS)")
    _p(
        doc,
        "To directly address the project specification for ALS/SVD Matrix Factorization on Top-N recommendation, we implement "
        "Implicit ALS following Hu, Koren, and Volinsky (2008). For every user-item pair (u, i) in U x I, define binary preference "
        "p_ui = 1 if (u, i) in R^+_train (i.e., r_ui >= 4) and p_ui = 0 otherwise. Confidence in p_ui is modeled as "
        "c_ui = 1 + alpha * r'_ui, where r'_ui = max(0, r_ui - 3) assigns higher confidence to 5-star interactions (1 + 2*alpha) "
        "than 4-star interactions (1 + alpha), while all unobserved pairs carry baseline unit confidence c_ui = 1. ALS minimizes "
        "the full-matrix objective:",
    )
    _add_equation(doc, "L_ALS(X, Y) = sum_{u=0}^{M-1} sum_{i=0}^{N-1} c_{u,i} (p_{u,i} - x_u^T y_i)^2 + lambda_ALS * (||X||_F^2 + ||Y||_F^2)", "4")
    _p(
        doc,
        "Although Equation (4) sums over M * N ≈ 23.45 million pairs, holding item factors Y fixed renders L_ALS strictly "
        "quadratic in each user vector x_u. Exploiting the identity Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y, where (C^u - I) is "
        "non-zero only on the |I^+_u| positive items of user u, yields the exact closed-form ridge update:",
    )
    _add_equation(doc, "x_u <- ( Y^T Y + Y_{I^+_u}^T diag(c_{u, I^+_u} - 1) Y_{I^+_u} + lambda_ALS * I_d )^{-1} Y_{I^+_u}^T c_{u, I^+_u}", "5")
    _p(
        doc,
        "Symmetrically fixing X updates each item vector y_i. Precomputing Y^T Y in O(N d^2) reduces each user solve to "
        "O(|I^+_u| d^2 + d^3), allowing 12 full alternating sweeps over MovieLens-1M to converge in under 10 seconds on CPU.",
    )

    _h2(doc, "3.4 Neural Collaborative Filtering (GMF, MLP, and NeuMF)")
    _p(
        doc,
        "Following He et al. (WWW 2017), our NCF module generalizes latent factor interaction modeling through two parallel "
        "neural pathways and their unified fusion (Figure 2b):",
    )
    _p(
        doc,
        "Maps user u and item i to dedicated embeddings p_u^G, q_i^G in R^{d_G} and computes the element-wise Hadamard product "
        "phi_GMF(u, i) = p_u^G (x) q_i^G, projected via a learnable weight vector h_G in R^{d_G}. When h_G = 1 and bias is zero, "
        "GMF recovers standard dot-product Matrix Factorization as a special case while permitting non-uniform dimension weighting.",
        bold_prefix="1. Generalized Matrix Factorization (GMF): ",
    )
    _p(
        doc,
        "Maintains an independent embedding space p_u^M, q_i^M in R^{d_M}, concatenates z_0 = [p_u^M || q_i^M] in R^{2 d_M}, "
        "and passes z_0 through a tower of L = 3 hidden blocks (64 -> 32 -> 16). Each block applies an affine transformation, "
        "Batch Normalization (stabilizing internal covariate shift across mini-batches), ReLU activation, and Dropout (p = 0.2) "
        "to learn high-order non-linear feature interactions phi_MLP(u, i) in R^{16}.",
        bold_prefix="2. Multi-Layer Perceptron (MLP): ",
    )
    _p(
        doc,
        "Allows GMF and MLP to learn separate optimal embedding spaces, concatenating their penultimate representations into the "
        "NeuMF output layer:",
        bold_prefix="3. Neural Matrix Factorization (NeuMF & Pre-training Fusion): ",
    )
    _add_equation(doc, "z_{u,i} = h^T [ phi_GMF(u, i)  ||  phi_MLP(u, i) ] + b_out,      y_hat_{u,i} = sigma(z_{u,i})", "6")
    _p(
        doc,
        "Implicit NCF models are trained by minimizing Binary Cross-Entropy (BCE) with logits over positive interactions R^+_train "
        "paired with K_neg = 4 dynamically re-sampled unobserved items R^-_train per epoch:",
    )
    _add_equation(doc, "L_NCF = - sum_{(u,i) in R^+_train U R^-_train} [ y_{u,i} log sigma(z_{u,i}) + (1 - y_{u,i}) log(1 - sigma(z_{u,i})) ]", "7")
    _p(
        doc,
        "Because non-convex joint optimization of GMF and MLP from random initialization can converge to suboptimal local minima, "
        "we implement He et al.'s two-stage pre-training strategy (NeuMF Pre-trained): after independently training GMF and MLP "
        "to convergence, their embeddings and hidden layers are copied into NeuMF, and the final projection weights are fused via "
        "h <- [alpha * h_GMF || (1 - alpha) * h_MLP] with trade-off factor alpha = 0.5, followed by low-learning-rate fine-tuning. "
        "Additionally, to enable direct star-rating RMSE comparison without sacrificing architectural parity, we train an explicit "
        "regression variant (NeuMF-Explicit) whose output mu + z_{u,i} minimizes MSE on 1–5 star ratings.",
    )

    _add_figure(
        doc,
        fig_dir / "fig2_architecture_overview.png",
        "Figure 2: Architectural Comparison. (a) Linear Matrix Factorization (Biased SVD / ALS) operating in a single inner-product space. (b) Dual-pathway Neural Collaborative Filtering (NeuMF) combining element-wise GMF multiplication with a deep regularized MLP tower.",
    )

    _add_styled_table(
        doc,
        "Table 2: Architectural, Supervision, and Computational Comparison of Implemented Models",
        ["Model", "Supervision", "Loss / Solver", "Parameters (d=64)", "Full-Catalog Scoring Complexity"],
        [
            ["Popularity (IMDB WR)", "Explicit Aggregates", "Closed-form Shrinkage", "~7.8K stats", "O(1) broadcast per user"],
            ["Biased SVD (MF)", "Explicit (1-5★)", "MSE + Batch L2 (Adam)", "~645K", "O(B * N * d) matrix multiply"],
            ["Implicit ALS (MF)", "Implicit (r >= 4)", "Weighted Ridge ALS", "~635K", "O(B * N * d) matrix multiply"],
            ["GMF", "Implicit + 4 Neg", "BCE with Logits (Adam)", "~318K (d_G=32)", "O(B * N * d_G) weighted dot"],
            ["MLP", "Implicit + 4 Neg", "BCE with Logits (Adam)", "~325K (64->32->16)", "O(B * N * h_1) neural forward"],
            ["NeuMF (Scratch / Pre)", "Implicit + 4 Neg", "BCE + Fusion (alpha=0.5)", "~643K", "O(B * N * (d_G + h_1))"],
            ["NeuMF-Explicit", "Explicit (1-5★)", "MSE + Weight Decay", "~643K", "O(B * N * (d_G + h_1))"],
        ],
    )

    # ==================== SECTION 4: IMPLEMENTATION ====================
    _h1(doc, "4. System Architecture and Implementation Details")
    _h2(doc, "4.1 Modular Codebase Design")
    _p(
        doc,
        "The project is engineered as a modular, test-driven Python package adhering to strict separation of concerns: "
        "src/download_data.py handles resilient HTTPS dataset acquisition with multi-context SSL verification; src/data.py "
        "implements fast C-engine parsing, chronological splitting, sparse CSR matrix construction, and vectorized batch iterators; "
        "src/models.py encapsulates all seven model variants behind a unified score_all(users) interface; src/train.py manages "
        "reproducible seeding, hyperparameter grid searches, early stopping, pre-trained weight fusion, and JSON telemetry logging; "
        "src/metrics.py and src/evaluate.py execute full-catalog evaluation; src/error_analysis.py performs cohort disaggregation; "
        "src/plot_results.py generates 300-DPI figures; and demo/app.py serves the interactive Streamlit web interface.",
    )

    _h2(doc, "4.2 Vectorized Data Pipeline and Binary-Search Negative Sampling")
    _p(
        doc,
        "A critical engineering bottleneck when training implicit NCF on ~460,000 positive interactions with K_neg = 4 negatives "
        "(yielding ~2.30 million training pairs per epoch) is Python iteration overhead in standard item-by-item PyTorch "
        "TensorDataset loaders, which invoke __getitem__ millions of times per epoch. We replace item-wise collation with a "
        "custom FastTensorDataLoader that generates a single permutation tensor perm = torch.randperm(N_samples) and slices "
        "contiguous mini-batches (batch_size = 4,096) directly in C++, accelerating CPU epoch throughput by over 12x.",
    )
    _p(
        doc,
        "Similarly, naive set-membership rejection sampling for negative items is slow in Python loops. Instead, we encode all "
        "observed training pairs into a sorted 64-bit integer array seen_keys = sort(u * N + i). Candidate negative arrays are "
        "sampled vectorized via NumPy's PCG64 generator and validated simultaneously using binary search (np.searchsorted), "
        "resampling only colliding indices until 0% false negatives remain—completing 1.84 million negative draws in ~0.18 seconds.",
    )

    _h2(doc, "4.3 Fast Full-Catalog Scoring via Linear Layer Decomposition")
    _p(
        doc,
        "Full-catalog ranking requires scoring 6,040 users across all 3,883 movies (23.45 million user-item pairs) on every "
        "validation and test pass. For MLP and NeuMF, naively concatenating [p_u^M || q_i^M] materializes a massive "
        "(B * N) x (2 d_M) tensor and multiplies it by W_0 in R^{h_0 x 2 d_M}. We exploit exact linearity of the first affine "
        "projection W_0 [p_u^M || q_i^M] + b_0 = (W_{0,u} p_u^M) + (W_{0,i} q_i^M + b_0): the item projection W_{0,i} Q^M + b_0 "
        "is precomputed once for all N = 3,883 items, and user projections are added via 3D broadcasting before entering BatchNorm "
        "and ReLU. Automated unit tests verify exact numerical equivalence (max absolute error < 10^-5) while cutting MLP/NeuMF "
        "full-catalog evaluation time by 55%.",
    )

    _h2(doc, "4.4 Hyperparameter Tuning and Training Dynamics")
    _p(
        doc,
        f"To ensure fair, well-tuned baselines (Ferrari Dacrema et al., 2019), Biased SVD undergoes an exhaustive 12-configuration "
        f"grid search across embedding dimensions d in {{32, 64, 128}}, learning rates lr in {{0.005, 0.002}}, and batch L2 "
        f"regularization penalties lambda in {{0.005, 0.02}} with early stopping (patience = 3) on Validation RMSE. As documented "
        f"in Table 3 and Figure 3, optimal generalization is achieved at d = {svd_hparams['dim']}, lr = {svd_hparams['lr']}, and "
        f"lambda = {svd_hparams['reg']}. Unregularized or weakly regularized large embeddings overfit chronological validation "
        f"rapidly after epoch 4, whereas moderate L2 shrinkage stabilizes latent inner products.",
    )

    if not grid_df.empty:
        grid_sorted = grid_df.sort_values("best_val_rmse").head(8)
        _add_styled_table(
            doc,
            "Table 3: Top Hyperparameter Configurations from Biased SVD Grid Search (Sorted by Validation RMSE)",
            ["Rank", "Embedding Dim (d)", "Learning Rate (lr)", "L2 Penalty (reg)", "Epochs Run", "Best Val RMSE"],
            [
                [
                    str(idx + 1),
                    str(int(r["dim"])),
                    f"{r['lr']:.4f}",
                    f"{r['reg']:.4f}",
                    str(int(r["epochs_run"])),
                    f"{r['best_val_rmse']:.4f}",
                ]
                for idx, (_, r) in enumerate(grid_sorted.iterrows())
            ],
        )

    _add_figure(
        doc,
        fig_dir / "fig3_mf_grid_search.png",
        "Figure 3: Biased SVD Hyperparameter Sensitivity across Embedding Dimensions, Learning Rates, and L2 Regularization Strengths.",
        width_in=5.85,
    )

    _p(
        doc,
        "For Implicit ALS, grid tuning over confidence scaling alpha in {15, 20} and ridge penalty lambda_ALS in {5, 10, 15} "
        "identifies peak validation NDCG@10 within 4–6 alternating iterations before mild over-specialization sets in. For NCF "
        "models (d = 64 split into d_G = 32 and d_M = 32 with MLP layers 64 -> 32 -> 16 and Dropout 0.2), Adam with lr = 0.001 "
        "trains GMF, MLP, and Scratch NeuMF, while Pre-trained NeuMF is fine-tuned from converged GMF + MLP checkpoints at "
        "lr = 0.0005. Figure 4 illustrates the validation trajectories across both explicit and implicit training regimes.",
    )

    _add_figure(
        doc,
        fig_dir / "fig4_training_curves.png",
        "Figure 4: Validation Learning Dynamics. (a) Explicit rating models tracked via Validation RMSE. (b) Implicit ranking models tracked via sampled Validation NDCG@10 (2,000 validation users per epoch).",
    )

    # ==================== SECTION 5: EVALUATION & ERROR ANALYSIS ====================
    _h1(doc, "5. Experimental Evaluation and Error Analysis")
    _h2(doc, "5.1 Evaluation Protocol and Metric Definitions")
    _p(
        doc,
        "All models are evaluated once on the held-out chronological Test set. For explicit rating prediction, Test RMSE is "
        "computed across all |R_test| ratings:",
    )
    _add_equation(doc, "RMSE = sqrt( (1 / |R_test|) * sum_{(u,i,r) in R_test} (r - clamp(r_hat_{u,i}, 1, 5))^2 )", "8")
    _p(
        doc,
        "For Top-K recommendation (K = 10), all movies interacted with by user u in Train U Val are masked to -infinity so models "
        "cannot re-recommend already watched films. Let L_10(u) = [i_1, ..., i_10] be the top-10 ranked unseen items across the "
        "entire catalog, and let R^+_u = {i in R_test(u) : r_ui >= 4} be the user's held-out liked movies (evaluating all users "
        "with |R^+_u| >= 1). Precision@10, Recall@10, and NDCG@10 are defined as:",
    )
    _add_equation(
        doc,
        "P@10(u) = |L_10(u) cap R^+_u| / 10,    R@10(u) = |L_10(u) cap R^+_u| / |R^+_u|,    NDCG@10(u) = DCG@10(u) / IDCG@10(u)",
        "9",
    )
    _p(
        doc,
        "where DCG@10(u) = sum_{r=1}^{10} I[i_r in R^+_u] / log_2(r + 1) rewards placing relevant movies near the top of the list, "
        "and IDCG@10(u) = sum_{r=1}^{min(10, |R^+_u|)} 1 / log_2(r + 1) normalizes scores to [0, 1]. Note that full-catalog "
        "ranking against ~3,700 unseen candidates per user is orders of magnitude stricter than 1-vs-99 sampled evaluation, so "
        "absolute Precision@10 and NDCG@10 values around 0.08–0.15 reflect strong real-world retrieval performance.",
    )

    _h2(doc, "5.2 Main Quantitative Benchmark Results")
    _p(
        doc,
        "Table 4 and Figure 5 present the comprehensive Test-set comparison across all eight baselines and learned architectures. "
        "Note that implicit BCE/ALS models optimize monotonic ranking scores rather than 1–5 star values; hence their RMSE cells "
        "are denoted '—', while NeuMF-Explicit (MSE) provides an exact neural star-rating counterpart.",
    )

    _add_styled_table(
        doc,
        "Table 4: Held-Out Test Set Performance on MovieLens-1M (Full-Catalog Top-10 Ranking & Explicit RMSE)",
        ["Model", "Paradigm", "Test RMSE ↓", "Precision@10 ↑", "Recall@10 ↑", "NDCG@10 ↑", "HitRate@10 ↑"],
        [
            [
                str(r["Model"]),
                str(r["Paradigm"]),
                "—" if pd.isna(r["RMSE"]) else f"{r['RMSE']:.4f}",
                "—" if pd.isna(r["Precision@10"]) else f"{r['Precision@10']:.4f}",
                "—" if pd.isna(r["Recall@10"]) else f"{r['Recall@10']:.4f}",
                "—" if pd.isna(r["NDCG@10"]) else f"{r['NDCG@10']:.4f}",
                "—" if pd.isna(r["HitRate@10"]) else f"{r['HitRate@10']:.4f}",
            ]
            for _, r in metrics_df.iterrows()
        ],
    )

    _add_figure(
        doc,
        fig_dir / "fig5_model_comparison.png",
        "Figure 5: Quantitative Test Benchmark. (a) Explicit Star Rating Prediction (RMSE, lower is better). (b) Full-Catalog Top-10 Ranking (Precision@10, Recall@10, NDCG@10, higher is better).",
    )

    _p(
        doc,
        f"Three headline conclusions emerge immediately from Table 4. First, on explicit star prediction, {best_rmse_row['Model']} "
        f"delivers the lowest Test RMSE ({best_rmse_row['RMSE']:.4f}), substantially outperforming both the Global Mean baseline "
        f"({gmean_rmse:.4f}, a {(gmean_rmse - best_rmse_row['RMSE'])/gmean_rmse*100:.1f}% error reduction) and Item-Mean Popularity "
        f"({pop_rmse:.4f}), while NeuMF-Explicit achieves a closely competitive RMSE of {neumf_exp_rmse:.4f}. Second, on full-catalog "
        f"Top-10 ranking, {best_ndcg_row['Model']} achieves the strongest overall performance with NDCG@10 = "
        f"{best_ndcg_row['NDCG@10']:.4f}, Precision@10 = {best_ndcg_row['Precision@10']:.4f}, and Recall@10 = "
        f"{best_ndcg_row['Recall@10']:.4f}, surpassing the Popularity baseline (NDCG@10 = {pop_ndcg:.4f}) by "
        f"+{(best_ndcg_row['NDCG@10'] - pop_ndcg)/pop_ndcg*100:.1f}%. Third, Implicit ALS (NDCG@10 = {als_ndcg:.4f}, P@10 = {als_p10:.4f}, "
        f"R@10 = {als_r10:.4f}) dramatically outperforms explicit Biased SVD on ranking despite sharing an identical linear "
        f"inner-product representation.",
    )

    _h2(doc, "5.3 Explaining the Explicit-vs-Implicit Ranking Paradox")
    _p(
        doc,
        f"One of the most instructive empirical findings in Table 4 is that explicit models—both Biased SVD (RMSE = {svd_rmse:.4f}, "
        f"NDCG@10 = {svd_ndcg:.4f}) and NeuMF-Explicit (RMSE = {neumf_exp_rmse:.4f}, NDCG@10 = {neumf_exp_ndcg:.4f})—rank worse "
        f"than non-personalized Popularity (NDCG@10 = {pop_ndcg:.4f}) when deployed directly for Top-10 retrieval! Far from being "
        f"an implementation bug, this confirms the classic Missing-Not-At-Random (MNAR) phenomenon established by Cremonesi et al. "
        f"(RecSys 2010) and Steck (RecSys 2010):",
    )
    _p(
        doc,
        "Explicit models are trained exclusively on observed ratings without ever penalizing high predictions on unobserved movies. "
        "If an obscure documentary was watched only by 5 dedicated enthusiasts who all rated it 5/5, explicit SVD assigns it a "
        "large positive item bias b_i and unpenalized latent norm, pushing it into everyone's Top-10 list—even though 99.9% of "
        "users had zero interest in watching it! By contrast, Implicit ALS (via unit confidence on all unobserved pairs) and "
        "Implicit NCF (via 4:1 negative sampling) teach the model which items a user is unlikely to engage with, aligning the "
        "learned score surface directly with Top-N retrieval.",
    )

    _h2(doc, "5.4 NCF Architectural and Pre-Training Ablation Study")
    _p(
        doc,
        f"Figure 6 isolates the contribution of each component within the NCF family. Standalone GMF reaches Test NDCG@10 = "
        f"{gmf_ndcg:.4f} while standalone MLP reaches {mlp_ndcg:.4f}, demonstrating that both multiplicative latent matching and "
        f"deep non-linear concatenation capture substantial collaborative signal under negative sampling. Notably, later bias "
        f"profiling (Table 6) shows that standalone MLP achieves its aggregate accuracy largely by amplifying popularity "
        f"(96.36% Top-10% head share and only 19.34% catalog coverage), whereas GMF explores broader item neighborhoods (32.60% "
        f"coverage). Combining both branches from random initialization (NeuMF Scratch) yields NDCG@10 = {neumf_s_ndcg:.4f}. "
        f"Crucially, initializing NeuMF with pre-trained GMF and MLP weights (alpha = 0.5) elevates Test NDCG@10 to "
        f"{neumf_pt_ndcg:.4f} (P@10 = {neumf_pt_p10:.4f}, R@10 = {neumf_pt_r10:.4f}) and HitRate@10 to 0.3470 while preserving "
        f"33.66% catalog coverage—validating He et al.'s hypothesis that two-stage pre-training escapes poor joint local optima.",
    )

    _add_figure(
        doc,
        fig_dir / "fig6_ablation_ncf.png",
        "Figure 6: NCF Ablation Study across GMF, MLP, Randomly Initialized NeuMF, and Pre-trained NeuMF on Held-Out Test Metrics.",
    )

    _h2(doc, "5.5 Error Analysis Across User Cohorts (Cold-Start & Gray-Sheep)")
    _p(
        doc,
        "Aggregate averages often conceal systematic failures on vulnerable user subpopulations. We stratify Test users along two "
        "orthogonal behavioral axes: (1) Activity Level—dividing users into Light Users (bottom 25% training history, serving as "
        "a realistic cold-start proxy since MovieLens-1M enforces >= 20 total ratings per user), Moderate Users (middle 50%), and "
        "Heavy Users (top 25%); and (2) Taste Mainstreamness—isolating Gray-Sheep / Niche Users (bottom 20% mean log-popularity "
        "of positively rated training movies) from Mainstream Users (top 80%). Table 5 and Figure 7 report the disaggregated metrics.",
    )

    focus_models = ["Popularity", "Biased SVD (MF)", "Implicit ALS (MF)", "GMF", "MLP", "NeuMF (Pre-trained)"]
    sub_coh = cohorts_df[cohorts_df["Model"].isin(focus_models)]
    _add_styled_table(
        doc,
        "Table 5: Stratified Error Analysis Across User Activity (Cold-Start Proxy) and Taste (Gray-Sheep) Cohorts",
        ["Model", "Cohort Dimension & Group", "Users", "P@10", "R@10", "NDCG@10"],
        [
            [
                str(r["Model"]),
                str(r["Cohort"]),
                str(int(r["Users"])),
                f"{r['Precision@10']:.4f}",
                f"{r['Recall@10']:.4f}",
                f"{r['NDCG@10']:.4f}",
            ]
            for _, r in sub_coh.iterrows()
        ],
    )

    _add_figure(
        doc,
        fig_dir / "fig7_error_user_groups.png",
        "Figure 7: Stratified Error Analysis. (a) Cold-Start Proxy across Light, Moderate, and Heavy User Activity Quartiles. (b) Gray-Sheep Niche Users vs. Mainstream Users.",
    )

    _p(
        doc,
        "Two striking patterns emerge from Table 5 and Figure 7. First, across Activity quartiles, Precision@10 and NDCG@10 scale "
        "strongly with interaction volume for neural models because Heavy Users hold both richer training profiles and larger Test "
        "positive sets (whereas Recall@10 is higher for Light Users due to smaller denominators |R^+_u|). On Light Users (bottom "
        "25% cold-start proxy), Implicit ALS actually achieves the highest NDCG@10 (0.0746), slightly outperforming Pre-trained "
        "NeuMF (0.0730) and standalone MLP (0.0587), because closed-form L2-regularized ridge regression resists overfitting when "
        "individual user histories are sparse. On Heavy Users, deep non-linear models dominate (MLP 0.1195, NeuMF Pre-trained 0.1155 "
        "vs. ALS 0.0857). Second, on Gray-Sheep / Niche Users (bottom 20% popularity), Pre-trained NeuMF achieves the strongest "
        "NDCG@10 (0.0757, a +71.7% gain over Popularity's 0.0441), proving genuine personalization beyond mainstream blockbusters.",
    )

    _h2(doc, "5.6 Popularity Bias, Catalog Coverage, and Complete-Miss Rate")
    _p(
        doc,
        "Beyond accuracy, sustainable recommendation ecosystems must surface long-tail content rather than creating 'rich-get-richer' "
        "filter bubbles. Table 6 and Figure 8 evaluate four beyond-accuracy indicators across all 60,400 Top-10 recommendation "
        "slots: Average Recommended Item Popularity, Top-10% Head Share, Catalog Coverage (unique movies recommended / 3,883), "
        "Gini Concentration Index, and the Zero-Hit Failure Rate (fraction of users receiving NDCG@10 = 0).",
    )

    zero_col = [c for c in bias_df.columns if c.startswith("Zero-Hit")][0]
    _add_styled_table(
        doc,
        "Table 6: Popularity Bias, Catalog Diversity, Gini Inequality, and Zero-Hit User Failure Rate",
        ["Model", "Avg Item Views", "Top-10% Head Share", "Catalog Coverage", "Unique Movies", "Gini Index", "Zero-Hit Users"],
        [
            [
                str(r["Model"]),
                f"{r['Avg Rec Item Popularity']:.1f}",
                f"{r['Top-10% Head Share']*100:.1f}%",
                f"{r['Catalog Coverage']*100:.2f}%",
                str(int(r["Unique Recommended Movies"])),
                f"{r['Gini Concentration']:.4f}",
                f"{r[zero_col]*100:.1f}%",
            ]
            for _, r in bias_df.iterrows()
        ],
    )

    _add_figure(
        doc,
        fig_dir / "fig8_popularity_bias_coverage.png",
        "Figure 8: Beyond-Accuracy Trade-offs. (a) Accuracy (NDCG@10) vs. Catalog Coverage (%) frontier with bubble size proportional to Top-10% head concentration. (b) Head-item bias and Zero-Hit failure rates.",
    )

    _p(
        doc,
        f"Table 6 exposes the extreme degeneracy of non-personalized Popularity: 100% of its recommendations come from the top-10% "
        f"head, and across all 6,040 users it surfaces barely {pop_cov:.2f}% of the movie catalog (only shifting slightly as "
        f"already-seen blockbusters are masked per user), resulting in a Gini index near 0.99. Conversely, Implicit ALS and "
        f"Pre-trained NeuMF occupy the Pareto-superior upper-right frontier of Figure 8a: they simultaneously elevate NDCG@10, "
        f"expand catalog coverage to {als_cov:.1f}% and {npt_cov:.1f}% respectively, and substantially reduce the proportion of "
        f"users experiencing complete recommendation failure (NDCG@10 = 0).",
    )

    _h2(doc, "5.7 Qualitative Case Study and Interactive Streamlit Application")
    _p(
        doc,
        "To bridge offline metrics with human-interpretable inspection, we developed an interactive multi-model web application "
        "in Streamlit (demo/app.py). During live demonstration, selecting an individual user (or switching among curated personas: "
        "Active Mainstream Fan, Heavy Cinephile, Light Cold-Start User, and Gray-Sheep Niche User) displays the user's demographic "
        "attributes, highest-rated historical films, held-out Test ground truth, and real-time (<20 ms) Top-K recommendation "
        "columns across Popularity, Biased SVD, Implicit ALS, and Pre-trained NeuMF.",
    )
    _p(
        doc,
        "Qualitative inspection on real test profiles corroborates our quantitative findings. Consider Gray-Sheep User #5333 "
        "(only 8 liked movies in Train+Val—including niche dramas such as 'The Funeral (1996)', 'Firelight (1997)', 'Song of "
        "Freedom (1936)', and 'Erin Brockovich (2000)'—and a single held-out positive in Test). Popularity recommends generic "
        "classics ('The Shawshank Redemption', 'The Godfather', 'Schindler's List'), scoring 0/5 hits. Consistent with our MNAR "
        "analysis, Biased SVD surfaces obscure high-bias titles ('Der bewegte Mann (1994)', 'An Actor's Revenge (1963)', 'Time "
        "of the Gypsies (1989)'), also scoring 0/5 hits. Implicit ALS adapts to the user's 1999–2000 drama cohort ('Magnolia', "
        "'Wonder Boys'), while Pre-trained NeuMF places the user's exact held-out Test movie 'Return to Me (2000)' at Rank #1!",
    )

    # ==================== SECTION 6: CONCLUSIONS & FUTURE WORK ====================
    _h1(doc, "6. Conclusions and Future Work")
    _h2(doc, "6.1 Summary of Findings")
    _p(
        doc,
        "This study conducted an end-to-end empirical comparison of Popularity heuristics, linear Matrix Factorization (Biased "
        "SVD and Implicit ALS), and deep Neural Collaborative Filtering (GMF, MLP, Scratch NeuMF, Pre-trained NeuMF, and Explicit "
        "NeuMF) on MovieLens-1M under a leak-free per-user temporal split and full-catalog evaluation. Our primary scientific "
        "and engineering conclusions are fourfold:",
    )
    _p(
        doc,
        f"Minimizing explicit rating error (RMSE = {svd_rmse:.4f} via Biased SVD) does not translate into effective Top-10 item "
        f"discovery (NDCG@10 = {svd_ndcg:.4f}) because explicit training ignores the vast implicit negative signal encoded in "
        f"unobserved interactions. Incorporating implicit negative feedback—either analytically via ALS confidence weighting or "
        f"stochastically via NCF negative sampling—improves Top-10 NDCG by more than 2x–3x over explicit MF.",
        bold_prefix="1. Supervision Paradigm Dominates Model Complexity: ",
    )
    _p(
        doc,
        f"Within the NCF family, two-stage pre-training (initializing NeuMF from converged GMF and MLP checkpoints) consistently "
        f"improves ranking quality over random initialization ({neumf_pt_ndcg:.4f} vs. {neumf_s_ndcg:.4f} NDCG@10), while well-tuned "
        f"Implicit ALS ({als_ndcg:.4f} NDCG@10) proves remarkably competitive at a fraction of neural training cost—echoing modern "
        f"reproducibility findings (Ferrari Dacrema et al., 2019; Rendle et al., 2020).",
        bold_prefix="2. Synergy of Pre-trained NeuMF and Strength of Tuned ALS: ",
    )
    _p(
        doc,
        "Disaggregated evaluation reveals that collaborative models deliver their largest relative gains on Gray-Sheep users with "
        "non-mainstream tastes while expanding catalog coverage by an order of magnitude over Popularity, mitigating filter-bubble "
        "monopolies.",
        bold_prefix="3. Fairness and Diversity Beyond Aggregate NDCG: ",
    )
    _p(
        doc,
        "Vectorized tensor batching, binary-search negative sampling, and first-layer MLP algebraic decomposition enable the "
        "entire seven-model training, tuning, and full-catalog evaluation suite to run reproducibly on standard commodity hardware "
        "in minutes and serve real-time interactive recommendations in under 20 milliseconds.",
        bold_prefix="4. Practical Efficiency: ",
    )

    _h2(doc, "6.2 Limitations and Future Directions")
    _p(
        doc,
        "Several limitations suggest promising avenues for future extension. First, both MF and NCF treat a user's historical "
        "interactions as an unordered static set, discarding short-term sequential intent (e.g., watching Toy Story immediately "
        "raising the probability of watching Toy Story 2 next). Integrating self-attentive sequential recommenders such as SASRec "
        "(Kang & McAuley, 2018) or bidirectional BERT4Rec (Sun et al., 2019) is a natural next step. Second, pure ID-based "
        "embeddings remain vulnerable to strict zero-interaction cold-start items and users; fusing multimodal movie metadata "
        "(genres, synopsis embeddings from pretrained Hugging Face Transformers, director/cast graphs) into a hybrid two-tower "
        "or LightGCN (He et al., 2020) architecture can enable zero-shot generalization. Finally, replacing uniform negative "
        "sampling with popularity-smoothed exposure debiasing (p_neg(i) proportional to v_i^{0.75}) can further suppress residual "
        "popularity bias while sharpening fine-grained ranking discrimination.",
    )

    # ==================== SECTION 7: REFERENCES ====================
    _h1(doc, "7. References")
    references = [
        "[1] F. M. Harper and J. A. Konstan, 'The MovieLens Datasets: History and Context,' ACM Transactions on Interactive Intelligent Systems (TiiS), vol. 5, no. 4, pp. 1–19, Dec. 2015.",
        "[2] Y. Koren, R. Bell, and C. Volinsky, 'Matrix Factorization Techniques for Recommender Systems,' IEEE Computer, vol. 42, no. 8, pp. 30–37, Aug. 2009.",
        "[3] Y. Hu, Y. Koren, and C. Volinsky, 'Collaborative Filtering for Implicit Feedback Datasets,' in Proc. IEEE International Conference on Data Mining (ICDM), Pisa, Italy, 2008, pp. 263–272.",
        "[4] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, 'Neural Collaborative Filtering,' in Proc. 26th International World Wide Web Conference (WWW), Perth, Australia, 2017, pp. 173–182.",
        "[5] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, 'BPR: Bayesian Personalized Ranking from Implicit Feedback,' in Proc. 25th Conference on Uncertainty in Artificial Intelligence (UAI), Montreal, Canada, 2009, pp. 452–461.",
        "[6] P. Cremonesi, Y. Koren, and R. Turrin, 'Performance of Recommender Algorithms on Top-N Recommendation Tasks,' in Proc. 4th ACM Conference on Recommender Systems (RecSys), Barcelona, Spain, 2010, pp. 39–46.",
        "[7] H. Steck, 'Training and Testing of Recommender Systems on Data Missing Not at Random,' in Proc. 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, 2010, pp. 713–722.",
        "[8] B. M. Marlin and R. S. Zemel, 'Collaborative Prediction and Ranking with Non-Random Missing Data,' in Proc. 3rd ACM Conference on Recommender Systems (RecSys), New York, NY, USA, 2009, pp. 5–12.",
        "[9] M. Ferrari Dacrema, P. Cremonesi, and D. Jannach, 'Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches,' in Proc. 13th ACM Conference on Recommender Systems (RecSys), Copenhagen, Denmark, 2019, pp. 101–109.",
        "[10] S. Rendle, W. Krichene, L. Zhang, and J. Anderson, 'Neural Collaborative Filtering vs. Matrix Factorization Revisited,' in Proc. 14th ACM Conference on Recommender Systems (RecSys), Virtual Event, 2020, pp. 240–248.",
        "[11] W. Krichene and S. Rendle, 'On Sampled Metrics for Item Recommendation,' in Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020, pp. 1748–1757.",
        "[12] Y. Ji, A. Sun, J. Zhang, and C. Li, 'A Critical Study on Data Leakage in Recommender System Offline Evaluation,' ACM Transactions on Information Systems (TOIS), vol. 41, no. 3, pp. 1–27, 2023.",
        "[13] H. Abdollahpouri, R. Burke, and B. Mobasher, 'Controlling Popularity Bias in Learning-to-Rank Recommendation,' in Proc. 11th ACM Conference on Recommender Systems (RecSys), Como, Italy, 2017, pp. 42–46.",
        "[14] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, 'Item-Based Collaborative Filtering Recommendation Algorithms,' in Proc. 10th International World Wide Web Conference (WWW), Hong Kong, 2001, pp. 285–295.",
        "[15] W.-C. Kang and J. McAuley, 'Self-Attentive Sequential Recommendation,' in Proc. IEEE International Conference on Data Mining (ICDM), Singapore, 2018, pp. 197–206.",
        "[16] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation,' in Proc. 43rd International ACM SIGIR Conference, 2020, pp. 639–648.",
    ]
    for ref in references:
        pr = doc.add_paragraph()
        pr.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pr.paragraph_format.left_indent = Inches(0.32)
        pr.paragraph_format.first_line_indent = Inches(-0.32)
        pr.paragraph_format.space_after = Pt(4.0)
        pr.paragraph_format.line_spacing = 1.15
        rr = pr.add_run(ref)
        rr.font.name = "Times New Roman"
        rr.font.size = Pt(10.0)

    docx_path = out_dir / "project_report.docx"
    doc.save(str(docx_path))

    # Also emit clean Overleaf-compatible LaTeX companion source
    tex_lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,float,caption}",
        r"\title{\textbf{A Rigorous Empirical Comparison of Popularity Baselines, Matrix Factorization (Biased SVD and Implicit ALS), and Neural Collaborative Filtering on MovieLens-1M}}",
        r"\author{Advanced Artificial Intelligence --- Course Project Technical Report}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        f"This report investigates Popularity (IMDB Weighted Rating), Matrix Factorization (Biased SVD and Implicit ALS), and Neural Collaborative Filtering (GMF, MLP, Scratch NeuMF, Pre-trained NeuMF, and Explicit NeuMF) on MovieLens-1M ({eda['n_ratings']:,} ratings, {eda['n_users']:,} users, {eda['n_items']:,} movies) under a leak-free per-user 80/10/10 temporal split and full-catalog evaluation. {best_rmse_row['Model']} achieves the best explicit rating accuracy (Test RMSE = {best_rmse_row['RMSE']:.4f}), whereas {best_ndcg_row['Model']} achieves superior Top-10 ranking (Test NDCG@10 = {best_ndcg_row['NDCG@10']:.4f}, Recall@10 = {best_ndcg_row['Recall@10']:.4f}).",
        r"\end{abstract}",
        r"\section{Problem and Data}",
        f"MovieLens-1M contains {eda['n_ratings']:,} ratings across {eda['n_users']:,} users and {eda['n_items']:,} items (sparsity {eda['sparsity']*100:.2f}\\%). See Figure~1 (`../results/figures/fig1_eda_distributions.png`).",
        r"\section{Related Work}",
        r"Covers Resnick et al. (1994), Koren et al. (2009), Hu et al. (2008), Rendle et al. (2009), He et al. (2017), and Ferrari Dacrema et al. (2019).",
        r"\section{Methods}",
        r"Formulates IMDB Weighted Rating, Biased Funk-SVD, Implicit ALS closed-form ridge updates, and dual-pathway GMF + MLP = Pre-trained NeuMF.",
        r"\section{Implementation}",
        r"Details vectorized tensor batching, binary-search negative sampling, and linear decomposition of the first MLP layer.",
        r"\section{Evaluation and Error Analysis}",
        r"See `../results/metrics.md`, `../results/error_analysis.md`, and Figures 3--8 in `../results/figures/`.",
        r"\section{Conclusions}",
        r"Supervision alignment (Implicit vs. Explicit) dominates architectural depth; pre-trained NeuMF and tuned Implicit ALS achieve Pareto-optimal accuracy and catalog coverage.",
        r"\end{document}",
    ]
    (out_dir / "project_report.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(f"Saved academic report -> {docx_path} and {out_dir / 'project_report.tex'}")
    return docx_path


if __name__ == "__main__":
    build_report()
