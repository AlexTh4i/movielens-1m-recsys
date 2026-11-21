# 5-Slide Project Presentation Summary

File generated: `docs/presentation_5_slides.pptx` (5 widescreen 16:9 slides)

## Slide 1: Movie Recommendation on MovieLens-1M — Problem & Temporal Protocol
- **Objective**: Compare Popularity, Matrix Factorization (`Biased SVD` & `Implicit ALS`), and Neural Collaborative Filtering (`GMF`, `MLP`, `NeuMF`) across `RMSE`, `Precision@10`, `Recall@10`, and `NDCG@10`.
- **Dataset**: MovieLens-1M (`1,000,209` ratings, `6,040` users, `3,883` movies, `95.74%` sparsity).
- **Split**: Strict per-user chronological `80/10/10` split preventing future leakage.

## Slide 2: Methodology — Linear Matrix Factorization vs. Dual-Pathway NCF
- **Popularity**: IMDB Weighted Rating (`q = 0.80`) + item mean baseline.
- **Matrix Factorization**: Explicit Biased SVD (MSE + L2) vs. Implicit ALS (Hu et al., 2008 confidence-weighted ridge regression).
- **Neural CF**: GMF + MLP (`64 -> 32 -> 16`) fused into **Pre-trained NeuMF** (`alpha = 0.5`) with 4:1 negative sampling and BCE loss.

## Slide 3: Quantitative Results & Ablation Study
- **Best Rating Predictor**: `Biased SVD (MF)` (`RMSE = 0.8935`).
- **Best Top-10 Ranker**: `NeuMF (Pre-trained)` (`NDCG@10 = 0.0801`, `Recall@10 = 0.0775`).
- **Explicit-Implicit Paradox**: Demonstrates why low RMSE does not imply strong Top-N ranking without negative feedback.

## Slide 4: Error Analysis — Cold-Start, Gray-Sheep Users & Popularity Bias
- Disaggregates performance across **Light / Moderate / Heavy** activity groups and **Gray-Sheep (Niche) vs. Mainstream** tastes.
- Quantifies the **Accuracy vs. Catalog Coverage** frontier (`Popularity` coverage `2.99%` vs. `NeuMF (Pre-trained)` `33.7%`).

## Slide 5: Interactive Streamlit Demo, Conclusions & Future Work
- Live real-time demonstration (`streamlit run demo/app.py`) with persona presets, Test-set hit verification, and pairwise Jaccard diversity.
- Extensions to Sequential RecSys (`SASRec`), Graph CF (`LightGCN`), and Transformer content embeddings.
