"""Automated test suite for the MovieLens-1M Recommender System pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.data import POS_THRESHOLD, eda_summary, explicit_loader, load_movielens, ncf_loader, to_csr
from src.metrics import evaluate_ranking, predict_ratings, rmse
from src.models import Popularity, build_model, load_model
from src.train import save_checkpoint


def test_movielens_1m_integrity_and_temporal_split():
    """Gate 1: Verify dataset dimensions, contiguous IDs, and zero temporal data leakage."""
    data = load_movielens("data/ml-1m")
    summary = eda_summary(data)

    assert summary["n_ratings"] == 1_000_209
    assert data.n_users == 6040
    assert data.n_items == 3883
    assert len(data.movies) == 3883
    assert len(data.users) == 6040

    assert data.train["u"].min() == 0 and data.train["u"].max() == data.n_users - 1
    assert data.movies["i"].tolist() == list(range(data.n_items))
    assert data.users["u"].tolist() == list(range(data.n_users))

    assert data.train["u"].nunique() == 6040
    assert data.val["u"].nunique() == 6040
    assert data.test["u"].nunique() == 6040

    tr_max = data.train.groupby("u")["ts"].max()
    va_min = data.val.groupby("u")["ts"].min()
    va_max = data.val.groupby("u")["ts"].max()
    te_min = data.test.groupby("u")["ts"].min()

    assert (tr_max <= va_min).all(), "Temporal leakage detected between Train and Validation!"
    assert (va_max <= te_min).all(), "Temporal leakage detected between Validation and Test!"


def test_negative_sampling_and_loaders():
    """Gate 1: Verify negative sampler never draws seen training items."""
    synth_train = pd.DataFrame(
        {
            "u": [0, 0, 1, 1, 2, 2],
            "i": [0, 1, 1, 2, 0, 3],
            "rating": [5.0, 3.0, 4.0, 5.0, 2.0, 4.0],
            "ts": [1, 2, 1, 2, 1, 2],
        }
    )
    n_items = 10
    seen_pairs = set(zip(synth_train["u"].tolist(), synth_train["i"].tolist()))
    rng = np.random.default_rng(42)

    loader = ncf_loader(synth_train, n_items=n_items, n_neg=4, batch=8, rng=rng)
    all_u, all_i, all_y = [], [], []
    for bu, bi, by in loader:
        all_u.extend(bu.tolist())
        all_i.extend(bi.tolist())
        all_y.extend(by.tolist())

    n_pos = int((synth_train["rating"] >= POS_THRESHOLD).sum())
    assert len(all_y) == n_pos * (1 + 4)
    for u, i, y in zip(all_u, all_i, all_y):
        if y == 0.0:
            assert (u, i) not in seen_pairs, f"Sampled negative {(u, i)} was actually seen in Train!"

    exp_loader = explicit_loader(synth_train, batch=4)
    total_exp = sum(len(bu) for bu, _, _ in exp_loader)
    assert total_exp == len(synth_train)


def test_models_forward_fast_scoring_and_checkpoint_roundtrip(tmp_path: Path):
    """Gate 2: Verify all model architectures, fast score_all equivalence, ALS fit, and pre-training fusion."""
    n_users, n_items = 12, 18
    rng = np.random.default_rng(7)
    train_df = pd.DataFrame(
        {
            "u": rng.integers(0, n_users, size=60),
            "i": rng.integers(0, n_items, size=60),
            "rating": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=60),
            "ts": np.arange(60),
        }
    ).drop_duplicates(["u", "i"])

    pop = Popularity(train_df, n_items)
    users = torch.arange(5, dtype=torch.long)
    assert pop.score_all(users).shape == (5, n_items)
    assert pop.predict_rating(train_df).shape == (len(train_df),)

    cfg = {"dim": 16, "gmf_dim": 8, "mlp_dim": 8, "layers": (16, 8), "dropout": 0.1, "reg": 0.1, "alpha": 10.0}
    for kind in ("mf", "als", "gmf", "mlp", "neumf", "neumf_explicit"):
        mdl = build_model(kind, n_users, n_items, cfg, mu=3.5)
        if kind == "als":
            csr = to_csr(train_df, n_users, n_items, positive_only=True)
            mdl.fit_als(csr, iters=3, seed=0)

        mdl.eval()
        scores = mdl.score_all(users)
        assert scores.shape == (5, n_items)

        # Verify fast score_all matches element-wise forward(u, i) to float32 precision
        grid_u = users.repeat_interleave(n_items)
        grid_i = torch.arange(n_items).repeat(len(users))
        with torch.no_grad():
            exact = mdl(grid_u, grid_i).view(len(users), n_items)
        assert torch.allclose(scores, exact, atol=1e-5), f"score_all mismatch for {kind}!"

        # Checkpoint round-trip
        ckpt = save_checkpoint(mdl, kind, cfg, 3.5, tmp_path)
        reloaded = load_model(ckpt, n_users, n_items, "cpu")
        assert torch.allclose(reloaded.score_all(users), scores, atol=1e-6)

    # Test NeuMF pre-trained fusion from GMF + MLP
    gmf_m = load_model(tmp_path / "gmf.pt", n_users, n_items)
    mlp_m = load_model(tmp_path / "mlp.pt", n_users, n_items)
    neumf_pt = build_model("neumf_pretrained", n_users, n_items, cfg)
    neumf_pt.load_pretrained_gmf_mlp(gmf_m, mlp_m, alpha=0.5)  # type: ignore[attr-defined]
    neumf_pt.eval()
    expected_fused = 0.5 * gmf_m.score_all(users) + 0.5 * mlp_m.score_all(users)
    assert torch.allclose(neumf_pt.score_all(users), expected_fused, atol=1e-5)


def test_metrics_exact_calculation():
    """Gate 2: Hand-verified test of RMSE, Precision@K, Recall@K, and NDCG@K."""
    assert np.isclose(rmse([3.0, 4.0], [3.0, 5.0]), np.sqrt(0.5))

    class DummyModel:
        def score_all(self, users):
            # 2 users, 5 items
            return torch.tensor(
                [
                    [10.0, 9.0, 8.0, 7.0, 1.0],  # item 0 is seen -> top-2 are [1, 2]
                    [1.0, 2.0, 9.0, 8.0, 10.0],  # none seen -> top-2 are [4, 2]
                ]
            )

    seen = sp.csr_matrix(([1], ([0], [0])), shape=(2, 5))
    # User 0 relevant: {1, 4} -> hits at rank 1 (item 1), miss at rank 2 (item 2): P@2=0.5, R@2=0.5, NDCG@2 = 1 / (1 + 1/log2(3))
    # User 1 relevant: {2}    -> miss at rank 1 (item 4), hit at rank 2 (item 2): P@2=0.5, R@2=1.0, NDCG@2 = (1/log2(3)) / 1
    rel = sp.csr_matrix(([1, 1, 1], ([0, 0, 1], [1, 4, 2])), shape=(2, 5))
    res = evaluate_ranking(DummyModel(), seen, rel, k=2)

    d2 = 1.0 / np.log2(3.0)
    expected_ndcg_u0 = 1.0 / (1.0 + d2)
    expected_ndcg_u1 = d2 / 1.0

    assert np.allclose(res["precision@2"], [0.5, 0.5])
    assert np.allclose(res["recall@2"], [0.5, 1.0])
    assert np.allclose(res["ndcg@2"], [expected_ndcg_u0, expected_ndcg_u1], atol=1e-6)

    # Also verify predict_ratings clamping
    df = pd.DataFrame({"u": [0, 1], "i": [0, 1]})
    m = build_model("mf", 2, 2, {"dim": 4}, mu=10.0)
    preds = predict_ratings(m, df)
    assert (preds <= 5.0).all() and (preds >= 1.0).all()


def test_streamlit_demo_app_end_to_end():
    """Gate 4: Execute the live Streamlit app via AppTest across personas and tabs."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("demo/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit raised exception on initial load: {at.exception}"
    assert len(at.title) == 1
    assert len(at.tabs) == 2
    assert len(at.dataframe) >= 6  # User history + Test GT + 4 model columns

    # Switch persona to Gray-Sheep / Niche User and adjust Top-K slider
    niche_options = [opt for opt in at.sidebar.selectbox[0].options if "Niche" in opt]
    assert niche_options, "Expected Niche / Gray-Sheep preset in sidebar"
    at.sidebar.selectbox[0].select(niche_options[0]).run()
    assert not at.exception, f"Streamlit raised exception after switching persona: {at.exception}"

    at.sidebar.slider[0].set_value(15).run()
    assert not at.exception, f"Streamlit raised exception after changing Top-K slider: {at.exception}"

