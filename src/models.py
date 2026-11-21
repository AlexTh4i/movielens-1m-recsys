"""Recommender models in PyTorch:
- Popularity baseline (IMDB Weighted Rating for Top-N + Item Mean for RMSE)
- Biased Matrix Factorization / Funk-SVD (Explicit MSE + L2)
- Implicit Alternating Least Squares / ALS (Hu, Koren & Volinsky, 2008)
- Neural Collaborative Filtering: GMF, MLP, NeuMF (Scratch & Pre-trained), and Explicit NeuMF (He et al., 2017)

Every model implements `score_all(users) -> Tensor[len(users), n_items]` for unified full-catalog ranking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# ------------------------------------------------------------------ Popularity
class Popularity:
    """Non-personalized baseline using IMDB Weighted Rating:
    WR_i = (v_i / (v_i + m)) * R_i + (m / (v_i + m)) * C."""

    def __init__(self, train, n_items: int, quantile: float = 0.8):
        g = train.groupby("i").rating.agg(["count", "mean"])
        v = np.zeros(n_items, dtype=np.float64)
        R = np.zeros(n_items, dtype=np.float64)
        v[g.index.values] = g["count"].values
        R[g.index.values] = g["mean"].values

        self.global_mean = float(train.rating.mean())
        self.counts = v
        m = float(np.quantile(v[v > 0], quantile)) if (v > 0).any() else 1.0
        wr = (v / (v + m)) * R + (m / (v + m)) * self.global_mean
        wr[v == 0] = -1.0
        self.wr = torch.as_tensor(wr, dtype=torch.float32)
        self.item_mean = np.where(v > 0, R, self.global_mean).astype(np.float32)

    def predict_rating(self, df) -> np.ndarray:
        """Baseline rating prediction: item mean rating (fallback to global mean)."""
        return self.item_mean[df.i.values]

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        return self.wr.unsqueeze(0).expand(len(users), -1).clone()


# ----------------------------------------------------------- Biased SVD / MF
class MF(nn.Module):
    """Biased Matrix Factorization (Funk-SVD / Koren, 2009):
    r_hat(u, i) = mu + b_u + b_i + <p_u, q_i>."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, mu: float = 0.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.P = nn.Embedding(n_users, dim)
        self.Q = nn.Embedding(n_items, dim)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))

        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (
            self.mu
            + self.bu(u).squeeze(-1)
            + self.bi(i).squeeze(-1)
            + (self.P(u) * self.Q(i)).sum(-1)
        )

    def l2(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Batch-level L2 regularization over active embeddings and biases."""
        return (
            (self.P(u) ** 2).sum(-1)
            + (self.Q(i) ** 2).sum(-1)
            + self.bu(u).squeeze(-1) ** 2
            + self.bi(i).squeeze(-1) ** 2
        ).mean()

    @torch.no_grad()
    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.mu + self.bu(users) + self.bi.weight.T + self.P(users) @ self.Q.weight.T


# ----------------------------------------------------------- Implicit ALS (MF)
class ImplicitALS(nn.Module):
    """Implicit Alternating Least Squares Matrix Factorization (Hu, Koren & Volinsky, ICDM 2008).
    Minimizes sum_{u,i} c_{ui} (p_{ui} - x_u^T y_i)^2 + reg * (||X||_F^2 + ||Y||_F^2),
    where p_{ui} = 1 for positive interactions and c_{ui} = 1 + alpha * r_{ui}."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, reg: float = 0.1, alpha: float = 15.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.reg = reg
        self.alpha = alpha
        self.U = nn.Parameter(torch.empty(n_users, dim), requires_grad=False)
        self.V = nn.Parameter(torch.empty(n_items, dim), requires_grad=False)
        nn.init.normal_(self.U, std=0.05)
        nn.init.normal_(self.V, std=0.05)

    def fit_als(
        self,
        user_item_csr: sp.csr_matrix,
        iters: int = 12,
        seed: int = 42,
        callback=None,
    ) -> list[dict]:
        """Run alternating ridge-regression updates on sparse confidence matrix."""
        rng = np.random.default_rng(seed)
        X = rng.normal(0.0, 0.05, size=(self.n_users, self.dim)).astype(np.float32)
        Y = rng.normal(0.0, 0.05, size=(self.n_items, self.dim)).astype(np.float32)

        C_ui = user_item_csr.tocsr()
        C_iu = C_ui.T.tocsr()
        eye = np.eye(self.dim, dtype=np.float32) * np.float32(self.reg)
        history = []

        for it in range(1, iters + 1):
            # Update user factors X given Y
            YtY = Y.T @ Y + eye
            for u in range(self.n_users):
                s, e = C_ui.indptr[u], C_ui.indptr[u + 1]
                if s == e:
                    X[u] = 0.0
                    continue
                idx = C_ui.indices[s:e]
                conf_excess = self.alpha * C_ui.data[s:e]  # c_{ui} - 1
                Y_u = Y[idx]
                A = YtY + (Y_u.T * conf_excess) @ Y_u
                b = Y_u.T @ (1.0 + conf_excess)
                X[u] = np.linalg.solve(A, b)

            # Update item factors Y given X
            XtX = X.T @ X + eye
            for i in range(self.n_items):
                s, e = C_iu.indptr[i], C_iu.indptr[i + 1]
                if s == e:
                    Y[i] = 0.0
                    continue
                idx = C_iu.indices[s:e]
                conf_excess = self.alpha * C_iu.data[s:e]
                X_i = X[idx]
                A = XtX + (X_i.T * conf_excess) @ X_i
                b = X_i.T @ (1.0 + conf_excess)
                Y[i] = np.linalg.solve(A, b)

            self.U.data.copy_(torch.from_numpy(X))
            self.V.data.copy_(torch.from_numpy(Y))
            if callback is not None:
                metrics = callback(it, self)
                if metrics is not None:
                    history.append(metrics)
        return history

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.U[u] * self.V[i]).sum(-1)

    @torch.no_grad()
    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.U[users] @ self.V.T


# ------------------------------------------------------------------------ NCF
class NCF(nn.Module):
    """Neural Collaborative Filtering (He et al., WWW 2017).
    Supports GMF, MLP, NeuMF (scratch or pre-trained), and Explicit-MSE NeuMF."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        layers: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_mlp: bool = True,
        explicit: bool = False,
        mu: float = 0.0,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.layers = tuple(layers)
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        self.explicit = explicit
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))

        out_in = 0
        if use_gmf:
            self.gmf_u = nn.Embedding(n_users, gmf_dim)
            self.gmf_i = nn.Embedding(n_items, gmf_dim)
            out_in += gmf_dim
        if use_mlp:
            self.mlp_u = nn.Embedding(n_users, mlp_dim)
            self.mlp_i = nn.Embedding(n_items, mlp_dim)
            mods: list[nn.Module] = []
            d = 2 * mlp_dim
            for h in layers:
                mods += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.mlp = nn.Sequential(*mods)
            out_in += layers[-1]

        self.out = nn.Linear(out_in, 1)
        for mod in self.modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=0.05)

    def load_pretrained_gmf_mlp(self, gmf_model: "NCF", mlp_model: "NCF", alpha: float = 0.5) -> None:
        """Initialize NeuMF from pre-trained GMF and MLP weights (He et al., 2017, Sec. 3.4.1)."""
        assert self.use_gmf and self.use_mlp, "Pre-training fusion requires both GMF and MLP branches."
        with torch.no_grad():
            self.gmf_u.weight.copy_(gmf_model.gmf_u.weight)
            self.gmf_i.weight.copy_(gmf_model.gmf_i.weight)
            self.mlp_u.weight.copy_(mlp_model.mlp_u.weight)
            self.mlp_i.weight.copy_(mlp_model.mlp_i.weight)
            self.mlp.load_state_dict(mlp_model.mlp.state_dict())
            self.out.weight.copy_(
                torch.cat([alpha * gmf_model.out.weight, (1.0 - alpha) * mlp_model.out.weight], dim=1)
            )
            self.out.bias.copy_(alpha * gmf_model.out.bias + (1.0 - alpha) * mlp_model.out.bias)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.use_gmf:
            feats.append(self.gmf_u(u) * self.gmf_i(i))
        if self.use_mlp:
            feats.append(self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=-1)))
        raw = self.out(torch.cat(feats, dim=-1)).squeeze(-1)
        return self.mu + raw if self.explicit else raw

    @torch.no_grad()
    def score_all(self, users: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """Fast full-catalog scoring exploiting linear decomposition of the first MLP layer."""
        self.eval()
        N = self.n_items
        if self.use_gmf and not self.use_mlp:
            # Pure GMF reduces to a weighted dot product: (gmf_u * w_out) @ gmf_i^T + b_out
            w = self.out.weight.squeeze(0)
            scores = (self.gmf_u(users) * w) @ self.gmf_i.weight.T + self.out.bias
            return self.mu + scores if self.explicit else scores

        # Precompute item-side projections once across chunks
        if self.use_mlp:
            lin0: nn.Linear = self.mlp[0]  # type: ignore[assignment]
            w_u = lin0.weight[:, : self.mlp_dim]
            w_i = lin0.weight[:, self.mlp_dim :]
            proj_i = self.mlp_i.weight @ w_i.T + lin0.bias  # [N, h0]
            rest_mlp = self.mlp[1:]

        out = []
        for s in range(0, len(users), chunk):
            u = users[s : s + chunk]
            B = len(u)
            feats = []
            if self.use_gmf:
                gu = self.gmf_u(u)  # [B, d]
                feats.append((gu.unsqueeze(1) * self.gmf_i.weight.unsqueeze(0)).reshape(B * N, -1))
            if self.use_mlp:
                proj_u = self.mlp_u(u) @ w_u.T  # [B, h0]
                h = (proj_u.unsqueeze(1) + proj_i.unsqueeze(0)).reshape(B * N, -1)
                feats.append(rest_mlp(h))
            raw = self.out(torch.cat(feats, dim=-1)).view(B, N)
            out.append(self.mu + raw if self.explicit else raw)
        return torch.cat(out, dim=0)


def build_model(kind: str, n_users: int, n_items: int, cfg: dict, mu: float = 0.0) -> nn.Module:
    if kind == "mf":
        return MF(n_users, n_items, dim=cfg.get("dim", 64), mu=mu)
    if kind == "als":
        return ImplicitALS(
            n_users,
            n_items,
            dim=cfg.get("dim", 64),
            reg=cfg.get("reg", 0.1),
            alpha=cfg.get("alpha", 15.0),
        )
    flags_map = {
        "gmf": (True, False, False),
        "mlp": (False, True, False),
        "neumf": (True, True, False),
        "neumf_pretrained": (True, True, False),
        "neumf_explicit": (True, True, True),
    }
    use_gmf, use_mlp, explicit = flags_map[kind]
    return NCF(
        n_users,
        n_items,
        gmf_dim=cfg.get("gmf_dim", 32),
        mlp_dim=cfg.get("mlp_dim", 32),
        layers=tuple(cfg.get("layers", (64, 32, 16))),
        dropout=cfg.get("dropout", 0.2),
        use_gmf=use_gmf,
        use_mlp=use_mlp,
        explicit=explicit,
        mu=mu,
    )


def load_model(path: str | Path, n_users: int, n_items: int, device: str = "cpu") -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["kind"], n_users, n_items, ck["cfg"], ck.get("mu", 0.0)).to(device)
    model.load_state_dict(ck["state"])
    return model.eval()
