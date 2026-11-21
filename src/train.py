"""Training, hyperparameter tuning, and checkpointing pipeline for:
- Biased Matrix Factorization / Funk-SVD (Explicit MSE + L2)
- Implicit Alternating Least Squares / ALS (Hu et al., 2008)
- Neural Collaborative Filtering: GMF, MLP, NeuMF (Scratch & Pre-trained), and Explicit NeuMF

Usage examples:
  python -m src.train --model mf --tune
  python -m src.train --model als
  python -m src.train --model gmf --dim 64
  python -m src.train --model mlp --dim 64
  python -m src.train --model neumf --dim 64
  python -m src.train --model neumf_pretrained --dim 64
  python -m src.train --model neumf_explicit --dim 64
  python -m src.train --model all --tune
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import POS_THRESHOLD, explicit_loader, load_movielens, ncf_loader, to_csr
from .metrics import evaluate_ranking, predict_ratings, rmse
from .models import ImplicitALS, NCF, build_model, load_model


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_json_dict(path: Path, key: str, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[key] = value
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def save_checkpoint(model, kind: str, cfg: dict, mu: float, out_dir: str | Path, extra: dict | None = None) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_path / f"{kind}.pt"
    torch.save(
        {"kind": kind, "cfg": cfg, "mu": float(mu), "state": model.state_dict(), **(extra or {})},
        ckpt_file,
    )
    return ckpt_file


def train_mf(
    data,
    dim: int,
    lr: float,
    reg: float,
    epochs: int,
    batch: int,
    device: str,
    patience: int = 3,
    verbose: bool = True,
) -> tuple[torch.nn.Module, float, list[dict]]:
    mu = float(data.train.rating.mean())
    model = build_model("mf", data.n_users, data.n_items, {"dim": dim}, mu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse, best_state, bad = 1e9, None, 0
    history: list[dict] = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for u, i, r in explicit_loader(data.train, batch):
            u, i, r = u.to(device), i.to(device), r.to(device)
            loss = F.mse_loss(model(u, i), r) + reg * model.l2(u, i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(u)

        train_loss = tot_loss / len(data.train)
        val_rmse = rmse(predict_ratings(model, data.val, device), data.val.rating.values)
        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
                "elapsed_sec": round(time.time() - t_start, 2),
            }
        )
        if verbose:
            print(f"[mf d={dim} lr={lr} reg={reg}] ep {ep:02d} | loss {train_loss:.4f} | val RMSE {val_rmse:.4f}")

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_rmse, history


def train_als(
    data,
    dim: int = 64,
    reg: float = 0.1,
    alpha: float = 15.0,
    iters: int = 12,
    val_users: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[ImplicitALS, float, list[dict]]:
    # Positive training interactions weighted by normalized rating excess (rating >= 4 -> 1..2)
    pos = data.train[data.train.rating >= POS_THRESHOLD]
    vals = (pos.rating.values - (POS_THRESHOLD - 1.0)).astype(np.float32)  # 4->1.0, 5->2.0
    import scipy.sparse as sp

    train_pos_csr = sp.csr_matrix((vals, (pos.u.values, pos.i.values)), shape=(data.n_users, data.n_items))
    seen_val = to_csr(data.train, data.n_users, data.n_items)
    rel_val = to_csr(data.val, data.n_users, data.n_items, positive_only=True)

    model = ImplicitALS(data.n_users, data.n_items, dim=dim, reg=reg, alpha=alpha)
    best_ndcg = -1.0
    best_state = None
    t_start = time.time()

    def _cb(it: int, mdl: ImplicitALS):
        nonlocal best_ndcg, best_state
        m = evaluate_ranking(mdl, seen_val, rel_val, k=10, device="cpu", max_users=val_users)["mean"]
        if verbose:
            print(
                f"[als d={dim} reg={reg} alpha={alpha}] iter {it:02d} | "
                f"val NDCG@10 {m['ndcg@10']:.4f} | P@10 {m['precision@10']:.4f} | R@10 {m['recall@10']:.4f}"
            )
        if m["ndcg@10"] > best_ndcg:
            best_ndcg = m["ndcg@10"]
            best_state = copy.deepcopy(mdl.state_dict())
        return {
            "epoch": it,
            "val_ndcg@10": float(m["ndcg@10"]),
            "val_precision@10": float(m["precision@10"]),
            "val_recall@10": float(m["recall@10"]),
            "elapsed_sec": round(time.time() - t_start, 2),
        }

    history = model.fit_als(train_pos_csr, iters=iters, seed=seed, callback=_cb)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ndcg, history


def train_ncf(
    data,
    kind: str,
    cfg: dict,
    lr: float,
    epochs: int,
    batch: int,
    n_neg: int,
    device: str,
    patience: int = 4,
    val_users: int = 2000,
    pretrained_dir: str | Path | None = None,
    weight_decay: float = 1e-6,
    verbose: bool = True,
) -> tuple[NCF, float, list[dict]]:
    model: NCF = build_model(kind, data.n_users, data.n_items, cfg).to(device)  # type: ignore[assignment]

    if kind == "neumf_pretrained":
        assert pretrained_dir is not None, "pretrained_dir is required for neumf_pretrained"
        gmf_ckpt = Path(pretrained_dir) / "gmf.pt"
        mlp_ckpt = Path(pretrained_dir) / "mlp.pt"
        if not gmf_ckpt.exists() or not mlp_ckpt.exists():
            raise FileNotFoundError("Pre-trained gmf.pt and mlp.pt must exist before training neumf_pretrained.")
        gmf_m = load_model(gmf_ckpt, data.n_users, data.n_items, device)
        mlp_m = load_model(mlp_ckpt, data.n_users, data.n_items, device)
        model.load_pretrained_gmf_mlp(gmf_m, mlp_m, alpha=0.5)  # type: ignore[arg-type]
        # He et al. (2017) recommend SGD or lower-LR Adam when fine-tuning pre-trained NeuMF
        opt: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    seen_val = to_csr(data.train, data.n_users, data.n_items)
    rel_val = to_csr(data.val, data.n_users, data.n_items, positive_only=True)
    seen_keys = np.sort(data.train.u.values.astype(np.int64) * data.n_items + data.train.i.values.astype(np.int64))
    rng = np.random.default_rng(0)

    best_ndcg, best_state, bad = -1.0, None, 0
    history: list[dict] = []
    t_start = time.time()

    # Evaluate initial pre-trained state at epoch 0 if neumf_pretrained
    if kind == "neumf_pretrained":
        m0 = evaluate_ranking(model, seen_val, rel_val, k=10, device=device, max_users=val_users)["mean"]
        best_ndcg = m0["ndcg@10"]
        best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[{kind}] init (fused GMF+MLP) | val NDCG@10 {m0['ndcg@10']:.4f}")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, n_samples = 0.0, 0
        for u, i, y in ncf_loader(data.train, data.n_items, n_neg, batch, rng, seen_keys=seen_keys):
            u, i, y = u.to(device), i.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(u, i), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(u)
            n_samples += len(u)

        train_loss = tot_loss / n_samples
        m = evaluate_ranking(model, seen_val, rel_val, k=10, device=device, max_users=val_users)["mean"]
        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_ndcg@10": float(m["ndcg@10"]),
                "val_precision@10": float(m["precision@10"]),
                "val_recall@10": float(m["recall@10"]),
                "elapsed_sec": round(time.time() - t_start, 2),
            }
        )
        if verbose:
            print(
                f"[{kind}] ep {ep:02d} | loss {train_loss:.4f} | "
                f"val NDCG@10 {m['ndcg@10']:.4f} | P@10 {m['precision@10']:.4f} | R@10 {m['recall@10']:.4f}"
            )
        if m["ndcg@10"] > best_ndcg + 1e-5:
            best_ndcg = m["ndcg@10"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_ndcg, history


def train_ncf_explicit(
    data,
    cfg: dict,
    lr: float,
    epochs: int,
    batch: int,
    device: str,
    patience: int = 3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> tuple[NCF, float, list[dict]]:
    """Train Explicit NeuMF with MSE loss on 1-5 star ratings for direct RMSE comparison."""
    mu = float(data.train.rating.mean())
    model: NCF = build_model("neumf_explicit", data.n_users, data.n_items, cfg, mu=mu).to(device)  # type: ignore[assignment]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse, best_state, bad = 1e9, None, 0
    history: list[dict] = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for u, i, r in explicit_loader(data.train, batch):
            u, i, r = u.to(device), i.to(device), r.to(device)
            loss = F.mse_loss(model(u, i), r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(u)

        train_loss = tot_loss / len(data.train)
        val_rmse = rmse(predict_ratings(model, data.val, device), data.val.rating.values)
        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
                "elapsed_sec": round(time.time() - t_start, 2),
            }
        )
        if verbose:
            print(f"[neumf_explicit] ep {ep:02d} | loss {train_loss:.4f} | val RMSE {val_rmse:.4f}")

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_rmse, history


def run_single_model(a, data, device: str, kind: str) -> None:
    history_file = Path("results") / "training_history.json"
    t0 = time.time()

    if kind == "mf":
        grid = (
            list(itertools.product([32, 64, 128], [0.005, 0.002], [0.005, 0.02]))
            if a.tune
            else [(a.dim, a.lr or 0.005, a.reg)]
        )
        best = (1e9, None, None, None)
        grid_records = []
        for dim, lr, reg in grid:
            set_seed(a.seed)
            model, v, hist = train_mf(data, dim, lr, reg, a.epochs, a.batch, device, verbose=not a.tune)
            print(f"[MF Grid] dim={dim:3d} lr={lr:.4f} reg={reg:.3f} -> best val RMSE {v:.4f} (epochs={len(hist)})")
            grid_records.append({"dim": dim, "lr": lr, "reg": reg, "best_val_rmse": float(v), "epochs_run": len(hist)})
            if v < best[0]:
                best = (v, {"dim": dim, "lr": lr, "reg": reg}, model, hist)

        v, cfg, model, best_hist = best
        print(f"Best MF configuration: {cfg} -> val RMSE {v:.4f}")
        if a.tune:
            Path("results").mkdir(parents=True, exist_ok=True)
            (Path("results") / "mf_grid_search.json").write_text(json.dumps(grid_records, indent=2), encoding="utf-8")
        save_checkpoint(
            model,
            "mf",
            {"dim": cfg["dim"]},
            float(data.train.rating.mean()),
            a.out_dir,
            {"hparams": cfg, "val_rmse": v},
        )
        _update_json_dict(history_file, "mf", {"hparams": cfg, "best_val_rmse": v, "epochs": best_hist})

    elif kind == "als":
        als_grid = [(64, 5.0, 15.0), (64, 15.0, 20.0), (128, 10.0, 20.0)] if a.tune else [(a.dim, 10.0, 15.0)]
        best_als = (-1.0, None, None, None)
        for dim, reg, alpha in als_grid:
            model, v, hist = train_als(data, dim=dim, reg=reg, alpha=alpha, iters=12, seed=a.seed)
            if v > best_als[0]:
                best_als = (v, {"dim": dim, "reg": reg, "alpha": alpha}, model, hist)
        v, cfg, model, best_hist = best_als
        print(f"Best ALS configuration: {cfg} -> val NDCG@10 {v:.4f}")
        save_checkpoint(model, "als", cfg, 0.0, a.out_dir, {"val_ndcg": v})
        _update_json_dict(history_file, "als", {"hparams": cfg, "best_val_ndcg@10": v, "epochs": best_hist})

    elif kind == "neumf_explicit":
        set_seed(a.seed)
        cfg = {
            "gmf_dim": a.dim // 2,
            "mlp_dim": a.dim // 2,
            "layers": [a.dim, a.dim // 2, a.dim // 4],
            "dropout": a.dropout,
        }
        mu = float(data.train.rating.mean())
        model, v, hist = train_ncf_explicit(data, cfg, a.lr or 0.002, min(a.epochs, 20), a.batch, device)
        print(f"[neumf_explicit] Best val RMSE = {v:.4f}")
        save_checkpoint(model, "neumf_explicit", cfg, mu, a.out_dir, {"val_rmse": v})
        _update_json_dict(history_file, "neumf_explicit", {"cfg": cfg, "best_val_rmse": v, "epochs": hist})

    else:
        set_seed(a.seed)
        cfg = {
            "gmf_dim": a.dim // 2,
            "mlp_dim": a.dim // 2,
            "layers": [a.dim, a.dim // 2, a.dim // 4],
            "dropout": a.dropout,
        }
        default_lr = 0.0005 if kind == "neumf_pretrained" else 0.001
        max_ep = min(a.epochs, 15) if kind == "neumf_pretrained" else a.epochs
        model, v, hist = train_ncf(
            data,
            kind,
            cfg,
            a.lr or default_lr,
            max_ep,
            a.batch,
            a.neg,
            device,
            pretrained_dir=a.out_dir,
        )
        print(f"[{kind}] Best val NDCG@10 = {v:.4f}")
        save_checkpoint(model, kind, cfg, 0.0, a.out_dir, {"val_ndcg": v})
        _update_json_dict(history_file, kind, {"cfg": cfg, "best_val_ndcg@10": v, "epochs": hist})

    print(f"Finished {kind} in {time.time() - t0:.1f}s -> {a.out_dir}/{kind}.pt\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Train MovieLens-1M recommender models.")
    p.add_argument(
        "--model",
        required=True,
        choices=["mf", "als", "gmf", "mlp", "neumf", "neumf_pretrained", "neumf_explicit", "all"],
    )
    p.add_argument("--data_dir", default="data/ml-1m")
    p.add_argument("--out_dir", default="saved_models")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--neg", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tune", action="store_true", help="Run hyperparameter grid search for MF/ALS.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MovieLens-1M from {a.data_dir} (device={device}) ...")
    data = load_movielens(a.data_dir)

    if a.model == "all":
        order = ["mf", "als", "gmf", "mlp", "neumf", "neumf_pretrained", "neumf_explicit"]
        for kind in order:
            run_single_model(a, data, device, kind)
    else:
        run_single_model(a, data, device, a.model)


if __name__ == "__main__":
    main()
