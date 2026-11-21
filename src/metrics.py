"""Unified evaluation metrics for rating prediction (RMSE) and full-catalog Top-K ranking
(Precision@K, Recall@K, NDCG@K, HitRate@K)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


@torch.no_grad()
def predict_ratings(model, df, device: str = "cpu", batch: int = 65536) -> np.ndarray:
    """Predict explicit 1-5 star ratings in mini-batches, clamped to [1.0, 5.0]."""
    model.eval()
    out = []
    u_all = df.u.values.astype(np.int64)
    i_all = df.i.values.astype(np.int64)
    for s in range(0, len(df), batch):
        u = torch.as_tensor(u_all[s : s + batch], device=device)
        i = torch.as_tensor(i_all[s : s + batch], device=device)
        out.append(model(u, i).clamp(1.0, 5.0).cpu().numpy())
    return np.concatenate(out)


def rmse(pred, true) -> float:
    """Root Mean Squared Error between predicted and ground-truth ratings."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


@torch.no_grad()
def evaluate_ranking(
    model,
    seen: sp.csr_matrix,
    relevant: sp.csr_matrix,
    k: int = 10,
    batch: int = 128,
    device: str = "cpu",
    max_users: int | None = None,
    seed: int = 0,
) -> dict:
    """Full-catalog Top-K ranking evaluation:
    - `seen`: CSR matrix of historical items masked out from candidate recommendations (-inf).
    - `relevant`: CSR matrix of ground-truth positive items (rating >= 4 in target split).
    Returns per-user arrays and mean Precision@K, Recall@K, NDCG@K, HitRate@K."""
    users = np.where(np.diff(relevant.indptr) > 0)[0]
    if max_users is not None and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))

    disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    idcg_table = torch.cumsum(disc, dim=0)

    P, R, N, H, TOP = [], [], [], [], []
    for s in range(0, len(users), batch):
        b = users[s : s + batch]
        scores = model.score_all(torch.as_tensor(b, dtype=torch.long, device=device))
        scores = scores.float().cpu().clone()
        scores[torch.as_tensor(seen[b].toarray() > 0)] = -float("inf")

        top = scores.topk(k, dim=1).indices
        rel = torch.as_tensor(relevant[b].toarray() > 0)
        hits = rel.gather(1, top).float()
        n_rel = rel.sum(dim=1).float()

        hit_cnt = hits.sum(dim=1)
        P.append(hit_cnt / k)
        R.append(hit_cnt / n_rel)
        H.append((hit_cnt > 0).float())
        dcg = (hits * disc).sum(dim=1)
        idcg = idcg_table[(n_rel.clamp(max=k) - 1).long()]
        N.append(dcg / idcg)
        TOP.append(top)

    P_arr, R_arr, N_arr, H_arr, TOP_arr = (torch.cat(x).numpy() for x in (P, R, N, H, TOP))
    return {
        "users": users,
        "top": TOP_arr,
        f"precision@{k}": P_arr,
        f"recall@{k}": R_arr,
        f"ndcg@{k}": N_arr,
        f"hitrate@{k}": H_arr,
        "mean": {
            f"precision@{k}": float(P_arr.mean()),
            f"recall@{k}": float(R_arr.mean()),
            f"ndcg@{k}": float(N_arr.mean()),
            f"hitrate@{k}": float(H_arr.mean()),
        },
    }
