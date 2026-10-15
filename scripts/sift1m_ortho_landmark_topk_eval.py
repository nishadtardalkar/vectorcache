"""
Full SIFT1M: compare original IP top-K vs orthonormal-landmark cossim top-K.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def ortho_landmarks(dim: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)), mode="reduced")
    return Q[:, :m].astype(np.float32)


def batched_topk(
    queries: np.ndarray,
    db: np.ndarray,
    k: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) each (nq, k), descending IP."""
    nq = queries.shape[0]
    idx = np.empty((nq, k), dtype=np.int32)
    sc = np.empty((nq, k), dtype=np.float32)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        scores = queries[s:e] @ db.T  # (b, N)
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_sc = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_sc, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        sc[s:e] = np.take_along_axis(part_sc, order, axis=1)
        print(f"  topk queries {e}/{nq}", flush=True)
    return idx, sc


def set_recall(gt: np.ndarray, pred: np.ndarray) -> float:
    hits = [len(set(gt[i].tolist()) & set(pred[i].tolist())) for i in range(len(gt))]
    return float(np.mean(hits) / gt.shape[1])


def true_nn_in_pred(gt: np.ndarray, pred: np.ndarray) -> float:
    nn = gt[:, 0]
    return float(np.mean([nn[i] in set(pred[i].tolist()) for i in range(len(gt))]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", type=Path, default=Path("data/sift-128-euclidean.hdf5"))
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--ms",
        type=int,
        nargs="+",
        default=[64, 96, 112, 128],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/.cache/ortho_landmark_cossim"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading SIFT1M...", flush=True)
    with h5py.File(args.hdf5, "r") as f:
        db = l2_normalize(np.asarray(f["train"][:], dtype=np.float32))
        qq = l2_normalize(np.asarray(f["test"][:], dtype=np.float32))
    print(f"db={db.shape} queries={qq.shape}", flush=True)

    gt_path = args.out_dir / f"gt_topk_ip_l2norm_n{db.shape[0]}_q{qq.shape[0]}_k{args.k}.npy"
    if gt_path.exists():
        print(f"loading cached GT {gt_path}", flush=True)
        gt = np.load(gt_path)
    else:
        print("computing full-dataset GT top-k (IP on L2-normalized)...", flush=True)
        t0 = time.perf_counter()
        gt, _ = batched_topk(qq, db, args.k, args.batch)
        np.save(gt_path, gt)
        print(f"GT done in {time.perf_counter() - t0:.1f}s -> {gt_path}", flush=True)

    results = []
    for m in args.ms:
        print(f"\nlandmarks m={m}", flush=True)
        t0 = time.perf_counter()
        L = ortho_landmarks(db.shape[1], m, args.seed)
        db_m = db @ L
        qq_m = qq @ L
        pred, _ = batched_topk(qq_m, db_m, args.k, args.batch)
        elapsed = time.perf_counter() - t0
        top1 = float(np.mean(gt[:, 0] == pred[:, 0]))
        nn_in = true_nn_in_pred(gt, pred)
        r_at_k = set_recall(gt, pred)
        row = {
            "m": m,
            "k": args.k,
            "n_db": int(db.shape[0]),
            "n_query": int(qq.shape[0]),
            "top1_agree": top1,
            "trueNN_in_pred_topk": nn_in,
            f"recall@{args.k}": r_at_k,
            "search_s": elapsed,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    out = args.out_dir / f"full_eval_k{args.k}.json"
    out.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
