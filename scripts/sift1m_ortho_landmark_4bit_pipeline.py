"""
Two-stage SIFT1M pipeline: FWHT + 4-bit first, then landmark shortlist.

  1) L2-normalize (128-D)
  2) 1 round fast Hadamard (perm + ±1 signs + block WH) — SRHT stand-in
  3) Scalar-quantize DB to 4 bits/dim (Lloyd-Max) + RaBitQ alpha
  4) Project query float + DB reconstruction onto m orthonormal landmarks
  5) Scan top-K in landmark space
  6) Exact re-rank shortlist with original 128-D float IP

Reports shortlist coverage of true top-{1,10,64,128} (quality after exact re-rank).
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


def block_size(dim: int) -> int:
    """Lowest set bit of dim (same as C++ `dim & -dim`)."""
    return dim & -dim


def fwht_blocks(X: np.ndarray, block: int, inv_sqrt: float) -> np.ndarray:
    """Normalized Walsh–Hadamard on last axis in contiguous blocks of `block`."""
    n, dim = X.shape
    assert dim % block == 0 and block >= 2 and (block & (block - 1)) == 0
    out = X.astype(np.float32, copy=True)
    for offset in range(0, dim, block):
        blk = out[:, offset : offset + block]
        h = 1
        while h < block:
            for i in range(0, block, 2 * h):
                a = blk[:, i : i + h].copy()
                b = blk[:, i + h : i + 2 * h]
                blk[:, i : i + h] = a + b
                blk[:, i + h : i + 2 * h] = a - b
            h <<= 1
        blk *= inv_sqrt
    return out


def make_fwht_round(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One SRHT-style round: Fisher–Yates perm + ±1 signs + block WH size."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim).astype(np.int64)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    b = block_size(dim)
    if b < 2 or dim % b != 0 or (b & (b - 1)) != 0:
        raise ValueError(f"dim={dim} has unusable WH block size {b}")
    return perm, signs, b, float(1.0 / np.sqrt(b))


def apply_fwht_round(
    X: np.ndarray, perm: np.ndarray, signs: np.ndarray, block: int, inv_sqrt: float
) -> np.ndarray:
    """Apply 1 round: gather(perm)*signs -> block FWHT."""
    Y = (X[:, perm] * signs[np.newaxis, :]).astype(np.float32, copy=False)
    return fwht_blocks(Y, block, inv_sqrt)


def lloyd_max_levels(samples: np.ndarray, n_levels: int, n_iter: int = 25) -> np.ndarray:
    """1-D Lloyd-Max on flat samples -> sorted reconstruction levels."""
    lo, hi = np.quantile(samples, [0.001, 0.999])
    cents = np.linspace(lo, hi, n_levels, dtype=np.float64)
    x = samples.astype(np.float64, copy=False)
    for _ in range(n_iter):
        bounds = 0.5 * (cents[:-1] + cents[1:])
        codes = np.searchsorted(bounds, x, side="left")
        for i in range(n_levels):
            sel = x[codes == i]
            if sel.size:
                cents[i] = sel.mean()
        cents.sort()
    return cents.astype(np.float32)


def quantize_rows(Y: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scalar-quantize each coord to nearest Lloyd-Max level.
    Returns codes uint8 (n,d), reconstruction float32 (n,d), alpha float32 (n,).
    alpha = <y, yhat> / ||yhat||^2  (RaBitQ-style IP scale).
    """
    bounds = 0.5 * (levels[:-1].astype(np.float64) + levels[1:].astype(np.float64))
    flat = Y.reshape(-1).astype(np.float64, copy=False)
    codes = np.searchsorted(bounds, flat, side="left").astype(np.uint8)
    codes = codes.reshape(Y.shape)
    yhat = levels[codes]
    dots = np.einsum("ij,ij->i", Y, yhat, dtype=np.float64)
    denom = np.einsum("ij,ij->i", yhat, yhat, dtype=np.float64)
    alpha = (dots / np.maximum(denom, 1e-12)).astype(np.float32)
    return codes, yhat.astype(np.float32, copy=False), alpha


def batched_topk_ip(
    queries: np.ndarray,
    db: np.ndarray,
    k: int,
    batch: int,
    db_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Top-k indices by IP; if db_scale given, score = (q @ db.T) * db_scale."""
    nq = queries.shape[0]
    idx = np.empty((nq, k), dtype=np.int32)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        scores = queries[s:e] @ db.T
        if db_scale is not None:
            scores *= db_scale[np.newaxis, :]
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_sc = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_sc, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        print(f"  topk queries {e}/{nq}", flush=True)
    return idx


def coverage(gt: np.ndarray, cand: np.ndarray, true_k: int) -> float:
    hits = []
    for i in range(len(gt)):
        cset = set(cand[i].tolist())
        hits.append(sum(1 for x in gt[i, :true_k] if x in cset) / true_k)
    return float(np.mean(hits))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", type=Path, default=Path("data/sift-128-euclidean.hdf5"))
    p.add_argument("--m", type=int, default=96, help="landmark dims (after FWHT+quant)")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--shortlist", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--codebook-sample", type=int, default=200_000)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/.cache/ortho_landmark_cossim"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_levels = 1 << args.bits

    print("loading SIFT1M...", flush=True)
    with h5py.File(args.hdf5, "r") as f:
        db0 = l2_normalize(np.asarray(f["train"][:], dtype=np.float32))
        qq0 = l2_normalize(np.asarray(f["test"][:], dtype=np.float32))
    dim = db0.shape[1]
    print(f"db={db0.shape} queries={qq0.shape}", flush=True)

    gt_path = (
        args.out_dir
        / f"gt_topk_ip_l2norm_n{db0.shape[0]}_q{qq0.shape[0]}_k{args.shortlist}.npy"
    )
    if not gt_path.exists():
        raise SystemExit(
            f"missing GT cache {gt_path}; run sift1m_ortho_landmark_topk_eval.py --k {args.shortlist} first"
        )
    gt = np.load(gt_path)

    # --- 1) FWHT on full dim (before quant + landmarks) ---
    perm, signs, wh_block, inv_sqrt = make_fwht_round(dim, seed=args.seed + 1)
    print(f"FWHT round: dim={dim} block={wh_block}", flush=True)
    t0 = time.perf_counter()
    db_r = apply_fwht_round(db0, perm, signs, wh_block, inv_sqrt)
    qq_r = apply_fwht_round(qq0, perm, signs, wh_block, inv_sqrt)
    print(f"FWHT {time.perf_counter() - t0:.1f}s", flush=True)

    # --- 2) 4-bit quantize in rotated 128-D ---
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.codebook_sample, db_r.shape[0])
    sample_rows = db_r[rng.choice(db_r.shape[0], size=sample_n, replace=False)]
    levels = lloyd_max_levels(sample_rows.reshape(-1), n_levels)
    print(f"4-bit levels[{n_levels}]: min={levels[0]:.4f} max={levels[-1]:.4f}", flush=True)

    t0 = time.perf_counter()
    codes, db_hat, alpha = quantize_rows(db_r, levels)
    print(f"quantize DB {time.perf_counter() - t0:.1f}s", flush=True)
    bytes_per = dim * args.bits / 8
    print(
        f"storage: {bytes_per:.0f} B/vec codes (+4 B alpha) vs {dim * 4} B float32",
        flush=True,
    )

    # --- 3) landmarks AFTER FWHT+quant (in rotated space) ---
    L = ortho_landmarks(dim, args.m, args.seed)
    print(f"landmark project m={args.m}...", flush=True)
    t0 = time.perf_counter()
    # float landmark baseline on rotated float (IP-identical to original landmarks)
    db_f = db_r @ L
    qq_f = qq_r @ L
    # quantized path: scale reconstruction by alpha, then landmark
    db_q = (alpha[:, np.newaxis] * db_hat) @ L
    qq_q = qq_f  # asymmetric: query stays float in landmark space
    print(f"landmark project {time.perf_counter() - t0:.1f}s", flush=True)

    print("\nfloat landmark scan...", flush=True)
    t0 = time.perf_counter()
    pred_f = batched_topk_ip(qq_f, db_f, args.shortlist, args.batch)
    t_float = time.perf_counter() - t0

    print("\nFWHT+4bit then landmark scan...", flush=True)
    t0 = time.perf_counter()
    pred_q = batched_topk_ip(qq_q, db_q, args.shortlist, args.batch)
    t_quant = time.perf_counter() - t0

    true_ks = (1, 10, 64, 128)
    results = {
        "order": "fwht -> 4bit -> landmarks",
        "dim": dim,
        "m": args.m,
        "bits": args.bits,
        "n_levels": n_levels,
        "fwht_rounds": 1,
        "fwht_block": wh_block,
        "shortlist": args.shortlist,
        "n_db": int(db0.shape[0]),
        "n_query": int(qq0.shape[0]),
        "seed": args.seed,
        "bytes_per_vec_codes": bytes_per,
        "levels": levels.tolist(),
        "float_scan_s": t_float,
        "quant_scan_s": t_quant,
        "float_landmarks": {
            f"true_top{tk}_in_shortlist": coverage(gt, pred_f, tk) for tk in true_ks
        },
        "fwht_quant4_then_landmarks": {
            f"true_top{tk}_in_shortlist": coverage(gt, pred_q, tk) for tk in true_ks
        },
    }
    print(json.dumps({k: v for k, v in results.items() if k != "levels"}, indent=2), flush=True)
    out = args.out_dir / f"pipeline_fwht1_b{args.bits}_then_m{args.m}_sl{args.shortlist}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    np.save(args.out_dir / f"fwht_perm_d{dim}.npy", perm)
    np.save(args.out_dir / f"fwht_signs_d{dim}.npy", signs)
    np.save(args.out_dir / f"codebook_fwht1_b{args.bits}_d{dim}.npy", levels)
    np.save(args.out_dir / f"codes_fwht1_b{args.bits}_d{dim}.npy", codes)
    np.save(args.out_dir / f"alpha_fwht1_b{args.bits}_d{dim}.npy", alpha)
    np.save(args.out_dir / f"landmarks_after_fwht_m{args.m}_seed{args.seed}.npy", L)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
