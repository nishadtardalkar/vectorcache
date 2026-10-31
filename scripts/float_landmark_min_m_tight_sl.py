#!/usr/bin/env python3
"""Min float orthonormal landmark dims for R@1@k>=target with shortlist in (k, k+16).

Uses holdout split matching query-bench --npy, exact top-1 from VCEX cache,
landmark-space top-S (heap), no 4-bit quant.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def load_vcex_nn(path: Path) -> np.ndarray:
    """Return (nq,) true NN ids from VCEX exact top-k cache (column 0)."""
    raw = path.read_bytes()
    assert raw[:4] == b"VCEX"
    version = struct.unpack_from("<I", raw, 4)[0]
    assert version == 1
    nq, k, index_n, seed = struct.unpack_from("<QQQQ", raw, 8)
    off = 8 + 32
    # skip 3 length-prefixed strings
    for _ in range(3):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4 + n
    ids = np.frombuffer(raw, dtype="<u8", count=nq * k, offset=off).reshape(nq, k)
    return ids[:, 0].astype(np.int64)


def ortho_landmarks(dim: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim), dtype=np.float64), mode="reduced")
    return Q[:, :m].astype(np.float32)


def recall_nn_in_topk(db_m: np.ndarray, qq_m: np.ndarray, nn: np.ndarray, S: int, batch: int) -> float:
    nq = qq_m.shape[0]
    hits = 0
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        scores = qq_m[s:e] @ db_m.T
        # top-S indices
        part = np.argpartition(-scores, kth=S - 1, axis=1)[:, :S]
        for i, qi in enumerate(range(s, e)):
            hits += int(nn[qi] in part[i])
    return hits / nq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", type=Path, default=Path("data/openai-1536-101k.npy"))
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--query-limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shortlist-offset", type=int, default=15)
    ap.add_argument("--lo", type=int, default=64)
    ap.add_argument("--hi", type=int, default=1536)
    args = ap.parse_args()

    S = args.k + args.shortlist_offset
    assert 0 < args.shortlist_offset < 16

    gt_path = Path(
        f".cache/exact_topk/npy-openai-1536-101k-1f06b38ae289398c_train_n{args.limit}"
        f"_qs-holdout_ql{args.query_limit}_k{args.k}_seed0.bin"
    )
    print(f"loading GT {gt_path}", flush=True)
    nn = load_vcex_nn(gt_path)
    assert nn.shape[0] == args.query_limit

    print(f"loading {args.npy}...", flush=True)
    X = np.load(args.npy, mmap_mode="r")
    need = args.limit + args.query_limit
    X = np.array(X[:need], dtype=np.float32, copy=True)
    # l2 normalize like typical IP/cosine setup
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    db = X[: args.limit]
    qq = X[args.limit : args.limit + args.query_limit]
    dim = db.shape[1]
    print(f"db={db.shape} qq={qq.shape} S={S} target R@1@{args.k}>={args.target}", flush=True)

    # Full orthonormal basis once; prefix columns = landmarks. Project once.
    print("QR + full project...", flush=True)
    Lfull = ortho_landmarks(dim, dim, args.seed)
    db_full = db @ Lfull
    qq_full = qq @ Lfull
    print("projected", flush=True)

    cache: dict[int, float] = {}

    def eval_m(m: int) -> float:
        m = max(8, (m + 7) // 8 * 8)
        m = min(m, dim)
        if m in cache:
            print(f"  m={m} R={cache[m]:.4f} (cached)", flush=True)
            return cache[m]
        print(f"  scan m={m}...", flush=True)
        r = recall_nn_in_topk(db_full[:, :m], qq_full[:, :m], nn, S, args.batch)
        cache[m] = r
        print(f"  m={m} R@1@{args.k}(sl={S})={r:.4f}", flush=True)
        return r

    lo, hi = args.lo, args.hi
    r_lo = eval_m(lo)
    if r_lo >= args.target:
        while lo > 8:
            nxt = max(8, lo // 2)
            if eval_m(nxt) >= args.target:
                lo = nxt
            else:
                break
        hi = args.lo if args.lo != lo else lo + 64
    else:
        r_hi = eval_m(hi)
        if r_hi < args.target:
            print(f"FAIL: m={hi} R={r_hi:.4f} < {args.target}", flush=True)
            # show a few points for diagnosis
            for m in [256, 512, 768, 1024, 1280]:
                if m not in cache:
                    eval_m(m)
            return 1

    left, right = lo, hi
    best = right
    while left + 8 < right:
        mid = (left + right) // 2
        mid = max(8, (mid + 7) // 8 * 8)
        if mid <= left:
            mid = left + 8
        if mid >= right:
            break
        if eval_m(mid) >= args.target:
            best = mid
            right = mid
        else:
            left = mid

    # confirm best and best-8
    r_best = eval_m(best)
    below = max(8, best - 8)
    r_below = eval_m(below)
    print(
        f"\nRESULT float landmarks: min_m={best} shortlist={S} "
        f"(window=({args.k},{args.k + 16})) R@1@{args.k}={r_best:.4f} "
        f"(m={below} -> {r_below:.4f})",
        flush=True,
    )
    return 0 if r_best >= args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
