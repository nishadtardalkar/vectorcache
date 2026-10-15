"""
Sweep: FWHT -> 4-bit -> landmarks shortlist -> exact 128-D re-rank top-64.

Reports Recall@1@64 = fraction of queries whose true NN is in the final top-64
(equiv. true NN hit the shortlist, since re-rank is exact).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sift1m_ortho_landmark_4bit_pipeline import (  # noqa: E402
    apply_fwht_round,
    batched_topk_ip,
    l2_normalize,
    lloyd_max_levels,
    make_fwht_round,
    ortho_landmarks,
    quantize_rows,
)


def exact_rerank_top64(
    qq: np.ndarray,
    db: np.ndarray,
    cand: np.ndarray,
    out_k: int = 64,
) -> np.ndarray:
    """Exact IP re-rank of candidate ids -> (nq, out_k) ids."""
    nq, sl = cand.shape
    out = np.empty((nq, out_k), dtype=np.int32)
    for i in range(nq):
        ids = cand[i]
        scores = qq[i] @ db[ids].T
        if sl > out_k:
            part = np.argpartition(-scores, out_k - 1)[:out_k]
            part = part[np.argsort(-scores[part])]
        else:
            part = np.argsort(-scores)[:out_k]
        out[i] = ids[part]
    return out


def recall_at_1_at_k(gt_nn: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([gt_nn[i] in set(pred[i].tolist()) for i in range(len(gt_nn))]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", type=Path, default=Path("data/sift-128-euclidean.hdf5"))
    p.add_argument("--ms", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--shortlists", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--final-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
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

    # true NN from cached GT (any k>=1 file works; prefer existing 1024)
    gt_path = args.out_dir / f"gt_topk_ip_l2norm_n{db0.shape[0]}_q{qq0.shape[0]}_k1024.npy"
    if gt_path.exists():
        gt_nn = np.load(gt_path)[:, 0]
        print(f"loaded true NN from {gt_path}", flush=True)
    else:
        print("computing true NN (top-1 IP)...", flush=True)
        gt1 = batched_topk_ip(qq0, db0, 1, batch=64)
        gt_nn = gt1[:, 0]

    # FWHT on full dim
    perm, signs, wh_block, inv_sqrt = make_fwht_round(dim, seed=args.seed + 1)
    print(f"FWHT dim={dim} block={wh_block}", flush=True)
    t0 = time.perf_counter()
    db_r = apply_fwht_round(db0, perm, signs, wh_block, inv_sqrt)
    qq_r = apply_fwht_round(qq0, perm, signs, wh_block, inv_sqrt)
    print(f"FWHT {time.perf_counter() - t0:.1f}s", flush=True)

    # 4-bit quantize DB
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.codebook_sample, db_r.shape[0])
    sample_rows = db_r[rng.choice(db_r.shape[0], size=sample_n, replace=False)]
    levels = lloyd_max_levels(sample_rows.reshape(-1), n_levels)
    t0 = time.perf_counter()
    _codes, db_hat, alpha = quantize_rows(db_r, levels)
    db_scaled = alpha[:, np.newaxis] * db_hat
    print(f"quantize {time.perf_counter() - t0:.1f}s  levels[{n_levels}]", flush=True)

    results = []
    for m in args.ms:
        L = ortho_landmarks(dim, m, args.seed)
        qq_m = qq_r @ L
        db_m = db_scaled @ L
        print(f"\n=== m={m} landmark scan space {qq_m.shape[1]} ===", flush=True)

        for sl in args.shortlists:
            print(f"shortlist={sl}...", flush=True)
            t0 = time.perf_counter()
            cand = batched_topk_ip(qq_m, db_m, sl, args.batch)
            scan_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            final = exact_rerank_top64(qq0, db0, cand, out_k=args.final_k)
            rerank_s = time.perf_counter() - t1

            r1_at_64 = recall_at_1_at_k(gt_nn, final)
            nn_in_sl = float(np.mean([gt_nn[i] in set(cand[i].tolist()) for i in range(len(gt_nn))]))
            row = {
                "m": m,
                "bits": args.bits,
                "shortlist": sl,
                "final_k": args.final_k,
                "recall@1@64": r1_at_64,
                "trueNN_in_shortlist": nn_in_sl,
                "scan_s": scan_s,
                "rerank_s": rerank_s,
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    out = {
        "order": "fwht -> 4bit -> landmarks -> exact rerank",
        "dim": dim,
        "fwht_block": wh_block,
        "n_db": int(db0.shape[0]),
        "n_query": int(qq0.shape[0]),
        "seed": args.seed,
        "results": results,
    }
    path = args.out_dir / "sweep_fwht4bit_landmark_r1at64.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}", flush=True)

    # compact table
    print("\nRecall@1@64 table (rows=m, cols=shortlist):", flush=True)
    header = "m\\" + "\t".join(str(s) for s in args.shortlists)
    print(header, flush=True)
    for m in args.ms:
        vals = []
        for sl in args.shortlists:
            r = next(x for x in results if x["m"] == m and x["shortlist"] == sl)
            vals.append(f"{r['recall@1@64']:.4f}")
        print(f"{m}\t" + "\t".join(vals), flush=True)


if __name__ == "__main__":
    main()
