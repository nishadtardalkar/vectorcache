"""Per-query latency breakdown for FWHT->4bit->landmark shortlist->exact re-rank."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sift1m_fwht4bit_landmark_sweep import exact_rerank_top64, recall_at_1_at_k
from sift1m_ortho_landmark_4bit_pipeline import (
    apply_fwht_round,
    batched_topk_ip,
    l2_normalize,
    lloyd_max_levels,
    make_fwht_round,
    ortho_landmarks,
    quantize_rows,
)


def main() -> None:
    out = Path("data/.cache/ortho_landmark_cossim")
    with h5py.File("data/sift-128-euclidean.hdf5", "r") as f:
        db0 = l2_normalize(np.asarray(f["train"][:], dtype=np.float32))
        qq0 = l2_normalize(np.asarray(f["test"][:], dtype=np.float32))
    nq, dim = qq0.shape[0], db0.shape[1]
    gt_nn = np.load(out / "gt_topk_ip_l2norm_n1000000_q10000_k1024.npy")[:, 0]

    seed, bits, batch = 0, 4, 16
    perm, signs, wh_block, inv_sqrt = make_fwht_round(dim, seed=seed + 1)

    t0 = time.perf_counter()
    qq_r = apply_fwht_round(qq0, perm, signs, wh_block, inv_sqrt)
    fwht_q_s = time.perf_counter() - t0
    db_r = apply_fwht_round(db0, perm, signs, wh_block, inv_sqrt)

    rng = np.random.default_rng(seed)
    sample = db_r[rng.choice(len(db_r), 200_000, replace=False)]
    levels = lloyd_max_levels(sample.reshape(-1), 1 << bits)
    _, db_hat, alpha = quantize_rows(db_r, levels)
    db_scaled = alpha[:, None] * db_hat

    rows = []
    for m in (64, 96):
        L = ortho_landmarks(dim, m, seed)
        t0 = time.perf_counter()
        qq_m = qq_r @ L
        proj_q_s = time.perf_counter() - t0
        db_m = db_scaled @ L
        for S in (1024, 4096, 16384):
            print(f"\n=== m={m} shortlist={S} ===", flush=True)
            t0 = time.perf_counter()
            cand = batched_topk_ip(qq_m, db_m, S, batch)
            scan_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            final = exact_rerank_top64(qq0, db0, cand, 64)
            rerank_s = time.perf_counter() - t0
            total = fwht_q_s + proj_q_s + scan_s + rerank_s
            r = recall_at_1_at_k(gt_nn, final)
            row = {
                "m": m,
                "shortlist": S,
                "recall@1@64": r,
                "ms_per_query": {
                    "fwht": 1e3 * fwht_q_s / nq,
                    "landmark_proj": 1e3 * proj_q_s / nq,
                    "landmark_scan": 1e3 * scan_s / nq,
                    "exact_rerank_top64": 1e3 * rerank_s / nq,
                    "total": 1e3 * total / nq,
                },
                "pct_of_total": {
                    "fwht": 100 * fwht_q_s / total,
                    "landmark_proj": 100 * proj_q_s / total,
                    "landmark_scan": 100 * scan_s / total,
                    "exact_rerank_top64": 100 * rerank_s / total,
                },
                "wall_s": {
                    "fwht_all_queries": fwht_q_s,
                    "proj_all_queries": proj_q_s,
                    "scan_all_queries": scan_s,
                    "rerank_all_queries": rerank_s,
                },
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    path = out / "latency_breakdown_m64_m96.json"
    path.write_text(
        json.dumps(
            {
                "n_query": nq,
                "n_db": int(db0.shape[0]),
                "batch": batch,
                "note": "ms/query = stage_wall_s / n_query (batched numpy)",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path}", flush=True)
    print("\nm   S      fwht   proj    scan  rerank  TOTAL  r@1@64", flush=True)
    for r in rows:
        ms = r["ms_per_query"]
        print(
            f"{r['m']:3d} {r['shortlist']:5d}  "
            f"{ms['fwht']:5.2f}  {ms['landmark_proj']:5.2f}  "
            f"{ms['landmark_scan']:6.2f}  {ms['exact_rerank_top64']:6.2f}  "
            f"{ms['total']:6.2f}  {r['recall@1@64']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
