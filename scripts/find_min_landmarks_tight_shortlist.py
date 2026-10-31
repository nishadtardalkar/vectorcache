#!/usr/bin/env python3
"""Min landmark dims for R@1@k >= target with mean shortlist in (k, k+16)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

R1_RE = re.compile(r"recall@1@(\d+)=([0-9.]+)")
RK_RE = re.compile(r"recall@(\d+)=([0-9.]+)")
T_RE = re.compile(r"tuned score_threshold=([0-9.eE+-]+)")
MEAN_SL_RE = re.compile(r"tuned score_threshold=.*?mean_shortlist=([0-9.]+)")
STAGE_MEAN_RE = re.compile(r"mean_shortlist=([0-9.]+) max_shortlist=")


def round8(m: int, dim: int) -> int:
    m = max(8, (m + 7) // 8 * 8)
    return min(m, dim - (dim % 8))


def run_bench(
    exe: Path,
    npy: Path,
    *,
    landmarks: int,
    shortlist: int,
    limit: int,
    query_limit: int,
    k: int,
    bits: int,
    omp: int,
) -> dict:
    cmd = [
        str(exe),
        "--npy",
        str(npy),
        "--limit",
        str(limit),
        "--query-limit",
        str(query_limit),
        "--bits",
        str(bits),
        "--landmarks",
        str(landmarks),
        "--shortlist",
        str(shortlist),
        "--k",
        str(k),
        "--tune-mean-shortlist",
        str(shortlist),
        "--timing-runs",
        "1",
        "--recall",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp)
    print(f"\n>>> L={landmarks} S={shortlist} q={query_limit}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(out[-3000:], flush=True)
        raise RuntimeError(f"query-bench rc={proc.returncode}")

    r1_m, rk_m = R1_RE.search(out), RK_RE.search(out)
    if not r1_m or not rk_m:
        print(out[-3000:], flush=True)
        raise RuntimeError("parse recall failed")
    r1, rk = float(r1_m.group(2)), float(rk_m.group(2))
    thr = float(t.group(1)) if (t := T_RE.search(out)) else float("nan")
    tune_mean = float(m.group(1)) if (m := MEAN_SL_RE.search(out)) else float("nan")
    stage_means = STAGE_MEAN_RE.findall(out)
    stage_mean = float(stage_means[-1]) if stage_means else float("nan")
    print(
        f"    R@1@{k}={r1:.4f} R@{k}={rk:.4f} T={thr} "
        f"tune_mean_sl={tune_mean:.2f} search_mean_sl={stage_mean:.2f}",
        flush=True,
    )
    return {
        "landmarks": landmarks,
        "shortlist": shortlist,
        "recall1": r1,
        "recallk": rk,
        "threshold": thr,
        "tune_mean_shortlist": tune_mean,
        "search_mean_shortlist": stage_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=Path("build/Release/query-bench.exe"))
    ap.add_argument("--npy", type=Path, default=Path("data/openai-1536-101k.npy"))
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--query-limit", type=int, default=200)
    ap.add_argument("--confirm-query-limit", type=int, default=500)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--omp", type=int, default=12)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--lo", type=int, default=768)
    ap.add_argument("--hi", type=int, default=1536)
    # Use largest allowed shortlist in (k, k+16) to minimize required L.
    ap.add_argument("--shortlist-offset", type=int, default=15, help="S = k + offset; need 0 < offset < 16")
    args = ap.parse_args()

    if not (0 < args.shortlist_offset < 16):
        print("shortlist-offset must be in 1..15", flush=True)
        return 2
    S = args.k + args.shortlist_offset
    lo = round8(args.lo, args.dim)
    hi = round8(args.hi, args.dim)
    print(
        f"Find min L s.t. R@1@{args.k}>={args.target} and S={S} in ({args.k}, {args.k + 16})\n"
        f"binary search L in [{lo}, {hi}] q={args.query_limit}",
        flush=True,
    )

    cache: dict[int, dict] = {}

    def eval_L(L: int, q: int | None = None) -> dict:
        L = round8(L, args.dim)
        q = args.query_limit if q is None else q
        key = (L, q)
        if key in cache:
            print(f"\n>>> L={L} S={S} q={q} (cached)", flush=True)
            return cache[key]
        r = run_bench(
            args.exe,
            args.npy,
            landmarks=L,
            shortlist=S,
            limit=args.limit,
            query_limit=q,
            k=args.k,
            bits=args.bits,
            omp=args.omp,
        )
        cache[key] = r
        return r

    # Probe endpoints
    r_lo = eval_L(lo)
    if r_lo["recall1"] >= args.target:
        # Push lower
        hi = lo
        lo = round8(max(8, lo // 2), args.dim)
        while True:
            r = eval_L(lo)
            if r["recall1"] >= args.target:
                hi = lo
                nxt = round8(max(8, lo // 2), args.dim)
                if nxt == lo:
                    break
                lo = nxt
            else:
                break
    else:
        r_hi = eval_L(hi)
        if r_hi["recall1"] < args.target:
            print(f"FAIL: L={hi} still R@1={r_hi['recall1']:.4f} < {args.target}", flush=True)
            return 1

    # Binary search: find smallest L with recall >= target
    best = None
    left, right = lo, hi
    # Ensure left fails (or is min), right succeeds
    if eval_L(left)["recall1"] >= args.target:
        best = eval_L(left)
        right = left
    else:
        while left + 8 < right:
            mid = round8((left + right) // 2, args.dim)
            if mid <= left:
                mid = left + 8
            if mid >= right:
                break
            r = eval_L(mid)
            if r["recall1"] >= args.target:
                best = r
                right = mid
            else:
                left = mid
        if best is None:
            best = eval_L(right)

    print(f"\n=== confirm q={args.confirm_query_limit} at L={best['landmarks']} ===", flush=True)
    confirm = eval_L(best["landmarks"], q=args.confirm_query_limit)
    # If confirm fails, bump L until it passes
    L = best["landmarks"]
    while confirm["recall1"] < args.target and L < args.dim:
        L = round8(L + 64, args.dim)
        print(f"confirm below target; bump L -> {L}", flush=True)
        confirm = eval_L(L, q=args.confirm_query_limit)
        best = confirm

    ok_sl = args.k < S < args.k + 16
    print(
        f"\nRESULT landmarks={confirm['landmarks']} shortlist={S} "
        f"(k={args.k}, window=({args.k},{args.k + 16})) "
        f"score_threshold={confirm['threshold']} "
        f"tune_mean_sl={confirm['tune_mean_shortlist']:.2f} "
        f"search_mean_sl={confirm['search_mean_shortlist']:.2f} "
        f"recall@1@{args.k}={confirm['recall1']:.4f} "
        f"recall@{args.k}={confirm['recallk']:.4f} "
        f"shortlist_ok={ok_sl}",
        flush=True,
    )
    return 0 if confirm["recall1"] >= args.target and ok_sl else 2


if __name__ == "__main__":
    sys.exit(main())
