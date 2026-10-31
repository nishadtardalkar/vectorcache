#!/usr/bin/env python3
"""Per-L independent min shortlist for R@1@k>=target (4-bit), then Pareto / min L*S."""

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


def round8(m: int, dim: int) -> int:
    m = max(8, (m + 7) // 8 * 8)
    return min(m, dim - (dim % 8))


def run_bench(exe, npy, *, L, S, limit, q, k, bits, omp) -> dict:
    cmd = [
        str(exe), "--npy", str(npy), "--limit", str(limit), "--query-limit", str(q),
        "--bits", str(bits), "--landmarks", str(L), "--shortlist", str(S), "--k", str(k),
        "--tune-mean-shortlist", str(S), "--timing-runs", "1", "--recall",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp)
    print(f"\n>>> L={L} S={S} q={q}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        print(out[-2500:], flush=True)
        raise RuntimeError(f"rc={p.returncode}")
    r1 = float(R1_RE.search(out).group(2))
    rk = float(RK_RE.search(out).group(2))
    thr = float(T_RE.search(out).group(1)) if T_RE.search(out) else float("nan")
    mean = float(MEAN_SL_RE.search(out).group(1)) if MEAN_SL_RE.search(out) else float("nan")
    print(f"    R@1@{k}={r1:.4f} R@{k}={rk:.4f} T={thr} tune_mean_sl={mean:.1f}", flush=True)
    return {"landmarks": L, "shortlist": S, "recall1": r1, "recallk": rk,
            "threshold": thr, "mean_shortlist": mean, "product": L * S}


def min_S(run, L: int, s_lo: int, s_hi: int, target: float) -> dict | None:
    """Geometric up from s_lo, then binary refine. Independent per L."""
    s = s_lo
    prev_fail = s_lo - 1
    hit = None
    while True:
        r = run(L, s)
        if r["recall1"] >= target:
            hit = r
            break
        prev_fail = s
        if s >= s_hi:
            return None
        nxt = min(max(s * 2, s + 512), s_hi)
        if nxt == s:
            return None
        s = nxt

    left, right = prev_fail + 1, hit["shortlist"]
    best = hit
    for _ in range(8):
        if left >= right:
            break
        mid = (left + right) // 2
        if mid <= prev_fail:
            mid = prev_fail + 1
        if mid >= right:
            break
        r = run(L, mid)
        if r["recall1"] >= target:
            best, right = r, mid
        else:
            left = mid + 1
    if best["shortlist"] != right:
        best = run(L, right)
    return best if best["recall1"] >= target else hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=Path("build/Release/query-bench.exe"))
    ap.add_argument("--npy", type=Path, default=Path("data/openai-1536-101k.npy"))
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--query-limit", type=int, default=500)
    ap.add_argument("--confirm-query-limit", type=int, default=1000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--omp", type=int, default=12)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--s-lo", type=int, default=2048)
    ap.add_argument("--s-hi", type=int, default=32768)
    ap.add_argument("--landmarks", type=int, nargs="*",
                    default=[1536, 1024, 768, 512, 384, 256])
    args = ap.parse_args()

    cache: dict[tuple[int, int, int], dict] = {}

    def run(L, S, q=None):
        L = round8(L, args.dim)
        q = args.query_limit if q is None else q
        key = (L, S, q)
        if key in cache:
            print(f"\n>>> L={L} S={S} q={q} (cached)", flush=True)
            c = cache[key]
            print(f"    R@1={c['recall1']:.4f}", flush=True)
            return c
        r = run_bench(args.exe, args.npy, L=L, S=S, limit=args.limit, q=q,
                      k=args.k, bits=args.bits, omp=args.omp)
        cache[key] = r
        return r

    grid = [round8(x, args.dim) for x in args.landmarks]
    print(f"4-bit independent min-S(L) for R@1@{args.k}>={args.target}\n"
          f"q={args.query_limit} L={grid}", flush=True)

    feasible = []
    for L in grid:
        print(f"\n===== L={L} =====", flush=True)
        best = min_S(run, L, args.s_lo, args.s_hi, args.target)
        if best is None:
            print(f"  L={L}: unreachable", flush=True)
            continue
        print(f"  >> L={L} minS={best['shortlist']} L*S={best['product']} "
              f"R@1={best['recall1']:.4f}", flush=True)
        feasible.append(best)

    if not feasible:
        print("FAIL: none feasible", flush=True)
        return 1

    print("\n=== min-S(L) sorted by L*S ===", flush=True)
    for p in sorted(feasible, key=lambda x: (x["product"], x["landmarks"])):
        print(f"  L={p['landmarks']:4d} S={p['shortlist']:5d} L*S={p['product']:8d} "
              f"R@1={p['recall1']:.4f}", flush=True)

    pick = min(feasible, key=lambda p: (p["product"], p["landmarks"], p["shortlist"]))
    print(f"\n=== pick min L*S: L={pick['landmarks']} S={pick['shortlist']} ===", flush=True)

    print(f"\n=== confirm q={args.confirm_query_limit} ===", flush=True)
    confirm = run(pick["landmarks"], pick["shortlist"], q=args.confirm_query_limit)
    s = pick["shortlist"]
    while confirm["recall1"] < args.target and s < args.s_hi:
        s = min(int(s * 1.15) + 64, args.s_hi)
        print(f"confirm below; bump S -> {s}", flush=True)
        confirm = run(pick["landmarks"], s, q=args.confirm_query_limit)
        pick = confirm

    # Also report absolute min S and min L among feasible
    min_s_pt = min(feasible, key=lambda p: (p["shortlist"], p["landmarks"]))
    min_l_pt = min(feasible, key=lambda p: (p["landmarks"], p["shortlist"]))
    print(
        f"\nRESULT min_product: landmarks={pick['landmarks']} shortlist={pick['shortlist']} "
        f"T={confirm['threshold']} R@1@{args.k}={confirm['recall1']:.4f}\n"
        f"min_shortlist_among_feasible: L={min_s_pt['landmarks']} S={min_s_pt['shortlist']}\n"
        f"min_landmarks_among_feasible: L={min_l_pt['landmarks']} S={min_l_pt['shortlist']}",
        flush=True,
    )
    return 0 if confirm["recall1"] >= args.target else 2


if __name__ == "__main__":
    sys.exit(main())
