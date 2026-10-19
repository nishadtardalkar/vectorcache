#!/usr/bin/env python3
"""Find (landmarks, shortlist) with Recall@1@k >= target, minimizing both.

For each L (descending): geometric ladder on S, then binary refine.
S*(L) non-increasing → warm-start lo from prior (higher-L) min S.
Pick Pareto knee by min L*S; confirm on full query set.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

R1_RE = re.compile(r"recall@1@(\d+)=([0-9.]+)")
RK_RE = re.compile(r"recall@(\d+)=([0-9.]+)")
T_RE = re.compile(r"tuned score_threshold=([0-9.eE+-]+)")
MEAN_SL_RE = re.compile(r"mean_shortlist=([0-9.]+)")
SCAN_RE = re.compile(r"shortlist_scan=([0-9.]+)")
MSQ_RE = re.compile(r"ms_per_query=([0-9.]+)")


def round_landmarks(m: int, dim: int = 1536) -> int:
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
    timing_runs: int = 1,
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
        str(timing_runs),
        "--recall",
    ]
    env = dict(__import__("os").environ)
    env["OMP_NUM_THREADS"] = str(omp)
    print(f"\n>>> L={landmarks} S={shortlist} q={query_limit}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(out[-2500:], flush=True)
        raise RuntimeError(f"query-bench failed rc={proc.returncode}")

    r1_m, rk_m = R1_RE.search(out), RK_RE.search(out)
    if not r1_m or not rk_m:
        print(out[-2500:], flush=True)
        raise RuntimeError("could not parse recall")
    r1, rk = float(r1_m.group(2)), float(rk_m.group(2))
    thr = float(t.group(1)) if (t := T_RE.search(out)) else float("nan")
    means = MEAN_SL_RE.findall(out)
    mean_sl = float(means[-1]) if means else float("nan")
    scan = float(s.group(1)) if (s := SCAN_RE.search(out)) else float("nan")
    msq = float(m.group(1)) if (m := MSQ_RE.search(out)) else float("nan")
    print(
        f"    R@1@{k}={r1:.4f} R@{k}={rk:.4f} T={thr} "
        f"mean_sl={mean_sl:.0f} scan={scan:.4f} ms/q={msq:.4f}",
        flush=True,
    )
    return {
        "landmarks": landmarks,
        "shortlist": shortlist,
        "recall1": r1,
        "recallk": rk,
        "threshold": thr,
        "mean_shortlist": mean_sl,
        "scan_ms": scan,
        "ms_per_query": msq,
        "product": landmarks * shortlist,
    }


def geometric_then_binsearch_S(
    run,
    landmarks: int,
    s_lo: int,
    s_hi: int,
    max_s: int,
    target: float,
) -> dict | None:
    """Ladder 2x from s_lo until hit (or max_s), then binary refine."""
    s_lo = max(64, s_lo)
    # Ladder
    prev_fail = s_lo - 1
    s = s_lo
    hit = None
    while True:
        r = run(landmarks, s)
        if r["recall1"] >= target:
            hit = r
            break
        prev_fail = s
        if s >= max_s:
            return None
        nxt = min(max(s * 2, s + 256), max_s)
        if nxt == s:
            return None
        s = nxt

    # Binary between prev_fail+1 and hit.S
    left = prev_fail + 1
    right = hit["shortlist"]
    best = hit
    # Cap refine iterations (coarse is enough given recall noise).
    for _ in range(6):
        if left >= right:
            break
        mid = (left + right) // 2
        if mid <= prev_fail:
            mid = prev_fail + 1
        if mid >= right:
            break
        r = run(landmarks, mid)
        if r["recall1"] >= target:
            best = r
            right = mid
        else:
            left = mid + 1
    if best["shortlist"] != right:
        best = run(landmarks, right)
    return best if best["recall1"] >= target else hit


def pareto_front(points: list[dict]) -> list[dict]:
    pts = sorted(points, key=lambda p: (p["landmarks"], p["shortlist"]))
    front, best_s = [], float("inf")
    for p in pts:
        if p["shortlist"] < best_s:
            front.append(p)
            best_s = p["shortlist"]
    return front


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
    ap.add_argument("--lo-shortlist", type=int, default=512)
    ap.add_argument("--max-shortlist", type=int, default=32768)
    ap.add_argument(
        "--landmarks",
        type=int,
        nargs="*",
        default=[1280, 1024, 896, 768, 640, 512, 384, 256],
    )
    args = ap.parse_args()

    cache: dict[tuple[int, int, int], dict] = {}

    def run(landmarks: int, shortlist: int, q: int | None = None) -> dict:
        landmarks = round_landmarks(landmarks, args.dim)
        q = args.query_limit if q is None else q
        key = (landmarks, shortlist, q)
        if key in cache:
            print(f"\n>>> L={landmarks} S={shortlist} q={q} (cached)", flush=True)
            c = cache[key]
            print(f"    R@1@{args.k}={c['recall1']:.4f} R@{args.k}={c['recallk']:.4f}", flush=True)
            return c
        r = run_bench(
            args.exe,
            args.npy,
            landmarks=landmarks,
            shortlist=shortlist,
            limit=args.limit,
            query_limit=q,
            k=args.k,
            bits=args.bits,
            omp=args.omp,
        )
        cache[key] = r
        return r

    grid = sorted({round_landmarks(x, args.dim) for x in args.landmarks}, reverse=True)
    print(
        f"target R@1@{args.k}>={args.target}  minimize both via Pareto / min L*S\n"
        f"npy={args.npy} n={args.limit} q_search={args.query_limit}\n"
        f"L grid desc: {grid}",
        flush=True,
    )

    feasible: list[dict] = []
    s_lo = args.lo_shortlist
    for L in grid:
        print(f"\n===== L={L}  S_lo={s_lo} =====", flush=True)
        best = geometric_then_binsearch_S(
            run, L, s_lo, args.max_shortlist, args.max_shortlist, args.target
        )
        if best is None:
            print(f"  L={L}: unreachable at S<={args.max_shortlist}", flush=True)
            continue
        print(
            f"  >> L={L} minS={best['shortlist']} L*S={best['product']} "
            f"R@1={best['recall1']:.4f}",
            flush=True,
        )
        feasible.append(best)
        s_lo = best["shortlist"]

    if not feasible:
        print("FAIL: no feasible (L,S)", flush=True)
        return 1

    front = pareto_front(feasible)
    print("\n=== Pareto front ===", flush=True)
    for p in front:
        print(
            f"  L={p['landmarks']:4d} S={p['shortlist']:5d} L*S={p['product']:8d} "
            f"R@1={p['recall1']:.4f}",
            flush=True,
        )

    pick = min(front, key=lambda p: (p["product"], p["landmarks"], p["shortlist"]))
    print(
        f"\n=== pick min L*S: L={pick['landmarks']} S={pick['shortlist']} "
        f"product={pick['product']} T={pick['threshold']} ===",
        flush=True,
    )

    print(f"\n=== confirm q={args.confirm_query_limit} ===", flush=True)
    confirm = run(pick["landmarks"], pick["shortlist"], q=args.confirm_query_limit)
    s = pick["shortlist"]
    while confirm["recall1"] < args.target and s < args.max_shortlist:
        s = min(int(s * 1.2) + 64, args.max_shortlist)
        print(f"confirm below target; bump S -> {s}", flush=True)
        confirm = run(pick["landmarks"], s, q=args.confirm_query_limit)
        pick = confirm

    print(
        f"\nRESULT landmarks={pick['landmarks']} shortlist={pick['shortlist']} "
        f"score_threshold={confirm['threshold']} "
        f"product={pick['landmarks'] * pick['shortlist']} "
        f"recall@1@{args.k}={confirm['recall1']:.4f} "
        f"recall@{args.k}={confirm['recallk']:.4f}",
        flush=True,
    )
    print("\nFeasible min-S(L) by product:", flush=True)
    for p in sorted(feasible, key=lambda x: x["product"]):
        star = (
            " <--"
            if p["landmarks"] == pick["landmarks"] and p["shortlist"] == pick["shortlist"]
            else ""
        )
        print(
            f"  L={p['landmarks']:4d} S={p['shortlist']:5d} L*S={p['product']:8d} "
            f"R@1={p['recall1']:.4f}{star}",
            flush=True,
        )
    return 0 if confirm["recall1"] >= args.target else 2


if __name__ == "__main__":
    sys.exit(main())
