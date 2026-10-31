#!/usr/bin/env python3
"""Refine min-S per L in known brackets (4-bit), q=500 then confirm pick at q=1000."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

R1 = re.compile(r"recall@1@64=([0-9.]+)")
RK = re.compile(r"recall@64=([0-9.]+)")
T = re.compile(r"tuned score_threshold=([0-9.eE+-]+)")

EXE = Path("build/Release/query-bench.exe")
NPY = Path("data/openai-1536-101k.npy")


def run(L: int, S: int, q: int) -> tuple[float, float, float]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "12"
    cmd = [
        str(EXE), "--npy", str(NPY), "--limit", "100000", "--query-limit", str(q),
        "--bits", "4", "--landmarks", str(L), "--shortlist", str(S), "--k", "64",
        "--tune-mean-shortlist", str(S), "--timing-runs", "1", "--recall",
    ]
    print(f"\n>>> L={L} S={S} q={q}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        print(out[-2000:], flush=True)
        raise RuntimeError(f"rc={p.returncode}")
    r1, rk = float(R1.search(out).group(1)), float(RK.search(out).group(1))
    thr = float(T.search(out).group(1))
    print(f"    R@1={r1:.4f} R@64={rk:.4f} T={thr} L*S={L*S}", flush=True)
    return r1, rk, thr


def min_s(L: int, lo: int, hi: int, q: int, target: float = 0.99) -> tuple[int, float] | None:
    r, _, t = run(L, hi, q)
    if r < target:
        while r < target and hi < 32768:
            hi = min(hi * 2, 32768)
            r, _, t = run(L, hi, q)
        if r < target:
            return None
    best, best_t = hi, t
    left, right = lo, hi
    for _ in range(7):
        if left >= right:
            break
        mid = (left + right) // 2
        r, _, t = run(L, mid, q)
        if r >= target:
            best, best_t, right = mid, t, mid
        else:
            left = mid + 1
    return best, best_t


def main() -> int:
    # Brackets from prior q=1000 refine + L=1536 from today's q=200 ladder
    brackets = [
        (1536, 4608, 6144),
        (1024, 4096, 6144),
        (768, 5120, 8192),
        (640, 6144, 9216),
        (512, 7168, 12288),
        (384, 8192, 12288),
        (256, 10240, 16384),
    ]
    q_search = 500
    results = []
    for L, lo, hi in brackets:
        print(f"\n===== refine L={L} q={q_search} =====", flush=True)
        ans = min_s(L, lo, hi, q_search)
        if ans:
            s, t = ans
            results.append((L, s, t, L * s))
            print(f"  CONFIRMED L={L} S={s} T={t} product={L*s}", flush=True)

    if not results:
        print("FAIL", flush=True)
        return 1

    print("\n=== SUMMARY (q=500, R@1@64>=0.99) ===", flush=True)
    results.sort(key=lambda x: x[3])
    for L, s, t, prod in results:
        print(f"  L={L:4d} S={s:5d} T={t:.6f} L*S={prod:8d}", flush=True)

    L, s, t, prod = results[0]
    print(f"\n=== confirm pick q=1000: L={L} S={s} ===", flush=True)
    r1, rk, t2 = run(L, s, 1000)
    while r1 < 0.99 and s < 32768:
        s = min(int(s * 1.1) + 64, 32768)
        print(f"bump S -> {s}", flush=True)
        r1, rk, t2 = run(L, s, 1000)

    min_s_row = min(results, key=lambda x: (x[1], x[0]))
    min_l_row = min(results, key=lambda x: (x[0], x[1]))
    print(
        f"\nRESULT min_L*S: landmarks={L} shortlist={s} T={t2} "
        f"R@1@64={r1:.4f} R@64={rk:.4f} product={L*s}\n"
        f"min_S: L={min_s_row[0]} S={min_s_row[1]}\n"
        f"min_L: L={min_l_row[0]} S={min_l_row[1]}",
        flush=True,
    )
    return 0 if r1 >= 0.99 else 2


if __name__ == "__main__":
    sys.exit(main())
