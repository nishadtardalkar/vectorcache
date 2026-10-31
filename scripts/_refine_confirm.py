import re, subprocess, os
exe = r"build\Release\query-bench.exe"
npy = r"data\openai-1536-101k.npy"
env = dict(os.environ); env["OMP_NUM_THREADS"] = "12"

def run(L, S):
    cmd = [exe, "--npy", npy, "--limit", "100000", "--query-limit", "1000",
           "--bits", "4", "--landmarks", str(L), "--shortlist", str(S), "--k", "64",
           "--tune-mean-shortlist", str(S), "--timing-runs", "1", "--recall"]
    print(f"\n>>> L={L} S={S}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    r1 = float(re.search(r"recall@1@64=([0-9.]+)", out).group(1))
    rk = float(re.search(r"recall@64=([0-9.]+)", out).group(1))
    t = float(re.search(r"tuned score_threshold=([0-9.eE+-]+)", out).group(1))
    print(f"    R@1={r1:.4f} R@64={rk:.4f} T={t} L*S={L*S}", flush=True)
    return r1, rk, t

def min_s(L, lo, hi, target=0.99):
    # ensure hi works
    r, _, t = run(L, hi)
    if r < target:
        while r < target and hi < 32768:
            hi = min(hi * 2, 32768)
            r, _, t = run(L, hi)
        if r < target:
            return None
    best = hi
    best_t = t
    left, right = lo, hi
    for _ in range(6):
        if left >= right: break
        mid = (left + right) // 2
        r, _, t = run(L, mid)
        if r >= target:
            best, best_t, right = mid, t, mid
        else:
            left = mid + 1
    return best, best_t

results = []
# From prior: L256 needs ~13k, L384 ~10.8k works, L640 ~8.1k works, L1024 ~4.7k at q500
for L, lo, hi in [(256, 10807, 13032), (384, 8645, 10838), (640, 6216, 8108), (1024, 4736, 8192)]:
    print(f"\n===== refine L={L} =====", flush=True)
    ans = min_s(L, lo, hi)
    if ans:
        s, t = ans
        results.append((L, s, t, L * s))
        print(f"  CONFIRMED L={L} S={s} T={t} product={L*s}", flush=True)

print("\n=== SUMMARY (q=1000, R@1@64>=0.99) ===", flush=True)
results.sort(key=lambda x: x[3])
for L, s, t, prod in results:
        print(f"  L={L:4d} S={s:5d} T={t:.6f} L*S={prod:8d}", flush=True)
if results:
    L, s, t, prod = results[0]
    print(f"\nPICK (min L*S): landmarks={L} shortlist={s} score_threshold={t} product={prod}", flush=True)
