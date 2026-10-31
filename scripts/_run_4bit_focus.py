import os, re, subprocess
from pathlib import Path

R1 = re.compile(r"recall@1@64=([0-9.]+)")
RK = re.compile(r"recall@64=([0-9.]+)")
T = re.compile(r"tuned score_threshold=([0-9.eE+-]+)")
EXE, NPY = Path("build/Release/query-bench.exe"), Path("data/openai-1536-101k.npy")

def run(L, S, q):
    env = dict(os.environ); env["OMP_NUM_THREADS"] = "12"
    cmd = [str(EXE),"--npy",str(NPY),"--limit","100000","--query-limit",str(q),
           "--bits","4","--landmarks",str(L),"--shortlist",str(S),"--k","64",
           "--tune-mean-shortlist",str(S),"--timing-runs","1","--recall"]
    print(f"\n>>> L={L} S={S} q={q}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + "\n" + p.stderr
    if p.returncode: print(out[-2000:]); raise SystemExit(p.returncode)
    r1, rk, thr = float(R1.search(out).group(1)), float(RK.search(out).group(1)), float(T.search(out).group(1))
    print(f"    R@1={r1:.4f} R@64={rk:.4f} T={thr} L*S={L*S}", flush=True)
    return r1, rk, thr

def min_s(L, lo, hi, q, target=0.99, iters=5):
    r,_,t = run(L, hi, q)
    if r < target:
        while r < target and hi < 32768:
            hi = min(hi*2, 32768); r,_,t = run(L, hi, q)
        if r < target: return None
    best, best_t, left, right = hi, t, lo, hi
    for _ in range(iters):
        if left >= right: break
        mid = (left+right)//2
        r,_,t = run(L, mid, q)
        if r >= target: best, best_t, right = mid, t, mid
        else: left = mid+1
    return best, best_t

# Focused brackets (coarse). L=1536 already ~5088 at q500.
brackets = [
    (1024, 4096, 6144),
    (768, 5120, 9216),
    (512, 7168, 12288),
    (384, 8192, 14336),
    (256, 10240, 16384),
]
q = 500
rows = [(1536, 5088, 0.196875)]  # from prior q500 refine (S=5088 hit 0.99)
print(f"seed L=1536 S=5088 from earlier q500 refine", flush=True)

for L, lo, hi in brackets:
    print(f"\n===== L={L} =====", flush=True)
    ans = min_s(L, lo, hi, q)
    if ans:
        s,t = ans
        rows.append((L,s,t))
        print(f"  >> L={L} minS={s} product={L*s}", flush=True)

print("\n=== SUMMARY q=500 ===", flush=True)
rows.sort(key=lambda x: x[0]*x[1] if len(x)==3 and isinstance(x[2], float) else x[0]*x[1])
# normalize
norm = []
for row in rows:
    L,S = row[0], row[1]
    T = row[2] if len(row)>2 else float('nan')
    norm.append((L,S,T,L*S))
norm.sort(key=lambda x: x[3])
for L,S,T,P in norm:
    print(f"  L={L:4d} S={S:5d} L*S={P:8d}", flush=True)

pick = norm[0]
print(f"\n=== confirm q=1000 L={pick[0]} S={pick[1]} ===", flush=True)
r1,rk,t = run(pick[0], pick[1], 1000)
S = pick[1]
while r1 < 0.99 and S < 32768:
    S = min(int(S*1.1)+64, 32768)
    print(f"bump S->{S}", flush=True)
    r1,rk,t = run(pick[0], S, 1000)
minS = min(norm, key=lambda x: x[1])
minL = min(norm, key=lambda x: x[0])
print(f"\nRESULT min_product L={pick[0]} S={S} R@1={r1:.4f} R@64={rk:.4f} T={t} prod={pick[0]*S}", flush=True)
print(f"min_S_row L={minS[0]} S={minS[1]}", flush=True)
print(f"min_L_row L={minL[0]} S={minL[1]}", flush=True)
