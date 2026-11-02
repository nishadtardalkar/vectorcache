# Paper outline: novelty claims & required benchmarks

Working title (placeholder): **Landmark FastScan over TurboQuant codes: subspace ANN with residual bound and optional exact re-rank**

Scope of this note: what is *defensibly novel*, what is *not*, and which measurements the codebase must produce for each comparison target. Dataset default = OpenAI DBpedia embeddings (`openai-1536` / `openai-3072`), n=100K, nq=1K, seed/holdout aligned with turbovec where possible.

---

## 1. What is actually novel (candidate contributions)

### 1.1 Core algorithmic claim (primary)

**Orthonormal landmark projection of TurboQuant reconstructions, then FastScan in landmark dimension `m ≪ d`, with a residual-energy correction.**

Pipeline (product path today):

1. L2-normalize → FWHT (ChaCha8 SRHT stand-in) → Lloyd-Max / α (TurboQuant-style encode into `α·ŷ`).
2. Project `ℓ = (α·ŷ) L` for orthonormal `L ∈ R^{d×m}`.
3. Quantize/pack landmark rows at `scan_bits`; **drop full-dim floats/codes from the hot index**.
4. Query: project `q` → `q_ℓ`; SIMD FastScan over landmark codes; score  
   \(s = s_L + \|q_⊥\|·\|r_i\|\)  
   (landmark IP + Cauchy–Schwarz residual **bound**, not residual IP).
5. Optional staged 4-bit path in landmark space (sign plane → shortlist → 4-bit exact).
6. Optional **exact float IP re-rank of the returned top-`k`** against unit corpus (application-realistic; timed with search).

**Why this can be a paper (vs turbovec / FAISS PQ / TurboQuant paper):**

| Piece | Prior art | Our twist |
|---|---|---|
| TurboQuant encode | Google TQ / turbovec | Same family; not claimed as ours |
| Full-dim FastScan / staged planes | turbovec | We scan an **`m`-dim** quantized landmark embedding of TQ codes |
| Residual term | RaBitQ-style ideas, energy bounds | Explicit **\|q_⊥\|·\|r\|** on landmark residuals in the product scorer |
| Exact re-rank of shortlist | Ubiquitous in IVF/HNSW+PQ | Positioned as **optional second stage** with originals “already in the app DB”; must be in the latency number when quality is post-re-rank |

**Sellable one-liner:** *Compress the search layout by projecting TQ reconstructions onto a small orthonormal landmark basis; keep competitive 1-recall / set-overlap after a cheap exact re-rank of `k`, at substantially lower scan cost than full-dim TQ FastScan.*

### 1.2 Systems / measurement claims (secondary, still useful)

- **Honest metrics:** report both **1-recall@t** (turbovec-comparable) and **t-recall@t** / set-overlap (ANN-Benchmarks / FAISS intersection). Literature on this corpus often mixes them; a short “metric hygiene” subsection is publishable as clarification, not as theory.
- **Pareto framing:** memory of **landmark codes** (`O(n·m·bits)`) vs full-dim TQ/PQ (`O(n·d·bits)`), vs float32; latency of ANN-only vs ANN+float-re-rank.
- **C++20 / OpenMP / AVX2–VNNI** reproduction harness next to turbovec’s published cells (`make turbovec` / `make compute`).

### 1.3 What is *not* novel (do not overclaim)

- TurboQuant / Lloyd-Max Beta / FWHT rotation themselves.
- Staged sign→4-bit FastScan as an idea (turbovec).
- Float re-ranking a shortlist (standard).
- Set-overlap Recall@k as a metric (ANN-Benchmarks, FAISS `knn_intersection_measure`).
- HNSW / IVF as competitors’ *algorithms* (only baselines).

### 1.4 Claims that need more evidence before writing them as novelty

- Residual **sketch** / residual landmarks (tried, reverted; only revisit with ablations that win).
- Bucketed / IVF layer in README — confirm it is still first-class in the paper build or drop from claims.
- ALGORITHM.md “full-dim quantized re-rank” — **not** what search does today; either restore as ablation or delete from paper narrative.

---

## 2. Suggested paper structure

1. **Intro:** RAG/ANN needs; TQ is strong at full-d scan; we ask whether a landmark subspace scan + light exact re-rank is a better operating point.
2. **Background:** TurboQuant, FastScan, 1-recall@k vs k-recall@k.
3. **Method:** landmark construction; encode; score with residual bound; staged search; optional float re-rank; complexity/memory.
4. **Experiments:** tables below; ablations (`m`, bits, planes on/off, float-re-rank on/off, `k`).
5. **Discussion:** when not to use (need graph at 10M+; residual bound misorders set-overlap without re-rank).
6. **Related work:** TQ, RaBitQ, FAISS PQ/FastScan, turbovec, Infino-style SQ.

---

## 3. Benchmark matrix (what to run, against whom)

Conventions for **all** rows unless noted:

- Same machine / thread pool (document `OMP_NUM_THREADS`, e.g. 8 to match turbovec cells).
- Same corpus slice and query split (prefer turbovec seed=42 100K+1K when comparing to turbovec).
- Report: ingest ms, search median ms, **ms/query**, RSS or codes-only bytes/vector, and **both** recall families when space allows.

### 3.1 vs turbovec (must-have)

| Goal | Their metric / cell | What we run | Flags / harness |
|---|---|---|---|
| Speed | Published ~0.12 ms/q openai-1536 4-bit staged mt | Our ANN-only (+ optional float-re-rank as second column) | `make turbovec` (theirs); `make compute DATASET=openai-1536 BITS=4 K=64 FLOAT_RESCORE=0` and `FLOAT_RESCORE=1` |
| 1-recall@k curve | Suite `recall_*` (true NN in top-t) | Same queries; print 1-recall@t for t∈{1,2,4,8,16,32,64} | Add/keep a **1-recall@t** reporter (today bench is set-overlap-only — **restore dual metrics for the paper**) |
| Fairness note | They search full-dim TQ | We search landmarks ± float re-rank | State clearly in table caption |

**Code needs:**

- [ ] Dual recall output: `recall1@t=` (1-recall) and `recall@t=` (set-overlap).
- [ ] `FLOAT_RESCORE=0/1` columns in every quality table.
- [ ] Landmarks / bits logged in banner (already).
- [ ] Side-by-side JSON or CSV dump for figures (optional script).

### 3.2 vs FAISS (must-have)

| Baseline | Why | Metric | Our matching run |
|---|---|---|---|
| `IndexPQ(m=d/2, nbits=8)` ≈ 4-bit rate | turbovec’s recall baseline | 1-recall@t **and** t-recall@t | Same n/nq; FAISS via Python harness (reuse `.ref-turbovec` suite or thin script) |
| `IndexPQFastScan` matched rate | Their **speed** baseline | ms/query only (+ optional recall if stable) | Same threads; document FastScan recall caveats |
| Optional: `IndexFlatIP` | Exact upper bound / GT | — | Already used for exact top-k cache |
| Optional: HNSW-Flat | Quality upper bound, not iso-memory | 1-recall + set-overlap @5/10 | Cite memory honestly |

**Code needs:**

- [ ] Documented FAISS recall+speed script (or `make faiss-recall` / wrap suite `recall_d1536_4bit.py` + speed harness).
- [ ] Bytes/vector and build-time for FAISS PQ vs our landmark codes + (optional) float corpus residency.

### 3.3 vs “application stack” (supports float-re-rank story)

| Claim | Benchmark |
|---|---|
| Exact re-rank of `k` is cheap vs ANN | Break out ANN ms vs re-rank ms (instrument `float_rescore_topk`) |
| Quality after re-rank | 1-recall@1 and set-overlap@t with re-rank on |
| Memory model | Index codes only vs codes + mmap/DB floats (don’t charge full float RAM as “index” if arguing DB already holds vectors) |

**Code needs:**

- [ ] Timing split: `ann_ms_per_query` + `rerank_ms_per_query` + `total`.
- [ ] Ablation: re-rank pool = `k` vs internal wider pool (if added later).

### 3.4 Ablations (needed for reviewers)

| Knob | Sweeps | Report |
|---|---|---|
| `m` (landmarks) | e.g. 128, 256, 512, 1024 | latency, memory, both recalls |
| `scan_bits` | 2 vs 4 | same |
| Staged planes | on / `TURBOVEC_4BIT_PLANES=0` | latency + recall |
| Float re-rank | on / off | quality delta vs latency delta |
| `k` | 10, 64, 256 | set-overlap@10 and @64; show pool effect under re-rank |

**Code needs:**

- [ ] One sweep script or Make targets that emit a single CSV.
- [ ] Fixed query set across sweeps (cached exact top-k already helps).

### 3.5 Extra datasets (recommended, not blocking)

| Dataset | Role |
|---|---|
| `openai-3072` | Scale dim; match turbovec cells |
| GloVe-200 | Hard low-dim regime (TQ literature) |
| SIFT1M | Classic ANN; optional |

Same protocol: bits ∈ {2,4}, with/without float-re-rank, dual recall.

---

## 4. Tables / figures to plan for

1. **Method diagram:** ingest → landmark pack → staged scan → optional float re-rank.
2. **Memory:** bytes/vector (float32 / TQ full-d / our landmark codes / PQ).
3. **Latency:** ours ANN-only vs ours+re-rank vs turbovec vs FAISS FastScan (openai-1536, 8 threads).
4. **Quality A (1-recall@t):** ours±re-rank vs turbovec vs FAISS PQ.
5. **Quality B (set-overlap@t):** ours±re-rank vs FAISS PQ (and Infino/TQ papers only where metric is verified).
6. **Ablation:** recall & ms/q vs `m`.
7. **Pareto:** set-overlap@10 or 1-recall@1 vs ms/q (points = configs).

---

## 5. Metric hygiene (short subsection worth writing)

| Name | Definition | Who uses it |
|---|---|---|
| **1-recall@t** | Fraction of queries with exact NN ∈ approx[:t] | turbovec suite / TQ marketing curves |
| **t-recall@t** (set-overlap) | \|approx[:t] ∩ exact[:t]\| / t | ANN-Benchmarks, FAISS intersection |
| **Hit@t** | Same as 1-recall@t | TurboVec arXiv |

**Rule for our paper:** always name the metric; never put 1-recall and set-overlap in one unlabeled column. Prefer reporting **both** on openai-1536.

Prefix set-overlap from a **fixed** search-`k` (our current bench) **decreases** in `t` by construction; curves that rise with `t` are usually 1-recall@t or a **new search at each t**. Say this once to avoid reviewer confusion.

---

## 6. Implementation checklist (code → paper)

Priority order:

1. **Dual recall in `query-bench`** (1-recall@t + set-overlap@t).
2. **Split timing** ANN vs float-re-rank.
3. **CSV/JSON result dump** for plots.
4. **FAISS / turbovec** one-command reproduction notes in README or `docs/repro.md`.
5. **Landmark sweep script** (`m`, bits, planes, re-rank).
6. Confirm paper build = LandmarkIndex path only (or explicitly include bucketed as appendix).
7. Align ALGORITHM.md with code (no phantom full-dim quantized re-rank unless restored).

---

## 7. Risk / honesty list for the draft

- Without float-re-rank, set-overlap@64 can look weak (~0.5) even when 1-recall@64 is excellent (~0.97); explain “coverage vs ranking.”
- Residual term is a **bound**, not residual IP — limits ranking among near-ties; motivates exact re-rank.
- Comparing to HNSW without matching memory is misleading.
- Infino / blog numbers: verify metric before citing as set-overlap (non-monotonic R@k is a hint, not a proof in their prose).
- Novelty gate: reviewers may say “IVF/PQ already does coarse + re-rank.” Counter: **TQ-aware landmark embedding of codes** + iso-comparison to **flat** TQ FastScan (turbovec), not to HNSW.

---

## 8. Minimal “paper-ready” experiment pack (ours only)

`make benchmark` runs **our** `query-bench` only (no turbovec / FAISS). openai-1536, n=100K, nq=1K, 8 threads:

```bash
make login       # once (login node)
make benchmark   # → results/paper_benchmark_openai-1536_<utc>.txt
```

| # | Run | Produces |
|---|---|---|
| A | `FLOAT_RESCORE=0` | ANN-only latency + dual recall + bytes/vector |
| B | `FLOAT_RESCORE=1` | End-to-end; `ann_ms` / `rerank_ms` / `ms_per_query` |
| C | m∈{256,512}, bits=2, planes-off | Ablations (`SKIP_ABLATIONS=1` to skip) |
| D | scraped key=value summary | Memory / Pareto table source |

**Baselines for the paper draft:** cite **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) for the encode family, and **FAISS PQ / published ANN** for iso-bit quality/speed — not turbovec README cells (third-party impl, self-reported, no paper). Keep `make turbovec` only as an optional engineering cross-check, outside the paper pack.

Knobs: `SKIP_ABLATIONS=1`, `BENCHMARK_OUT=path`.
