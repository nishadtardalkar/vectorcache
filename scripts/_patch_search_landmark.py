from pathlib import Path

p = Path("src/search/search.cpp")
t = p.read_text(encoding="utf-8")
start = t.find("SearchResults search_landmark_rerank(")
end = t.find("\nvoid merge_search_results(")
assert start > 0 and end > start, (start, end)

new_fn = r'''SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, std::size_t shortlist,
                                     std::span<const float> landmark_basis, std::size_t n_landmarks,
                                     std::size_t scan_dim, std::span<const float> lm_centroids,
                                     std::size_t scan_bits, std::span<const std::uint8_t> blocked_codes,
                                     std::size_t n_blocks, std::span<const float> lm_scales,
                                     std::span<const float> residual_norms, const Rotation& rotation,
                                     float score_threshold, LandmarkStageTimings* timings,
                                     bool shortlist_only) {
  using clock = std::chrono::steady_clock;
  auto ms_since = [](clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };

  SearchResults out;
  out.nq = nq;
  out.k = k;
  out.scores.assign(nq * k, 0.f);
  out.ids.assign(nq * k, 0);
  if (nq == 0 || lm_scales.empty() || k == 0) {
    return out;
  }
  if (scan_dim < n_landmarks || scan_dim % 8 != 0) {
    throw std::invalid_argument("search_landmark_rerank: scan_dim must be >= m and multiple of 8");
  }
  if (scan_bits < 2 || scan_bits > 4) {
    throw std::invalid_argument("search_landmark_rerank: scan_bits must be 2, 3, or 4");
  }
  if (residual_norms.size() != lm_scales.size()) {
    throw std::invalid_argument("search_landmark_rerank: residual_norms size mismatch");
  }
  if (!std::isfinite(score_threshold)) {
    throw std::invalid_argument(
        "search_landmark_rerank: finite score_threshold required (no top-S heap shortlist)");
  }
  if (shortlist == 0 && !shortlist_only) {
    throw std::invalid_argument(
        "search_landmark_rerank: shortlist is the mean target "
        "(buffer capacity = 2xshortlist) and must be > 0");
  }

  const std::size_t n_vectors = lm_scales.size();
  const std::size_t sl = shortlist == 0 ? n_vectors : std::min(2 * shortlist, n_vectors);
  const std::size_t kk = std::min(k, n_vectors);

  std::vector<float> q_land(nq * scan_dim, 0.f);
  std::vector<float> q_rot(nq * dim);
  std::vector<float> q_perp(nq, 0.f);

  auto t0 = clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> scratch(dim);
    std::vector<float> row(dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      auto q = queries.subspan(qis * dim, dim);
      const float nrm = row_norm(q);
      const float inv = (nrm > kMinInputNorm) ? (1.f / nrm) : 0.f;
      rotation.apply_scaled_into(q, inv, row, scratch);
      std::copy(row.begin(), row.end(), q_rot.begin() + static_cast<std::ptrdiff_t>(qis * dim));
    }
  }
  const double rotate_ms = ms_since(t0);

  t0 = clock::now();
  if (dim % 8 != 0) {
    throw std::invalid_argument("search_landmark_rerank: dim must be multiple of 8");
  }
  if (landmark_basis.size() < n_landmarks * dim) {
    throw std::invalid_argument("search_landmark_rerank: landmark_basis too short");
  }
  project_onto_landmarks(q_rot.data(), nq, dim, landmark_basis.data(), n_landmarks, q_land.data(),
                         scan_dim);
  // ||q_perp||^2 = ||q_rot||^2 - ||q_L||^2 (orthonormal landmarks).
  for (std::size_t qi = 0; qi < nq; ++qi) {
    double q2 = 0.0;
    double l2 = 0.0;
    const float* qr = q_rot.data() + qi * dim;
    const float* ql = q_land.data() + qi * scan_dim;
    for (std::size_t d = 0; d < dim; ++d) q2 += static_cast<double>(qr[d]) * qr[d];
    for (std::size_t j = 0; j < n_landmarks; ++j) l2 += static_cast<double>(ql[j]) * ql[j];
    q_perp[qi] = static_cast<float>(std::sqrt(std::max(0.0, q2 - l2)));
  }
  const double landmark_project_ms = ms_since(t0);
  const double project_ms = rotate_ms + landmark_project_ms;

  t0 = clock::now();
  auto prep = prepare_from_rows(q_land, nq, scan_dim, lm_centroids, scan_bits);
  const double shortlist_prep_ms = ms_since(t0);

  if (shortlist_only && shortlist == 0) {
    // Count-only uncapped (tune-mean-shortlist): capacity 0 outs ⇒ hit counts only.
    t0 = clock::now();
    std::vector<std::size_t> hit_counts(nq, 0);
#if defined(__x86_64__) || defined(_M_X64)
    if (prep.backend == SearchBackendKind::VmPermuteDot ||
        prep.backend == SearchBackendKind::VmVnni) {
      const bool use_pd = prep.backend == SearchBackendKind::VmPermuteDot;
      std::size_t n_threads_co = 1;
#ifdef _OPENMP
      n_threads_co = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
      const std::size_t nq_batch = nq_batch_for(use_pd, n_threads_co, nq);
      const std::size_t n_quads = (nq + nq_batch - 1) / nq_batch;
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel for schedule(dynamic)
#endif
      for (int bi = 0; bi < static_cast<int>(n_quads); ++bi) {
        const std::size_t qi_start = static_cast<std::size_t>(bi) * nq_batch;
        const std::size_t batch_nq = std::min(nq_batch, nq - qi_start);
        std::size_t local_counts[kNqBatchWide] = {};
        float rs_local[kNqBatchWide];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          rs_local[i] = q_perp[qi_start + i];
        }
        if (use_pd) {
          const QueryPermuteDot* pd_ptrs[kNqBatchWide];
          for (std::size_t i = 0; i < batch_nq; ++i) {
            pd_ptrs[i] = &prep.pds[qi_start + i];
          }
          score_queries_permute_dot_threshold_fill(
              pd_ptrs, batch_nq, blocked_codes, lm_scales, prep.n_byte_groups, n_vectors, n_blocks,
              score_threshold, /*capacity=*/0, /*out_scores=*/nullptr, /*out_ids=*/nullptr,
              local_counts, nullptr, nullptr, residual_norms.data(), rs_local);
        } else {
          const std::uint8_t* lut_ptrs[kNqBatchWide];
          float lut_scales[kNqBatchWide];
          float lut_biases[kNqBatchWide];
          for (std::size_t i = 0; i < batch_nq; ++i) {
            const std::size_t qi = qi_start + i;
            lut_ptrs[i] = prep.split_luts[qi].data();
            lut_scales[i] = prep.lut_scales[qi];
            lut_biases[i] = prep.lut_biases[qi];
          }
          score_queries_vnni_threshold_fill(
              lut_ptrs, lut_scales, lut_biases, batch_nq, blocked_codes, lm_scales,
              prep.n_byte_groups, n_vectors, n_blocks, score_threshold, /*capacity=*/0, nullptr,
              nullptr, local_counts, nullptr, nullptr, residual_norms.data(), rs_local);
        }
        for (std::size_t i = 0; i < batch_nq; ++i) {
          hit_counts[qi_start + i] = local_counts[i];
        }
      }
    } else
#endif
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        const std::size_t qis = static_cast<std::size_t>(qi);
        std::size_t count = 0;
        const auto& lut = prep.luts[qis];
        const float rs = q_perp[qis];
        for (std::size_t b = 0; b < n_blocks; ++b) {
          const std::size_t base_vec = b * kBlock;
          for (std::size_t lane = 0; lane < kBlock; ++lane) {
            const std::size_t vi = base_vec + lane;
            if (vi >= n_vectors) break;
            float score = lut.bias;
            for (std::size_t g = 0; g < prep.n_byte_groups; ++g) {
              const auto byte_val = static_cast<std::size_t>(
                  read_code(blocked_codes, prep.bits, prep.n_byte_groups, b, g, lane));
              score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + (byte_val >> 4)]);
              score += lut.scale *
                       static_cast<float>(lut.uint8_luts[g * 32 + 16 + (byte_val & 0x0F)]);
            }
            score *= lm_scales[vi];
            score += rs * residual_norms[vi];
            if (score >= score_threshold) ++count;
          }
        }
        hit_counts[qis] = count;
      }
    }
    const double shortlist_scan_ms = ms_since(t0);
    double mean_sl = 0.0;
    double max_sl = 0.0;
    for (std::size_t c : hit_counts) {
      mean_sl += static_cast<double>(c);
      max_sl = std::max(max_sl, static_cast<double>(c));
    }
    if (nq > 0) mean_sl /= static_cast<double>(nq);
    if (timings != nullptr) {
      timings->project_ms = project_ms;
      timings->rotate_ms = rotate_ms;
      timings->landmark_project_ms = landmark_project_ms;
      timings->shortlist_prep_ms = shortlist_prep_ms;
      timings->shortlist_scan_ms = shortlist_scan_ms;
      timings->shortlist_buf_ms = 0;
      timings->shortlist_kernel_ms = shortlist_scan_ms;
      timings->rerank_prep_ms = 0;
      timings->rerank_ms = 0;
      timings->rerank_score_ms = 0;
      timings->rerank_select_ms = 0;
      timings->mean_shortlist = mean_sl;
      timings->max_shortlist = max_sl;
    }
    out.k = kk;
    return out;
  }

  LandmarkStageTimings scan_detail;
  t0 = clock::now();
  SearchResults cand =
      score_prepared(prep, sl, blocked_codes, n_blocks, lm_scales, {}, {}, score_threshold,
                     timings ? &scan_detail : nullptr, residual_norms, q_perp);
  const double shortlist_scan_ms = ms_since(t0);

  double mean_sl = 0.0;
  double max_sl = 0.0;
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const std::size_t n_cand =
        cand.counts.empty() ? cand.k : std::min(cand.counts[qi], cand.k);
    mean_sl += static_cast<double>(n_cand);
    max_sl = std::max(max_sl, static_cast<double>(n_cand));
  }
  if (nq > 0) mean_sl /= static_cast<double>(nq);

  if (!shortlist_only) {
    // Final top-k from residual-corrected landmark scores (no full-dim re-rank).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::size_t n_cand =
          cand.counts.empty() ? cand.k : std::min(cand.counts[qis], cand.k);
      std::vector<std::size_t> order(n_cand);
      for (std::size_t j = 0; j < n_cand; ++j) order[j] = j;
      const float* cs = cand.scores.data() + qis * cand.k;
      const std::uint64_t* ci = cand.ids.data() + qis * cand.k;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (cs[a] != cs[b]) return cs[a] > cs[b];
        return ci[a] < ci[b];
      });
      const std::size_t take = std::min(kk, n_cand);
      for (std::size_t j = 0; j < take; ++j) {
        out.scores[qis * kk + j] = cs[order[j]];
        out.ids[qis * kk + j] = ci[order[j]];
      }
      for (std::size_t j = take; j < kk; ++j) {
        out.scores[qis * kk + j] = 0.f;
        out.ids[qis * kk + j] = 0;
      }
    }
  }

  out.k = kk;
  if (timings != nullptr) {
    timings->project_ms = project_ms;
    timings->rotate_ms = rotate_ms;
    timings->landmark_project_ms = landmark_project_ms;
    timings->shortlist_prep_ms = shortlist_prep_ms;
    timings->shortlist_scan_ms = shortlist_scan_ms;
    timings->shortlist_buf_ms = scan_detail.shortlist_buf_ms;
    timings->shortlist_kernel_ms = scan_detail.shortlist_kernel_ms;
    timings->shortlist_copy_ms = 0;
    timings->rerank_prep_ms = 0;
    timings->rerank_ms = 0;
    timings->rerank_score_ms = 0;
    timings->rerank_select_ms = 0;
    timings->mean_shortlist = mean_sl;
    timings->max_shortlist = max_sl;
  }
  return out;
}

'''
p.write_text(t[:start] + new_fn + t[end:], encoding="utf-8")
print("ok", end - start, "->", len(new_fn))
