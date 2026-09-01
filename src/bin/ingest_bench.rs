//! Stage-level ingestion profiler (timing lives in support scripts, not the engine).
//!
//! ```text
//! cargo run --release --bin ingest-bench
//! cargo run --release --features glove --bin ingest-bench -- --dataset glove
//! cargo run --release --bin ingest-bench -- --npy data/openai-1536.npy --limit 50000
//! cargo run --release --bin ingest-bench -- --synthetic --limit 10000 --dim 200
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;
use vectorcache::datasets::npy::NpyReader;
use vectorcache::datasets::reader::{DatasetMeta, DatasetReader, DatasetSplit};
#[cfg(feature = "glove")]
use vectorcache::datasets::reader::open_dataset;
use vectorcache::datasets::DatasetKind;
use vectorcache::ingest::{IngestionEngine, TimingSummary};
use vectorcache::quantize::{l1_words_per_vector, quantize_4d_to_1bit_into};
use vectorcache::transform::{l2_normalize_in_place, padded_dim, SrhtRotation};

#[derive(Debug, Parser)]
#[command(name = "ingest-bench", about = "Profile ingestion stage hot paths")]
struct Args {
    /// Pre-extracted float32 NPY matrix
    #[arg(long, conflicts_with_all = ["dataset", "synthetic"])]
    npy: Option<PathBuf>,

    /// Dataset name (default: VECTORCACHE_DATASET env or glove)
    #[arg(long, env = "VECTORCACHE_DATASET", conflicts_with_all = ["npy", "synthetic"])]
    dataset: Option<String>,

    /// Generate synthetic data instead of reading a dataset
    #[arg(long, conflicts_with_all = ["npy", "dataset"])]
    synthetic: bool,

    /// Data directory (default: VECTORCACHE_DATA_DIR env or data)
    #[arg(long, env = "VECTORCACHE_DATA_DIR", default_value = "data")]
    data_dir: PathBuf,

    /// HDF5 split for GloVe (train or test)
    #[arg(long, default_value = "train")]
    split: String,

    /// Cap vectors profiled (default: full dataset)
    #[arg(long)]
    limit: Option<usize>,

    /// Vector dimension for --synthetic only
    #[arg(long, default_value_t = 200)]
    dim: usize,

    /// SRHT seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Write synthetic NPY here when --synthetic is used
    #[arg(long, default_value = "data/.cache/bench-synthetic.npy")]
    synthetic_path: PathBuf,
}

enum BenchReader {
    Npy(NpyReader),
    #[cfg(feature = "glove")]
    Open(vectorcache::datasets::reader::OpenDatasetReader),
}

impl DatasetReader for BenchReader {
    fn meta(&self) -> DatasetMeta {
        match self {
            Self::Npy(r) => r.meta(),
            #[cfg(feature = "glove")]
            Self::Open(r) => r.meta(),
        }
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        match self {
            Self::Npy(r) => r.next_vector_into(out),
            #[cfg(feature = "glove")]
            Self::Open(r) => r.next_vector_into(out),
        }
    }
}

struct LimitedReader<R> {
    inner: R,
    remaining: usize,
}

impl<R: DatasetReader> DatasetReader for LimitedReader<R> {
    fn meta(&self) -> DatasetMeta {
        self.inner.meta()
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }
        if !self.inner.next_vector_into(out)? {
            bail!("reader exhausted before reaching ingest limit");
        }
        self.remaining -= 1;
        Ok(true)
    }
}

#[derive(Debug, Default, Clone)]
struct StageTotals {
    read_ns: u64,
    normalize_ns: u64,
    srht_ns: u64,
    quantize_ns: u64,
    store_ns: u64,
    batch_copy_ns: u64,
    vectors: u64,
}

impl StageTotals {
    fn total_ns(&self) -> u64 {
        self.read_ns + self.normalize_ns + self.srht_ns + self.quantize_ns + self.store_ns + self.batch_copy_ns
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.synthetic && args.limit == Some(0) {
        bail!("--limit must be > 0");
    }

    let (mut reader, source_label) = open_reader(&args)?;
    let meta = reader.meta();
    let limit = args.limit.unwrap_or(meta.count).min(meta.count);
    if limit == 0 {
        bail!("dataset has no vectors to profile");
    }

    let padded = padded_dim(meta.dim);

    println!(
        "Ingest bench: {} (dim={}, padded={}, vectors={})",
        source_label, meta.dim, padded, limit
    );
    if args.limit.is_some() && args.limit.unwrap() < meta.count {
        println!("  (capped from {} vectors in dataset)", meta.count);
    }
    println!();

    let stages = profile_stages_inner(&mut reader, meta.dim, padded, args.seed, limit)?;
    print_stage_report("Per-stage (sequential micro-profile)", &stages);

    let (reader2, _) = open_reader(&args)?;
    let wall_ns = profile_engine_inner(reader2, meta.dim, args.seed, limit)?;
    print_wall_report(wall_ns, limit, &stages);

    Ok(())
}

fn open_reader(args: &Args) -> Result<(BenchReader, String)> {
    if let Some(path) = &args.npy {
        let reader = NpyReader::open(path, "bench")?;
        let label = format!("{} ({})", path.display(), reader.meta().label);
        return Ok((BenchReader::Npy(reader), label));
    }

    if args.synthetic {
        let count = args.limit.unwrap_or(10_000);
        if count == 0 {
            bail!("--limit must be > 0 for --synthetic");
        }
        ensure_synthetic_npy(&args.synthetic_path, count, args.dim)?;
        let reader = NpyReader::open(&args.synthetic_path, "synthetic")?;
        return Ok((
            BenchReader::Npy(reader),
            format!("{} (synthetic)", args.synthetic_path.display()),
        ));
    }

    let name = args.dataset.as_deref().unwrap_or("glove");
    let kind = DatasetKind::parse(name).with_context(|| format!("unknown dataset '{name}'"))?;
    let split = match args.split.as_str() {
        "train" => DatasetSplit::Train,
        "test" => DatasetSplit::Test,
        other => bail!("unknown split '{other}'; use train or test"),
    };

    #[cfg(feature = "glove")]
    {
        let reader = open_dataset(kind, &args.data_dir, split)?;
        let label = format!("{name} ({})", args.data_dir.display());
        Ok((BenchReader::Open(reader), label))
    }

    #[cfg(not(feature = "glove"))]
    {
        let _ = (kind, split);
        if matches!(kind, DatasetKind::OpenAi1536 | DatasetKind::OpenAi3072) {
            let path = kind.path(&args.data_dir);
            let reader = NpyReader::open(&path, kind.label())?;
            let label = format!("{name} ({})", path.display());
            return Ok((BenchReader::Npy(reader), label));
        }
        bail!(
            "dataset '{name}' requires --npy, --synthetic, or build with --features glove"
        );
    }
}

fn ensure_synthetic_npy(path: &PathBuf, count: usize, dim: usize) -> Result<()> {
    if path.is_file() {
        let reader = NpyReader::open(path, "synthetic")?;
        if reader.meta().dim == dim && reader.meta().count >= count {
            return Ok(());
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let data: Vec<f32> = (0..count * dim)
        .map(|i| (i as f32 * 0.001313).sin() * 0.5 + (i as f32 * 0.0007).cos() * 0.3)
        .collect();
    let array = Array2::from_shape_vec((count, dim), data).context("invalid synthetic shape")?;
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    array
        .write_npy(&mut file)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("Wrote synthetic NPY: {} ({count} x {dim})", path.display());
    Ok(())
}

fn profile_stages_inner<R: DatasetReader>(
    reader: &mut R,
    dim: usize,
    padded: usize,
    seed: u64,
    limit: usize,
) -> Result<StageTotals> {
    use vectorcache::ingest::INGEST_BATCH_SIZE;

    let rotation = SrhtRotation::new(dim, seed);
    let l1_words = l1_words_per_vector(padded);
    let batch_cap = INGEST_BATCH_SIZE.min(limit.max(1));

    let mut read_buf = vec![0.0; dim];
    let mut batch_inputs = vec![0.0_f32; batch_cap * dim];
    let mut batch_work: Vec<(Vec<f32>, Vec<f32>, Vec<u64>)> = (0..batch_cap)
        .map(|_| (vec![0.0; dim], vec![0.0; padded], vec![0_u64; l1_words]))
        .collect();

    let mut store = vectorcache::ingest::BlockStore::with_capacity(l1_words, limit);
    let mut totals = StageTotals::default();
    let mut processed = 0usize;

    while processed < limit {
        let mut batch_len = 0usize;
        while batch_len < batch_cap && processed + batch_len < limit {
            let t0 = Instant::now();
            if !reader.next_vector_into(&mut read_buf)? {
                bail!("reader exhausted at {} vectors", processed + batch_len);
            }
            totals.read_ns += t0.elapsed().as_nanos() as u64;

            let dst = batch_len * dim;
            batch_inputs[dst..dst + dim].copy_from_slice(&read_buf);
            batch_len += 1;
        }
        if batch_len == 0 {
            break;
        }

        let inputs = {
            let t_copy = Instant::now();
            let copied = batch_inputs[..batch_len * dim].to_vec();
            totals.batch_copy_ns += t_copy.elapsed().as_nanos() as u64;
            copied
        };

        for i in 0..batch_len {
            let (normalized, rotated, l1) = &mut batch_work[i];
            normalized.copy_from_slice(&inputs[i * dim..(i + 1) * dim]);

            let t_norm = Instant::now();
            l2_normalize_in_place(normalized);
            totals.normalize_ns += t_norm.elapsed().as_nanos() as u64;

            let t_srht = Instant::now();
            rotation.apply(normalized, rotated)?;
            totals.srht_ns += t_srht.elapsed().as_nanos() as u64;

            let t_quant = Instant::now();
            quantize_4d_to_1bit_into(rotated, l1);
            totals.quantize_ns += t_quant.elapsed().as_nanos() as u64;

            let t_store = Instant::now();
            store.push_l1_codes(l1)?;
            totals.store_ns += t_store.elapsed().as_nanos() as u64;
        }

        processed += batch_len;
        totals.vectors = processed as u64;
    }

    Ok(totals)
}

fn profile_engine_inner<R: DatasetReader>(
    reader: R,
    dim: usize,
    seed: u64,
    limit: usize,
) -> Result<u64> {
    let mut limited = LimitedReader {
        inner: reader,
        remaining: limit,
    };
    let mut engine = IngestionEngine::with_rotation(dim, seed);
    engine.reserve_vectors(limit);

    let start = Instant::now();
    let report = engine.ingest(&mut limited)?;
    let wall_ns = start.elapsed().as_nanos() as u64;

    if report.vectors_ingested as usize != limit {
        bail!(
            "engine ingested {} vectors, expected {}",
            report.vectors_ingested,
            limit
        );
    }

    Ok(wall_ns)
}

fn print_stage_report(label: &str, stages: &StageTotals) {
    let total = stages.total_ns().max(1);
    let v = stages.vectors.max(1) as f64;

    println!("{label}");
    println!("  vectors: {}", stages.vectors);
    println!(
        "  total:   {} ({:.0} ns)",
        TimingSummary::format_duration(total),
        total
    );
    println!();

    let rows = [
        ("read (mmap → buffer)", stages.read_ns),
        ("batch copy (engine-style to_vec)", stages.batch_copy_ns),
        ("L2 normalize", stages.normalize_ns),
        ("SRHT (3× sign + FWHT)", stages.srht_ns),
        ("L1 quantize", stages.quantize_ns),
        ("store (push_l1_codes)", stages.store_ns),
    ];

    println!("  {:<34} {:>12} {:>10} {:>7}", "stage", "total", "per-vec", "share");
    println!("  {}", "-".repeat(67));
    for (name, ns) in rows {
        let pct = ns as f64 / total as f64 * 100.0;
        println!(
            "  {:<34} {:>12} {:>10} {:>6.1}%",
            name,
            TimingSummary::format_duration(ns),
            TimingSummary::format_duration((ns as f64 / v) as u64),
            pct
        );
    }
    println!();
}

fn print_wall_report(wall_ns: u64, vectors: usize, stages: &StageTotals) {
    let v = vectors.max(1) as f64;
    let seq_total = stages.total_ns();

    println!("Engine ingest (parallel batches, release)");
    println!(
        "  wall:    {} ({:.0} ns)",
        TimingSummary::format_duration(wall_ns),
        wall_ns
    );
    println!(
        "  per-vec: {} ({:.0} ns)",
        TimingSummary::format_duration((wall_ns as f64 / v) as u64),
        wall_ns as f64 / v
    );
    println!();

    if wall_ns > 0 {
        let parallel_speedup = seq_total as f64 / wall_ns as f64;
        println!(
            "  parallel speedup vs sequential stages: {:.2}x",
            parallel_speedup
        );
    }

    println!();
    println!("Hot paths (by sequential stage share):");
    let mut ranked = [
        ("SRHT / FWHT", stages.srht_ns),
        ("batch input copy (to_vec per batch)", stages.batch_copy_ns),
        ("L1 quantize", stages.quantize_ns),
        ("read I/O", stages.read_ns),
        ("L2 normalize", stages.normalize_ns),
        ("store", stages.store_ns),
    ];
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (name, ns)) in ranked.iter().enumerate() {
        let pct = *ns as f64 / seq_total.max(1) as f64 * 100.0;
        println!("  {}. {} — {:.1}%", i + 1, name, pct);
    }
}
