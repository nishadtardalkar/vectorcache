//! Ingest a sample of vectors and report per-vector dimension variance.
//!
//! Pre-ingestion variance is computed on raw dataset vectors.
//! Post-ingestion variance is computed on L2-normalized + SRHT-rotated vectors.
//!
//! Usage:
//! ```text
//! cargo run --release --bin ingest-sample -- --npy data/.cache/glove-sample-100.npy
//! cargo run --release --features glove --bin ingest-sample -- --dataset glove --limit 100
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use vectorcache::datasets::npy::NpyReader;
use vectorcache::datasets::reader::{DatasetReader, DatasetSplit};
#[cfg(feature = "glove")]
use vectorcache::datasets::reader::open_dataset;
use vectorcache::datasets::DatasetKind;
use vectorcache::ingest::{IngestionEngine, TimingSummary, VectorHook};
use vectorcache::transform::{l2_normalize_in_place, padded_dim, SrhtRotation};

#[derive(Debug, Parser)]
#[command(
    name = "ingest-sample",
    about = "Ingest a limited number of vectors and report per-vector dimension variance"
)]
struct Args {
    /// Pre-extracted float32 NPY matrix (bypasses dataset reader)
    #[arg(long)]
    npy: Option<PathBuf>,

    /// Dataset name (default: VECTORCACHE_DATASET env or glove)
    #[arg(long, env = "VECTORCACHE_DATASET", conflicts_with = "npy")]
    dataset: Option<String>,

    /// Data directory (default: VECTORCACHE_DATA_DIR env or data)
    #[arg(long, env = "VECTORCACHE_DATA_DIR", default_value = "data")]
    data_dir: PathBuf,

    /// Maximum number of vectors to ingest
    #[arg(long, default_value_t = 100)]
    limit: usize,

    /// HDF5 split for GloVe (train or test)
    #[arg(long, default_value = "train")]
    split: String,

    /// SRHT rotation seed (each round uses seed + round index)
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of consecutive SRHT rounds to apply during ingestion
    #[arg(long, default_value_t = 1)]
    rounds: usize,

    /// Print the stored vector at this index after ingestion
    #[arg(long)]
    show_index: Option<usize>,
}

enum SampleReader {
    Npy(NpyReader),
    #[cfg(feature = "glove")]
    Open(vectorcache::datasets::reader::OpenDatasetReader),
}

impl DatasetReader for SampleReader {
    fn meta(&self) -> vectorcache::datasets::reader::DatasetMeta {
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
    pre_variances: Vec<f64>,
}

impl<R: DatasetReader> DatasetReader for LimitedReader<R> {
    fn meta(&self) -> vectorcache::datasets::reader::DatasetMeta {
        self.inner.meta()
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }

        if !self
            .inner
            .next_vector_into(out)?
        {
            bail!("reader exhausted before reaching ingest limit");
        }
        self.pre_variances.push(variance_across_dims(out));
        self.remaining -= 1;
        Ok(true)
    }
}

struct VarianceHook {
    post_variances: Vec<f64>,
    vectors: Vec<Vec<f32>>,
    capture_vectors: bool,
}

impl VarianceHook {
    fn new(capture_vectors: bool) -> Self {
        Self {
            post_variances: Vec::new(),
            vectors: Vec::new(),
            capture_vectors,
        }
    }
}

impl VectorHook for VarianceHook {
    fn on_vector(&mut self, _global_id: u64, vector: &[f32]) -> Result<()> {
        self.post_variances.push(variance_across_dims(vector));
        if self.capture_vectors {
            self.vectors.push(vector.to_vec());
        }
        Ok(())
    }
}

struct MultiRoundReader<R> {
    inner: R,
    remaining: usize,
    pre_variances: Vec<f64>,
    rounds: usize,
    seed: u64,
    round_variances: Vec<Vec<f64>>,
    raw: Vec<f32>,
    scratch: Vec<f32>,
    output: Vec<f32>,
}

impl<R: DatasetReader> DatasetReader for MultiRoundReader<R> {
    fn meta(&self) -> vectorcache::datasets::reader::DatasetMeta {
        self.inner.meta()
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }

        if !self.inner.next_vector_into(&mut self.raw)? {
            bail!("reader exhausted before reaching ingest limit");
        }
        self.pre_variances.push(variance_across_dims(&self.raw));
        self.remaining -= 1;

        self.output.copy_from_slice(&self.raw);
        l2_normalize_in_place(&mut self.output);
        for round in 0..self.rounds {
            let rot = SrhtRotation::new(self.output.len(), self.seed + round as u64);
            self.scratch.resize(rot.padded_dim(), 0.0);
            rot.apply(&self.output, &mut self.scratch)?;
            if self.round_variances.len() <= round {
                self.round_variances.push(Vec::new());
            }
            self.round_variances[round].push(variance_across_dims(&self.scratch));
            self.output.copy_from_slice(&self.scratch);
        }

        if out.len() != self.output.len() {
            bail!(
                "output buffer dimension mismatch: expected {}, got {}",
                self.output.len(),
                out.len()
            );
        }
        out.copy_from_slice(&self.output);
        Ok(true)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let split = match args.split.as_str() {
        "train" => DatasetSplit::Train,
        "test" => DatasetSplit::Test,
        other => bail!("unknown split '{other}'; use train or test"),
    };

    if args.rounds == 0 {
        bail!("--rounds must be at least 1");
    }

    let reader = if let Some(path) = &args.npy {
        SampleReader::Npy(NpyReader::open(path, "glove-sample")?)
    } else {
        let name = args.dataset.as_deref().unwrap_or("glove");
        let kind = DatasetKind::parse(name)
            .with_context(|| format!("unknown dataset '{name}'"))?;
        #[cfg(feature = "glove")]
        {
            SampleReader::Open(open_dataset(kind, &args.data_dir, split)?)
        }
        #[cfg(not(feature = "glove"))]
        {
            let _ = (kind, split);
            bail!(
                "dataset '{name}' requires the 'glove' feature or pass --npy with a pre-extracted sample",
            );
        }
    };

    let meta = reader.meta();
    let limit = args.limit.min(meta.count);
    let padded = padded_dim(meta.dim);
    let capture_vectors = args.show_index.is_some();

    println!(
        "Dataset: {} (dim={}, padded={}, available={}, ingesting={}, srht_seed={}, rounds={})",
        meta.label, meta.dim, padded, meta.count, limit, args.seed, args.rounds
    );

    let mut hook = VarianceHook::new(capture_vectors);
    hook.post_variances.reserve(limit);
    if capture_vectors {
        hook.vectors.reserve(limit);
    }

    let ingest_start = Instant::now();

    let (pre_variances, round_variances, report, engine) = if args.rounds == 1 {
        let mut limited = LimitedReader {
            inner: reader,
            remaining: limit,
            pre_variances: Vec::with_capacity(limit),
        };
        let mut engine = IngestionEngine::with_rotation(meta.dim, args.seed);
        engine.reserve_vectors(limit);
        let mut hook_ref = Some(&mut hook);
        let report = engine.ingest(&mut limited, &mut hook_ref)?;
        (limited.pre_variances, Vec::new(), report, engine)
    } else {
        let mut limited = MultiRoundReader {
            inner: reader,
            remaining: limit,
            pre_variances: Vec::with_capacity(limit),
            rounds: args.rounds,
            seed: args.seed,
            round_variances: Vec::new(),
            raw: vec![0.0; meta.dim],
            scratch: Vec::new(),
            output: vec![0.0; meta.dim],
        };
        let store_dim = {
            let mut dim = meta.dim;
            for _ in 0..args.rounds {
                dim = padded_dim(dim);
            }
            dim
        };
        let mut engine = IngestionEngine::from_rotated(store_dim);
        engine.reserve_vectors(limit);
        let mut hook_ref = Some(&mut hook);
        let report = engine.ingest(&mut limited, &mut hook_ref)?;
        (
            limited.pre_variances,
            limited.round_variances,
            report,
            engine,
        )
    };

    let elapsed_ns = ingest_start.elapsed().as_nanos() as u64;

    print_variance_stats("Pre-ingestion (raw)", &pre_variances);
    for (i, round_stats) in round_variances.iter().enumerate() {
        print_variance_stats(&format!("After SRHT round {}", i + 1), round_stats);
    }
    print_variance_stats("Post-ingestion (stored)", &hook.post_variances);

    println!("Ingested: {} vectors", report.vectors_ingested);
    println!(
        "Total ingestion time: {} ({elapsed_ns:.0} ns)",
        TimingSummary::format_duration(elapsed_ns)
    );
    if report.vectors_ingested > 0 {
        let per_vec = elapsed_ns as f64 / report.vectors_ingested as f64;
        println!(
            "  per-vector: {} ({per_vec:.0} ns)",
            TimingSummary::format_duration(per_vec as u64)
        );
    }
    println!(
        "Blocks: {} full, {} in partial block (L1 words/vec: {})",
        report.full_blocks,
        report.partial_len,
        engine.store().l1_words_per_vec()
    );

    if let Some(index) = args.show_index {
        if index >= report.vectors_ingested as usize {
            bail!(
                "--show-index {index} out of range (ingested {})",
                report.vectors_ingested
            );
        }
        print_stored_l1_codes(index, &engine, padded);
        if let Some(vector) = hook.vectors.get(index) {
            print_rotated_vector(index, vector);
        }
    }

    Ok(())
}

fn print_stored_l1_codes(index: usize, engine: &IngestionEngine, padded_dim: usize) {
    let words_per_vec = engine.store().l1_words_per_vec();
    let num_bits = (padded_dim + 3) / 4;
    let block_idx = index / vectorcache::ingest::BLOCK_SIZE;
    let vec_in_block = index % vectorcache::ingest::BLOCK_SIZE;

    let block = if block_idx < engine.store().block_count() {
        engine.store().get_block(block_idx).unwrap()
    } else {
        engine.store().partial_block()
    };

    let offset = vec_in_block * words_per_vec;
    let codes = &block.as_slice()[offset..offset + words_per_vec];
    let hex: Vec<String> = codes.iter().map(|w| format!("{w:#018x}")).collect();
    println!(
        "Stored L1 codes at index {index} ({num_bits} bits, {} u64 words):",
        words_per_vec
    );
    println!("  [{}]", hex.join(", "));
}

fn print_rotated_vector(index: usize, vector: &[f32]) {
    println!(
        "Rotated f32 at index {index} (dim={}, L2 norm={:.6}):",
        vector.len(),
        l2_norm(vector)
    );
    let preview = 16.min(vector.len());
    let head: Vec<String> = vector[..preview]
        .iter()
        .map(|x| format!("{x:+.6}"))
        .collect();
    println!("  [{}]", head.join(", "));
    if vector.len() > preview {
        println!("  ... ({} more dims)", vector.len() - preview);
    }
}

fn l2_norm(vector: &[f32]) -> f64 {
    vector.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

fn print_variance_stats(label: &str, variances: &[f64]) {
    if variances.is_empty() {
        println!("{label}: (no vectors)");
        return;
    }
    let avg = variances.iter().sum::<f64>() / variances.len() as f64;
    let min = variances.iter().copied().fold(f64::INFINITY, f64::min);
    let max = variances.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("{label} per-vector variance across dims:");
    println!("  average: {avg:.8}");
    println!("  min:     {min:.8}");
    println!("  max:     {max:.8}");
}

/// Population variance of a single vector's components across its dimensions.
fn variance_across_dims(vector: &[f32]) -> f64 {
    if vector.is_empty() {
        return 0.0;
    }

    let n = vector.len() as f64;
    let mean = vector.iter().map(|&x| x as f64).sum::<f64>() / n;
    let sum_sq = vector
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>();
    sum_sq / n
}
