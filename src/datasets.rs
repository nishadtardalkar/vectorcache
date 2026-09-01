use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub mod reader;
#[cfg(feature = "glove")]
mod hdf5;
pub mod npy;

use anyhow::{bail, Context, Result};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use reqwest::blocking::Client;

const GLOVE_URL: &str = "http://ann-benchmarks.com/glove-200-angular.hdf5";
const GLOVE_FILENAME: &str = "glove-200-angular.hdf5";
const GLOVE_MIN_BYTES: u64 = 100_000_000;
const OPENAI_SHARDS: usize = 26;
const HF_DATASET_BASE: &str = "https://huggingface.co/datasets";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetKind {
    Glove,
    OpenAi1536,
    OpenAi3072,
}

impl DatasetKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "glove" => Some(Self::Glove),
            "openai-1536" => Some(Self::OpenAi1536),
            "openai-3072" => Some(Self::OpenAi3072),
            "all" => None,
            _ => None,
        }
    }

    pub fn all() -> [Self; 3] {
        [Self::Glove, Self::OpenAi1536, Self::OpenAi3072]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Glove => "glove",
            Self::OpenAi1536 => "openai-1536",
            Self::OpenAi3072 => "openai-3072",
        }
    }

    pub fn expected_dim(self) -> usize {
        match self {
            Self::Glove => 200,
            Self::OpenAi1536 => 1536,
            Self::OpenAi3072 => 3072,
        }
    }

    pub fn path(self, data_dir: &Path) -> PathBuf {
        match self {
            Self::Glove => data_dir.join(GLOVE_FILENAME),
            Self::OpenAi1536 => data_dir.join("openai-1536.npy"),
            Self::OpenAi3072 => data_dir.join("openai-3072.npy"),
        }
    }
}

pub fn fetch(kind: DatasetKind, data_dir: &Path, force: bool) -> Result<()> {
    fs::create_dir_all(data_dir)?;

    match kind {
        DatasetKind::Glove => fetch_glove(data_dir, force),
        DatasetKind::OpenAi1536 => fetch_openai(1536, data_dir, force),
        DatasetKind::OpenAi3072 => fetch_openai(3072, data_dir, force),
    }
}

fn fetch_glove(data_dir: &Path, force: bool) -> Result<()> {
    let dest = data_dir.join(GLOVE_FILENAME);
    if !force && validate_glove(&dest).is_ok() {
        print_dataset_status("GloVe", &dest, "train + test HDF5 datasets", true)?;
        return Ok(());
    }

    println!("Downloading {GLOVE_URL} ...");
    let client = http_client()?;
    let mut response = client
        .get(GLOVE_URL)
        .send()
        .context("failed to download GloVe dataset")?
        .error_for_status()
        .context("GloVe download returned an error status")?;

    let tmp = dest.with_extension("hdf5.tmp");
    write_atomic(&mut response, &tmp, &dest)?;
    validate_glove(&dest).context("downloaded GloVe file failed validation")?;
    print_dataset_status("GloVe", &dest, "train + test HDF5 datasets", false)?;
    Ok(())
}

fn fetch_openai(dim: usize, data_dir: &Path, force: bool) -> Result<()> {
    let filename = format!("openai-{dim}.npy");
    let dest = data_dir.join(&filename);
    if !force && validate_openai(&dest, dim).is_ok() {
        let shape = read_openai_shape(&dest)?;
        print_dataset_status(
            &format!("OpenAI-{dim}"),
            &dest,
            &format!("shape ({}, {})", shape.0, shape.1),
            true,
        )?;
        return Ok(());
    }

    let repo_id = format!(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    );
    let column = format!("text-embedding-3-large-{dim}-embedding");
    println!("Downloading {repo_id} ({OPENAI_SHARDS} parquet shards) ...");

    let client = http_client()?;
    let cache_dir = data_dir.join(".cache").join(format!("openai-{dim}"));
    fs::create_dir_all(&cache_dir)?;

    let mut vectors = Vec::with_capacity(1_000_000 * dim);
    for shard in 0..OPENAI_SHARDS {
        let shard_name = format!("data/train-{shard:05}-of-{OPENAI_SHARDS:05}.parquet");
        println!("  shard {}/{}: {shard_name}", shard + 1, OPENAI_SHARDS);
        let path = download_hf_dataset_file(&client, &repo_id, &shard_name, &cache_dir)?;
        append_embeddings_from_parquet(&path, &column, dim, &mut vectors)?;
    }

    if vectors.len() / dim != 1_000_000 {
        bail!(
            "expected 1,000,000 vectors for OpenAI-{dim}, got {}",
            vectors.len() / dim
        );
    }

    let tmp = dest.with_extension("npy.tmp");
    write_openai_npy(&tmp, &vectors, dim)?;
    fs::rename(&tmp, &dest).with_context(|| format!("failed to move {} into place", dest.display()))?;

    validate_openai(&dest, dim).context("downloaded OpenAI file failed validation")?;
    print_dataset_status(
        &format!("OpenAI-{dim}"),
        &dest,
        &format!("shape (1000000, {dim})"),
        false,
    )?;
    Ok(())
}

fn http_client() -> Result<Client> {
    Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .context("failed to build HTTP client")
}

fn download_hf_dataset_file(
    client: &Client,
    repo_id: &str,
    file_path: &str,
    cache_dir: &Path,
) -> Result<std::path::PathBuf> {
    let file_name = file_path.rsplit('/').next().unwrap_or(file_path);
    let dest = cache_dir.join(file_name);
    if dest.is_file() {
        return Ok(dest);
    }

    let url = format!("{HF_DATASET_BASE}/{repo_id}/resolve/main/{file_path}");
    let mut response = client
        .get(&url)
        .send()
        .with_context(|| format!("failed to download {url}"))?
        .error_for_status()
        .with_context(|| format!("download returned an error status for {url}"))?;

    let tmp = dest.with_extension("parquet.tmp");
    write_atomic(&mut response, &tmp, &dest)?;
    Ok(dest)
}

fn write_atomic(reader: &mut impl Read, tmp: &Path, dest: &Path) -> Result<()> {
    if let Some(parent) = tmp.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(tmp)
        .with_context(|| format!("failed to create temporary file {}", tmp.display()))?;
    io::copy(reader, &mut file)
        .with_context(|| format!("failed while writing {}", tmp.display()))?;
    file.sync_all()?;
    drop(file);

    fs::rename(tmp, dest)
        .with_context(|| format!("failed to move {} into place", dest.display()))?;
    Ok(())
}

fn validate_glove(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("file does not exist: {}", path.display());
    }

    let metadata = fs::metadata(path)?;
    if metadata.len() < GLOVE_MIN_BYTES {
        bail!(
            "GloVe file is too small ({} bytes); expected at least {GLOVE_MIN_BYTES}",
            metadata.len()
        );
    }

    let mut header = [0_u8; 8];
    let mut file = File::open(path)?;
    file.read_exact(&mut header)?;
    const HDF5_SIGNATURE: &[u8] = b"\x89HDF\r\n\x1a\n";
    if header != HDF5_SIGNATURE {
        bail!("file does not look like HDF5: {}", path.display());
    }

    #[cfg(feature = "glove")]
    {
        let file = hdf5::File::open(path).context("failed to open GloVe HDF5 file")?;
        file.dataset("train")
            .context("missing 'train' dataset in GloVe HDF5")?;
        file.dataset("test")
            .context("missing 'test' dataset in GloVe HDF5")?;
    }

    Ok(())
}

fn validate_openai(path: &Path, dim: usize) -> Result<()> {
    let (rows, cols) = read_openai_shape(path)?;
    if rows != 1_000_000 || cols != dim {
        bail!("unexpected OpenAI shape: ({rows}, {cols}), expected (1000000, {dim})");
    }
    Ok(())
}

fn read_openai_shape(path: &Path) -> Result<(usize, usize)> {
    let array: Array2<f32> = ndarray_npy::read_npy(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok((array.nrows(), array.ncols()))
}

fn append_embeddings_from_parquet(
    path: &Path,
    column_name: &str,
    dim: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read parquet {}", path.display()))?;
    let reader = builder.build()?;

    for batch in reader {
        let batch = batch.with_context(|| format!("failed to read batch from {}", path.display()))?;
        let column = batch
            .column_by_name(column_name)
            .with_context(|| format!("missing column '{column_name}' in {}", path.display()))?;

        let values = extract_fixed_size_list_f32(column, dim, path)?;
        out.extend_from_slice(&values);
    }

    Ok(())
}

fn extract_fixed_size_list_f32(column: &dyn Array, dim: usize, path: &Path) -> Result<Vec<f32>> {
    let list = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .with_context(|| {
            format!(
                "expected fixed-size list column in {}, got {:?}",
                path.display(),
                column.data_type()
            )
        })?;

    if list.value_length() as usize != dim {
        bail!(
            "unexpected embedding width in {}: expected {dim}, got {}",
            path.display(),
            list.value_length()
        );
    }

    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .context("expected float32 values inside embedding list")?;

    if values.len() % dim != 0 {
        bail!(
            "embedding values in {} are not divisible by dimension {dim}",
            path.display()
        );
    }

    Ok(values.values().to_vec())
}

fn write_openai_npy(path: &Path, vectors: &[f32], dim: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let array = Array2::from_shape_vec((vectors.len() / dim, dim), vectors.to_vec())
        .context("invalid embedding buffer shape")?;
    let mut file = File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    array
        .write_npy(&mut file)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.sync_all()?;
    Ok(())
}

fn print_dataset_status(label: &str, path: &Path, detail: &str, skipped: bool) -> Result<()> {
    let size_mb = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len() as f64
        / (1024.0 * 1024.0);
    let prefix = if skipped { "[skip]" } else { "Saved" };
    println!(
        "{prefix} {label}: {} ({detail}, {size_mb:.0} MB)",
        path.display()
    );
    Ok(())
}
