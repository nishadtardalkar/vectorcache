//! Download TurboVec/TurboQuant benchmark datasets.
//!
//! Dependencies: a network connection and, for OpenAI embeddings, access to
//! HuggingFace datasets (no token required for these public Qdrant repos).
//!
//! Usage:
//! ```text
//! cargo run --release --bin fetch-datasets -- all
//! cargo run --release --bin fetch-datasets -- glove openai-1536
//! cargo run --release --bin fetch-datasets -- --data-dir data --force openai-3072
//! ```

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use vectorcache::datasets::{self, DatasetKind};

#[derive(Debug, Parser)]
#[command(
    name = "fetch-datasets",
    about = "Download TurboVec/TurboQuant benchmark datasets into data/"
)]
struct Args {
    /// Output directory for downloaded files (default: ./data)
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    /// Re-download even when a valid file already exists
    #[arg(long)]
    force: bool,

    /// Datasets to fetch: glove, openai-1536, openai-3072, or all
    #[arg(required = true)]
    targets: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kinds = resolve_targets(&args.targets)?;

    for kind in kinds {
        println!("Fetching {} ...", kind.label());
        datasets::fetch(kind, &args.data_dir, args.force)?;
    }

    Ok(())
}

fn resolve_targets(targets: &[String]) -> Result<Vec<DatasetKind>> {
    if targets.iter().any(|t| t == "all") {
        if targets.len() > 1 {
            bail!("pass either 'all' or explicit dataset names, not both");
        }
        return Ok(DatasetKind::all().to_vec());
    }

    let mut kinds = Vec::with_capacity(targets.len());
    for target in targets {
        match DatasetKind::parse(target) {
            Some(kind) => kinds.push(kind),
            None => bail!(
                "unknown dataset '{target}'. Available: glove, openai-1536, openai-3072, all"
            ),
        }
    }
    Ok(kinds)
}
