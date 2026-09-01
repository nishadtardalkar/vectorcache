use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::datasets::DatasetKind;

use super::npy::NpyReader;
#[cfg(feature = "glove")]
use super::hdf5::Hdf5GloveReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Test,
}

impl DatasetSplit {
    pub fn hdf5_dataset_name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetMeta {
    pub dim: usize,
    pub count: usize,
    pub label: &'static str,
}

pub trait DatasetReader {
    fn meta(&self) -> DatasetMeta;

    /// Read the next vector into `out`. Returns `Ok(true)` on success, `Ok(false)` at EOF.
    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool>;

    /// Allocate and return the next vector (convenience wrapper).
    fn next_vector(&mut self) -> Result<Option<Vec<f32>>> {
        let dim = self.meta().dim;
        let mut buf = vec![0.0; dim];
        if self.next_vector_into(&mut buf)? {
            Ok(Some(buf))
        } else {
            Ok(None)
        }
    }
}

pub enum OpenDatasetReader {
    Npy(NpyReader),
    #[cfg(feature = "glove")]
    Hdf5(Hdf5GloveReader),
}

impl DatasetReader for OpenDatasetReader {
    fn meta(&self) -> DatasetMeta {
        match self {
            Self::Npy(r) => r.meta(),
            #[cfg(feature = "glove")]
            Self::Hdf5(r) => r.meta(),
        }
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        match self {
            Self::Npy(r) => r.next_vector_into(out),
            #[cfg(feature = "glove")]
            Self::Hdf5(r) => r.next_vector_into(out),
        }
    }
}

pub fn open_dataset(
    kind: DatasetKind,
    data_dir: &Path,
    split: DatasetSplit,
) -> Result<OpenDatasetReader> {
    let path = kind.path(data_dir);

    match kind {
        DatasetKind::Glove => {
            #[cfg(not(feature = "glove"))]
            {
                let _ = (path, split);
                bail!(
                    "GloVe ingestion requires the 'glove' feature (HDF5 C library must be installed to build it)"
                );
            }
            #[cfg(feature = "glove")]
            {
                if !path.is_file() {
                    bail!("GloVe dataset not found: {}", path.display());
                }
                Ok(OpenDatasetReader::Hdf5(Hdf5GloveReader::open(&path, split)?))
            }
        }
        DatasetKind::OpenAi1536 | DatasetKind::OpenAi3072 => {
            if split == DatasetSplit::Test {
                bail!("OpenAI NPY datasets do not have a test split");
            }
            if !path.is_file() {
                bail!("OpenAI dataset not found: {}", path.display());
            }
            Ok(OpenDatasetReader::Npy(NpyReader::open(&path, kind.label())?))
        }
    }
}

pub fn dataset_path(kind: DatasetKind, data_dir: &Path) -> PathBuf {
    kind.path(data_dir)
}
