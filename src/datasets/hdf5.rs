//! HDF5 streaming reader for GloVe benchmark files.
//!
//! Requires the HDF5 C library to be installed on the build machine.
//! Enable via the `glove` crate feature.

use std::path::Path;

use anyhow::{bail, Context, Result};
use hdf5::File;

use super::reader::{DatasetMeta, DatasetReader, DatasetSplit};
use crate::ingest::BLOCK_SIZE;

const GLOVE_LABEL: &str = "glove";

/// HDF5 reader that keeps the file handle and reloads chunks efficiently.
pub struct Hdf5GloveReader {
    file: File,
    dataset_name: &'static str,
    dim: usize,
    count: usize,
    index: usize,
    chunk_buffer: Vec<f32>,
    chunk_len: usize,
    chunk_pos: usize,
}

impl Hdf5GloveReader {
    pub fn open(path: &Path, split: DatasetSplit) -> Result<Self> {
        let file = File::open(path).context("failed to open GloVe HDF5 file")?;
        let dataset_name = split.hdf5_dataset_name();
        let dataset = file
            .dataset(dataset_name)
            .with_context(|| format!("missing '{dataset_name}' dataset in GloVe HDF5"))?;

        let shape = dataset.shape();
        if shape.len() != 2 {
            bail!(
                "expected 2-D GloVe dataset '{dataset_name}', got shape {:?}",
                shape
            );
        }

        Ok(Self {
            file,
            dataset_name,
            dim: shape[1],
            count: shape[0],
            index: 0,
            chunk_buffer: Vec::new(),
            chunk_len: 0,
            chunk_pos: 0,
        })
    }

    fn ensure_chunk(&mut self) -> Result<()> {
        if self.chunk_pos < self.chunk_len {
            return Ok(());
        }
        if self.index >= self.count {
            return Ok(());
        }

        let dataset = self
            .file
            .dataset(self.dataset_name)
            .with_context(|| format!("missing '{}' dataset in GloVe HDF5", self.dataset_name))?;

        let remaining = self.count - self.index;
        let chunk_rows = remaining.min(BLOCK_SIZE);
        let buffer_len = chunk_rows * self.dim;

        self.chunk_buffer.resize(buffer_len, 0.0);
        self.chunk_len = chunk_rows;
        self.chunk_pos = 0;

        dataset
            .read_slice(
                &mut self.chunk_buffer,
                ([self.index, 0], [self.index + chunk_rows, self.dim]),
            )
            .with_context(|| {
                format!(
                    "failed to read HDF5 rows [{}..{}) from '{}'",
                    self.index,
                    self.index + chunk_rows,
                    self.dataset_name
                )
            })?;

        Ok(())
    }
}

impl DatasetReader for Hdf5GloveReader {
    fn meta(&self) -> DatasetMeta {
        DatasetMeta {
            dim: self.dim,
            count: self.count,
            label: GLOVE_LABEL,
        }
    }

    fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
        if self.index >= self.count {
            return Ok(false);
        }
        if out.len() != self.dim {
            bail!(
                "buffer dimension mismatch: expected {}, got {}",
                self.dim,
                out.len()
            );
        }

        self.ensure_chunk()?;

        let offset = self.chunk_pos * self.dim;
        let end = offset + self.dim;
        out.copy_from_slice(&self.chunk_buffer[offset..end]);
        self.chunk_pos += 1;
        self.index += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn glove_integration_if_present() {
        let path = Path::new("data/glove-200-angular.hdf5");
        if !path.is_file() {
            return;
        }

        let mut reader = Hdf5GloveReader::open(path, DatasetSplit::Train).unwrap();
        let meta = reader.meta();
        assert_eq!(meta.dim, 200);
        assert!(meta.count > 0);

        let v = reader.next_vector().unwrap().unwrap();
        assert_eq!(v.len(), 200);
    }
}
