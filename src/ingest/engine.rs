use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::datasets::reader::DatasetReader;
use crate::quantize::{l1_words_per_vector, quantize_4d_to_1bit_into};
use crate::transform::{l2_normalize_in_place, SrhtRotation};

use super::hook::VectorHook;
use super::store::BlockStore;

/// Vectors processed per parallel batch during ingestion.
pub const INGEST_BATCH_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IngestReport {
    pub full_blocks: usize,
    pub partial_len: usize,
    pub vectors_ingested: u64,
}

struct VectorWork {
    normalized: Vec<f32>,
    rotated: Vec<f32>,
    l1: Vec<u64>,
}

pub struct IngestionEngine {
    store: BlockStore,
    rotation: Option<SrhtRotation>,
    quantize_only: bool,
    input_dim: usize,
    padded_dim: usize,
    l1_words_per_vec: usize,
    read_buf: Vec<f32>,
    batch_inputs: Vec<f32>,
    batch_work: Vec<VectorWork>,
}

impl IngestionEngine {
    /// Ingest already-rotated vectors: L1-quantize and store bitcodes only.
    pub fn from_rotated(padded_dim: usize) -> Self {
        let l1_words = l1_words_per_vector(padded_dim);
        Self {
            store: BlockStore::new(l1_words),
            rotation: None,
            quantize_only: true,
            input_dim: padded_dim,
            padded_dim,
            l1_words_per_vec: l1_words,
            read_buf: vec![0.0; padded_dim],
            batch_inputs: Vec::new(),
            batch_work: Vec::new(),
        }
    }

    pub fn with_rotation(original_dim: usize, seed: u64) -> Self {
        let rotation = SrhtRotation::new(original_dim, seed);
        let padded = rotation.padded_dim();
        let l1_words = l1_words_per_vector(padded);
        Self {
            store: BlockStore::new(l1_words),
            rotation: Some(rotation),
            quantize_only: false,
            input_dim: original_dim,
            padded_dim: padded,
            l1_words_per_vec: l1_words,
            read_buf: vec![0.0; original_dim],
            batch_inputs: Vec::new(),
            batch_work: Vec::new(),
        }
    }

    /// Pre-size internal buffers and block store when vector count is known.
    pub fn reserve_vectors(&mut self, count: usize) {
        self.store = BlockStore::with_capacity(self.l1_words_per_vec, count);
        self.ensure_batch_capacity(INGEST_BATCH_SIZE.min(count.max(1)));
    }

    pub fn rotation(&self) -> Option<&SrhtRotation> {
        self.rotation.as_ref()
    }

    pub fn push_l1_codes(&mut self, codes: &[u64]) -> Result<()> {
        self.store.push_l1_codes(codes)
    }

    /// Ingest vectors with no per-vector hook (default).
    pub fn ingest(&mut self, reader: &mut dyn DatasetReader) -> Result<IngestReport> {
        let mut hook: Option<&mut crate::ingest::NoopHook> = None;
        self.ingest_with_hook(reader, &mut hook)
    }

    /// Ingest vectors, invoking `hook` on each rotated vector before L1 storage.
    pub fn ingest_with_hook<H: VectorHook + ?Sized>(
        &mut self,
        reader: &mut dyn DatasetReader,
        hook: &mut Option<&mut H>,
    ) -> Result<IngestReport> {
        let meta_count = reader.meta().count;
        if meta_count > 0 {
            self.reserve_vectors(meta_count);
        } else {
            self.ensure_batch_capacity(INGEST_BATCH_SIZE);
        }

        let mut global_id = 0_u64;

        loop {
            let batch_len = self.read_batch(reader)?;
            if batch_len == 0 {
                break;
            }

            self.process_batch_parallel(batch_len)?;

            for i in 0..batch_len {
                let work = &self.batch_work[i];
                if let Some(h) = hook.as_mut() {
                    h.on_vector(global_id, &work.rotated)?;
                }
                self.store.push_l1_codes(&work.l1)?;
                global_id += 1;
            }
        }

        Ok(IngestReport {
            full_blocks: self.store.block_count(),
            partial_len: self.store.partial_block().len(),
            vectors_ingested: global_id,
        })
    }

    fn ensure_batch_capacity(&mut self, batch_cap: usize) {
        if self.batch_work.len() < batch_cap {
            self.batch_work.resize_with(batch_cap, || VectorWork {
                normalized: vec![0.0; self.input_dim],
                rotated: vec![0.0; self.padded_dim],
                l1: vec![0_u64; self.l1_words_per_vec],
            });
        }
        let input_bytes = batch_cap * self.input_dim;
        if self.batch_inputs.len() < input_bytes {
            self.batch_inputs.resize(input_bytes, 0.0);
        }
    }

    fn read_batch(&mut self, reader: &mut dyn DatasetReader) -> Result<usize> {
        let max_batch = self.batch_work.len();
        let mut count = 0;
        while count < max_batch {
            if !reader.next_vector_into(&mut self.read_buf)? {
                break;
            }
            let dst_start = count * self.input_dim;
            self.batch_inputs[dst_start..dst_start + self.input_dim]
                .copy_from_slice(&self.read_buf);
            count += 1;
        }
        Ok(count)
    }

    fn process_batch_parallel(&mut self, batch_len: usize) -> Result<()> {
        let input_dim = self.input_dim;
        let rotation = self.rotation.clone();
        let quantize_only = self.quantize_only;
        let inputs = self.batch_inputs[..batch_len * input_dim].to_vec();

        self.batch_work[..batch_len]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(i, work)| {
                let input = &inputs[i * input_dim..(i + 1) * input_dim];
                if let Some(ref rot) = rotation {
                    work.normalized.copy_from_slice(input);
                    l2_normalize_in_place(&mut work.normalized);
                    rot.apply(&work.normalized, &mut work.rotated)?;
                } else if quantize_only {
                    work.rotated.copy_from_slice(input);
                } else {
                    bail!("IngestionEngine requires with_rotation() or from_rotated()");
                }
                quantize_4d_to_1bit_into(&work.rotated, &mut work.l1);
                Ok::<(), anyhow::Error>(())
            })?;

        Ok(())
    }

    pub fn store(&self) -> &BlockStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::datasets::reader::{DatasetMeta, DatasetReader};
    use crate::quantize::{l1_words_per_vector, quantize_4d_to_1bit};
    use crate::transform::padded_dim;

    use crate::ingest::block::BLOCK_SIZE;

    use super::*;
    struct MockReader {
        vectors: Vec<Vec<f32>>,
        index: usize,
        dim: usize,
    }

    impl MockReader {
        fn new(vectors: Vec<Vec<f32>>, dim: usize) -> Self {
            Self {
                vectors,
                index: 0,
                dim,
            }
        }
    }

    impl DatasetReader for MockReader {
        fn meta(&self) -> DatasetMeta {
            DatasetMeta {
                dim: self.dim,
                count: self.vectors.len(),
                label: "mock",
            }
        }

        fn next_vector_into(&mut self, out: &mut [f32]) -> Result<bool> {
            if self.index >= self.vectors.len() {
                return Ok(false);
            }
            let vector = &self.vectors[self.index];
            out.copy_from_slice(vector);
            self.index += 1;
            Ok(true)
        }
    }

    struct CountingHook {
        count: u64,
    }

    impl VectorHook for CountingHook {
        fn on_vector(&mut self, _global_id: u64, _vector: &[f32]) -> Result<()> {
            self.count += 1;
            Ok(())
        }
    }

    struct CapturingHook {
        last: Option<Vec<f32>>,
    }

    impl VectorHook for CapturingHook {
        fn on_vector(&mut self, _global_id: u64, vector: &[f32]) -> Result<()> {
            self.last = Some(vector.to_vec());
            Ok(())
        }
    }

    #[test]
    fn ingest_default_has_no_hook() {
        let dim = 4;
        let n = BLOCK_SIZE * 2 + 5;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![i as f32, i as f32 + 1.0, i as f32 + 2.0, i as f32 + 3.0])
            .collect();

        let mut reader = MockReader::new(vectors, dim);
        let mut engine = IngestionEngine::with_rotation(dim, 42);
        let report = engine.ingest(&mut reader).unwrap();

        assert_eq!(report.vectors_ingested, n as u64);
        assert_eq!(report.full_blocks, 2);
        assert_eq!(report.partial_len, 5);
        assert_eq!(engine.store().total_vectors(), n);
    }

    #[test]
    fn ingest_with_hook_counts_vectors() {
        let dim = 4;
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut reader = MockReader::new(vectors, dim);
        let mut engine = IngestionEngine::with_rotation(dim, 42);
        let mut hook = CountingHook { count: 0 };
        let mut hook_ref = Some(&mut hook);
        let report = engine.ingest_with_hook(&mut reader, &mut hook_ref).unwrap();
        assert_eq!(hook.count, 1);
        assert_eq!(report.vectors_ingested, 1);
    }

    #[test]
    fn ingest_with_rotation_stores_l1_codes() {
        let dim = 4;
        let n = 3;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![i as f32, i as f32 + 1.0, i as f32 + 2.0, i as f32 + 3.0])
            .collect();

        let mut reader = MockReader::new(vectors, dim);
        let mut engine = IngestionEngine::with_rotation(dim, 42);
        let mut hook = CapturingHook { last: None };

        let padded = padded_dim(dim);
        let words_per_vec = l1_words_per_vector(padded);
        assert_eq!(engine.store().l1_words_per_vec(), words_per_vec);

        let mut hook_ref = Some(&mut hook);
        let report = engine.ingest_with_hook(&mut reader, &mut hook_ref).unwrap();

        assert_eq!(report.vectors_ingested, n as u64);
        assert_eq!(engine.store().total_vectors(), n);

        let rotated = hook.last.unwrap();
        let (expected_codes, _) = quantize_4d_to_1bit(&rotated);
        let stored = engine.store().partial_block().as_slice();
        let offset = (n - 1) * words_per_vec;
        assert_eq!(&stored[offset..offset + words_per_vec], expected_codes.as_slice());
    }

    #[test]
    fn ingest_with_rotation_normalizes_before_srht() {
        let dim = 4;
        let vectors = vec![vec![3.0_f32, 4.0, 0.0, 0.0]];

        let mut reader = MockReader::new(vectors, dim);
        let mut engine = IngestionEngine::with_rotation(dim, 42);
        let mut hook = CapturingHook { last: None };

        let mut hook_ref = Some(&mut hook);
        engine.ingest_with_hook(&mut reader, &mut hook_ref).unwrap();

        let rotated = hook.last.unwrap();
        let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "SRHT output should be unit norm after L2 normalization, got {norm}"
        );
    }

    #[test]
    fn from_rotated_quantizes_without_srht() {
        let dim = 4;
        let vector = vec![0.6_f32, 0.8, 0.0, 0.0];
        let (expected, _) = quantize_4d_to_1bit(&vector);

        let mut reader = MockReader::new(vec![vector], dim);
        let mut engine = IngestionEngine::from_rotated(dim);

        engine.ingest(&mut reader).unwrap();

        let stored = engine.store().partial_block().as_slice();
        assert_eq!(stored, expected.as_slice());
    }
}
