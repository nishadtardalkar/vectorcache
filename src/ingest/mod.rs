mod block;
mod engine;
mod hook;
mod store;
mod timing;

pub use block::{VectorBlock, BLOCK_SIZE};
pub use engine::{IngestionEngine, IngestReport, INGEST_BATCH_SIZE};
pub use hook::{NoopHook, VectorHook};
pub use store::BlockStore;
pub use timing::{IngestTimings, TimingSummary};

pub use crate::quantize::{
    quantize_4d_to_1bit, quantize_4d_to_1bit_into, l1_words_per_vector, HADAMARD_4D,
};
pub use crate::transform::SrhtRotation;
