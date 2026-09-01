use anyhow::Result;

/// Per-vector callback invoked during ingestion (e.g. future quantization).
pub trait VectorHook {
    fn on_vector(&mut self, global_id: u64, vector: &[f32]) -> Result<()>;
}

/// Default hook that performs no processing.
#[derive(Debug, Default)]
pub struct NoopHook;

impl VectorHook for NoopHook {
    fn on_vector(&mut self, _global_id: u64, _vector: &[f32]) -> Result<()> {
        Ok(())
    }
}
