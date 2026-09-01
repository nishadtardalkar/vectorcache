use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::fwht::{apply_signs_i8, fwht_orthonormal_in_place, padded_dim};

/// TurboQuant-style 3-round SRHT: H·D₃·H·D₂·H·D₁ with orthonormal H.
///
/// One seeded instance is shared across all vectors in an index (data-oblivious).
#[derive(Debug, Clone)]
pub struct SrhtRotation {
    original_dim: usize,
    padded_dim: usize,
    signs1: Vec<i8>,
    signs2: Vec<i8>,
    signs3: Vec<i8>,
}

impl SrhtRotation {
    pub fn new(original_dim: usize, seed: u64) -> Self {
        assert!(original_dim > 0, "original_dim must be > 0");
        let n = padded_dim(original_dim);
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            original_dim,
            padded_dim: n,
            signs1: rademacher_i8(&mut rng, n),
            signs2: rademacher_i8(&mut rng, n),
            signs3: rademacher_i8(&mut rng, n),
        }
    }

    pub fn original_dim(&self) -> usize {
        self.original_dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Zero-pad `vector`, apply SRHT, write result into `out` (length `padded_dim`).
    pub fn apply(&self, vector: &[f32], out: &mut [f32]) -> Result<()> {
        if vector.len() != self.original_dim {
            bail!(
                "vector dimension mismatch: expected {}, got {}",
                self.original_dim,
                vector.len()
            );
        }
        if out.len() != self.padded_dim {
            bail!(
                "output buffer dimension mismatch: expected {}, got {}",
                self.padded_dim,
                out.len()
            );
        }

        out.fill(0.0);
        out[..self.original_dim].copy_from_slice(vector);

        apply_signs_i8(out, &self.signs1);
        fwht_orthonormal_in_place(out)?;

        apply_signs_i8(out, &self.signs2);
        fwht_orthonormal_in_place(out)?;

        apply_signs_i8(out, &self.signs3);
        fwht_orthonormal_in_place(out)?;

        Ok(())
    }
}

fn rademacher_i8(rng: &mut StdRng, n: usize) -> Vec<i8> {
    (0..n)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn pad_vector(v: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0; n];
        out[..v.len()].copy_from_slice(v);
        out
    }

    #[test]
    fn same_seed_same_output() {
        let a = SrhtRotation::new(4, 42);
        let b = SrhtRotation::new(4, 42);
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out_a = vec![0.0; 4];
        let mut out_b = vec![0.0; 4];
        a.apply(&input, &mut out_a).unwrap();
        b.apply(&input, &mut out_b).unwrap();
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn different_seed_different_output() {
        let a = SrhtRotation::new(4, 42);
        let b = SrhtRotation::new(4, 99);
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out_a = vec![0.0; 4];
        let mut out_b = vec![0.0; 4];
        a.apply(&input, &mut out_a).unwrap();
        b.apply(&input, &mut out_b).unwrap();
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn preserves_norm_with_padding() {
        let rot = SrhtRotation::new(200, 7);
        assert_eq!(rot.padded_dim(), 256);
        let input: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let mut out = vec![0.0; rot.padded_dim()];
        rot.apply(&input, &mut out).unwrap();
        let norm_in = l2_norm(&input);
        let norm_out = l2_norm(&out);
        assert!((norm_in - norm_out).abs() < 1e-4, "{norm_in} vs {norm_out}");
    }

    #[test]
    fn preserves_inner_product_with_padding() {
        let rot = SrhtRotation::new(1536, 123);
        assert_eq!(rot.padded_dim(), 2048);
        let x: Vec<f32> = (0..1536).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..1536).map(|i| (i as f32).cos()).collect();
        let mut rx = vec![0.0; rot.padded_dim()];
        let mut ry = vec![0.0; rot.padded_dim()];
        rot.apply(&x, &mut rx).unwrap();
        rot.apply(&y, &mut ry).unwrap();

        let x_pad = pad_vector(&x, rot.padded_dim());
        let y_pad = pad_vector(&y, rot.padded_dim());
        let ip_before = dot(&x_pad, &y_pad);
        let ip_after = dot(&rx, &ry);
        assert!(
            (ip_before - ip_after).abs() < 1e-3,
            "{ip_before} vs {ip_after}"
        );
    }

    #[test]
    fn padded_dims_for_benchmark_datasets() {
        assert_eq!(SrhtRotation::new(200, 0).padded_dim(), 256);
        assert_eq!(SrhtRotation::new(1536, 0).padded_dim(), 2048);
        assert_eq!(SrhtRotation::new(3072, 0).padded_dim(), 4096);
    }
}
