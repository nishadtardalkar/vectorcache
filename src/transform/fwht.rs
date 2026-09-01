use anyhow::{bail, Result};
use wide::f32x8;

/// Next power of two >= `dim` (minimum 1 for `dim > 0`).
pub fn padded_dim(dim: usize) -> usize {
    dim.max(1).next_power_of_two()
}

/// In-place unnormalized Walsh-Hadamard butterfly transform.
///
/// `buf.len()` must be a power of two. After this call, multiply by `1/√N`
/// for an orthonormal Hadamard step.
pub fn fwht_in_place(buf: &mut [f32]) -> Result<()> {
    let n = buf.len();
    if n == 0 {
        bail!("FWHT buffer must be non-empty");
    }
    if !n.is_power_of_two() {
        bail!("FWHT buffer length must be a power of two, got {n}");
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let a = buf[j];
                let b = buf[j + h];
                buf[j] = a + b;
                buf[j + h] = a - b;
            }
        }
        h *= 2;
    }
    Ok(())
}

/// Apply orthonormal FWHT: unnormalized butterfly followed by `1/√N` scaling.
pub fn fwht_orthonormal_in_place(buf: &mut [f32]) -> Result<()> {
    fwht_in_place(buf)?;
    scale_in_place(buf, 1.0 / (buf.len() as f32).sqrt());
    Ok(())
}

/// Multiply all elements by `scale` using SIMD where possible.
#[inline]
pub fn scale_in_place(buf: &mut [f32], scale: f32) {
    let scale_v = f32x8::splat(scale);
    let (main, tail) = buf.as_chunks_mut::<8>();
    for chunk in main {
        let v = f32x8::new(*chunk) * scale_v;
        *chunk = v.to_array();
    }
    for x in tail {
        *x *= scale;
    }
}

/// Element-wise `buf[i] *= signs[i]` where `signs[i]` is +1 or -1.
#[inline]
pub fn apply_signs_i8(buf: &mut [f32], signs: &[i8]) {
    debug_assert_eq!(buf.len(), signs.len());
    let len = buf.len();
    let mut i = 0;
    while i + 8 <= len {
        let mut v = f32x8::new(buf[i..i + 8].try_into().unwrap());
        let s = f32x8::new([
            signs[i] as f32,
            signs[i + 1] as f32,
            signs[i + 2] as f32,
            signs[i + 3] as f32,
            signs[i + 4] as f32,
            signs[i + 5] as f32,
            signs[i + 6] as f32,
            signs[i + 7] as f32,
        ]);
        v *= s;
        buf[i..i + 8].copy_from_slice(&v.to_array());
        i += 8;
    }
    while i < len {
        buf[i] *= signs[i] as f32;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_dim_values() {
        assert_eq!(padded_dim(200), 256);
        assert_eq!(padded_dim(1536), 2048);
        assert_eq!(padded_dim(3072), 4096);
        assert_eq!(padded_dim(1), 1);
        assert_eq!(padded_dim(4), 4);
    }

    #[test]
    fn orthonormal_single_impulse() {
        let mut buf = [1.0_f32, 0.0, 0.0, 0.0];
        fwht_orthonormal_in_place(&mut buf).unwrap();
        let expected = 0.5_f32;
        for &v in &buf {
            assert!((v - expected).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn orthonormal_preserves_norm() {
        let mut buf = [3.0_f32, -1.0, 2.0, 4.0];
        let norm_before: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        fwht_orthonormal_in_place(&mut buf).unwrap();
        let norm_after: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn rejects_non_power_of_two() {
        let mut buf = [1.0_f32, 2.0, 3.0];
        assert!(fwht_in_place(&mut buf).is_err());
    }

    #[test]
    fn apply_signs_i8_negates() {
        let mut buf = [1.0_f32, 2.0, 3.0, 4.0];
        apply_signs_i8(&mut buf, &[-1, 1, -1, 1]);
        assert_eq!(buf, [-1.0, 2.0, -3.0, 4.0]);
    }
}
