use wide::f32x8;

/// L2-normalize `vector` in place. Zero vectors are left unchanged.
pub fn l2_normalize_in_place(vector: &mut [f32]) {
    let norm_sq = norm_sq_simd(vector);
    if norm_sq > 0.0 {
        let inv = norm_sq.sqrt().recip();
        let inv_v = f32x8::splat(inv);
        let (main, tail) = vector.as_chunks_mut::<8>();
        for chunk in main {
            let v = f32x8::new(*chunk) * inv_v;
            *chunk = v.to_array();
        }
        for x in tail {
            *x *= inv;
        }
    }
}

#[inline]
fn norm_sq_simd(vector: &[f32]) -> f32 {
    let mut sum = f32x8::ZERO;
    let (main, tail) = vector.as_chunks::<8>();
    for chunk in main {
        let v = f32x8::new(*chunk);
        sum += v * v;
    }
    let mut norm_sq = sum.reduce_add();
    for &x in tail {
        norm_sq += x * x;
    }
    norm_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_vector_unchanged_up_to_scale() {
        let mut v = [3.0_f32, 4.0];
        l2_normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_stays_zero() {
        let mut v = [0.0_f32; 4];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, [0.0; 4]);
    }
}
