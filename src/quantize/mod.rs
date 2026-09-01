/// L1 quantization: 4D hyperplane projection to 1 bit per 4D block.

/// 4 orthogonal Hadamard basis vectors for 4D hyperplane projections.
pub const HADAMARD_4D: [[f32; 4]; 4] = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0],
];

/// Number of `u64` words needed to store L1 bitcodes for a vector of `dim` floats.
pub fn l1_words_per_vector(dim: usize) -> usize {
    let num_bits = (dim + 3) / 4;
    (num_bits + 63) / 64
}

/// Number of L1 bits for a vector of `dim` floats.
#[inline]
pub fn l1_bits_per_vector(dim: usize) -> usize {
    (dim + 3) / 4
}

/// Quantize a rotated vector to L1 4D→1bit bitcodes, writing into `out`.
///
/// Returns the number of valid bits (`⌈dim / 4⌉`). `out.len()` must equal
/// `l1_words_per_vector(dim)`.
pub fn quantize_4d_to_1bit_into(vector: &[f32], out: &mut [u64]) -> usize {
    let dim = vector.len();
    let num_4d_blocks = l1_bits_per_vector(dim);
    debug_assert_eq!(out.len(), l1_words_per_vector(dim));

    out.fill(0);

    let mut b = 0;

    while b + 64 <= num_4d_blocks {
        let mut word = 0_u64;
        for bit in 0..64 {
            let block = b + bit;
            let offset = block * 4;
            let basis = HADAMARD_4D[block & 3];
            let proj = vector[offset] * basis[0]
                + vector[offset + 1] * basis[1]
                + vector[offset + 2] * basis[2]
                + vector[offset + 3] * basis[3];
            word |= ((proj >= 0.0) as u64) << bit;
        }
        out[b / 64] = word;
        b += 64;
    }

    if b < num_4d_blocks {
        let mut word = 0_u64;
        for bit in 0..(num_4d_blocks - b) {
            let block = b + bit;
            let offset = block * 4;
            let basis = HADAMARD_4D[block & 3];
            let mut proj = 0.0f32;
            for i in 0..4 {
                if offset + i < dim {
                    proj += vector[offset + i] * basis[i];
                }
            }
            word |= ((proj >= 0.0) as u64) << bit;
        }
        out[b / 64] = word;
    }

    num_4d_blocks
}

/// Allocating wrapper around [`quantize_4d_to_1bit_into`].
pub fn quantize_4d_to_1bit(vector: &[f32]) -> (Vec<u64>, usize) {
    let num_words = l1_words_per_vector(vector.len());
    let mut words = vec![0_u64; num_words];
    let num_bits = quantize_4d_to_1bit_into(vector, &mut words);
    (words, num_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_4d_block_positive_projection() {
        let mut words = [0_u64];
        let num_bits = quantize_4d_to_1bit_into(&[1.0, 1.0, 1.0, 1.0], &mut words);
        assert_eq!(num_bits, 1);
        assert_eq!(words, [1]);
    }

    #[test]
    fn single_4d_block_negative_projection() {
        let mut words = [0_u64];
        let num_bits = quantize_4d_to_1bit_into(&[-1.0, -1.0, -1.0, -1.0], &mut words);
        assert_eq!(num_bits, 1);
        assert_eq!(words, [0]);
    }

    #[test]
    fn cyclic_basis_changes_bit() {
        let mut w0 = [0_u64];
        quantize_4d_to_1bit_into(&[1.0, 2.0, 0.0, 0.0], &mut w0);
        let mut v = vec![0.0; 8];
        v[4..8].copy_from_slice(&[1.0, 2.0, 0.0, 0.0]);
        let mut w1 = [0_u64];
        quantize_4d_to_1bit_into(&v, &mut w1);
        assert_eq!(w0[0] & 1, 1);
        assert_eq!(w1[0] & 2, 0);
    }

    #[test]
    fn padding_partial_last_block() {
        let mut words = [0_u64];
        let num_bits = quantize_4d_to_1bit_into(&[1.0, 1.0, 1.0, 1.0, 1.0], &mut words);
        assert_eq!(num_bits, 2);
        assert_eq!(words[0], 0b11);
    }

    #[test]
    fn benchmark_dims() {
        assert_eq!(l1_words_per_vector(256), 1);
        let mut w = [0_u64];
        assert_eq!(quantize_4d_to_1bit_into(&vec![0.0; 256], &mut w), 64);

        assert_eq!(l1_words_per_vector(2048), 8);
        let mut w8 = vec![0_u64; 8];
        assert_eq!(quantize_4d_to_1bit_into(&vec![0.0; 2048], &mut w8), 512);

        assert_eq!(l1_words_per_vector(4096), 16);
        let mut w16 = vec![0_u64; 16];
        assert_eq!(quantize_4d_to_1bit_into(&vec![0.0; 4096], &mut w16), 1024);
    }

    #[test]
    fn into_matches_allocating() {
        let v: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
        let (alloc, n1) = quantize_4d_to_1bit(&v);
        let mut into = vec![0_u64; alloc.len()];
        let n2 = quantize_4d_to_1bit_into(&v, &mut into);
        assert_eq!(n1, n2);
        assert_eq!(alloc, into);
    }
}
