use anyhow::{bail, Result};

pub const BLOCK_SIZE: usize = 1024;

/// Contiguous storage for up to `BLOCK_SIZE` L1 bitcode vectors laid out as
/// `[v0_words.., v1_words.., ...]`.
#[derive(Debug, Clone)]
pub struct VectorBlock {
    pub l1_words_per_vec: usize,
    data: Vec<u64>,
    len: usize,
}

impl VectorBlock {
    pub fn new(l1_words_per_vec: usize) -> Self {
        Self {
            l1_words_per_vec,
            data: Vec::with_capacity(BLOCK_SIZE * l1_words_per_vec),
            len: 0,
        }
    }

    pub fn with_capacity(l1_words_per_vec: usize, vector_capacity: usize) -> Self {
        let cap = vector_capacity.min(BLOCK_SIZE);
        Self {
            l1_words_per_vec,
            data: Vec::with_capacity(cap * l1_words_per_vec),
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == BLOCK_SIZE
    }

    /// Write L1 codes for the next vector directly into block storage.
    pub fn push_l1(&mut self, codes: &[u64]) -> Result<()> {
        if codes.len() != self.l1_words_per_vec {
            bail!(
                "L1 word count mismatch: expected {}, got {}",
                self.l1_words_per_vec,
                codes.len()
            );
        }
        if self.is_full() {
            bail!("block is already full ({BLOCK_SIZE} vectors)");
        }
        let offset = self.len * self.l1_words_per_vec;
        if offset + self.l1_words_per_vec > self.data.len() {
            self.data.resize(offset + self.l1_words_per_vec, 0);
        }
        self.data[offset..offset + self.l1_words_per_vec].copy_from_slice(codes);
        self.len += 1;
        Ok(())
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.data[..self.len * self.l1_words_per_vec]
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Take ownership of a full block's data, leaving this block empty with reserved capacity.
    pub fn take_full(&mut self) -> Self {
        let l1_words_per_vec = self.l1_words_per_vec;
        let len = self.len;
        let mut replacement = Vec::with_capacity(BLOCK_SIZE * l1_words_per_vec);
        std::mem::swap(&mut self.data, &mut replacement);
        self.len = 0;
        Self {
            l1_words_per_vec,
            data: replacement,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_layout() {
        let words_per_vec = 2;
        let mut block = VectorBlock::new(words_per_vec);
        block.push_l1(&[1, 2]).unwrap();
        block.push_l1(&[3, 4]).unwrap();
        assert_eq!(block.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(block.len(), 2);
    }

    #[test]
    fn full_block_detection() {
        let mut block = VectorBlock::new(1);
        for i in 0..BLOCK_SIZE {
            block.push_l1(&[i as u64]).unwrap();
        }
        assert!(block.is_full());
        assert_eq!(block.len(), BLOCK_SIZE);
    }
}
