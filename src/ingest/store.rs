use anyhow::Result;

use super::block::VectorBlock;

/// In-memory store of full L1 bitcode blocks plus one in-progress partial block.
#[derive(Debug)]
pub struct BlockStore {
    l1_words_per_vec: usize,
    blocks: Vec<VectorBlock>,
    partial: VectorBlock,
}

impl BlockStore {
    pub fn new(l1_words_per_vec: usize) -> Self {
        Self {
            l1_words_per_vec,
            blocks: Vec::new(),
            partial: VectorBlock::new(l1_words_per_vec),
        }
    }

    /// Pre-allocate block containers when the total vector count is known upfront.
    pub fn with_capacity(l1_words_per_vec: usize, vector_count: usize) -> Self {
        let full_blocks = vector_count / super::BLOCK_SIZE;
        let partial_reserve = (vector_count % super::BLOCK_SIZE).max(1);
        Self {
            l1_words_per_vec,
            blocks: Vec::with_capacity(full_blocks),
            partial: VectorBlock::with_capacity(l1_words_per_vec, partial_reserve),
        }
    }

    pub fn l1_words_per_vec(&self) -> usize {
        self.l1_words_per_vec
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn total_vectors(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum::<usize>() + self.partial.len()
    }

    pub fn get_block(&self, index: usize) -> Option<&VectorBlock> {
        self.blocks.get(index)
    }

    pub fn partial_block(&self) -> &VectorBlock {
        &self.partial
    }

    pub fn blocks(&self) -> &[VectorBlock] {
        &self.blocks
    }

    pub fn push_l1_codes(&mut self, codes: &[u64]) -> Result<()> {
        self.partial.push_l1(codes)?;
        if self.partial.is_full() {
            let full = self.partial.take_full();
            self.blocks.push(full);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::BLOCK_SIZE;

    #[test]
    fn two_full_blocks_and_partial() {
        let words_per_vec = 2;
        let mut store = BlockStore::new(words_per_vec);
        let total = BLOCK_SIZE * 2 + 3;

        for i in 0..total {
            let v = [i as u64, i as u64 + 1];
            store.push_l1_codes(&v).unwrap();
        }

        assert_eq!(store.block_count(), 2);
        assert_eq!(store.partial_block().len(), 3);
        assert_eq!(store.total_vectors(), total);

        let block0 = store.get_block(0).unwrap();
        assert_eq!(block0.len(), BLOCK_SIZE);
        assert_eq!(block0.as_slice()[0], 0);
        assert_eq!(block0.as_slice()[1], 1);

        let last_in_block0 = (BLOCK_SIZE - 1) as u64;
        let offset = (BLOCK_SIZE - 1) * words_per_vec;
        assert_eq!(block0.as_slice()[offset], last_in_block0);
        assert_eq!(block0.as_slice()[offset + 1], last_in_block0 + 1);
    }

    #[test]
    fn with_capacity_preallocates_containers() {
        let store = BlockStore::with_capacity(1, BLOCK_SIZE + 5);
        assert_eq!(store.block_count(), 0);
        assert_eq!(store.l1_words_per_vec(), 1);
    }
}
