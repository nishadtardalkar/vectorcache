use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use memmap2::Mmap;

use super::reader::{DatasetMeta, DatasetReader};

/// Memory-mapped streaming reader for NumPy `.npy` float32 matrices shaped `(N, dim)`.
pub struct NpyReader {
    mmap: Mmap,
    dim: usize,
    count: usize,
    index: usize,
    label: &'static str,
    data_offset: usize,
}

impl NpyReader {
    pub fn open(path: &Path, label: &'static str) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe {
            Mmap::map(&file).with_context(|| format!("failed to mmap {}", path.display()))?
        };

        let (count, dim, data_offset) = parse_npy_f32_matrix(&mmap)
            .with_context(|| format!("failed to parse NPY header in {}", path.display()))?;

        Ok(Self {
            mmap,
            dim,
            count,
            index: 0,
            label,
            data_offset,
        })
    }
}

impl DatasetReader for NpyReader {
    fn meta(&self) -> DatasetMeta {
        DatasetMeta {
            dim: self.dim,
            count: self.count,
            label: self.label,
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

        let byte_offset = self.data_offset + self.index * self.dim * 4;
        let end = byte_offset + self.dim * 4;
        if end > self.mmap.len() {
            bail!(
                "NPY data truncated at vector {} (need {} bytes, have {})",
                self.index,
                end,
                self.mmap.len()
            );
        }

        let floats: &[f32] = cast_slice(&self.mmap[byte_offset..end]);
        out.copy_from_slice(floats);

        self.index += 1;
        Ok(true)
    }
}

fn parse_npy_f32_matrix(mmap: &[u8]) -> Result<(usize, usize, usize)> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if mmap.len() < MAGIC.len() + 4 {
        bail!("NPY file too small");
    }
    if &mmap[..MAGIC.len()] != MAGIC {
        bail!("invalid NPY magic bytes");
    }

    let major = mmap[6];
    let minor = mmap[7];
    let (header_len, header_start): (usize, usize) = match (major, minor) {
        (1, 0) => {
            if mmap.len() < 10 {
                bail!("NPY v1 header truncated");
            }
            let len = u16::from_le_bytes([mmap[8], mmap[9]]) as usize;
            (len, 10)
        }
        (2, 0) => {
            if mmap.len() < 12 {
                bail!("NPY v2 header truncated");
            }
            let len = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
            (len, 12)
        }
        _ => bail!("unsupported NPY version {major}.{minor}"),
    };

    let header_end = header_start + header_len;
    if header_end > mmap.len() {
        bail!("NPY header extends past end of file");
    }

    let header = &mmap[header_start..header_end];
    let header_str = std::str::from_utf8(header)
        .context("NPY header is not valid UTF-8")?;

    if !header_str.contains("'descr': '<f4'") && !header_str.contains("\"descr\": \"<f4\"") {
        bail!("expected float32 ('<f4') NPY array, got header: {header_str}");
    }

    let shape = parse_shape(header_str).context("failed to parse NPY shape")?;
    if shape.len() != 2 {
        bail!("expected 2-D NPY matrix, got shape with {} dimensions", shape.len());
    }

    let count = shape[0];
    let dim = shape[1];
    let data_offset = header_end;

    let expected_bytes = data_offset + count * dim * 4;
    if mmap.len() < expected_bytes {
        bail!(
            "NPY data truncated: expected at least {expected_bytes} bytes, got {}",
            mmap.len()
        );
    }

    Ok((count, dim, data_offset))
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let key = "shape";
    let start = header
        .find(key)
        .context("missing 'shape' in NPY header")?;
    let after_key = &header[start + key.len()..];
    let open_paren = after_key
        .find('(')
        .context("missing '(' after shape key")?;
    let close_paren = after_key
        .find(')')
        .context("missing ')' in shape tuple")?;
    let inner = &after_key[open_paren + 1..close_paren];

    let mut shape = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed
            .parse::<usize>()
            .with_context(|| format!("invalid shape component '{trimmed}'"))?;
        shape.push(value);
    }

    if shape.is_empty() {
        bail!("empty shape tuple in NPY header");
    }

    Ok(shape)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ndarray::Array2;
    use ndarray_npy::WriteNpyExt;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn round_trip_small_npy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.npy");

        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        )
        .unwrap();
        let mut file = File::create(&path).unwrap();
        data.write_npy(&mut file).unwrap();

        let mut reader = NpyReader::open(&path, "test").unwrap();
        assert_eq!(reader.meta().count, 3);
        assert_eq!(reader.meta().dim, 2);

        let mut buf = [0.0_f32; 2];
        assert!(reader.next_vector_into(&mut buf).unwrap());
        assert_eq!(buf, [1.0, 2.0]);
        assert!(reader.next_vector_into(&mut buf).unwrap());
        assert_eq!(buf, [3.0, 4.0]);
        assert!(reader.next_vector_into(&mut buf).unwrap());
        assert_eq!(buf, [5.0, 6.0]);
        assert!(!reader.next_vector_into(&mut buf).unwrap());
    }
}
