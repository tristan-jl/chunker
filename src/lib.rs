use pyo3::prelude::*;
use rayon::prelude::*;
use tiktoken_rs::{cl100k_base, CoreBPE};

#[pyclass]
#[derive(Debug, Clone)]
struct Chunker {
    encoder: CoreBPE,
}

#[pymethods]
impl Chunker {
    #[new]
    fn new() -> Self {
        Self {
            encoder: cl100k_base().unwrap(),
        }
    }

    fn __call__(
        &self,
        text: String,
        max_chunk_size: usize,
        overlap: usize,
    ) -> (Vec<String>, Vec<Vec<usize>>, usize) {
        let tokens = self.encoder.encode_ordinary(&text);
        let total_tokens = tokens.len();
        if total_tokens <= max_chunk_size {
            return (vec![text], vec![tokens], total_tokens);
        }

        let (chunks, chunks_encoded) = tokens
            .chunks(max_chunk_size - overlap)
            .map(|chunk_encoded| {
                (
                    String::from_utf8(self.encoder._decode_native(chunk_encoded)).unwrap(),
                    chunk_encoded.to_vec(),
                )
            })
            .unzip();

        (chunks, chunks_encoded, total_tokens)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct Chunker2 {
    encoder: CoreBPE,
}

#[pymethods]
impl Chunker2 {
    #[new]
    fn new() -> Self {
        Self {
            encoder: cl100k_base().unwrap(),
        }
    }

    fn __call__(
        &self,
        text: String,
        max_chunk_size: usize,
        overlap: usize,
    ) -> (Vec<String>, Vec<Vec<usize>>, usize) {
        let tokens = self.encoder.encode_ordinary(&text);
        let total_tokens = tokens.len();
        if total_tokens <= max_chunk_size {
            return (vec![text], vec![tokens], total_tokens);
        }

        let (chunks, chunks_encoded) = tokens
            .into_par_iter()
            .chunks(max_chunk_size - overlap)
            .map(|chunk_encoded| {
                (
                    String::from_utf8(self.encoder._decode_native(&chunk_encoded)).unwrap(),
                    chunk_encoded,
                )
            })
            .unzip();

        (chunks, chunks_encoded, total_tokens)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn chunker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Chunker>()?;
    m.add_class::<Chunker2>()?;
    Ok(())
}
