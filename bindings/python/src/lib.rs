use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bertalign_rs::aligner::{Aligner, AlignerBuilder};
use bertalign_rs::embed::{Embed, LaBSE};
use bertalign_rs::error;
use bertalign_rs::error::BertAlignError;
use bertalign_rs::similarity;

pub struct BertAlignErrorWrapper(error::BertAlignError);
impl From<BertAlignError> for BertAlignErrorWrapper {
    fn from(other: BertAlignError) -> Self {
        Self(other)
    }
}
impl From<BertAlignErrorWrapper> for PyErr {
    fn from(error: BertAlignErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

#[pyfunction]
fn cosine_similarity(py: Python, a: Vec<f32>, b: Vec<f32>) -> Result<f32, BertAlignErrorWrapper> {
    py.allow_threads(|| {
        similarity::cosine_similarity(&a, &b).map_err(|err| BertAlignError::from(err).into())
    })
}

#[pyclass(module = "bertalign_rs")]
#[pyo3(name = "LaBSE")]
pub struct LaBSEWrapper(Arc<LaBSE>);

#[pymethods]
impl LaBSEWrapper {
    #[new]
    #[pyo3(signature = (use_safetensors=true, batch_size=32))]
    pub fn new(use_safetensors: bool, batch_size: usize) -> Result<Self, BertAlignErrorWrapper> {
        Ok(LaBSEWrapper(Arc::new(LaBSE::new(
            Some(use_safetensors),
            Some(batch_size),
        )?)))
    }

    pub fn embed(
        &self,
        py: Python<'_>,
        sents: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, BertAlignErrorWrapper> {
        // we need to convert the strings
        let sents: Vec<&str> = sents.iter().map(|s| s.as_str()).collect();

        // I think it's a good idea to release the GIL for this too.
        let embeddings = py.allow_threads(move || self.0.embed(&sents))?;
        Ok(embeddings)
    }
}

#[pyclass(module = "bertalign_rs")]
#[pyo3(name = "BertAlign")]
pub struct BertAlignWrapper(Aligner);

#[pymethods]
impl BertAlignWrapper {
    #[new]
    #[pyo3(signature = (embed, max_align=5, top_k=3, win=5, skip=-0.1, margin=true, len_penalty=true))]
    pub fn new(
        embed: &LaBSEWrapper,
        max_align: usize,
        top_k: usize,
        win: usize,
        skip: f32,
        margin: bool,
        len_penalty: bool,
    ) -> Result<Self, BertAlignErrorWrapper> {
        #[rustfmt::skip]
        let builder = AlignerBuilder::new(embed.0.clone())
            .max_align(max_align).map_err(|err| BertAlignError::from(err))?
            .top_k(top_k).map_err(|err| BertAlignError::from(err))?
            .win(win)
            .skip(skip)
            .margin(margin)
            .len_penalty(len_penalty);

        Ok(BertAlignWrapper(builder.build()))
    }

    pub fn align(
        &self,
        py: Python<'_>,
        src_lines: Vec<String>,
        tgt_lines: Vec<String>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, BertAlignErrorWrapper> {
        let src_lines: Vec<&str> = src_lines.iter().map(|s| s.as_str()).collect();
        let tgt_lines: Vec<&str> = tgt_lines.iter().map(|s| s.as_str()).collect();

        let alignments = py.allow_threads(move || self.0.align(&src_lines, &tgt_lines))?;
        Ok(alignments)
    }
}

#[pymodule]
#[pyo3(name = "bertalign_rs")]
fn _bertalign_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BertAlignWrapper>()?;
    m.add_class::<LaBSEWrapper>()?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;

    Ok(())
}
