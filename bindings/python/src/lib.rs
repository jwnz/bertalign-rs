use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bertalign_rs::aligner::{AlignArgs, Aligner};
use bertalign_rs::embed::{Embed, LaBSE};
use bertalign_rs::error;
use bertalign_rs::error::BertAlignError;

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
        sents: Vec<&str>,
    ) -> Result<Vec<Vec<f32>>, BertAlignErrorWrapper> {
        // I think it's a good idea to release the GIL for this too.
        let embeddings = py.allow_threads(move || self.0.embed(&sents))?;
        Ok(embeddings)
    }
}

#[pyclass(module = "bertalign_rs")]
#[pyo3(name = "BertAlignArgs")]
#[derive(Clone)]
pub struct BertAlignArgWrapper(AlignArgs);

#[pymethods]
impl BertAlignArgWrapper {
    #[new]
    #[pyo3(signature = (max_align=5, top_k=3, win=5, skip=-0.1, margin=true, len_penalty=true))]
    pub fn new(
        max_align: usize,
        top_k: usize,
        win: usize,
        skip: f32,
        margin: bool,
        len_penalty: bool,
    ) -> BertAlignArgWrapper {
        BertAlignArgWrapper(AlignArgs::new(
            max_align,
            top_k,
            win,
            skip,
            margin,
            len_penalty,
        ))
    }
}

#[pyclass(module = "bertalign_rs")]
#[pyo3(name = "BertAlign")]
pub struct BertAlignWrapper(Aligner);

#[pymethods]
impl BertAlignWrapper {
    #[new]
    pub fn new(
        embed: &LaBSEWrapper,
        args: &BertAlignArgWrapper,
    ) -> Result<Self, BertAlignErrorWrapper> {
        let aligner = Aligner::new(args.0.clone(), embed.0.clone());
        Ok(BertAlignWrapper(aligner))
    }

    pub fn align(
        &self,
        py: Python<'_>,
        src_lines: Vec<&str>,
        tgt_lines: Vec<&str>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, BertAlignErrorWrapper> {
        let alignments = py.allow_threads(move || self.0.align(&src_lines, &tgt_lines))?;
        Ok(alignments)
    }
}

#[pymodule]
#[pyo3(name = "bertalign_rs")]
fn _bertalign_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<BertAlignArgWrapper>()?;
    m.add_class::<BertAlignWrapper>()?;
    m.add_class::<LaBSEWrapper>()?;

    Ok(())
}
