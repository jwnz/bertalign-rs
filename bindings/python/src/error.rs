use bertalign_rs::error::{
    AlignBuilderError, BertAlignError, CosineSimilarityError, EmbeddingError, LabseError,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub struct CosineSimilarityErrorWrapper(CosineSimilarityError);
impl From<CosineSimilarityError> for CosineSimilarityErrorWrapper {
    fn from(other: CosineSimilarityError) -> CosineSimilarityErrorWrapper {
        Self(other)
    }
}
impl From<CosineSimilarityErrorWrapper> for PyErr {
    fn from(error: CosineSimilarityErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

pub struct LabseErrorWrapper(LabseError);
impl From<LabseError> for LabseErrorWrapper {
    fn from(other: LabseError) -> LabseErrorWrapper {
        Self(other)
    }
}
impl From<LabseErrorWrapper> for PyErr {
    fn from(error: LabseErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

pub struct EmbeddingErrorWrapper(EmbeddingError);
impl From<EmbeddingError> for EmbeddingErrorWrapper {
    fn from(other: EmbeddingError) -> EmbeddingErrorWrapper {
        Self(other)
    }
}
impl From<EmbeddingErrorWrapper> for PyErr {
    fn from(error: EmbeddingErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

pub struct BertAlignErrorWrapper(BertAlignError);
impl From<BertAlignError> for BertAlignErrorWrapper {
    fn from(other: BertAlignError) -> BertAlignErrorWrapper {
        Self(other)
    }
}
impl From<BertAlignErrorWrapper> for PyErr {
    fn from(error: BertAlignErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

pub struct AlignBuilderErrorWrapper(AlignBuilderError);
impl From<AlignBuilderError> for AlignBuilderErrorWrapper {
    fn from(other: AlignBuilderError) -> AlignBuilderErrorWrapper {
        Self(other)
    }
}
impl From<AlignBuilderErrorWrapper> for PyErr {
    fn from(error: AlignBuilderErrorWrapper) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}
