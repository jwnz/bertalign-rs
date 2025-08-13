use thiserror::Error;

pub type Result<T, E = BertAlignError> = std::result::Result<T, E>;

#[derive(Error, Debug)]
pub enum BertAlignError {
    #[error("Embeddings must have the same length: {0}")]
    EmbeddingsLengthMismatchError(String),

    #[error("Embeddings cannot be empty: {0}")]
    EmptyEmbeddingsError(String),

    #[error("Value must be nonzero: {0}")]
    NonZeroValueError(String),

    #[error("String cannot be empty")]
    EmptyStringError,

    // need less ambiguous error handling here?
    #[error("Error related to the Candle framework")]
    CandleError(#[from] candle_core::error::Error),

    #[error("Error related to the tokenizer framework")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("Error related to hf hub")]
    HFHubError(#[from] hf_hub::api::sync::ApiError),

    #[error("Error reading serde json")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("IO Error")]
    StdIOError(#[from] std::io::Error),

    #[error("YieldOverlapError: {0}")]
    YieldOverlapError(#[from] YieldOverlapError),

    #[error("CosineSimilarityError: {0}")]
    CosineSimilarityError(#[from] CosineSimilarityError),
}

#[derive(Error, Debug)]
pub enum YieldOverlapError {
    #[error("num_overlaps ({num_overlaps:?}), line count ({line_count:?})")]
    OverlapsExceedsLineCount {
        num_overlaps: usize,
        line_count: usize,
    },

    #[error("NonEmptyStringError: {0}")]
    NonEmptyStringError(#[from] NonEmptyStringError),
}

#[derive(Error, Debug)]
pub enum NonEmptyStringError {
    #[error("String cannot be empty")]
    EmptyString,
}

#[derive(Error, Debug)]
pub enum CosineSimilarityError {
    #[error("Cosine similarity of 0 sized vectors is undefined")]
    ZeroSizedVectorSimUndefined,

    #[error("Cosine similarity of vectors of different lengths (lhs: {lhs:?}, rhs: {rhs:?}) is undefined")]
    DifferentLenVectorSimUndefined { lhs: usize, rhs: usize },
}
