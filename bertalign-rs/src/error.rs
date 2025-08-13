use thiserror::Error;

pub type Result<T, E = BertAlignError> = std::result::Result<T, E>;

#[derive(Error, Debug)]
pub enum BertAlignError {
    #[error("Embeddings must have the same length: {0}")]
    EmbeddingsLengthMismatchError(String),

    #[error("Embeddings cannot be empty: {0}")]
    EmptyEmbeddingsError(String),

    #[error("Value ({0}) must be nonzero")]
    NonZeroValueError(String),

    #[error("String cannot be empty")]
    EmptyStringError,

    #[error("YieldOverlapError: {0}")]
    YieldOverlapError(#[from] YieldOverlapError),

    #[error("CosineSimilarityError: {0}")]
    CosineSimilarityError(#[from] CosineSimilarityError),

    #[error("AlignBuilderError: {0}")]
    AlignBuilderError(#[from] AlignBuilderError),

    #[error("CandleError: {0}")]
    CandleError(#[from] candle_core::error::Error),

    #[error("TokenizersError: {0}")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("HFHubError: {0}")]
    HFHubError(#[from] hf_hub::api::sync::ApiError),

    #[error("SerdeJsonError: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("IO Error: {0}")]
    StdIOError(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum AlignBuilderError {
    #[error("max_align ({0}) must be > 1")]
    MaxAlignTooSmall(usize),

    #[error("top_k ({0}) must be >= 1")]
    TopKTooSmall(usize),
}

#[derive(Error, Debug)]
pub enum YieldOverlapError {
    #[error("num_overlaps ({num_overlaps:?}), line count ({line_count:?})")]
    OverlapsExceedsLineCount {
        num_overlaps: usize,
        line_count: usize,
    },

    #[error("num_overlaps ({0}), cannot be 0")]
    OverlapsCantBeZero(usize),

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
