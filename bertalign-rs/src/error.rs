use thiserror::Error;

#[derive(Error, Debug)]
pub enum BertAlignError {
    #[error("TransformError: {0}")]
    TransformError(#[from] TransformError),

    #[error("FindTopKError: {0}")]
    FindTopKError(#[from] FindTopKError),

    #[error("PlaceholderError: {0}")]
    PlaceHolderError(#[from] PlaceholderError),
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

#[derive(Error, Debug)]
pub enum FindTopKError {
    #[error("Embeddings can't be empty")]
    EmbeddingsCantBeEmpty,

    #[error("Token-level embeddings can't be empty")]
    TokenLevelEmbeddingsCantBeEmpty,

    #[error("CosineSimilarityError: {0}")]
    CosineSimilarityError(#[from] CosineSimilarityError),
}

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("Embeddings can't be empty")]
    EmbeddingsCantBeEmpty,

    #[error("Index ({0}) out of bounds error for sentence_embeddings")]
    SentenceEmbeddingIndexOutOfBounds(usize),

    #[error("YieldOverlapError: {0}")]
    YieldOverlapError(#[from] YieldOverlapError),

    #[error("CandleError: {0}")]
    CandleError(#[from] candle_core::error::Error),

    #[error("EmbeddingError: {0}")]
    EmbeddingError(#[from] EmbeddingError),
}

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("LabseError: {0}")]
    LabseError(#[from] LabseError),
}

#[derive(Error, Debug)]
pub enum LabseError {
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

#[derive(Debug, Error)]
pub enum LoadSafeTensorError {
    #[error("CandleError: {0}")]
    CandleError(#[from] candle_core::error::Error),
}

// placeholder until I figure out a better way to handle some errors
#[derive(Debug, Error)]
pub enum PlaceholderError {
    #[error("CosineSimilarityError: {0}")]
    CosineSimilarityError(#[from] CosineSimilarityError),
}
