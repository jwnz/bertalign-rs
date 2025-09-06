pub mod bert;
pub mod pooling;
pub mod sentence_transformer;
mod utils;

use crate::error::EmbeddingError;

pub trait Embed {
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}
