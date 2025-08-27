pub mod bert;
pub mod labse;
pub mod pooling;
pub mod types;
mod utils;

pub use crate::embed::labse::LaBSE;
use crate::error::EmbeddingError;

pub trait Embed {
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}
