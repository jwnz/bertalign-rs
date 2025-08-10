pub mod bert;
pub mod labse;

pub use crate::embed::labse::LaBSE;
use crate::error::Result;

pub trait Embed {
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>>;
}
