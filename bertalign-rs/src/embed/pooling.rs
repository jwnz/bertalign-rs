use candle_core::Tensor;
use candle_nn::{Linear, Module};
use serde::Deserialize;

use crate::error::PoolingError;

pub enum Which {
    MeanPooling,
    SentenceTransformerPooling { hf_hub_config_path: String },
}

pub enum PoolingStrategy {
    MeanPooling,
    SentenceTransformerPooling(Linear),
}

impl PoolingStrategy {
    pub fn pool(&self, last_hidden_state: Tensor) -> Result<Tensor, PoolingError> {
        match self {
            PoolingStrategy::MeanPooling => {
                todo!()
            }
            PoolingStrategy::SentenceTransformerPooling(linear) => {
                let cls_pooling = last_hidden_state.get_on_dim(1, 0)?.contiguous()?;
                let emb = linear.forward(&cls_pooling)?.tanh()?;
                Ok(emb.broadcast_div(&emb.sqr()?.sum_keepdim(1)?.sqrt()?)?)
            }
        }
    }
}

#[derive(Deserialize)]
pub struct SentenceTransformersPoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}

impl SentenceTransformersPoolingConfig {
    pub fn word_embedding_dimension(&self) -> usize {
        self.word_embedding_dimension
    }
}
