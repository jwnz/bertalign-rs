use candle_core::Tensor;
use candle_nn::Linear;

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
                // first we get the embeddings from the cls
                let cls_token_embeddings = last_hidden_state.get_on_dim(1, 0)?;
                todo!()
            }
        }
    }
}

struct SentenceTransformersPoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}
