use std::fmt;

use candle_core::Device;
use candle_nn::linear;
use serde_json;
use tokenizers::{Tokenizer, TruncationParams};

use super::{
    bert::{BertModel, Config, DTYPE},
    pooling::PoolingStrategy,
    pooling::Which as PoolingWhich,
    utils::{download_hf_model, load_safetensors},
};
use crate::embed::{pooling::SentenceTransformersPoolingConfig, utils, Embed};
use crate::error::{EmbeddingError, SentenceTransformerBuilderError};

const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_WITH_SAFETENSORS: bool = false;

pub enum Which {
    AllMiniLML6v2,
    AllMiniLML12v2,
    ParaphraseMultilingualMiniLML6v2,
    ParaphraseMultilingualMiniLML12v2,
    LaBSE,
}

impl fmt::Display for Which {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Which::AllMiniLML6v2 => write!(f, "sentence-transformers/all-MiniLM-L6-v2"),
            Which::AllMiniLML12v2 => write!(f, "sentence-transformers/all-MiniLM-L12-v2"),
            Which::ParaphraseMultilingualMiniLML6v2 => {
                write!(f, "sentence-transformers/paraphrase-MiniLM-L6-v2")
            }
            Which::ParaphraseMultilingualMiniLML12v2 => {
                write!(
                    f,
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            }
            Which::LaBSE => write!(f, "sentence-transformers/LaBSE"),
        }
    }
}

pub struct SentenceTransformerBuilder {
    model_id: String,
    with_safetensors: bool,
    batch_size: Option<usize>,
    device: Option<Device>,
    pooling: Option<PoolingWhich>,
}

impl SentenceTransformerBuilder {
    pub fn new(model_id: impl AsRef<str>) -> Self {
        Self {
            model_id: model_id.as_ref().to_string(),
            with_safetensors: DEFAULT_WITH_SAFETENSORS,
            batch_size: None,
            device: None,
            pooling: None,
        }
    }

    pub fn with_sentence_transformer(model: Which) -> SentenceTransformerBuilder {
        match model {
            Which::LaBSE
            | Which::AllMiniLML6v2
            | Which::AllMiniLML12v2
            | Which::ParaphraseMultilingualMiniLML6v2
            | Which::ParaphraseMultilingualMiniLML12v2 => {
                SentenceTransformerBuilder::new(model.to_string())
                    .with_safetensors()
                    .with_pooling(PoolingWhich::SentenceTransformerPooling {
                        hf_hub_config_path: "1_Pooling/config.json".to_string(),
                    })
            }
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> SentenceTransformerBuilder {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_safetensors(mut self) -> SentenceTransformerBuilder {
        self.with_safetensors = true;
        self
    }

    pub fn with_device(mut self, device: &Device) -> SentenceTransformerBuilder {
        self.device = Some(device.clone());
        self
    }

    pub fn with_pooling(mut self, pooling: PoolingWhich) -> SentenceTransformerBuilder {
        self.pooling = Some(pooling);
        self
    }

    pub fn build(self) -> Result<SentenceTransformer, SentenceTransformerBuilderError> {
        // do our checks first
        let device = self
            .device
            .ok_or_else(|| SentenceTransformerBuilderError::DeviceNotSpecified)?;

        let pooling_method = self
            .pooling
            .ok_or_else(|| SentenceTransformerBuilderError::PoolingMethodNotSpecified)?;

        // Load the model config
        let config_filename = download_hf_model(&self.model_id, "config.json")?;
        let config = serde_json::from_str::<Config>(&std::fs::read_to_string(config_filename)?)?;

        // Load the Tokenizer
        let tokenizer_filename = download_hf_model(&self.model_id, "tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_position_embeddings, // the max for LaBSE is 512
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            direction: tokenizers::TruncationDirection::Right,
            stride: 0,
        }))?;

        // Load the VarBuilder for loading model weights
        let vb = if self.with_safetensors {
            let weights_filename = download_hf_model(&self.model_id, "model.safetensors")?;
            load_safetensors(&[weights_filename], DTYPE, &device)?
        } else {
            let weights_filename = download_hf_model(&self.model_id, "pytorch_model.bin")?;
            candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };

        // load the appropriate pooler
        let pooler = match pooling_method {
            PoolingWhich::MeanPooling => PoolingStrategy::MeanPooling,
            PoolingWhich::SentenceTransformerPooling {
                hf_hub_config_path: path,
            } => {
                let pooling_config_filename = download_hf_model(&self.model_id, &path)?;
                let pooling_config = serde_json::from_str::<SentenceTransformersPoolingConfig>(
                    &std::fs::read_to_string(pooling_config_filename)?,
                )?;

                let word_emb_size = pooling_config.word_embedding_dimension();
                PoolingStrategy::SentenceTransformerPooling(linear::linear(
                    word_emb_size,
                    word_emb_size,
                    vb.pp("pooler.dense"),
                )?)
            }
        };

        let bert = BertModel::load(vb, &config)?;

        let batch_size = self.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        Ok(SentenceTransformer {
            batch_size: batch_size,
            device: device,
            tokenizer: tokenizer,
            bert: bert,
            pooling: pooler,
        })
    }
}

pub struct SentenceTransformer {
    batch_size: usize,
    device: Device,
    tokenizer: Tokenizer,
    bert: BertModel,
    pooling: PoolingStrategy,
}

impl Embed for SentenceTransformer {
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut embeddings: Vec<Vec<f32>> = vec![];

        let mut token_ids = self
            .tokenizer
            .encode_batch(lines.to_vec(), true)?
            .iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect::<Vec<Vec<u32>>>();

        let batches =
            utils::fast_token_based_batching(&mut token_ids, self.batch_size, &self.device)?;

        for batch in batches.batches.iter() {
            let batch_embeddings = self.bert.forward(
                &batch.input_ids,
                &batch.token_type_ids,
                Some(&batch.attention_mask),
            )?;

            let batch_embeddings = self.pooling.pool(batch_embeddings)?;

            // add batch embeddings to final embeddings - still unsorted at this point
            for emb in batch_embeddings.to_vec2()?.into_iter() {
                embeddings.push(emb);
            }
        }

        let mut sorted_embeddings = vec![vec![]; embeddings.len()];
        for (emb, (idx, _)) in embeddings.into_iter().zip(batches.original_ids) {
            sorted_embeddings[idx] = emb;
        }

        Ok(sorted_embeddings)
    }
}

#[cfg(test)]
mod tests {}
