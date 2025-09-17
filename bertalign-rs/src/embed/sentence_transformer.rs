use std::fmt;

use candle_core::Device;
use candle_nn::{seq, Module, Sequential};
use serde_json;
use tokenizers::{Tokenizer, TruncationParams};

use super::{
    bert::{BertModel, Config, DTYPE},
    pooling::PoolingStrategy,
    utils::{download_hf_model, load_safetensors},
};
use crate::embed::{
    dense::{Dense, SentenceTransformersDenseConfig},
    normalize::Normalize,
    pooling::PoolingConfig,
    utils, Embed,
};
use crate::error::{EmbeddingError, SentenceTransformerBuilderError};

const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_WITH_SAFETENSORS: bool = false;
const DEFAULT_WITH_NORMALIZE: bool = false;

pub enum Which {
    AllMiniLML6v2,
    AllMiniLML12v2,
    ParaphraseMiniLML6v2,
    ParaphraseMultilingualMiniLML12v2,
    LaBSE,
}

impl fmt::Display for Which {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Which::AllMiniLML6v2 => write!(f, "sentence-transformers/all-MiniLM-L6-v2"),
            Which::AllMiniLML12v2 => write!(f, "sentence-transformers/all-MiniLM-L12-v2"),
            Which::ParaphraseMiniLML6v2 => {
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
    with_normalization: bool,
    batch_size: Option<usize>,
    device: Option<Device>,
    pooling_path: Option<String>,
    dense_paths: Vec<String>,
}

impl SentenceTransformerBuilder {
    pub fn new(model_id: impl AsRef<str>) -> Self {
        Self {
            model_id: model_id.as_ref().to_string(),
            with_safetensors: DEFAULT_WITH_SAFETENSORS,
            with_normalization: DEFAULT_WITH_NORMALIZE,
            batch_size: None,
            device: None,
            pooling_path: None,
            dense_paths: vec![],
        }
    }

    pub fn with_sentence_transformer(model: Which) -> SentenceTransformerBuilder {
        match model {
            Which::LaBSE => SentenceTransformerBuilder::new(model.to_string())
                .with_safetensors()
                .with_normalization()
                .with_pooling("1_Pooling".to_string())
                .with_dense("2_Dense".to_string()),
            Which::AllMiniLML6v2
            | Which::AllMiniLML12v2
            | Which::ParaphraseMiniLML6v2
            | Which::ParaphraseMultilingualMiniLML12v2 => {
                SentenceTransformerBuilder::new(model.to_string())
                    .with_safetensors()
                    .with_normalization()
                    .with_pooling("1_Pooling".to_string())
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

    pub fn with_normalization(mut self) -> SentenceTransformerBuilder {
        self.with_normalization = true;
        self
    }

    pub fn with_device(mut self, device: &Device) -> SentenceTransformerBuilder {
        self.device = Some(device.clone());
        self
    }

    pub fn with_pooling(mut self, pooling: String) -> SentenceTransformerBuilder {
        self.pooling_path = Some(pooling);
        self
    }

    pub fn with_dense(mut self, dense_path: String) -> SentenceTransformerBuilder {
        self.dense_paths.push(dense_path);
        self
    }

    pub fn build(self) -> Result<SentenceTransformer, SentenceTransformerBuilderError> {
        // Device must be specified
        let device = self
            .device
            .ok_or_else(|| SentenceTransformerBuilderError::DeviceNotSpecified)?;

        // The pooling method must also be specified
        let pooling_method = self
            .pooling_path
            .ok_or_else(|| SentenceTransformerBuilderError::PoolingMethodNotSpecified)?;
        let pooling_method = format!("{pooling_method}/config.json");

        let batch_size = self.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        // Load the transformer model config
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

        // Load the transformer
        let vb = if self.with_safetensors {
            let weights_filename = download_hf_model(&self.model_id, "model.safetensors")?;
            load_safetensors(&[weights_filename], DTYPE, &device)?
        } else {
            let weights_filename = download_hf_model(&self.model_id, "pytorch_model.bin")?;
            candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };
        let bert = BertModel::load(vb, &config)?;

        // load the pooler
        let pooling_config_filename = download_hf_model(&self.model_id, &pooling_method)?;
        let pooling_config = serde_json::from_str::<PoolingConfig>(&std::fs::read_to_string(
            pooling_config_filename,
        )?)?;
        let pooler = PoolingStrategy::from_config(pooling_config);

        // Load the dense layers
        let mut dense_layers = vec![];
        for dense_path in self.dense_paths.iter() {
            let dense_config_filename =
                download_hf_model(&self.model_id, &format!("{dense_path}/config.json"))?;
            let dense_config = serde_json::from_str::<SentenceTransformersDenseConfig>(
                &std::fs::read_to_string(dense_config_filename)?,
            )?;

            let dense_vb = if self.with_safetensors {
                let weights_filename =
                    download_hf_model(&self.model_id, &format!("{dense_path}/model.safetensors"))?;
                load_safetensors(&[weights_filename], DTYPE, &device)?
            } else {
                let weights_filename =
                    download_hf_model(&self.model_id, &format!("{dense_path}/pytorch_model.bin"))?;
                candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
            };

            let layer = Dense::from_config(dense_vb, dense_config)?;
            dense_layers.push(layer);
        }

        // create a sequntial layer and add the dense layers
        let mut sequential = seq();
        for layer in dense_layers.into_iter() {
            sequential = sequential.add(layer)
        }

        // ...then add normalization
        if self.with_normalization {
            sequential = sequential.add(Normalize);
        }

        Ok(SentenceTransformer {
            batch_size: batch_size,
            device: device,
            tokenizer: tokenizer,
            transformer: bert,
            pooler: pooler,
            sequential: sequential,
        })
    }
}

pub struct SentenceTransformer {
    batch_size: usize,
    device: Device,
    tokenizer: Tokenizer,
    transformer: BertModel,
    pooler: PoolingStrategy,
    sequential: Sequential,
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
            // transformer
            let batch_embeddings = &self.transformer.forward(
                &batch.input_ids,
                &batch.token_type_ids,
                Some(&batch.attention_mask),
            )?;

            // pool
            let batch_embeddings = &self
                .pooler
                .forward(batch_embeddings, &batch.attention_mask)?;

            // dense and norm
            let batch_embeddings = self.sequential.forward(&batch_embeddings)?;

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
