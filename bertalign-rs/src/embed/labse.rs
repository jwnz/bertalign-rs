use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, Module};
use hf_hub::{api::sync::Api, Repo};
use serde_json;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

use super::bert::{BertModel, Config, DTYPE};
use crate::embed::Embed;
use crate::error::{EmbeddingError, LabseError};

pub struct LaBSE {
    pub tokenizer: Tokenizer,
    pub bert: BertModel,
    pub pooling: Linear,
    pub batch_size: usize,
}

impl std::fmt::Debug for LaBSE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaBSE")
            .field("tokenizer", &self.tokenizer)
            // Skip the bert field or represent it as a placeholder
            .field("bert", &"<BertModel>")
            .field("pooling", &self.pooling)
            .finish()
    }
}

impl LaBSE {
    pub fn new(
        use_safetensors: Option<bool>,
        batch_size: Option<usize>,
    ) -> Result<Self, LabseError> {
        // check if compiled with the cuda option, fallback to cpu if gpu
        // isn't visible
        let device = match candle_core::utils::cuda_is_available() {
            true => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
            _ => candle_core::Device::Cpu,
        };

        Ok(LaBSE::load_model(
            "sentence-transformers/LaBSE",
            device,
            use_safetensors,
            batch_size,
        )?)
    }

    fn load_model(
        model_name: &str,
        device: candle_core::Device,
        use_safetensors: Option<bool>,
        batch_size: Option<usize>,
    ) -> Result<LaBSE, LabseError> {
        let repo = Repo::model(model_name.to_string());

        // download stuff from hf hub
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = match use_safetensors {
                Some(true) => api.get("model.safetensors")?,
                _ => api.get("pytorch_model.bin")?,
            };

            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        // load model weights
        let vb = match use_safetensors {
            Some(true) => unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
            },
            _ => candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?,
        };

        // load model & set padding params
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        let padding_params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));

        // Initialize the model(s)
        // this works only if pooling is above bert.
        let pooling = linear::linear(768, 768, vb.pp("pooler.dense"))?;
        let bert = BertModel::load(vb, &config)?;

        let batch_size = match batch_size {
            Some(bsz) => bsz,
            None => 32,
        };
        Ok(LaBSE {
            tokenizer: tokenizer,
            pooling: pooling,
            bert: bert,
            batch_size,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[tracing::instrument]
fn cls_pooling(last_hidden_state: &Tensor) -> Result<Tensor, LabseError> {
    Ok(last_hidden_state.get_on_dim(1, 0)?)
}

#[tracing::instrument]
fn encode_text(
    sentences: &[&str],
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor), LabseError> {
    let encoding = tokenizer.encode_batch(sentences.to_vec(), true)?;

    let (token_ids, attn_mask): (Vec<_>, Vec<_>) = encoding
        .into_iter()
        .map(|x| {
            let token_ids = x.get_ids().to_vec();
            let attn_mask = x.get_attention_mask().to_vec();

            let token_ids = Tensor::new(token_ids.as_slice(), device).unwrap(); // how do we propogate these errors?
            let attn_mask = Tensor::new(attn_mask.as_slice(), device).unwrap();

            (token_ids, attn_mask)
        })
        .unzip();

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attn_mask = Tensor::stack(&attn_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;

    Ok((token_ids, token_type_ids, attn_mask))
}

#[tracing::instrument]
pub fn embed_text(
    sentences: &[&str],
    model: &LaBSE,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, LabseError> {
    let mut embeddings = Vec::new();

    for batch in sentences.chunks(batch_size) {
        let (token_ids, token_type_ids, _attn_mask) =
            encode_text(batch, model.tokenizer(), device)?;

        let token_ids = token_ids
            .narrow(1, 0, std::cmp::min(512, *token_ids.dims().last().unwrap()))?
            .contiguous()?;
        let _attn_mask = _attn_mask
            .narrow(1, 0, std::cmp::min(512, *_attn_mask.dims().last().unwrap()))?
            .contiguous()?;
        let token_type_ids = token_type_ids
            .narrow(
                1,
                0,
                std::cmp::min(512, *token_type_ids.dims().last().unwrap()),
            )?
            .contiguous()?;

        let batch_embeddings =
            model
                .bert
                .forward(&token_ids, &token_type_ids, Some(&_attn_mask))?;

        let batch_embeddings = cls_pooling(&batch_embeddings)?.contiguous()?;
        let batch_embeddings = &model.pooling.forward(&batch_embeddings)?.tanh()?;
        let batch_embeddings =
            batch_embeddings.broadcast_div(&batch_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        embeddings.push(batch_embeddings);
    }

    let embeddings = Tensor::cat(&embeddings, 0)?.squeeze(1)?.to_vec2()?;
    Ok(embeddings)
}

impl Embed for LaBSE {
    #[tracing::instrument]
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let device = &self.bert.device;
        let embeddings = embed_text(lines, self, device, self.batch_size)?;

        Ok(embeddings)
    }
}
