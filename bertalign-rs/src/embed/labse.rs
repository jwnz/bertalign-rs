use candle_core::Tensor;
use candle_nn::{linear, Linear, Module};
use hf_hub::api::sync::Api;
use serde_json;
use tokenizers::{Tokenizer, TruncationParams};

use super::bert::{BertModel, Config, DTYPE};
use crate::embed::Embed;
use crate::error::{EmbeddingError, LabseError};

pub struct LaBSE {
    pub tokenizer: Tokenizer,
    pub bert: BertModel,
    pub pooling: Linear,
    pub batch_size: usize,
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
        let batch_size = match batch_size {
            Some(bsz) => bsz,
            None => 2048,
        };

        // download stuff from hf hub
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?.model(model_name.to_string());
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
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_position_embeddings, // the max for LaBSE is 512
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            direction: tokenizers::TruncationDirection::Right,
            stride: 0,
        }))?;

        // Initialize the model(s)
        // this works only if pooling is above bert.
        let pooling = linear::linear(768, 768, vb.pp("pooler.dense"))?;
        let bert = BertModel::load(vb, &config)?;

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

fn cls_pooling(last_hidden_state: &Tensor) -> Result<Tensor, LabseError> {
    Ok(last_hidden_state.get_on_dim(1, 0)?)
}

pub fn embed_text(
    sentences: &[&str],
    model: &LaBSE,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, LabseError> {
    let mut embeddings: Vec<Vec<f32>> = vec![];

    // encode all sentences
    let mut token_ids = model
        .tokenizer
        .encode_batch(sentences.to_vec(), true)?
        .iter()
        .map(|enc| enc.get_ids().to_vec())
        .collect::<Vec<Vec<u32>>>();

    // sort the sentences based on length as if it was already paded to the first multiple-of-8th token
    let mut sentence_lens = token_ids
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let tok_len = 8 * ((t.len() + 7) / 8);
            (i, tok_len)
        })
        .collect::<Vec<(usize, usize)>>();
    sentence_lens.sort_by_key(|(_, len)| *len);

    // collate the data into batches that fill the batch_size
    let mut batches: Vec<Vec<usize>> = vec![];
    let mut current_batch: Vec<usize> = vec![];
    let mut current_token_count = 0;

    for (idx, tok_len) in sentence_lens.iter() {
        let idx = *idx;
        let tok_len = *tok_len;

        if current_token_count + tok_len > batch_size {
            batches.push(std::mem::take(&mut current_batch));
            current_batch.push(idx);
            current_token_count = tok_len;
        } else {
            current_batch.push(idx);
            current_token_count += tok_len;
        }
    }
    if current_batch.len() > 0 {
        batches.push(std::mem::take(&mut current_batch));
    }

    // for each batch run embeddings
    for b in batches.iter() {
        let mut current_token_ids = b
            .iter()
            .map(|idx| std::mem::take(&mut token_ids[*idx]))
            .collect::<Vec<Vec<u32>>>();

        let mut max_seq_length = current_token_ids.iter().map(|tok| tok.len()).max().unwrap();
        max_seq_length = 8 * ((max_seq_length + 7) / 8);

        let mut current_attn_mask = vec![];

        // pad tokens
        for i in 0..current_token_ids.len() {
            // attention mask first, since it depends on len of input_ids before padding
            let mut this_attn_mask = vec![1u32; current_token_ids[i].len()];
            this_attn_mask.resize(max_seq_length, 0u32);
            current_attn_mask.push(std::mem::take(&mut this_attn_mask));

            // pad token_ids
            current_token_ids[i].resize(max_seq_length, 0u32);
        }

        // convert input_ids to tensor
        let (a, b) = (current_token_ids.len(), max_seq_length);
        let current_token_ids = Tensor::from_vec(
            current_token_ids
                .into_iter()
                .flatten()
                .collect::<Vec<u32>>(),
            a * b,
            &model.bert.device,
        )?
        .reshape((a, b))?;

        // convert attn_mask to tensor
        let current_attn_mask = Tensor::from_vec(
            current_attn_mask
                .into_iter()
                .flatten()
                .collect::<Vec<u32>>(),
            a * b,
            &model.bert.device,
        )?
        .reshape((a, b))?;

        // create token_type_ids
        let current_token_type_ids = Tensor::zeros_like(&current_token_ids)?;

        // run embedding
        let batch_embeddings = model.bert.forward(
            &current_token_ids,
            &current_token_type_ids,
            Some(&current_attn_mask),
        )?;

        let batch_embeddings = cls_pooling(&batch_embeddings)?.contiguous()?;
        let batch_embeddings = &model.pooling.forward(&batch_embeddings)?.tanh()?;
        let batch_embeddings =
            batch_embeddings.broadcast_div(&batch_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        for emb in batch_embeddings.to_vec2()?.into_iter() {
            embeddings.push(emb);
        }
    }

    // restore the original order
    let mut sorted_embeddings: Vec<Vec<f32>> = vec![vec![]; embeddings.len()];
    for (embedding, (idx, _)) in embeddings.into_iter().zip(sentence_lens) {
        sorted_embeddings[idx] = embedding;
    }

    Ok(sorted_embeddings)
}

impl Embed for LaBSE {
    fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let embeddings = embed_text(lines, self, self.batch_size)?;

        Ok(embeddings)
    }
}
