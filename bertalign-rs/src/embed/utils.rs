use candle_core::{DType, Device};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use hf_hub::api::sync::Api;

use std::path::PathBuf;

use crate::error::{DownloadHFModel, LoadSafeTensorError};

/// memmap is unsafe because the file can be modified during or after
/// the file is read.
pub fn load_safetensors<'a>(
    filepath: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, LoadSafeTensorError> {
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[filepath], dtype, device) }?)
}

pub enum DownloadType {
    Tokenizer,
    ModelConfig,
    WeightsPyTorch,
    WeightsSafeTensors,
}

impl DownloadType {
    pub fn filename(&self) -> &'static str {
        match self {
            DownloadType::Tokenizer => "tokenizer.json",
            DownloadType::ModelConfig => "config.json",
            DownloadType::WeightsPyTorch => "pytorch_model.bin",
            DownloadType::WeightsSafeTensors => "model.safetensors",
        }
    }
}

pub fn download_hf_model(
    model_id: &str,
    download_type: DownloadType,
) -> Result<PathBuf, DownloadHFModel> {
    let api = Api::new()?.model(model_id.to_string());
    let filepath = api.get(download_type.filename())?;
    Ok(filepath)
}
