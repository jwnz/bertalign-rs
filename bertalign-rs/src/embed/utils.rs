use candle_core::{DType, Device};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use hf_hub::api::sync::Api;

use std::path::PathBuf;

use crate::error::{DownloadHFModelError, LoadSafeTensorError};

/// memmap is unsafe because the file can be modified during or after
/// the file is read.
pub fn load_safetensors<'a, P: AsRef<std::path::Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, LoadSafeTensorError> {
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device) }?)
}

/// downloads a specific file from a specified HuggingFace Repo
pub fn download_hf_model(model_id: &str, filename: &str) -> Result<PathBuf, DownloadHFModelError> {
    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let filepath = model_repo.get(filename)?;
    Ok(filepath)
}
