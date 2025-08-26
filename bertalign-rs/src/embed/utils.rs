use candle_core::{DType, Device};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use hf_hub::api::sync::{Api, ApiError};

use std::path::PathBuf;

use crate::error::{DownloadHFModelError, LoadSafeTensorError};

/// memmap is unsafe because the file can be modified during or after
/// the file is read.
pub fn load_safetensors<'a>(
    filepath: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, LoadSafeTensorError> {
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[filepath], dtype, device) }?)
}

/// downloads a specific file from a specified HuggingFace Repo
///
/// ```no_run
/// let path = download_hf_model("sentence-transformers/LaBSE", "config.json")?;
/// println!("File downloaded to {:?}", path);
/// ```
pub fn download_hf_model(model_id: &str, filename: &str) -> Result<PathBuf, DownloadHFModelError> {
    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let filepath = model_repo.get(filename)?;
    Ok(filepath)
}
