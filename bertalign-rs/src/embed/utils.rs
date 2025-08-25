use candle_core::{DType, Device};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};

use crate::error::LoadSafeTensorError;

/// memmap is unsafe because the file can be modified during or after
/// the file is read
pub fn load_safetensors<'a>(
    filepath: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, LoadSafeTensorError> {
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[filepath], dtype, device) }?)
}
