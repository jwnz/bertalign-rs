use candle_core::Tensor;

pub struct Batch {
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
}

impl Batch {
    pub fn new(input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
        }
    }

    pub fn input_ids(&self) -> &Tensor {
        &self.input_ids
    }

    pub fn attention_mask(&self) -> &Tensor {
        &self.attention_mask
    }

    pub fn token_type_ids(&self) -> &Tensor {
        &self.token_type_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let device = candle_core::Device::Cpu;
        let dtype = candle_core::DType::U32;

        let input_ids = Tensor::zeros((2, 3), dtype, &device).unwrap();
        let attention_mask = Tensor::zeros((2, 3), dtype, &device).unwrap();
        let token_type_ids = Tensor::zeros((2, 3), dtype, &device).unwrap();

        let input_ids_clone = input_ids.clone();
        let attention_mask_clone = attention_mask.clone();
        let token_type_ids_clone = token_type_ids.clone();

        let batch = Batch::new(input_ids, attention_mask, token_type_ids);

        assert_eq!(
            batch.input_ids().to_vec2::<u32>().unwrap(),
            input_ids_clone.to_vec2::<u32>().unwrap()
        );

        assert_eq!(
            batch.attention_mask().to_vec2::<u32>().unwrap(),
            attention_mask_clone.to_vec2::<u32>().unwrap()
        );

        assert_eq!(
            batch.token_type_ids().to_vec2::<u32>().unwrap(),
            token_type_ids_clone.to_vec2::<u32>().unwrap()
        );
    }
}
