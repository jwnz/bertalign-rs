use crate::error::{BertAlignError, Result};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() == 0 || b.len() == 0 {
        Err(BertAlignError::EmptyEmbeddingsError(
            "Cosine similarity of 0 sized vectors is undefined".to_string(),
        ))
    } else if a.len() != b.len() {
        Err(BertAlignError::EmbeddingsLengthMismatchError(
            "Cosine similarity of vectors of different lengths is undefined".to_string(),
        ))
    } else {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        let sim = match (norm_a, norm_b) {
            (0.0, _) | (_, 0.0) => 0.0,
            _ => dot_product / (norm_a * norm_b),
        };

        Ok(sim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b).unwrap();

        // due to floating point errors, we'll check within some error margin
        let error_margin = 1e-6;
        assert!((1.0 - result).abs() < error_margin);

        // vectors of different sizes is undefined for the cosine similarity
        let c = vec![1.0, 2.0];
        assert!(matches!(
            cosine_similarity(&a, &c),
            Err(BertAlignError::EmbeddingsLengthMismatchError(_))
        ));

        // The vectors also shouldn't be empty
        let d = vec![];
        assert!(matches!(
            cosine_similarity(&c, &d),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
        assert!(matches!(
            cosine_similarity(&d, &c),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
        assert!(matches!(
            cosine_similarity(&d, &d),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
    }
}
