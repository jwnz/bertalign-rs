use crate::error::CosineSimilarityError;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, CosineSimilarityError> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => Err(CosineSimilarityError::ZeroSizedVectorSimUndefined),
        (lhs, rhs) if lhs != rhs => {
            Err(CosineSimilarityError::DifferentLenVectorSimUndefined { lhs: lhs, rhs: rhs })
        }
        _ => {
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
    }

    #[test]
    fn test_cosine_similarity_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];

        let _a_len = a.len();
        let _b_len = b.len();

        assert!(matches!(
            cosine_similarity(&a, &b),
            Err(CosineSimilarityError::DifferentLenVectorSimUndefined {
                lhs: _a_len,
                rhs: _b_len,
            })
        ));
    }

    #[test]
    fn test_cosine_similarity_empty_embeddings() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![];
        assert!(matches!(
            cosine_similarity(&a, &b),
            Err(CosineSimilarityError::ZeroSizedVectorSimUndefined)
        ));
        assert!(matches!(
            cosine_similarity(&b, &a),
            Err(CosineSimilarityError::ZeroSizedVectorSimUndefined)
        ));
        assert!(matches!(
            cosine_similarity(&b, &b),
            Err(CosineSimilarityError::ZeroSizedVectorSimUndefined)
        ));
    }
}
