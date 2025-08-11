pub mod core;

use std::sync::Arc;

use crate::embed::Embed;
use crate::error::{BertAlignError, Result};

pub struct AlignerBuilder {
    max_align: usize,
    top_k: usize,
    win: usize,
    skip: f32,
    margin: bool,
    len_penalty: bool,
    model: Arc<dyn Embed + Send + Sync>,
}

impl AlignerBuilder {
    pub fn new(model: Arc<dyn Embed + Send + Sync>) -> AlignerBuilder {
        AlignerBuilder {
            max_align: 5,
            top_k: 3,
            win: 5,
            skip: -0.1,
            margin: true,
            len_penalty: true,
            model: model,
        }
    }

    pub fn max_align(mut self, max_align: usize) -> AlignerBuilder {
        self.max_align = max_align;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> AlignerBuilder {
        self.top_k = top_k;
        self
    }

    pub fn win(mut self, win: usize) -> AlignerBuilder {
        self.win = win;
        self
    }

    pub fn skip(mut self, skip: f32) -> AlignerBuilder {
        self.skip = skip;
        self
    }

    pub fn margin(mut self, margin: bool) -> AlignerBuilder {
        self.margin = margin;
        self
    }

    pub fn len_penalty(mut self, len_penalty: bool) -> AlignerBuilder {
        self.len_penalty = len_penalty;
        self
    }

    pub fn build(self) -> Aligner {
        Aligner {
            max_align: self.max_align,
            top_k: self.top_k,
            win: self.win,
            skip: self.skip,
            margin: self.margin,
            len_penalty: self.len_penalty,
            model: self.model,
        }
    }
}

pub struct Aligner {
    max_align: usize,
    top_k: usize,
    win: usize,
    skip: f32,
    margin: bool,
    len_penalty: bool,
    model: Arc<dyn Embed + Send + Sync>,
}

impl Aligner {
    pub fn align(
        &self,
        src_sents: &[&str],
        tgt_sents: &[&str],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let src_num = src_sents.len();
        let tgt_num = tgt_sents.len();

        let num_overlaps = std::num::NonZeroUsize::new(self.max_align - 1).ok_or(
            BertAlignError::NonZeroValueError(
                "The number of overlaps (max_align - 1) should be > 0".to_string(),
            ),
        )?;
        let top_k = std::num::NonZeroUsize::new(self.top_k).ok_or(
            BertAlignError::NonZeroValueError("top_k should be > 0".to_string()),
        )?;

        let (src_vecs, src_len_vecs) = core::transform(&self.model, &src_sents, num_overlaps)?;
        let (tgt_vecs, tgt_len_vecs) = core::transform(&self.model, &tgt_sents, num_overlaps)?;

        let (top_k_distances, top_k_indicies) =
            core::find_top_k_sents(&src_vecs, &tgt_vecs, top_k)?;

        let first_alignment_types = core::get_alignment_types(2);

        let (first_w, first_path) = core::find_first_search_path(src_num, tgt_num, None, None);

        let first_pointers = core::first_pass_align(
            src_num,
            tgt_num,
            first_w,
            &first_path,
            &first_alignment_types,
            &top_k_distances,
            &top_k_indicies,
        );

        let mut first_alignment = core::first_back_track(
            src_num,
            tgt_num,
            &first_pointers,
            &first_path,
            &first_alignment_types,
        );

        let second_alignment_types = core::get_alignment_types(self.max_align);

        let (second_w, second_path) =
            core::find_second_search_path(&mut first_alignment, self.win, src_num, tgt_num);

        let sum_f32 = |v: &Vec<usize>| v.iter().map(|&x| x as f32).sum::<f32>();
        let char_ratio = sum_f32(&src_len_vecs[0]) / sum_f32(&tgt_len_vecs[0]);

        let second_pointers = core::second_pass_align(
            &src_vecs,
            &tgt_vecs,
            &src_len_vecs,
            &tgt_len_vecs,
            second_w,
            &second_path,
            &second_alignment_types,
            char_ratio,
            self.skip,
            Some(self.margin),
            Some(self.len_penalty),
        )?;

        let second_alignment = core::second_back_track(
            src_num,
            tgt_num,
            second_pointers,
            second_path,
            second_alignment_types,
        );

        Ok(second_alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel;
    impl Embed for MockModel {
        fn embed(&self, _lines: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![]])
        }
    }

    #[test]
    fn test_aligner_builder() {
        let embedding_model = Arc::new(MockModel {});
        let _aligner = AlignerBuilder::new(embedding_model.clone())
            .max_align(5)
            .top_k(3)
            .win(5)
            .skip(-0.1)
            .margin(true)
            .len_penalty(true)
            .build();
    }
}
