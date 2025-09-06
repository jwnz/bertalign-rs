mod core;

use std::sync::Arc;

use crate::embed::Embed;
use crate::error::{AlignBuilderError, BertAlignError};

pub struct AlignerBuilder {
    max_align: usize,
    top_k: usize,
    win: usize,
    skip: f32,
    margin: bool,
    len_penalty: bool,
}

impl Default for AlignerBuilder {
    fn default() -> Self {
        AlignerBuilder {
            max_align: 5,
            top_k: 3,
            win: 5,
            skip: -0.1,
            margin: true,
            len_penalty: true,
        }
    }
}

impl AlignerBuilder {
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

    pub fn build(self) -> Result<Aligner, AlignBuilderError> {
        // make sure max align is correct
        match self.max_align {
            0 | 1 => Err(AlignBuilderError::MaxAlignTooSmall(self.max_align)),
            _ => Ok(()),
        }?;

        // make sure top_k is correct
        match self.top_k {
            0 => Err(AlignBuilderError::TopKTooSmall(self.top_k)),
            _ => Ok(()),
        }?;

        let aligner = Aligner {
            max_align: self.max_align,
            top_k: self.top_k,
            win: self.win,
            skip: self.skip,
            margin: self.margin,
            len_penalty: self.len_penalty,
        };
        Ok(aligner)
    }
}

pub struct Aligner {
    max_align: usize,
    top_k: usize,
    win: usize,
    skip: f32,
    margin: bool,
    len_penalty: bool,
}

impl Aligner {
    pub fn align(
        &self,
        model: Arc<dyn Embed>,
        src_sents: &[&str],
        tgt_sents: &[&str],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, BertAlignError> {
        self._align(model.clone(), src_sents, tgt_sents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligner_builder() {
        let aligner = AlignerBuilder::default()
            .max_align(5)
            .top_k(3)
            .win(5)
            .skip(-0.1)
            .margin(true)
            .len_penalty(true)
            .build();

        assert!(aligner.is_ok());
    }

    #[test]
    fn test_aligner_builder_max_align_too_small() {
        // test when max_align == 0
        let aligner_builder = AlignerBuilder::default();
        assert!(matches!(
            aligner_builder.max_align(0).build(),
            Err(AlignBuilderError::MaxAlignTooSmall(0))
        ));

        // test when max_align == 1
        let aligner_builder = AlignerBuilder::default();
        assert!(matches!(
            aligner_builder.max_align(1).build(),
            Err(AlignBuilderError::MaxAlignTooSmall(1))
        ));
    }

    #[test]
    fn test_aligner_builder_top_k_too_small() {
        let aligner_builder = AlignerBuilder::default();

        let _top_k = 0;
        assert!(matches!(
            aligner_builder.top_k(_top_k).build(),
            Err(AlignBuilderError::TopKTooSmall(_top_k))
        ));
    }
}
