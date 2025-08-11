pub mod core;

use std::sync::Arc;

use crate::embed::Embed;
use crate::error::Result;

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
        self._align(src_sents, tgt_sents)
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
