use std::sync::Arc;

use super::Aligner;
use crate::{
    embed::Embed,
    error::{BertAlignError, FindTopKError, Result},
    similarity, utils,
};

impl Aligner {
    pub fn _align(
        &self,
        src_sents: &[&str],
        tgt_sents: &[&str],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let src_num = src_sents.len();
        let tgt_num = tgt_sents.len();

        let num_overlaps = self.max_align.saturating_sub(1);

        let (src_vecs, src_len_vecs) = transform(&self.model, &src_sents, num_overlaps)?;
        let (tgt_vecs, tgt_len_vecs) = transform(&self.model, &tgt_sents, num_overlaps)?;

        let (top_k_distances, top_k_indicies) = find_top_k_sents(&src_vecs, &tgt_vecs, self.top_k)?;

        let first_alignment_types = get_alignment_types(2);

        let (first_w, first_path) = find_first_search_path(src_num, tgt_num, None, None);

        let first_pointers = first_pass_align(
            src_num,
            tgt_num,
            first_w,
            &first_path,
            &first_alignment_types,
            &top_k_distances,
            &top_k_indicies,
        );

        let mut first_alignment = first_back_track(
            src_num,
            tgt_num,
            &first_pointers,
            &first_path,
            &first_alignment_types,
        );

        let second_alignment_types = get_alignment_types(self.max_align);

        let (second_w, second_path) =
            find_second_search_path(&mut first_alignment, self.win, src_num, tgt_num);

        let sum_f32 = |v: &Vec<usize>| v.iter().map(|&x| x as f32).sum::<f32>();
        let char_ratio = sum_f32(&src_len_vecs[0]) / sum_f32(&tgt_len_vecs[0]);

        let second_pointers = second_pass_align(
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

        let second_alignment = second_back_track(
            src_num,
            tgt_num,
            second_pointers,
            second_path,
            second_alignment_types,
        );

        Ok(second_alignment)
    }
}

pub fn transform(
    model: &Arc<dyn Embed + Send + Sync>,
    sents: &[&str],
    num_overlaps: usize,
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<usize>>)> {
    let overlaps = utils::yield_overlaps(&sents, num_overlaps)?;

    // actually embeddable text segments
    let embeddable_cnt = overlaps.iter().filter(|x| x.is_some()).count();
    let mut embeddable_overlaps = Vec::with_capacity(embeddable_cnt);

    // mask for which indices originally contained padding
    let mut pad_mask = vec![true; overlaps.len()];

    // vector containing the embedded text including a zeroed tensor for padding
    let mut embeddings_with_pad = Vec::with_capacity(overlaps.len());

    // vector containing the lengths for each group
    let mut len_vecs = vec![vec![0; sents.len()]; num_overlaps];

    for (i, overlap) in overlaps.iter().enumerate() {
        let group_num = i / sents.len();
        let segment_num = i % sents.len();

        // This will only panic if the length of overlaps != sents.len() * num_overlaps
        if let Some(text) = overlap {
            len_vecs[group_num][segment_num] = text.chars().count();
            pad_mask[i] = false;
            embeddable_overlaps.push(text.as_str());
        }
    }

    // embed overlaps
    let sent_vecs = model.embed(&embeddable_overlaps)?;
    let zero_tensor;
    if let Some(sent_vec) = sent_vecs.get(0) {
        zero_tensor = vec![0 as f32; sent_vec.len()];
    } else {
        return Err(BertAlignError::EmptyEmbeddingsError(
            "src embeddings cannot be empty".to_string(),
        ));
    }

    let mut curr_idx = 0;

    for is_pad in pad_mask.iter() {
        if *is_pad {
            embeddings_with_pad.push(zero_tensor.clone())
        } else {
            let sent_vec =
                sent_vecs
                    .get(curr_idx)
                    .ok_or(BertAlignError::EmbeddingsLengthMismatchError(
                        "Embedded vectors count doesn't match length of input sentences"
                            .to_string(),
                    ))?;
            embeddings_with_pad.push(sent_vec.clone());
            curr_idx += 1;
        }
    }
    let sent_vecs = embeddings_with_pad
        .chunks(embeddings_with_pad.len() / num_overlaps)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<Vec<Vec<f32>>>>();
    Ok((sent_vecs, len_vecs))
}

pub fn get_alignment_types(max_alignment_size: usize) -> Vec<[usize; 2]> {
    vec![[0, 1], [1, 0]]
        .into_iter()
        .chain(
            (1..max_alignment_size)
                .flat_map(|x| (1..max_alignment_size).map(move |y| [x, y]))
                .filter(|[x, y]| x + y <= max_alignment_size),
        )
        .collect()
}

pub fn find_first_search_path(
    src_len: usize,
    tgt_len: usize,
    min_win_size: Option<usize>,
    percent: Option<f32>,
) -> (usize, Vec<[usize; 2]>) {
    // These are set as defaults in the original repo, but never changed
    let min_win_size = min_win_size.unwrap_or(250);
    let percent = percent.unwrap_or(0.06);

    let mut search_path: Vec<[usize; 2]> = vec![];

    let max_len = std::cmp::max(src_len, tgt_len);
    let win_size_from_percent = (max_len as f32 * percent) as usize; // should we round?
    let win_size = std::cmp::max(min_win_size, win_size_from_percent);

    let yx_ratio = tgt_len as f32 / src_len as f32;

    for i in 0..=src_len {
        let center = (yx_ratio * i as f32) as usize;
        let win_start = center.saturating_sub(win_size);
        let win_end = std::cmp::min(center + win_size, tgt_len);
        search_path.push([win_start, win_end]);
    }
    (win_size, search_path)
}

pub fn first_pass_align(
    src_len: usize,
    _tgt_len: usize, // unused in original repo
    w: usize,
    search_path: &[[usize; 2]],
    align_types: &[[usize; 2]],
    dist: &[Vec<f32>],
    index: &[Vec<usize>],
) -> Vec<Vec<u8>> {
    let mut cost = vec![vec![0.0 as f32; 2 * w + 1]; src_len + 1];
    let mut pointers = vec![vec![0 as u8; 2 * w + 1]; src_len + 1];

    let top_k = index[0].len();

    for i in 0..src_len + 1 {
        let i_start = search_path[i][0];
        let i_end = search_path[i][1];
        for j in i_start..i_end + 1 {
            if i + j == 0 {
                continue;
            }
            let mut best_score = f32::MIN;
            let mut best_a: Option<u8> = None;

            for a in 0..align_types.len() {
                let a_1 = align_types[a][0];
                let a_2 = align_types[a][1];
                let (prev_i, prev_i_overflowed) = i.overflowing_sub(a_1);
                let (prev_j, prev_j_overflowed) = j.overflowing_sub(a_2);
                if prev_i_overflowed || prev_j_overflowed {
                    continue;
                }
                let prev_i_start = search_path[prev_i][0];
                let prev_i_end = search_path[prev_i][1];
                if prev_j < prev_i_start || prev_j > prev_i_end {
                    continue;
                }
                let prev_j_offset = prev_j - prev_i_start;
                let mut score = cost[prev_i][prev_j_offset];

                if a_1 > 0 && a_2 > 0 {
                    for k in 0..top_k {
                        if index[i - 1][k] == j - 1 {
                            score += dist[i - 1][k];
                        }
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_a = Some(a as u8);
                }
            }
            let j_offset = j - i_start;
            cost[i][j_offset] = best_score;
            pointers[i][j_offset] = best_a.unwrap(); // Should never be None
        }
    }
    pointers
}

pub fn first_back_track(
    i: usize,
    j: usize,
    pointers: &[Vec<u8>],
    search_path: &[[usize; 2]],
    a_types: &[[usize; 2]],
) -> Vec<(usize, usize)> {
    let mut alignment = vec![];
    let mut i = i;
    let mut j = j;

    loop {
        let j_offset = j - search_path[i][0];
        let a = pointers[i][j_offset];
        let s = a_types[a as usize][0];
        let t = a_types[a as usize][1];
        if a == 2 {
            // best 1-1 alignment
            alignment.push((i, j));
        }
        i = i - s;
        j = j - t;

        // Will we never get -1 here no overflow?
        if i == 0 && j == 0 {
            break;
        }
    }

    alignment.reverse();
    alignment
}

pub fn find_second_search_path(
    align: &mut Vec<(usize, usize)>,
    w: usize,
    src_len: usize,
    tgt_len: usize,
) -> (usize, Vec<(usize, usize)>) {
    let last_bead_src = align[align.len() - 1].0;
    let last_bead_tgt = align[align.len() - 1].1;
    if last_bead_src != src_len {
        if last_bead_tgt == tgt_len {
            align.pop();
        }
        align.push((src_len, tgt_len));
    } else {
        if last_bead_tgt != tgt_len {
            align.pop();
            align.push((src_len, tgt_len));
        }
    }

    // Find the search path for each row.
    let mut prev_src: i32 = 0;
    let mut prev_tgt: i32 = 0;
    let mut path = vec![];
    let mut max_w: Option<usize> = None;

    for (src, tgt) in align.iter() {
        // Limit the search path in a rectangle with the width
        // along the Y axis being (upper_bound - lower_bound).
        let lower_bound: i32 = std::cmp::max(0, prev_tgt - w as i32);
        let upper_bound: i32 = std::cmp::min(tgt_len, tgt + w) as i32;
        for _ in (prev_src + 1) as usize..src + 1 {
            path.push((lower_bound, upper_bound));
        }
        prev_src = *src as i32;
        prev_tgt = *tgt as i32;
        let width = upper_bound - lower_bound;
        if max_w.is_none() || width as usize > max_w.unwrap() as usize {
            max_w = Some(width as usize);
        }
    }

    let mut final_path = vec![path[0]];
    final_path.extend(path.iter().cloned());

    let path = final_path
        .iter()
        .map(|(a, b)| (*a as usize, *b as usize))
        .collect();

    (max_w.unwrap() + 1, path)
}

pub fn second_pass_align(
    src_vecs: &[Vec<Vec<f32>>],
    tgt_vecs: &[Vec<Vec<f32>>],
    src_lens: &[Vec<usize>],
    tgt_lens: &[Vec<usize>],
    w: usize,
    search_path: &[(usize, usize)],
    align_types: &[[usize; 2]],
    char_ratio: f32,
    skip: f32,
    margin: Option<bool>,
    len_penalty: Option<bool>,
) -> Result<Vec<Vec<u8>>> {
    let margin = margin.unwrap_or(false);
    let len_penalty = len_penalty.unwrap_or(false);

    let src_len = src_vecs[0].len();
    let tgt_len = tgt_vecs[0].len();
    let mut cost = vec![vec![0.0 as f32; w]; src_len + 1];
    let mut pointers = vec![vec![0 as u8; w]; src_len + 1];

    for i in 0..src_len + 1 {
        let i_start = search_path[i].0;
        let i_end = search_path[i].1;

        for j in i_start..i_end + 1 {
            if i + j == 0 {
                continue;
            }

            let mut best_score = f32::MIN;
            let mut best_a: Option<u8> = None;

            for a in 0..align_types.len() {
                let a_1 = align_types[a][0];
                let a_2 = align_types[a][1];
                let (prev_i, prev_i_overflowed) = i.overflowing_sub(a_1);
                let (prev_j, prev_j_overflowed) = j.overflowing_sub(a_2);
                if prev_i_overflowed || prev_j_overflowed {
                    continue;
                }
                let prev_i_start = search_path[prev_i].0;
                let prev_i_end = search_path[prev_i].1;
                if prev_j < prev_i_start || prev_j > prev_i_end {
                    continue;
                }
                let prev_j_offset = prev_j - prev_i_start;
                let mut score = cost[prev_i][prev_j_offset];

                let mut cur_score = 0.0;
                if a_1 == 0 || a_2 == 0 {
                    cur_score = skip;
                } else {
                    cur_score = calculate_similarity_score(
                        &src_vecs,
                        &tgt_vecs,
                        i,
                        j,
                        a_1,
                        a_2,
                        src_len,
                        tgt_len,
                        Some(margin),
                    )
                    .unwrap();
                    if len_penalty {
                        let penalty = calculate_length_penalty(
                            src_lens, tgt_lens, i, j, a_1, a_2, char_ratio,
                        );
                        cur_score *= penalty;
                    }
                }
                score += cur_score;
                if score > best_score {
                    best_score = score;
                    best_a = Some(a as u8);
                }
            }

            let j_offset = j - i_start;
            cost[i][j_offset] = best_score;
            pointers[i][j_offset] = best_a.unwrap(); // this should also never be None
        }
    }
    Ok(pointers)
}

pub fn calculate_similarity_score(
    src_vecs: &[Vec<Vec<f32>>],
    tgt_vecs: &[Vec<Vec<f32>>],
    src_idx: usize,
    tgt_idx: usize,
    src_overlap: usize,
    tgt_overlap: usize,
    src_len: usize,
    tgt_len: usize,
    margin: Option<bool>,
) -> Result<f32> {
    let margin = margin.unwrap_or(false);
    let src_v = &src_vecs[src_overlap - 1][src_idx - 1];
    let tgt_v = &tgt_vecs[tgt_overlap - 1][tgt_idx - 1];

    let mut similarity = similarity::cosine_similarity(&src_v, &tgt_v)?;

    if margin {
        let tgt_neighbor_ave_sim =
            calculate_neighbor_similarity(&src_v, tgt_overlap, tgt_idx, tgt_len, tgt_vecs)?;

        let src_neighbor_ave_sim =
            calculate_neighbor_similarity(&tgt_v, src_overlap, src_idx, src_len, src_vecs)?;

        let neighbor_ave_sim = (tgt_neighbor_ave_sim + src_neighbor_ave_sim) / 2.0;
        similarity -= neighbor_ave_sim;
    }

    Ok(similarity)
}

pub fn calculate_neighbor_similarity(
    vec: &[f32],
    overlap: usize,
    sent_idx: usize,
    sent_len: usize,
    db: &[Vec<Vec<f32>>],
) -> Result<f32> {
    let left_idx = sent_idx - overlap;
    let right_idx = sent_idx + 1;

    let mut neighbor_right_sim: f32 = 0.0;
    let mut neighbor_left_sim: f32 = 0.0;
    let mut neighbor_ave_sim = neighbor_left_sim + neighbor_right_sim;

    if right_idx <= sent_len {
        let right_embed = &db[0][right_idx - 1];
        neighbor_right_sim = similarity::cosine_similarity(vec, &right_embed)?;
    }

    if left_idx > 0 {
        let left_embed = &db[0][left_idx - 1];
        neighbor_left_sim = similarity::cosine_similarity(vec, &left_embed)?;
    }

    if neighbor_right_sim * neighbor_left_sim > 0.0 {
        neighbor_ave_sim /= 2.0;
    }

    Ok(neighbor_ave_sim)
}

pub fn calculate_length_penalty(
    src_lens: &[Vec<usize>],
    tgt_lens: &[Vec<usize>],
    src_idx: usize,
    tgt_idx: usize,
    src_overlap: usize,
    tgt_overlap: usize,
    char_ratio: f32,
) -> f32 {
    let src_l = src_lens[src_overlap - 1][src_idx - 1];
    let tgt_l = tgt_lens[tgt_overlap - 1][tgt_idx - 1];

    let tgt_l = tgt_l as f32 * char_ratio;
    let min_len = f32::min(src_l as f32, tgt_l);
    let max_len = f32::max(src_l as f32, tgt_l);
    let length_penalty = (1.0 + min_len / max_len).log2();

    length_penalty
}

pub fn second_back_track(
    i: usize,
    j: usize,
    pointers: Vec<Vec<u8>>,
    search_path: Vec<(usize, usize)>,
    a_types: Vec<[usize; 2]>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut alignment = vec![];

    let mut i = i;
    let mut j = j;

    loop {
        let j_offset = j - search_path[i].0;
        let a = pointers[i][j_offset];
        let s = a_types[a as usize][0];
        let t = a_types[a as usize][1];
        let src_range: Vec<usize> = (0..s).map(|offset| i - offset - 1).rev().collect();
        let tgt_range: Vec<usize> = (0..t).map(|offset| j - offset - 1).rev().collect();
        alignment.push((src_range, tgt_range));

        i = i - s;
        j = j - t;

        if i == 0 && j == 0 {
            break;
        }
    }

    alignment.reverse();
    alignment
}

pub fn find_top_k_sents(
    src_vecs: &[Vec<Vec<f32>>],
    tgt_vecs: &[Vec<Vec<f32>>],
    top_k: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<usize>>)> {
    // The first index here grabs the non concatenated sentences
    // also embedding shouldn't be empty
    let src_vecs = src_vecs
        .get(0)
        .ok_or_else(|| FindTopKError::EmbeddingsCantBeEmpty)?;

    let tgt_vecs = tgt_vecs
        .get(0)
        .ok_or_else(|| FindTopKError::EmbeddingsCantBeEmpty)?;

    // internal embeddings (token-level) shouldn't be empty either
    if src_vecs.is_empty() {
        return Err(FindTopKError::TokenLevelEmbeddingsCantBeEmpty.into());
    }
    if tgt_vecs.is_empty() {
        return Err(FindTopKError::TokenLevelEmbeddingsCantBeEmpty.into());
    }

    let mut topk_scores = Vec::with_capacity(src_vecs.len());
    let mut topk_indices = Vec::with_capacity(tgt_vecs.len());

    for src_vec in src_vecs.iter() {
        let mut sims = Vec::with_capacity(tgt_vecs.len());

        for (idx, tgt_vec) in tgt_vecs.iter().enumerate() {
            let sim = similarity::cosine_similarity(src_vec, tgt_vec)?;
            sims.push((sim, idx));
        }

        // maxheap is faster, but sorting is probably good enough
        sims.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Extract top_k scores and indices
        let top_k = std::cmp::min(top_k, sims.len());
        let (current_top_k_scores, current_top_k_indices) = sims
            .iter()
            .take(top_k)
            .map(|(sim, idx)| (*sim, *idx))
            .unzip();

        topk_scores.push(current_top_k_scores);
        topk_indices.push(current_top_k_indices);
    }

    Ok((topk_scores, topk_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock embedder for testing
    struct MockEmbedder {
        embeddings: Vec<Vec<f32>>,
    }
    impl MockEmbedder {
        fn new(embeddings: Vec<Vec<f32>>) -> Self {
            Self { embeddings }
        }

        fn generate_embeddings(num_sents: usize, num_overlaps: usize) -> Vec<Vec<f32>> {
            // The number of embeddings is sents.len() * num_overlaps minus n!, where n is num_overlaps - 1
            let total = num_sents * num_overlaps;
            let pad_cnt = match num_overlaps {
                ..=1 => 0, // product is weird and returns 1 for the range 0..0
                _ => (0..(num_overlaps.saturating_sub(1))).product::<usize>(),
            };
            (0..(total.saturating_sub(pad_cnt)))
                .map(|_| vec![0.1, 0.2, 0.3])
                .collect()
        }
    }

    impl Embed for MockEmbedder {
        fn embed(&self, _lines: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.embeddings.clone())
        }
    }

    #[test]
    fn test_get_alignment_types() {
        let alignments = get_alignment_types(0);
        assert_eq!(alignments, [[0, 1], [1, 0]]);

        let alignments = get_alignment_types(2);
        assert_eq!(alignments, [[0, 1], [1, 0], [1, 1]]);

        let alignments = get_alignment_types(5);
        assert_eq!(
            alignments,
            [
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, 1],
                [3, 2],
                [4, 1]
            ]
        );
    }

    #[test]
    fn test_find_topk() {
        let src_vecs = vec![vec![
            vec![1.0, 0.0], // src 1
            vec![0.0, 1.0], // src 2
        ]];
        let tgt_vecs = vec![vec![
            vec![1.0, 0.0], // tgt 1
            vec![0.0, 1.0], // tgt 2
            vec![1.0, 1.0], // tgt 3
        ]];

        let top_k = 1;
        let (_, indicies) = find_top_k_sents(&src_vecs, &tgt_vecs, top_k).unwrap();
        assert_eq!(indicies, [[0], [1]]);

        // top_k might be larger than size of tgt_vecs
        let top_k = 100;
        let (_, indicies) = find_top_k_sents(&src_vecs, &tgt_vecs, top_k).unwrap();
        assert_eq!(indicies, [[0, 2, 1], [1, 2, 0]]);

        // The embeddings should never be empty
        let empty_embeddings = vec![];
        assert!(matches!(
            find_top_k_sents(&src_vecs, &empty_embeddings, top_k),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
        assert!(matches!(
            find_top_k_sents(&empty_embeddings, &src_vecs, top_k),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
        assert!(matches!(
            find_top_k_sents(&empty_embeddings, &empty_embeddings, top_k),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));

        // The internal embeddings also shouldn't be empty
        let internal_empty_embeddings = vec![vec![]];
        assert!(matches!(
            find_top_k_sents(&src_vecs, &internal_empty_embeddings, top_k),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
        assert!(matches!(
            find_top_k_sents(&internal_empty_embeddings, &src_vecs, top_k),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));
    }

    #[test]
    fn test_transform() {
        let sents = vec!["a", "b", "c"];
        let num_overlaps = 3;
        let embeddings = MockEmbedder::generate_embeddings(sents.len(), num_overlaps);

        let model: Arc<dyn Embed + Send + Sync> = Arc::new(MockEmbedder::new(embeddings));

        let (embeddings, lengths) = transform(&model, &sents, num_overlaps).unwrap();
        assert_eq!(
            embeddings,
            [
                [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
            ]
        );
        assert_eq!(lengths, [[1, 1, 1], [0, 3, 3], [0, 0, 5]]);

        // The model shouldn't be allowed to return empty embeddings
        let embeddings = vec![];
        let model: Arc<dyn Embed + Send + Sync> = Arc::new(MockEmbedder::new(embeddings));
        assert!(matches!(
            transform(&model, &sents, num_overlaps),
            Err(BertAlignError::EmptyEmbeddingsError(_))
        ));

        // The input sentence count shouldn't be different from whatever the model returns
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.1, 0.2, 0.3]];
        let model: Arc<dyn Embed + Send + Sync> = Arc::new(MockEmbedder::new(embeddings));
        assert!(matches!(
            transform(&model, &sents, num_overlaps),
            Err(BertAlignError::EmbeddingsLengthMismatchError(_))
        ));

        // example where the embeddings is longer than the input sentences
        let embeddings = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.1, 0.2, 0.3],
            vec![0.1, 0.2, 0.3],
            vec![0.1, 0.2, 0.3],
            vec![0.1, 0.2, 0.3],
        ];
        let model: Arc<dyn Embed + Send + Sync> = Arc::new(MockEmbedder::new(embeddings));
        assert!(matches!(
            transform(&model, &sents, num_overlaps),
            Err(BertAlignError::EmbeddingsLengthMismatchError(_))
        ));
    }
}
