use crate::error::{BertAlignError, Result};
use std::num::NonZeroUsize;

const SPACE: &'static str = " ";

#[derive(Debug)]
struct NonEmptyString(String);
impl NonEmptyString {
    fn new(s: &str) -> Result<Self> {
        if s.trim().is_empty() {
            Err(BertAlignError::EmptyStringError)
        } else {
            Ok(NonEmptyString(s.to_string()))
        }
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

pub fn yield_overlaps(lines: &[&str], num_overlaps: NonZeroUsize) -> Result<Vec<Option<String>>> {
    // Check if lines are non-empty
    let lines = lines
        .into_iter()
        .map(|s| NonEmptyString::new(s).map(|s| s.into_inner()))
        .collect::<Result<Vec<String>>>()?;

    // make sure num_overlaps doesn't exceed line cnt
    let num_overlaps = std::cmp::min(num_overlaps.get(), lines.len());

    let mut overlaps: Vec<Option<String>> = vec![];

    for overlap in 1..=num_overlaps {
        for _ in 0..std::cmp::min(overlap.saturating_sub(1), lines.len()) {
            overlaps.push(None);
        }

        // end is the line count minus the overlap count
        let end = match lines.len().checked_sub(overlap) {
            Some(val) => val,
            None => unreachable!(), // should be unreachable
        };

        for i in 0..=end {
            let _ = match lines.get(i..i + overlap) {
                Some(line_chunk) => {
                    let line_chunk = line_chunk.join(SPACE).chars().take(10_000).collect();
                    overlaps.push(Some(line_chunk));
                }
                None => unreachable!(), // should be unreachable
            };
        }
    }

    Ok(overlaps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_empty_string() {
        assert_eq!(NonEmptyString::new("Hello").unwrap().into_inner(), "Hello");

        assert!(matches!(
            NonEmptyString::new(""),
            Err(BertAlignError::EmptyStringError)
        ));
        assert!(matches!(
            NonEmptyString::new("      "),
            Err(BertAlignError::EmptyStringError)
        ));

        // These invisible character related edge cases are currently not
        // covered, as the string is technically not non-empty
        assert_eq!(NonEmptyString::new("‎").unwrap().into_inner(), "‎");
    }

    #[test]
    fn test_yield_overlaps() {
        assert_eq!(
            yield_overlaps(
                &["한국어", "hello", "你好", "わたし"],
                NonZeroUsize::new(4).unwrap(),
            )
            .unwrap(),
            [
                Some("한국어".to_string()),
                Some("hello".to_string()),
                Some("你好".to_string()),
                Some("わたし".to_string()),
                None,
                Some("한국어 hello".to_string()),
                Some("hello 你好".to_string()),
                Some("你好 わたし".to_string()),
                None,
                None,
                Some("한국어 hello 你好".to_string()),
                Some("hello 你好 わたし".to_string()),
                None,
                None,
                None,
                Some("한국어 hello 你好 わたし".to_string())
            ]
        );

        // Any empty lines should have been removed prior
        assert!(matches!(
            yield_overlaps(&["hello", "", "world"], NonZeroUsize::new(2).unwrap()),
            Err(BertAlignError::EmptyStringError)
        ));

        // The overlap count should not be larger than the number of actual sentences
        assert_eq!(
            yield_overlaps(&["hi"], NonZeroUsize::new(5).unwrap()).unwrap(),
            [Some("hi".to_string())]
        );

        // Empty lists should just be empty
        assert_eq!(
            yield_overlaps(&[], NonZeroUsize::new(5).unwrap()).unwrap(),
            []
        );
    }
}
