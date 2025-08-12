use crate::error::{NonEmptyStringError, YieldOverlapError};
use std::num::NonZeroUsize;

const SPACE: &'static str = " ";

struct NonEmptyString(String);

impl NonEmptyString {
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::str::FromStr for NonEmptyString {
    type Err = NonEmptyStringError;
    fn from_str(s: &str) -> Result<Self, NonEmptyStringError> {
        if s.trim().is_empty() {
            Err(NonEmptyStringError::EmptyString)
        } else {
            Ok(NonEmptyString(s.to_string()))
        }
    }
}

pub fn yield_overlaps(
    lines: &[&str],
    num_overlaps: NonZeroUsize,
) -> Result<Vec<Option<String>>, YieldOverlapError> {
    // Check if lines are non-empty
    let lines = lines
        .into_iter()
        .map(|s| s.parse::<NonEmptyString>().map(|s| s.into_inner()))
        .collect::<Result<Vec<String>, NonEmptyStringError>>()?;

    let num_overlaps = num_overlaps.get();

    // make sure num_overlaps doesn't exceed line cnt
    if num_overlaps > lines.len() {
        return Err(YieldOverlapError::OverlapsExceedsLineCount {
            num_overlaps: num_overlaps,
            line_count: lines.len(),
        });
    }

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
        let s = "Hello".parse::<NonEmptyString>().unwrap();
        assert_eq!(s.into_inner(), "Hello".to_string());
    }

    #[test]
    fn test_non_empty_string_is_empty() {
        assert!(matches!(
            "".parse::<NonEmptyString>(),
            Err(NonEmptyStringError::EmptyString)
        ));

        assert!(matches!(
            "      ".parse::<NonEmptyString>(),
            Err(NonEmptyStringError::EmptyString)
        ));
    }

    #[test]
    fn test_non_empty_string_invisible_chars() {
        // Left-to-Right Mark (U+200E)
        assert_eq!("‎".parse::<NonEmptyString>().unwrap().into_inner(), "‎");

        // Right-to-Left Mark (U+200F)
        assert_eq!("‏".parse::<NonEmptyString>().unwrap().into_inner(), "‏");

        // Zero-Width Space (U+200B)
        assert_eq!("​".parse::<NonEmptyString>().unwrap().into_inner(), "​");

        // Zero-Width Non-Joiner (U+200C)
        assert_eq!("‌".parse::<NonEmptyString>().unwrap().into_inner(), "‌");

        // Zero-Width Joiner (U+200D)
        assert_eq!("‍".parse::<NonEmptyString>().unwrap().into_inner(), "‍");

        // Word Joiner (U+2060)
        assert_eq!("⁠".parse::<NonEmptyString>().unwrap().into_inner(), "⁠");

        // Hangul Filler (U+3164)
        assert_eq!("ㅤ".parse::<NonEmptyString>().unwrap().into_inner(), "ㅤ");

        // Braille Pattern Blank (U+2800)
        assert_eq!("⠀".parse::<NonEmptyString>().unwrap().into_inner(), "⠀");

        // Zero Width No-Break Space / Byte Order Mark (U+FEFF)
        assert_eq!("﻿".parse::<NonEmptyString>().unwrap().into_inner(), "﻿");

        // Combining Grapheme Joiner (U+034F)
        assert_eq!("͏".parse::<NonEmptyString>().unwrap().into_inner(), "͏");
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
    }

    #[test]
    fn test_yield_overlaps_empty_string() {
        assert!(matches!(
            yield_overlaps(&["hello", "", "world"], NonZeroUsize::new(2).unwrap()),
            Err(YieldOverlapError::NonEmptyStringError(_))
        ));
    }

    #[test]
    fn test_yield_overlaps_overlap_exceeds_lc() {
        assert!(matches!(
            yield_overlaps(&["hi"], NonZeroUsize::new(5).unwrap()),
            Err(YieldOverlapError::OverlapsExceedsLineCount {
                num_overlaps: 5,
                line_count: 1
            })
        ));
    }
}
