# Bertalign-rs

Rust port of [bertalign](https://github.com/bfsujason/bertalign), an automatic multilingual sentence aligner.

**building**

You can build with cuda support or mkl support by adding the feature flags, `cuda`, `mkl`, `metal`.
```bash
cargo build --release --features cuda
```

note: batch_size refers to the max number of tokens in a batch

```rust
fn main() -> error::Result<()> {
    let device = candle_core::Device::new_cuda(0)?;
    let model = SentenceTransformerBuilder::with_sentence_transformer(
        SentenceTransformerWhich::LaBSE
    )
    .batch_size(2048)
    .with_device(&device)
    .build()?;
    let model = Arc::new(model);

    let aligner = AlignerBuilder::default()
            .max_align(5)?
            .top_k(3)?
            .win(5)
            .skip(-0.1)
            .margin(true)
            .len_penalty(true)
            .build();

    let lines = vec![
        "The weather was warm and sunny.",
        "We decided to go for a walk in the park.",
        "For lunch we had Denny's and went home and slept all day.",
        "It was a perfect day to relax.",
    ];

    let lines2 = vec![
        "날씨가 따뜻하고 화창해요.",
        "우리는 공원에서 산책을 하기로 했어요",
        "점심 때 데니스 먹었고,",
        "집에 가서 하루종일 잤어요.",
        "쉬기에 완벽한 하루였어요.",
    ];

    let alignments = aligner.align(model.clone(), &lines, &lines2)?;
    println!("{:?}", alignments);

    for (src, tgt) in alignments {
        println!(
            "{}",
            src.iter()
                .map(|i| lines[*i])
                .collect::<Vec<&str>>()
                .join(" ")
        );
        println!(
            "{}",
            tgt.iter()
                .map(|i| lines2[*i])
                .collect::<Vec<&str>>()
                .join(" ")
        );
        println!();
    }
    Ok(())
}

```
