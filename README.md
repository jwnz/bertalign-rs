# Bertalign-rs

Rust port of [bertalign](https://github.com/bfsujason/bertalign), an automatic multilingual sentence aligner, with [Python](#python) bindings.

# Rust

**building**

You can build with cuda support or mkl support by adding the feature flags, `cuda`, `mkl`, `metal`.
```bash
cargo build --release --features cuda
```

note: batch_size refers to the max number of tokens in a batch

```rust
fn main() -> error::Result<()> {
    let labse = Arc::new(LaBSE::new(Some(true), Some(2048)).unwrap()); // embedding batch_size = 2048
    let aligner = AlignerBuilder::new(embedding_model.clone())
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

    let alignments = aligner.align(&lines, &lines2)?;
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

# Python
 
**Install**

You can install the python package by first building with maturin then installing the whl file. You may have to specify your python interpreter version as shown below.

```bash
cd bindings/python
maturin build --release

# specify python version
maturin build --release --interpreter python3.10

# enable cuda
maturin build --release --features cuda

# enable mkl
maturin build --release --features mkl

# enable metal
maturin build --release features metal
```

**Usage**

note: batch_size refers to the max number of tokens in a batch

```python
import bertalign_rs
labse = bertalign_rs.LaBSE(batch_size=2048)
aligner = bertalign_rs.BertAlign(labse, top_k=3, max_align=5)

src = [
    "The weather was warm and sunny.",
    "We decided to go for a walk in the park.",
    "For lunch we had Denny's and went home and slept all day.",
    "It was a perfect day to relax.",
]

tgt = [
    "날씨가 따뜻하고 화창해요.",
    "우리는 공원에서 산책을 하기로 했어요",
    "점심 때 데니스 먹었고,",
    "집에 가서 하루종일 잤어요.",
    "쉬기에 완벽한 하루였어요.",
]

for src_list, tgt_list in aligner.align(src, tgt):
    s = " ".join(map(lambda x: src[x], src_list))
    t = " ".join(map(lambda x: tgt[x], tgt_list))

    print(s)
    print(t)
    print()
```

**Embedding**

You can use the model for embedding text and get the vectors as `list[list[float]]`. 

```Python
embeddings = labse.embed(src)

print(embeddings[0][:5])
# [-0.0358, -0.0017, 0.0394, -0.0324, 0.0072]
```

The cosine similarity function is also exposed, and it's non-blocking too!

```Python
# get the embeddings
a = labse.embed(["Good Morning"])[0]
b = labse.embed(["Guten Morgen"])[0]

# calculate their similarity
bertalign_rs.cosine_similarity(a, b)
```

**Free gpu memory**

You can free the memory on the GPU by deleting both the `labse` and the `aligner` objects. You have to delete all references to the labse object.

```Python
del labse
del aligner
```



:warning: This project is a WIP and the api is subject to change.
