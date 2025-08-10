# Bertalign-rs

Rust port of [bertalign](https://github.com/bfsujason/bertalign), an automatic multilingual sentence aligner, with [Python](#python) bindings.
 
**Install**

You can install the python package as follows. This installs CUDA support by default.

```bash
cd bindings/python
pip install . 
```

**Usage**

```python
import bertalign_rs
labse = bertalign_rs.LaBSE(batch_size=32)
args = bertalign_rs.BertAlignArgs(top_k=3)
aligner = bertalign_rs.BertAlign(labse, args)

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

You can use the model for embedding text and get the vectors as `list[list[float]]`

```Python
embeddings = labse.embed(src)

print(embeddings[0][:5])
# [-0.0358, -0.0017, 0.0394, -0.0324, 0.0072]
```

**Free gpu memory**

You can free the memory on the GPU by deleting both the `labse` and the `aligner` objects. You have to delete all references to the labse object.

```Python
del labse
del aligner
```



:warning: This project is a WIP and the api is subject to change.