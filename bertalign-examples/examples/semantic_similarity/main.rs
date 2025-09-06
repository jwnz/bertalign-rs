use bertalign_rs::embed::sentence_transformer::{
    SentenceTransformerBuilder, Which as SentenceTransformerWhich,
};
use bertalign_rs::embed::Embed;
use bertalign_rs::similarity::cosine_similarity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    let device = candle_core::Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = candle_core::Device::Cpu;

    let model =
        SentenceTransformerBuilder::with_sentence_transformer(SentenceTransformerWhich::LaBSE)
            .batch_size(2048)
            .with_device(&device)
            .build()?;

    let sent1 = "Let’s explore the key differences and improvements in Gemma 3.";
    let sent2 = "Gemma 3의 주요 차이점과 개선 사항을 살펴보겠습니다.";

    let sent1_emb = model.embed(&[sent1])?;
    let sent2_emb = model.embed(&[sent2])?;

    let sim = cosine_similarity(&sent1_emb[0], &sent2_emb[0])?;

    println!("{:?}", sim);

    Ok(())
}
