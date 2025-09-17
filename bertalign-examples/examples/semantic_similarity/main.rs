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

    let which_model = SentenceTransformerWhich::ParaphraseMultilingualMiniLML12v2;
    let model = SentenceTransformerBuilder::with_sentence_transformer(which_model)
        .batch_size(2048)
        .with_device(&device)
        .build()?;

    let sents_1 = vec![
        "Let’s explore the key differences and improvements in Gemma 3.",
        "Midnight Commander is a feature-rich, full-screen, text-mode application that allows you to copy, move, and delete files and entire directory trees, search for files, and execute commands in the subshell. Internal viewer, editor and diff viewer are included.",
        "את יכולה לחזור על זה?",
    ];
    let sents_2 = vec![
        "garblegarble fargle",
        "미드나이트 커맨더(Midnight Commander)는 다양한 기능을 갖춘 풀스크린 텍스트 모드 애플리케이션으로, 파일과 전체 디렉터리 트리를 복사, 이동, 삭제할 수 있으며, 파일 검색과 서브셸에서의 명령 실행을 지원합니다. 또한 내부 뷰어, 편집기, 그리고 차이 비교 뷰어(diff viewer)가 포함되어 있습니다.",
        "Could you repeat that?"
    ];

    let sent1_emb = model.embed(&sents_1)?;
    let sent2_emb = model.embed(&sents_2)?;

    for (s1, s2) in sent1_emb.iter().zip(&sent2_emb) {
        let sim = cosine_similarity(&s1, &s2)?;
        println!("{:?}", sim);
    }

    Ok(())
}
