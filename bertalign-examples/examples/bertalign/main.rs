use std::sync::Arc;

use bertalign_rs::aligner::AlignerBuilder;
use bertalign_rs::embed::sentence_transformer::{
    SentenceTransformerBuilder, Which as SentenceTransformerWhich,
};

fn get_sentences() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    let en_sents = vec![
        "As mentioned in the “Storing Values with Variables” section, by default, variables are immutable.",
        "This is one of many nudges Rust gives you to write your code in a way that takes advantage of the safety and easy concurrency that Rust offers.",
        "However, you still have the option to make your variables mutable.",
        "Let’s explore how and why Rust encourages you to favor immutability and why sometimes you might want to opt out.",
        "When a variable is immutable, once a value is bound to a name, you can’t change that value.",
        "To illustrate this, generate a new project called variables in your projects directory by using cargo new variables.",
        "Then, in your new variables directory, open src/main.rs and replace its code with the following code, which won’t compile just yet:",
    ];
    let ko_sents = vec![
        "‘변수에 값 저장하기’ 에서 언급했듯이, 변수는 기본적으로 불변 (immutable) 입니다.",
        "이것은 러스트가 제공하는 안정성과 쉬운 동시성을 활용하는 방식으로 코드를 작성할 수 있도록 하는 넛지 (nudge, 슬며시 선택을 유도하기) 중 하나입니다.",
        "하지만 여러분은 여전히 변수를 가변 (mutable) 으로 만들 수 있습니다.",
        "어떻게 하는지 살펴보고 왜 러스트가 불변성을 권하는지와 어떨 때 가변성을 써야 하는지 알아봅시다.",
        "변수가 불변일 때, 어떤 이름에 한번 값이 묶이면 그 값은 바꿀 수 없습니다.",
        "이를 표현하기 위해, cargo new variables로 projects 디렉터리 안에 variables라는 프로젝트를 만들어 봅시다.",
        "그리고, 새 variables 디렉터리의 src/main.rs 파일을 열어서 다음의 코드로 교체하세요 (아직은 컴파일되지 않습니다):",
    ];
    let de_sents = vec![
        "Wie im Abschnitt „Speichern von Werten mit Variablen“ erwähnt, sind Variablen standardmäßig unveränderbar.",
        "Dies ist einer der vielen Stupser, die Rust dir gibt, um deinen Code so zu schreiben, dass du die Vorteile von Sicherheit (safety) und einfacher Nebenläufigkeit (easy concurrency) nutzt, die Rust bietet.",
        "Du hast jedoch immer noch die Möglichkeit, deine Variablen veränderbar (mutable) zu machen.",
        "Lass uns untersuchen, wie und warum Rust dich dazu ermutigt, die Unveränderbarkeit (immutability) zu bevorzugen, und warum du manchmal vielleicht davon abweichen möchtest.",
        "Wenn eine Variable unveränderbar ist, kannst du deren Wert nicht mehr ändern, sobald ein Wert gebunden ist.",
        "Um dies zu veranschaulichen, lege ein neues Projekt namens variables in deinem projects-Verzeichnis an, indem du cargo new variables aufrufst.",
        "Öffne dann in deinem neuen Verzeichnis variables die Datei src/main.rs und ersetze dessen Code durch folgenden Code, der sich sich noch nicht kompilieren lässt:",
    ];

    (en_sents, ko_sents, de_sents)
}

fn print_alignments(
    lhs_lines: &[&str],
    rhs_lines: &[&str],
    alignments: Vec<(Vec<usize>, Vec<usize>)>,
) {
    for (src, tgt) in alignments {
        println!(
            "{}",
            src.iter()
                .map(|i| lhs_lines[*i])
                .collect::<Vec<&str>>()
                .join(" ")
        );
        println!(
            "{}",
            tgt.iter()
                .map(|i| rhs_lines[*i])
                .collect::<Vec<&str>>()
                .join(" ")
        );
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = candle_core::Device::new_cuda(0)?;
    let model = SentenceTransformerBuilder::with_sentence_transformer(
        SentenceTransformerWhich::AllMiniLML6v2,
    )
    .batch_size(2048)
    .with_device(&device)
    .build()?;
    let model = Arc::new(model);

    let aligner = AlignerBuilder::default()
        .max_align(5)
        .top_k(3)
        .win(5)
        .skip(-0.1)
        .margin(true)
        .len_penalty(true)
        .build()?;

    let (en_sents, ko_sents, de_sents) = get_sentences();

    let en2ko_alignments = aligner.align(model.clone(), &en_sents, &ko_sents)?;
    let en2de_alignments = aligner.align(model.clone(), &en_sents, &de_sents)?;
    let ko2de_alignments = aligner.align(model.clone(), &ko_sents, &de_sents)?;

    print_alignments(&en_sents, &ko_sents, en2ko_alignments);
    print_alignments(&en_sents, &de_sents, en2de_alignments);
    print_alignments(&ko_sents, &de_sents, ko2de_alignments);

    Ok(())
}
