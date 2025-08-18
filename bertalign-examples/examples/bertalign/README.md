# Bertalign

Example of how to align multilingual sentences using the bertalign algorithm

# Run the Example

```Bash
$ cargo run --example bertalign --release
```

```
As mentioned in the “Storing Values with Variables” section, by default, variables are immutable.
‘변수에 값 저장하기’ 에서 언급했듯이, 변수는 기본적으로 불변 (immutable) 입니다.

This is one of many nudges Rust gives you to write your code in a way that takes advantage of the safety and easy concurrency that Rust offers.
이것은 러스트가 제공하는 안정성과 쉬운 동시성을 활용하는 방식으로 코드를 작성할 수 있도록 하는 넛지 (nudge, 슬며시 선택을 유도하기) 중 하나입니다.

However, you still have the option to make your variables mutable.
하지만 여러분은 여전히 변수를 가변 (mutable) 으로 만들 수 있습니다.

Let’s explore how and why Rust encourages you to favor immutability and why sometimes you might want to opt out.
어떻게 하는지 살펴보고 왜 러스트가 불변성을 권하는지와 어떨 때 가변성을 써야 하는지 알아봅시다.

When a variable is immutable, once a value is bound to a name, you can’t change that value.
변수가 불변일 때, 어떤 이름에 한번 값이 묶이면 그 값은 바꿀 수 없습니다.

To illustrate this, generate a new project called variables in your projects directory by using cargo new variables.
이를 표현하기 위해, cargo new variables로 projects 디렉터리 안에 variables라는 프로젝트를 만들어 봅시다.

Then, in your new variables directory, open src/main.rs and replace its code with the following code, which won’t compile just yet:
그리고, 새 variables 디렉터리의 src/main.rs 파일을 열어서 다음의 코드로 교체하세요 (아직은 컴파일되지 않습니다):

As mentioned in the “Storing Values with Variables” section, by default, variables are immutable.
Wie im Abschnitt „Speichern von Werten mit Variablen“ erwähnt, sind Variablen standardmäßig unveränderbar.

This is one of many nudges Rust gives you to write your code in a way that takes advantage of the safety and easy concurrency that Rust offers.
Dies ist einer der vielen Stupser, die Rust dir gibt, um deinen Code so zu schreiben, dass du die Vorteile von Sicherheit (safety) und einfacher Nebenläufigkeit (easy concurrency) nutzt, die Rust bietet.

However, you still have the option to make your variables mutable.
Du hast jedoch immer noch die Möglichkeit, deine Variablen veränderbar (mutable) zu machen.

Let’s explore how and why Rust encourages you to favor immutability and why sometimes you might want to opt out.
Lass uns untersuchen, wie und warum Rust dich dazu ermutigt, die Unveränderbarkeit (immutability) zu bevorzugen, und warum du manchmal vielleicht davon abweichen möchtest.

When a variable is immutable, once a value is bound to a name, you can’t change that value.
Wenn eine Variable unveränderbar ist, kannst du deren Wert nicht mehr ändern, sobald ein Wert gebunden ist.

To illustrate this, generate a new project called variables in your projects directory by using cargo new variables.
Um dies zu veranschaulichen, lege ein neues Projekt namens variables in deinem projects-Verzeichnis an, indem du cargo new variables aufrufst.

Then, in your new variables directory, open src/main.rs and replace its code with the following code, which won’t compile just yet:
Öffne dann in deinem neuen Verzeichnis variables die Datei src/main.rs und ersetze dessen Code durch folgenden Code, der sich sich noch nicht kompilieren lässt:

‘변수에 값 저장하기’ 에서 언급했듯이, 변수는 기본적으로 불변 (immutable) 입니다.
Wie im Abschnitt „Speichern von Werten mit Variablen“ erwähnt, sind Variablen standardmäßig unveränderbar.

이것은 러스트가 제공하는 안정성과 쉬운 동시성을 활용하는 방식으로 코드를 작성할 수 있도록 하는 넛지 (nudge, 슬며시 선택을 유도하기) 중 하나입니다.
Dies ist einer der vielen Stupser, die Rust dir gibt, um deinen Code so zu schreiben, dass du die Vorteile von Sicherheit (safety) und einfacher Nebenläufigkeit (easy concurrency) nutzt, die Rust bietet.

하지만 여러분은 여전히 변수를 가변 (mutable) 으로 만들 수 있습니다.
Du hast jedoch immer noch die Möglichkeit, deine Variablen veränderbar (mutable) zu machen.

어떻게 하는지 살펴보고 왜 러스트가 불변성을 권하는지와 어떨 때 가변성을 써야 하는지 알아봅시다.
Lass uns untersuchen, wie und warum Rust dich dazu ermutigt, die Unveränderbarkeit (immutability) zu bevorzugen, und warum du manchmal vielleicht davon abweichen möchtest.

변수가 불변일 때, 어떤 이름에 한번 값이 묶이면 그 값은 바꿀 수 없습니다.
Wenn eine Variable unveränderbar ist, kannst du deren Wert nicht mehr ändern, sobald ein Wert gebunden ist.

이를 표현하기 위해, cargo new variables로 projects 디렉터리 안에 variables라는 프로젝트를 만들어 봅시다.
Um dies zu veranschaulichen, lege ein neues Projekt namens variables in deinem projects-Verzeichnis an, indem du cargo new variables aufrufst.
```

