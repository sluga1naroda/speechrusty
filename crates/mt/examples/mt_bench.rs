//! cargo run -p sr-mt --release --example mt_bench -- <model.gguf> [--cpu]
//! (requires a built speechrusty-mt alongside, or SR_MT_SIDECAR)
use std::time::Instant;
use sr_core::Lang::{self, *};
use sr_mt::{ContextTurn, LlmTranslator, build_prompt, clean_output, locate_sidecar, max_tokens_for};

const CASES: &[(Lang, Lang, &str)] = &[
    (En, Ru, "Hi everyone, can you hear me well?"),
    (En, Ru, "The quarterly budget is ready, I'll share my screen in a sec."),
    (Ru, En, "Отлично, давайте начнём с бюджета и потом обсудим сроки."),
    (De, Ru, "Ja, alles klar. Wir können anfangen."),
    (De, En, "Entschuldigung, die Verbindung war kurz weg."),
    (Es, De, "Perfecto, también tengo algunas preguntas sobre el plan de marketing."),
    (Es, Ru, "¿Podemos mover la reunión al jueves por la tarde?"),
    (Fr, Ru, "Je pense qu'on devrait réduire les coûts de publicité de dix pour cent."),
    (Fr, De, "D'accord, je vous envoie le document après l'appel."),
    (Ru, Fr, "Мне кажется, это слишком оптимистичный прогноз."),
    (En, Es, "Let's wrap up and meet again next Monday."),
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gpu = !args.iter().any(|a| a == "--cpu");
    let side = locate_sidecar().expect("speechrusty-mt not found");
    let t = Instant::now();
    let mut tr = LlmTranslator::start(&side, args[1].as_ref(), gpu)?;
    println!("load {:?} on {:?}", t.elapsed(), tr.devices);
    let mut total = 0.0;
    for (i, (s, d, text)) in CASES.iter().enumerate() {
        let msgs = build_prompt(*s, *d, text, &[]);
        let t = Instant::now();
        let raw = tr.translate(&msgs, max_tokens_for(text))?;
        if std::env::var_os("RAW").is_some() { println!("        RAW: {raw:?}"); }
        let out = clean_output(&raw);
        let ms = t.elapsed().as_millis();
        if i > 0 { total += ms as f64; }
        println!("{:>5}ms {}→{} | {text}\n        => {out}", ms, s.code(), d.code());
    }
    println!("avg (excluding first): {:.0} ms", total / (CASES.len() - 1) as f64);
    // context: "it" → the gender in Russian depends on the previous turn
    let ctx = [ContextTurn { lang: En, original: "Did you see the new presentation?".into(), translation: "Вы видели новую презентацию?".into() }];
    let msgs = build_prompt(En, Ru, "Yes, it looks great, but it is too long.", &ctx);
    println!("ctx: {}", clean_output(&tr.translate(&msgs, 64)?));
    Ok(())
}
