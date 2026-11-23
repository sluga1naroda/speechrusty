//! build.cmd run -p sr-asr --release --example bench -- <model.bin> <wav16k>... [--cpu] [--full-ctx]
use std::time::Instant;
use sr_asr::{AsrConfig, Engine, Outcome, Pass};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let full = args.iter().any(|a| a == "--full-ctx");
    let no_flash = args.iter().any(|a| a == "--no-flash");
    let no_maxtok = args.iter().any(|a| a == "--no-maxtok");
    let margin: i32 = args.iter().position(|a| a == "--margin").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(256);
    let files: Vec<&String> = args[2..].iter().filter(|a| a.ends_with(".wav")).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let t = Instant::now();
    let mut cfg = AsrConfig::new(args[1].clone().into(), !cpu, threads);
    cfg.audio_ctx_margin = (!full).then_some(margin);
    cfg.flash_attn = !no_flash;
    if no_maxtok { cfg.max_tokens_per_sec = None; }
    let mut e = Engine::load(cfg.clone())?;
    println!("load {:?} gpu={} margin={:?} flash={} maxtok={:?}", t.elapsed(), !cpu, cfg.audio_ctx_margin, cfg.flash_attn, cfg.max_tokens_per_sec);
    let (mut total_audio, mut total_t) = (0.0f32, 0.0f32);
    for (i, f) in files.iter().enumerate() {
        let a: Vec<f32> = hound::WavReader::open(f)?.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
        let secs = a.len() as f32 / 16000.0;
        let t = Instant::now();
        let (lang, p) = e.detect_language(&a)?;
        let tl = t.elapsed();
        let t = Instant::now();
        let out = e.transcribe(&a, lang, Pass::Final)?;
        let ta = t.elapsed();
        if i > 0 { total_audio += secs; total_t += (tl + ta).as_secs_f32(); } // first one is warm-up
        let text = match out { Outcome::Text(t) => format!("{} (lp {:.2})", t.text, t.avg_logprob), Outcome::NotRecognized => "<not recognized>".into() };
        println!("{secs:4.1}s {lang:?} {p:.2} | lang {:4}ms asr {:5}ms | {text}", tl.as_millis(), ta.as_millis());
    }
    println!("RTF (excluding warm-up): {:.2}", total_t / total_audio.max(0.01));
    Ok(())
}
