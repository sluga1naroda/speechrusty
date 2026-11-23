//! Spike: whisper.cpp on Vulkan. usage: asr <model.bin> <wav16k>... [--cpu]
use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const LANGS: [&str; 5] = ["en", "ru", "de", "es", "fr"];

fn read_wav(path: &str) -> Vec<f32> {
    let r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    assert!(spec.sample_rate == 16000 && spec.channels == 1, "need 16k mono");
    r.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let files: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    for d in whisper_rs::vulkan::list_devices() {
        println!("GPU #{} {}  vram free {:.2} / total {:.2} GB", d.id, d.name, d.vram.free as f64 / 1e9, d.vram.total as f64 / 1e9);
    }
    let t = Instant::now();
    let mut cp = WhisperContextParameters::default();
    cp.use_gpu(!cpu);
    cp.flash_attn(!cpu);
    let wctx = WhisperContext::new_with_params(&args[1], cp)?;
    let mut st = wctx.create_state()?;
    println!("whisper loaded in {:?} (gpu={})", t.elapsed(), !cpu);
    for f in files {
        let audio = read_wav(f);
        let dur = audio.len() as f32 / 16000.0;
        let t = Instant::now();
        st.pcm_to_mel(&audio, threads)?;
        let (_, probs) = st.lang_detect(0, threads)?;
        let (lang, p) = LANGS.iter()
            .map(|l| (*l, probs[whisper_rs::get_lang_id(l).unwrap() as usize]))
            .max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let t_lang = t.elapsed();
        let mut top: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<String> = top[..5].iter().map(|(i, p)| format!("{}={:.2}", whisper_rs::get_lang_str(*i as i32).unwrap_or("?"), p)).collect();
        println!("top langs: {}", top.join(" "));
        let t = Instant::now();
        let mut fp = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        fp.set_language(Some(lang));
        fp.set_n_threads(threads as i32);
        fp.set_print_progress(false);
        fp.set_print_realtime(false);
        fp.set_print_special(false);
        fp.set_print_timestamps(false);
        st.full(fp, &audio)?;
        let mut text = String::new();
        let mut nsp: f32 = 0.0;
        for s in st.as_iter() {
            text.push_str(&s.to_string());
            nsp = nsp.max(s.no_speech_probability());
        }
        let t_asr = t.elapsed();
        println!("--- {f} ({dur:.1}s) lang={lang} p={p:.2} ({t_lang:?}) no_speech={nsp:.2}");
        println!("ASR {t_asr:?} (RTF {:.2}): {}", t_asr.as_secs_f32() / dur, text.trim());
    }
    Ok(())
}
