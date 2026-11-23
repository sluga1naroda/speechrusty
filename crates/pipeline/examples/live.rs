//! build.cmd run -p sr-pipeline --example live -- <seconds> [--mic] [--dump <dir>] [--model <ggml.bin>] [--mt <model.gguf>] [--mt-interim] [--cpu]
use std::time::{Duration, Instant};
use sr_core::{AudioSource, Lang, PipelineEvent, SessionConfig, UtteranceStatus};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,whisper_rs=warn")).init();
    sr_pipeline::init_asr_logging();
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let secs: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let asr = arg("--model").map(|m| sr_pipeline::AsrConfig::new(m.into(), !args.iter().any(|a| a == "--cpu"), threads));
    let cfg = SessionConfig { target: Lang::Ru, source: None, mic_enabled: args.iter().any(|a| a == "--mic"), asr_model: String::new(), mt_model: String::new() };
    let t0 = Instant::now();
    let mt = arg("--mt").map(|m| sr_pipeline::MtConfig { sidecar: sr_pipeline::locate_sidecar().expect("speechrusty-mt"), model: m.into(), gpu: !args.iter().any(|a| a == "--cpu"), translate_interim: args.iter().any(|a| a == "--mt-interim") });
    let diarization = arg("--diar").map(|m| sr_pipeline::DiarizationConfig { model: m.into(), tracker: std::sync::Arc::new(std::sync::Mutex::new(sr_pipeline::SpeakerTracker::new(sr_pipeline::TrackerConfig::default()))) });
    let (mut h, rx) = sr_pipeline::start_live(cfg, sr_pipeline::LiveOptions { asr, mt, diarization, dump_dir: arg("--dump").map(Into::into) }, 0, t0);
    let mut speech_ticks = 0;
    let deadline = Duration::from_secs(secs);
    let mut stopped = false;
    loop {
        if !stopped && t0.elapsed() >= deadline {
            h.stop();
            stopped = true;
        }
        let ev = match rx.recv_blocking() { Ok(e) => e, Err(_) => break };
        let at = t0.elapsed().as_secs_f32();
        match ev {
            PipelineEvent::Level { source: AudioSource::System, speech, .. } => speech_ticks += speech as u32,
            PipelineEvent::Utterance(u) => {
                let tag = match u.status { UtteranceStatus::Interim => "…", UtteranceStatus::Final => "✓", UtteranceStatus::NotRecognized => "✗", UtteranceStatus::TranslationFailed(_) => "!" };
                println!("[{at:5.1}s] {tag} #{} {:?} {:?} start={:.1}s: {}{}", u.id, u.speaker, u.src_lang, u.start_ms as f32 / 1000.0, u.original, u.translation.as_deref().map(|t| format!("  ⇒ {t}")).unwrap_or_default());
            }
            PipelineEvent::Retract(id) => println!("[{at:5.1}s] retract #{id}"),
            PipelineEvent::SpeakerAssigned { id, speaker } => println!("[{at:5.1}s] speaker #{id} = {speaker:?}"),
            PipelineEvent::AsrLoading(b) => println!("[{at:5.1}s] asr loading: {b}"),
            PipelineEvent::MtLoading(b) => println!("[{at:5.1}s] mt loading: {b}"),

            PipelineEvent::Notice(m) | PipelineEvent::Failed(m) => println!("[{at:5.1}s] NOTICE: {m}"),
            _ => {}
        }
    }
    println!("done; speech ticks {speech_ticks}");
}
