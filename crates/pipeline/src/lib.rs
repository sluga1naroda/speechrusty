//! Audio → text → translation pipeline. For now this only holds a demo source that
//! plays back a prepared dialogue with interim results, like the real thing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sr_core::{AudioSource, Lang, PipelineEvent, SessionConfig, Speaker, Utterance, UtteranceStatus};

mod asr_worker;
mod diarize_worker;
mod live;
mod mt_worker;
mod queue;
pub use live::{LiveOptions, start_live};
pub use diarize_worker::DiarizationConfig;
pub use mt_worker::MtConfig;
pub use sr_diarize::{EMBEDDING_MODEL_FILE, SpeakerTracker, TrackerConfig};
pub use sr_asr::{AsrConfig, init_logging as init_asr_logging, model_file as asr_model_file};
pub use sr_mt::{client::locate_sidecar, model_file as mt_model_file};

pub type EventRx = async_channel::Receiver<PipelineEvent>;

/// A running pipeline. Stop it with `stop()` or by dropping it.
pub struct PipelineHandle {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    /// Called after the sources stop (e.g. to close the recognition queue).
    on_stopped: Option<Box<dyn FnOnce() + Send>>,
}

impl PipelineHandle {
    pub(crate) fn new(stop: Arc<AtomicBool>, threads: Vec<JoinHandle<()>>) -> Self {
        Self { stop, threads, on_stopped: None }
    }

    pub(crate) fn with_on_stopped(mut self, f: impl FnOnce() + Send + 'static) -> Self {
        self.on_stopped = Some(Box::new(f));
        self
    }

    /// Stops capture. Recognition of already recorded phrases finishes in the background.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        if let Some(f) = self.on_stopped.take() {
            f();
        }
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Starts the demo pipeline. `id_base` is the id to start numbering utterances from,
/// so ids don't repeat after a pause; `t0` is the conversation start time.
pub fn start_demo(cfg: SessionConfig, id_base: u64, t0: Instant) -> (PipelineHandle, EventRx) {
    let (tx, rx) = async_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let thread = std::thread::Builder::new()
        .name("sr-demo-pipeline".into())
        .spawn(move || run_demo(cfg, id_base, t0, tx, stop2))
        .expect("spawn demo pipeline");
    (PipelineHandle::new(stop, vec![thread]), rx)
}

struct Line {
    speaker: Speaker,
    lang: Lang,
    text: &'static str,
    /// Translations into (ru, en); for other target languages the demo uses English.
    ru: &'static str,
    en: &'static str,
}

const SCRIPT: &[Line] = &[
    Line { speaker: Speaker::Remote(1), lang: Lang::En, text: "Hi everyone, can you hear me well?", ru: "Всем привет, меня хорошо слышно?", en: "Hi everyone, can you hear me well?" },
    Line { speaker: Speaker::Remote(2), lang: Lang::De, text: "Ja, alles klar. Wir können anfangen.", ru: "Да, всё понятно. Можем начинать.", en: "Yes, all clear. We can start." },
    Line { speaker: Speaker::Me, lang: Lang::Ru, text: "Отлично, давайте начнём с бюджета.", ru: "Отлично, давайте начнём с бюджета.", en: "Great, let's start with the budget." },
    Line { speaker: Speaker::Remote(1), lang: Lang::En, text: "The quarterly budget is ready, I will share my screen in a moment.", ru: "Квартальный бюджет готов, сейчас я покажу свой экран.", en: "The quarterly budget is ready, I will share my screen in a moment." },
    Line { speaker: Speaker::Remote(3), lang: Lang::Es, text: "Perfecto, también tengo algunas preguntas sobre el plan de marketing.", ru: "Отлично, у меня тоже есть вопросы по маркетинговому плану.", en: "Perfect, I also have some questions about the marketing plan." },
    Line { speaker: Speaker::Remote(2), lang: Lang::De, text: "", ru: "", en: "" },
    Line { speaker: Speaker::Remote(2), lang: Lang::De, text: "Entschuldigung, die Verbindung war kurz weg.", ru: "Извините, связь ненадолго пропала.", en: "Sorry, the connection dropped for a moment." },
];

fn sleep_checked(stop: &AtomicBool, d: Duration) -> bool {
    let end = Instant::now() + d;
    while Instant::now() < end {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    !stop.load(Ordering::Relaxed)
}

fn run_demo(cfg: SessionConfig, id_base: u64, t0: Instant, tx: async_channel::Sender<PipelineEvent>, stop: Arc<AtomicBool>) {
    let send = |e| tx.send_blocking(e).is_ok();
    if !send(PipelineEvent::Started) {
        return;
    }
    let mut id = id_base;
    'outer: for line in SCRIPT.iter().cycle() {
        if line.speaker == Speaker::Me && !cfg.mic_enabled {
            continue;
        }
        if !sleep_checked(&stop, Duration::from_millis(900)) {
            break;
        }
        id += 1;
        let start_ms = t0.elapsed().as_millis() as u64;
        let mut u = Utterance {
            id,
            speaker: line.speaker,
            start_ms,
            src_lang: Some(line.lang),
            dst_lang: cfg.target,
            original: String::new(),
            translation: None,
            status: UtteranceStatus::Interim,
        };
        if line.text.is_empty() {
            // unintelligible chunk of speech
            if !sleep_checked(&stop, Duration::from_millis(1200)) {
                break;
            }
            u.status = UtteranceStatus::NotRecognized;
            if !send(PipelineEvent::Utterance(u)) {
                break;
            }
            continue;
        }
        // interim results, word by word
        let words: Vec<&str> = line.text.split(' ').collect();
        for n in 1..=words.len() {
            if !sleep_checked(&stop, Duration::from_millis(220)) {
                break 'outer;
            }
            u.original = words[..n].join(" ");
            let level = 0.3 + 0.4 * ((n % 3) as f32 / 2.0);
            let source = if line.speaker == Speaker::Me { AudioSource::Mic } else { AudioSource::System };
            if !send(PipelineEvent::Level { source, level, speech: true }) || !send(PipelineEvent::Utterance(u.clone())) {
                break 'outer;
            }
        }
        if !sleep_checked(&stop, Duration::from_millis(400)) {
            break;
        }
        u.translation = if line.lang == cfg.target {
            None
        } else if cfg.target == Lang::Ru {
            Some(line.ru.into())
        } else {
            Some(line.en.into())
        };
        u.status = UtteranceStatus::Final;
        let source = if line.speaker == Speaker::Me { AudioSource::Mic } else { AudioSource::System };
        if !send(PipelineEvent::Level { source, level: 0.0, speech: false }) || !send(PipelineEvent::Utterance(u)) {
            break;
        }
    }
    let _ = send(PipelineEvent::Stopped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_emits_and_stops() {
        let cfg = SessionConfig { target: Lang::Ru, source: None, mic_enabled: true, asr_model: "x".into(), mt_model: "y".into() };
        let (mut h, rx) = start_demo(cfg, 100, Instant::now());
        assert_eq!(rx.recv_blocking().unwrap(), PipelineEvent::Started);
        let mut final_seen = None;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Ok(PipelineEvent::Utterance(u)) = rx.recv_blocking() {
                assert!(u.id > 100);
                if u.status == UtteranceStatus::Final {
                    final_seen = Some(u);
                    break;
                }
            }
        }
        let u = final_seen.expect("final utterance");
        assert_eq!(u.translation.as_deref(), Some("Всем привет, меня хорошо слышно?"));
        h.stop();
        // after stop the channel yields the remainder and Stopped, then closes
        let rest: Vec<_> = std::iter::from_fn(|| rx.recv_blocking().ok()).collect();
        assert_eq!(rest.last(), Some(&PipelineEvent::Stopped));
    }
}
