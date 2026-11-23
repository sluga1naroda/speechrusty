//! Recognition thread: phrase queue from sources → whisper → utterances for the UI.
//!
//! Priorities: final phrases always go before interim ones; of the interim ones
//! only the freshest snapshot of each phrase is taken, stale ones are dropped.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sr_asr::{AsrConfig, Engine, Outcome, Pass};
use sr_core::{AudioSource, Lang, PipelineEvent, Speaker, Utterance, UtteranceStatus};

use crate::mt_worker::{MtJob, MtQueue};
use crate::queue::{PriorityQueue, QueueItem};

#[derive(Debug)]
pub enum JobKind {
    Interim(Vec<f32>),
    Final(Vec<f32>),
    Cancel,
}

#[derive(Debug)]
pub struct Job {
    pub utt_id: u64,
    pub source: AudioSource,
    pub start_ms: u64,
    pub kind: JobKind,
    /// For the mic: system audio was playing during the phrase (possible echo).
    pub system_overlap: bool,
}

/// A recent final utterance, used to detect echo by matching text.
struct Recent {
    id: u64,
    source: AudioSource,
    start_ms: u64,
    text: String,
}

/// Time window and similarity threshold for text-based echo detection.
const ECHO_WINDOW_MS: u64 = 8_000;
const ECHO_TEXT_SIMILARITY: f32 = 0.5;

impl QueueItem for Job {
    fn key(&self) -> u64 {
        self.utt_id
    }
    fn is_interim(&self) -> bool {
        matches!(self.kind, JobKind::Interim(_))
    }
}

pub type JobQueue = PriorityQueue<Job>;

/// The engine outlives recordings: restarting does not reload the model.
static ENGINE: Mutex<Option<(String, Engine)>> = Mutex::new(None);

fn engine_key(c: &AsrConfig) -> String {
    format!("{}|gpu={}", c.model_path.display(), c.use_gpu)
}

pub struct AsrWorker {
    pub cfg: AsrConfig,
    pub target: Lang,
    /// Language of the other participants, if set manually.
    pub source_lang: Option<Lang>,
    pub queue: Arc<JobQueue>,
    pub tx: async_channel::Sender<PipelineEvent>,
    /// Translation queue (`None` means translation is off). Closed when the worker exits.
    pub mt: Option<Arc<MtQueue>>,
    /// Also translate interim text.
    pub mt_interim: bool,
    /// Last system-audio speaker from diarization; the provisional speaker for new utterances.
    pub last_speaker: Arc<std::sync::atomic::AtomicU8>,
    /// Diarization: only final system-audio phrases recognized as speech
    /// (non-speech sounds must not create "speakers").
    pub diar: Option<std::sync::mpsc::Sender<crate::diarize_worker::DiarJob>>,
}

/// Index of a recent utterance from *another* source with nearly the same text close in time.
fn is_echo_text(recent: &std::collections::VecDeque<Recent>, source: AudioSource, start_ms: u64, text: &str) -> Option<usize> {
    recent.iter().position(|r| {
        r.source != source && r.start_ms.abs_diff(start_ms) <= ECHO_WINDOW_MS && sr_diarize::text_similarity(&r.text, text) >= ECHO_TEXT_SIMILARITY
    })
}

impl Drop for AsrWorker {
    fn drop(&mut self) {
        if let Some(mt) = &self.mt {
            mt.close();
        }
    }
}

/// Language detected for the phrase.
#[derive(Clone, Copy)]
struct Detected {
    lang: Lang,
    prob: f32,
}

impl AsrWorker {
    pub fn run(self) {
        let send = |e: PipelineEvent| self.tx.send_blocking(e).is_ok();
        let mut guard = ENGINE.lock().expect("engine poisoned");
        let key = engine_key(&self.cfg);
        if guard.as_ref().map(|(k, _)| k != &key).unwrap_or(true) {
            *guard = None; // free the old model's memory before loading the new one
            send(PipelineEvent::AsrLoading(true));
            let t = Instant::now();
            let loaded = Engine::load(self.cfg.clone()).and_then(|mut e| e.warmup().map(|_| e));
            send(PipelineEvent::AsrLoading(false));
            match loaded {
                Ok(e) => {
                    log::info!("ASR model ready in {:?}", t.elapsed());
                    *guard = Some((key, e));
                }
                Err(e) => {
                    log::error!("ASR load failed: {e:#}");
                    send(PipelineEvent::Failed(format!("{e:#}")));
                    // drain the queue so sources don't pile up jobs
                    while self.queue.next().is_some() {}
                    return;
                }
            }
        }
        let engine = &mut guard.as_mut().expect("engine loaded").1;

        let mut langs: HashMap<u64, Detected> = HashMap::new();
        let mut last_lang: HashMap<AudioSource, Detected> = HashMap::new();
        let mut shown: std::collections::HashSet<u64> = Default::default();
        let mut recent: std::collections::VecDeque<Recent> = Default::default();

        while let Some(job) = self.queue.next() {
            let (audio, pass) = match job.kind {
                JobKind::Cancel => {
                    if shown.remove(&job.utt_id) {
                        send(PipelineEvent::Retract(job.utt_id));
                    }
                    langs.remove(&job.utt_id);
                    continue;
                }
                JobKind::Interim(a) => (a, Pass::Interim),
                JobKind::Final(a) => (a, Pass::Final),
            };
            let t = Instant::now();
            let lang = match self.source_lang {
                Some(l) => l,
                None => {
                    // For the final, re-detect the language on the whole phrase if the interim was uncertain.
                    let cached = langs.get(&job.utt_id).copied().filter(|d| d.prob >= 0.7 || pass == Pass::Interim);
                    let d = match cached {
                        Some(d) => d,
                        None => match engine.detect_language(&audio) {
                            Ok((lang, prob)) => {
                                // Uncertain: stick to the language this source spoke last.
                                match last_lang.get(&job.source) {
                                    Some(prev) if prob < 0.4 => *prev,
                                    _ => Detected { lang, prob },
                                }
                            }
                            Err(e) => {
                                log::warn!("language detection failed: {e:#}");
                                last_lang.get(&job.source).copied().unwrap_or(Detected { lang: self.target, prob: 0.0 })
                            }
                        },
                    };
                    langs.insert(job.utt_id, d);
                    d.lang
                }
            };
            let outcome = match engine.transcribe(&audio, lang, pass) {
                Ok(o) => o,
                Err(e) => {
                    log::warn!("transcription failed: {e:#}");
                    Outcome::NotRecognized
                }
            };
            log::debug!("ASR #{} {:?} {:.1}s audio in {:?}: {:?}", job.utt_id, pass, audio.len() as f32 / 16_000.0, t.elapsed(), outcome);
            if let (Some(d), Pass::Final, AudioSource::System, Outcome::Text(_)) = (&self.diar, pass, job.source, &outcome) {
                let _ = d.send(crate::diarize_worker::DiarJob { utt_id: job.utt_id, audio: audio.clone() });
            }

            let speaker = match job.source {
                AudioSource::Mic => Speaker::Me,
                AudioSource::System => Speaker::Remote(self.last_speaker.load(std::sync::atomic::Ordering::Relaxed).max(1)),
            };
            let mut u = Utterance {
                id: job.utt_id,
                speaker,
                start_ms: job.start_ms,
                src_lang: Some(lang),
                dst_lang: self.target,
                original: String::new(),
                translation: None,
                status: UtteranceStatus::Interim,
            };
            let is_mic = job.source == AudioSource::Mic;
            match (pass, outcome) {
                (Pass::Interim, Outcome::NotRecognized) => continue,
                // Unintelligible mic audio while the speakers were playing is almost certainly echo.
                (Pass::Final, Outcome::NotRecognized) if is_mic && job.system_overlap => {
                    log::info!("mic #{} dropped: unrecognized while system audio was playing", job.utt_id);
                    if shown.remove(&job.utt_id) {
                        send(PipelineEvent::Retract(job.utt_id));
                    }
                    langs.remove(&job.utt_id);
                    continue;
                }
                (Pass::Final, Outcome::Text(tr)) if is_echo_text(&recent, job.source, job.start_ms, &tr.text).is_some() && is_mic => {
                    log::info!("mic #{} dropped: same text as system audio (echo)", job.utt_id);
                    if shown.remove(&job.utt_id) {
                        send(PipelineEvent::Retract(job.utt_id));
                    }
                    langs.remove(&job.utt_id);
                    continue;
                }
                (Pass::Interim, Outcome::Text(tr)) => {
                    u.original = tr.text;
                    shown.insert(job.utt_id);
                }
                (Pass::Final, Outcome::Text(tr)) => {
                    // The system phrase matches an already shown mic phrase, so that one was echo.
                    if !is_mic {
                        while let Some(i) = is_echo_text(&recent, job.source, job.start_ms, &tr.text) {
                            let mic = recent.remove(i).expect("index valid");
                            log::info!("mic #{} retracted: echo of system #{}", mic.id, job.utt_id);
                            send(PipelineEvent::Retract(mic.id));
                        }
                    }
                    recent.push_back(Recent { id: job.utt_id, source: job.source, start_ms: job.start_ms, text: tr.text.clone() });
                    recent.retain(|r| r.start_ms + 2 * ECHO_WINDOW_MS >= job.start_ms);
                    u.original = tr.text;
                    u.status = UtteranceStatus::Final;
                    if let Some(d) = langs.get(&job.utt_id) {
                        last_lang.insert(job.source, *d);
                    }
                    shown.remove(&job.utt_id);
                    langs.remove(&job.utt_id);
                }
                (Pass::Final, Outcome::NotRecognized) => {
                    u.status = UtteranceStatus::NotRecognized;
                    shown.remove(&job.utt_id);
                    langs.remove(&job.utt_id);
                }
            }
            // Translation is needed if the phrase language differs from the target.
            let needs_mt = u.src_lang != Some(self.target) && !u.original.is_empty();
            let interim = u.status == UtteranceStatus::Interim;
            if let (Some(mt), true) = (&self.mt, needs_mt && (!interim || self.mt_interim)) {
                mt.push(MtJob { utt: u.clone() });
            }
            if !send(PipelineEvent::Utterance(u)) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(id: u64, source: AudioSource, start_ms: u64, text: &str) -> Recent {
        Recent { id, source, start_ms, text: text.into() }
    }

    #[test]
    fn echo_text_matching() {
        let recent: std::collections::VecDeque<Recent> =
            [r(1, AudioSource::System, 10_000, "Hello everyone, can you hear me well?"), r(2, AudioSource::Mic, 20_000, "Да, слышно.")].into();
        // the mic repeated a system phrase close in time: echo
        assert_eq!(is_echo_text(&recent, AudioSource::Mic, 10_300, "hello everyone can you hear me"), Some(0));
        // same text, but too far apart in time
        assert_eq!(is_echo_text(&recent, AudioSource::Mic, 30_000, "hello everyone can you hear me"), None);
        // different text
        assert_eq!(is_echo_text(&recent, AudioSource::Mic, 10_300, "Давайте начнём."), None);
        // a system phrase finds mic echo that arrived earlier
        assert_eq!(is_echo_text(&recent, AudioSource::System, 19_800, "Да, слышно"), Some(1));
        // same source is not echo
        assert_eq!(is_echo_text(&recent, AudioSource::System, 10_000, "Hello everyone, can you hear me well?"), None);
    }
}
