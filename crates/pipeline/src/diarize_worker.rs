//! Diarization thread: final system-audio phrases → voice embedding → speaker number.
//! The result reaches the UI as a separate `SpeakerAssigned` event and is applied on top of
//! any utterance updates (recognition and translation only know the provisional speaker).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use sr_core::{PipelineEvent, Speaker};
use sr_diarize::{SpeakerEmbedder, SpeakerTracker};

#[derive(Clone)]
pub struct DiarizationConfig {
    pub model: PathBuf,
    /// Conversation speakers: persist across recording pauses (owned by the app).
    pub tracker: Arc<Mutex<SpeakerTracker>>,
}

impl std::fmt::Debug for DiarizationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizationConfig").field("model", &self.model).finish_non_exhaustive()
    }
}

pub struct DiarJob {
    pub utt_id: u64,
    pub audio: Vec<f32>,
}

static EMBEDDER: Mutex<Option<(PathBuf, SpeakerEmbedder)>> = Mutex::new(None);

pub fn run(cfg: DiarizationConfig, rx: mpsc::Receiver<DiarJob>, tx: async_channel::Sender<PipelineEvent>, last: Arc<AtomicU8>) {
    let mut guard = EMBEDDER.lock().expect("embedder poisoned");
    if guard.as_ref().map(|(p, _)| p != &cfg.model).unwrap_or(true) {
        match SpeakerEmbedder::load(&cfg.model) {
            Ok(e) => *guard = Some((cfg.model.clone(), e)),
            Err(e) => {
                log::error!("speaker model failed to load: {e:#}");
                let _ = tx.send_blocking(PipelineEvent::Notice(format!("speaker separation unavailable: {e:#}")));
                return; // utterances keep the provisional speaker
            }
        }
    }
    let embedder = &guard.as_ref().expect("loaded").1;
    for job in rx {
        let secs = job.audio.len() as f32 / 16_000.0;
        let t = Instant::now();
        let speaker = match embedder.embed(&job.audio) {
            Ok(emb) => {
                let a = cfg.tracker.lock().expect("tracker poisoned").assign(&emb, secs);
                log::debug!("diarize #{} {secs:.1}s in {:?}: speaker {} sim {:.2}{}", job.utt_id, t.elapsed(), a.speaker, a.similarity, if a.is_new { " (new)" } else { "" });
                a.speaker
            }
            // Phrase too short: same speaker as the last one.
            Err(_) => last.load(Ordering::Relaxed).max(1),
        };
        last.store(speaker, Ordering::Relaxed);
        if tx.send_blocking(PipelineEvent::SpeakerAssigned { id: job.utt_id, speaker: Speaker::Remote(speaker) }).is_err() {
            break;
        }
    }
}
