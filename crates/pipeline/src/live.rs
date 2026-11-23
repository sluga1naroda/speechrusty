//! Live pipeline: system audio and mic capture → VAD → phrases → recognition.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use sr_asr::AsrConfig;
use sr_audio::{Capture, CaptureMsg, SegEvent, Segmenter, SegmenterConfig, SourceKind, default_vad, meter_level};
use sr_core::{AudioSource, PipelineEvent, SessionConfig};
use sr_diarize::{EchoDetector, LoopbackHistory};

use crate::asr_worker::{AsrWorker, Job, JobKind, JobQueue};
use crate::diarize_worker::{DiarJob, DiarizationConfig};
// DiarJob is sent by the ASR worker: voices are clustered only from phrases recognized as speech.
use crate::mt_worker::{MtConfig, MtQueue, MtWorker};
use crate::{EventRx, PipelineHandle};

/// The level meter updates every 64 ms (1024 samples @ 16 kHz).
const METER_SAMPLES: usize = 1024;

#[derive(Clone, Debug, Default)]
pub struct LiveOptions {
    /// Recognition. `None` means capture and segmentation only (debugging).
    pub asr: Option<AsrConfig>,
    /// Translation. `None` means no translation (only the original is shown).
    pub mt: Option<MtConfig>,
    /// Speaker separation in system audio. `None` means all utterances are "Speaker 1".
    pub diarization: Option<DiarizationConfig>,
    /// Where to write each phrase as a WAV file (for debugging segmentation).
    pub dump_dir: Option<PathBuf>,
}

/// `id_base` is the id to start numbering utterances from (ids must not repeat after a pause).
pub fn start_live(cfg: SessionConfig, opts: LiveOptions, id_base: u64, t0: Instant) -> (PipelineHandle, EventRx) {
    let (tx, rx) = async_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let _ = tx.send_blocking(PipelineEvent::Started);
    let queue = Arc::new(JobQueue::default());
    let ids = Arc::new(AtomicU64::new(id_base));
    let last_speaker = Arc::new(AtomicU8::new(1));

    // Diarization runs on its own thread: a voice embedding takes 0.2–0.9 s on the CPU.
    let diar_tx = opts.diarization.clone().filter(|_| opts.asr.is_some()).map(|dc| {
        if let Some(l) = dc.tracker.lock().ok().and_then(|t| t.last()) {
            last_speaker.store(l, Ordering::Relaxed);
        }
        let (dtx, drx) = mpsc::channel::<DiarJob>();
        let (tx, last) = (tx.clone(), last_speaker.clone());
        std::thread::Builder::new()
            .name("sr-diarize".into())
            .spawn(move || crate::diarize_worker::run(dc, drx, tx, last))
            .expect("spawn diarize worker");
        dtx
    });
    // System audio loudness history, used to detect echo in the mic.
    let history = Arc::new(Mutex::new(LoopbackHistory::default()));

    if let Some(asr) = opts.asr.clone() {
        // Translation runs on its own thread: recognition keeps going while the LLM translates.
        let (mt, mt_interim) = match opts.mt.clone() {
            Some(mc) => {
                let q = Arc::new(MtQueue::default());
                let interim = mc.translate_interim;
                let w = MtWorker { cfg: mc, queue: q.clone(), tx: tx.clone() };
                std::thread::Builder::new().name("sr-mt".into()).spawn(move || w.run()).expect("spawn mt worker");
                (Some(q), interim)
            }
            None => (None, false),
        };
        let worker = AsrWorker {
            cfg: asr,
            target: cfg.target,
            source_lang: cfg.source,
            queue: queue.clone(),
            tx: tx.clone(),
            mt,
            mt_interim,
            last_speaker: last_speaker.clone(),
            diar: diar_tx,
        };
        // Not part of the handle: after stop it finishes the final phrases and exits on its own.
        std::thread::Builder::new().name("sr-asr".into()).spawn(move || worker.run()).expect("spawn asr worker");
    } else {
        queue.close();
    }

    let mut sources = vec![(AudioSource::System, SourceKind::Loopback)];
    if cfg.mic_enabled {
        sources.push((AudioSource::Mic, SourceKind::Microphone));
    }
    let asr_on = opts.asr.is_some();
    let threads = sources
        .into_iter()
        .map(|(src, kind)| {
            let w = SourceWorker {
                src,
                kind,
                tx: tx.clone(),
                stop: stop.clone(),
                base_ms: t0.elapsed().as_millis() as u64,
                dump_dir: opts.dump_dir.clone(),
                queue: asr_on.then(|| queue.clone()),
                ids: ids.clone(),
                utt: HashMap::new(),
                history: history.clone(),
                echo: (src == AudioSource::Mic).then(EchoDetector::default),
                samples: 0,
            };
            std::thread::Builder::new().name(format!("sr-source-{src:?}")).spawn(move || w.run()).expect("spawn source worker")
        })
        .collect();
    let handle = PipelineHandle::new(stop, threads).with_on_stopped(move || queue.close());
    (handle, rx)
}

struct SourceWorker {
    src: AudioSource,
    kind: SourceKind,
    tx: async_channel::Sender<PipelineEvent>,
    stop: Arc<AtomicBool>,
    /// Stream start relative to the conversation start.
    base_ms: u64,
    dump_dir: Option<PathBuf>,
    queue: Option<Arc<JobQueue>>,
    ids: Arc<AtomicU64>,
    /// Segmenter phrase id → utterance id (shared across all sources).
    utt: HashMap<u64, (u64, u64)>,
    history: Arc<Mutex<LoopbackHistory>>,
    /// Mic only: drop phrases that turned out to be echo from the speakers.
    echo: Option<EchoDetector>,
    /// Number of 16 kHz samples received since the stream started.
    samples: u64,
}

impl SourceWorker {
    fn run(mut self) {
        let (ctx, crx) = mpsc::channel();
        let _capture = Capture::spawn(self.kind, ctx);
        let mut seg = Segmenter::new(default_vad(), SegmenterConfig::default());
        let mut meter = Vec::with_capacity(METER_SAMPLES * 2);
        let mut events = Vec::new();

        loop {
            if self.stop.load(Ordering::Relaxed) {
                seg.flush(&mut events);
                self.handle(events.drain(..));
                break;
            }
            match crx.recv_timeout(Duration::from_millis(50)) {
                Ok(CaptureMsg::Audio(a)) => {
                    if self.src == AudioSource::System {
                        let at = self.base_ms + self.samples * 1000 / 16_000;
                        self.history.lock().expect("history poisoned").push(at, &a);
                    }
                    self.samples += a.len() as u64;
                    seg.push(&a, &mut events);
                    meter.extend_from_slice(&a);
                    if meter.len() >= METER_SAMPLES {
                        let ok = self.send(PipelineEvent::Level { source: self.src, level: meter_level(&meter), speech: seg.in_speech() });
                        meter.clear();
                        if !ok {
                            break;
                        }
                    }
                }
                Ok(CaptureMsg::Opened { device, rate, channels }) => {
                    log::info!("{:?}: capturing from '{device}' ({rate} Hz, {channels} ch)", self.src);
                }
                Ok(CaptureMsg::Problem(m)) => {
                    if !self.send(PipelineEvent::Notice(format!("{}: {m}", source_name(self.src)))) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            self.handle(std::mem::take(&mut events).into_iter());
        }
        self.send(PipelineEvent::Level { source: self.src, level: 0.0, speech: false });
    }

    fn send(&self, e: PipelineEvent) -> bool {
        self.tx.send_blocking(e).is_ok()
    }

    fn handle(&mut self, events: impl Iterator<Item = SegEvent>) {
        for e in events {
            match e {
                SegEvent::Start { id, start_ms } => {
                    let utt_id = self.ids.fetch_add(1, Ordering::Relaxed) + 1;
                    self.utt.insert(id, (utt_id, self.base_ms + start_ms));
                }
                SegEvent::Update { id, audio } => {
                    if !self.is_echo(id, &audio) {
                        self.job(id, JobKind::Interim(audio));
                    }
                }
                SegEvent::Cancel { id } => {
                    log::debug!("{:?} #{id} cancelled (too short)", self.src);
                    self.job(id, JobKind::Cancel);
                    self.utt.remove(&id);
                }
                SegEvent::End { id, start_ms, end_ms, audio, forced } => {
                    log::info!(
                        "{:?} #{id} phrase {:.2}–{:.2} s{}",
                        self.src,
                        (self.base_ms + start_ms) as f32 / 1000.0,
                        (self.base_ms + end_ms) as f32 / 1000.0,
                        if forced { " (forced cut)" } else { "" }
                    );
                    if let Some(dir) = &self.dump_dir {
                        dump_wav(dir, self.src, id, self.base_ms + start_ms, &audio);
                    }
                    if self.is_echo(id, &audio) {
                        log::info!("{:?} #{id} dropped: echo of system audio", self.src);
                        self.job(id, JobKind::Cancel);
                        self.utt.remove(&id);
                        continue;
                    }
                    self.job(id, JobKind::Final(audio));
                    self.utt.remove(&id);
                }
            }
        }
    }

    /// Is the mic phrase an echo of system audio?
    fn is_echo(&self, seg_id: u64, audio: &[f32]) -> bool {
        let (Some(det), Some(&(_, start_ms))) = (&self.echo, self.utt.get(&seg_id)) else { return false };
        let hist = self.history.lock().expect("history poisoned");
        match det.correlation(audio, start_ms, &hist) {
            Some(c) => {
                log::debug!("mic #{seg_id} echo correlation {c:.2}");
                c >= det.threshold
            }
            None => false,
        }
    }

    fn job(&self, seg_id: u64, kind: JobKind) {
        let (Some(q), Some(&(utt_id, start_ms))) = (&self.queue, self.utt.get(&seg_id)) else { return };
        let system_overlap = match (&self.echo, &kind) {
            (Some(det), JobKind::Final(a) | JobKind::Interim(a)) => {
                let dur = a.len() as u64 * 1000 / 16_000;
                self.history.lock().expect("history poisoned").active_fraction(start_ms, dur, det.min_loopback_log_rms) >= 0.3
            }
            _ => false,
        };
        q.push(Job { utt_id, source: self.src, start_ms, kind, system_overlap });
    }
}

fn source_name(src: AudioSource) -> &'static str {
    match src {
        AudioSource::System => "system audio",
        AudioSource::Mic => "microphone",
    }
}

fn dump_wav(dir: &std::path::Path, src: AudioSource, id: u64, start_ms: u64, audio: &[f32]) {
    let path = dir.join(format!("{src:?}-{start_ms:08}-{id}.wav"));
    let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let res = std::fs::create_dir_all(dir).map_err(|e| e.to_string()).and_then(|_| {
        let mut w = hound::WavWriter::create(&path, spec).map_err(|e| e.to_string())?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|e| e.to_string())?;
        }
        w.finalize().map_err(|e| e.to_string())
    });
    if let Err(e) = res {
        log::warn!("failed to dump {path:?}: {e}");
    }
}
