//! Translation thread: utterances from recognition → `speechrusty-mt` → translated utterances.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sr_core::{Lang, PipelineEvent, Utterance, UtteranceStatus};
use sr_mt::{ContextTurn, LlmTranslator, build_prompt, clean_output, max_tokens_for};

use crate::queue::{PriorityQueue, QueueItem};

/// How many previous utterances to give the model as context.
const CONTEXT_TURNS: usize = 3;
/// Interim text shorter than this is not translated (little value, and the GPU is busy).
const MIN_INTERIM_WORDS: usize = 3;

#[derive(Clone, Debug)]
pub struct MtConfig {
    pub sidecar: PathBuf,
    pub model: PathBuf,
    pub gpu: bool,
    pub translate_interim: bool,
}

#[derive(Debug)]
pub struct MtJob {
    pub utt: Utterance,
}

impl QueueItem for MtJob {
    fn key(&self) -> u64 {
        self.utt.id
    }
    fn is_interim(&self) -> bool {
        self.utt.status == UtteranceStatus::Interim
    }
}

pub type MtQueue = PriorityQueue<MtJob>;

/// The translator process outlives recordings.
static TRANSLATOR: Mutex<Option<(String, LlmTranslator)>> = Mutex::new(None);

fn key(c: &MtConfig) -> String {
    format!("{}|gpu={}", c.model.display(), c.gpu)
}

fn ensure_loaded<'a>(
    slot: &'a mut Option<(String, LlmTranslator)>,
    cfg: &MtConfig,
    send: &dyn Fn(PipelineEvent) -> bool,
) -> Result<&'a mut LlmTranslator, String> {
    let k = key(cfg);
    let alive = match slot.as_mut() {
        Some((sk, t)) => *sk == k && t.is_alive(),
        None => false,
    };
    if !alive {
        *slot = None; // the old process exits before the new one starts (GPU memory)
        send(PipelineEvent::MtLoading(true));
        let t = Instant::now();
        let res = LlmTranslator::start(&cfg.sidecar, &cfg.model, cfg.gpu);
        send(PipelineEvent::MtLoading(false));
        match res {
            Ok(tr) => {
                log::info!("translator ready in {:?}", t.elapsed());
                *slot = Some((k, tr));
            }
            Err(e) => return Err(format!("{e:#}")),
        }
    }
    Ok(&mut slot.as_mut().expect("loaded").1)
}

pub struct MtWorker {
    pub cfg: MtConfig,
    pub queue: Arc<MtQueue>,
    pub tx: async_channel::Sender<PipelineEvent>,
}

impl MtWorker {
    pub fn run(self) {
        let send = |e: PipelineEvent| self.tx.send_blocking(e).is_ok();
        let mut guard = TRANSLATOR.lock().expect("translator poisoned");
        let mut load_error: Option<String> = match ensure_loaded(&mut guard, &self.cfg, &send) {
            Ok(_) => None,
            Err(e) => {
                log::error!("translator failed to start: {e}");
                send(PipelineEvent::Notice(e.clone()));
                Some(e)
            }
        };
        let mut context: VecDeque<ContextTurn> = VecDeque::new();

        while let Some(job) = self.queue.next() {
            let u = job.utt;
            let interim = u.status == UtteranceStatus::Interim;
            if interim && (u.original.split_whitespace().count() < MIN_INTERIM_WORDS || self.queue.has_final_for(u.id)) {
                continue;
            }
            let Some(src) = u.src_lang else { continue };

            if let Some(e) = &load_error {
                if !interim {
                    send(failed(u, e.clone()));
                }
                continue;
            }
            let ctx: Vec<ContextTurn> = context.iter().cloned().collect();
            let t = Instant::now();
            let result = translate(&mut guard, &self.cfg, &send, src, u.dst_lang, &u.original, &ctx);
            log::debug!("MT #{} {} {:?}: {:?}", u.id, if interim { "interim" } else { "final" }, t.elapsed(), result);
            match result {
                Ok(text) if !text.is_empty() => {
                    if !interim {
                        context.push_back(ContextTurn { lang: src, original: u.original.clone(), translation: text.clone() });
                        while context.len() > CONTEXT_TURNS {
                            context.pop_front();
                        }
                    }
                    let out = Utterance { translation: Some(text), ..u };
                    if !send(PipelineEvent::Utterance(out)) {
                        break;
                    }
                }
                Ok(_) if interim => {}
                Ok(_) => {
                    send(failed(u, "empty translation".into()));
                }
                Err(e) => {
                    log::warn!("translation failed: {e}");
                    if !interim {
                        send(failed(u, e));
                    }
                    // The process crashed and failed to restart; give up.
                    if guard.is_none() {
                        load_error = Some("translator is not running".into());
                    }
                }
            }
        }
    }
}

fn failed(u: Utterance, reason: String) -> PipelineEvent {
    PipelineEvent::Utterance(Utterance { translation: None, status: UtteranceStatus::TranslationFailed(reason), ..u })
}

/// Translate, retrying once with a process restart if it crashed.
fn translate(
    slot: &mut Option<(String, LlmTranslator)>,
    cfg: &MtConfig,
    send: &dyn Fn(PipelineEvent) -> bool,
    src: Lang,
    dst: Lang,
    text: &str,
    ctx: &[ContextTurn],
) -> Result<String, String> {
    let messages = build_prompt(src, dst, text, ctx);
    for attempt in 0..2 {
        let tr = match ensure_loaded(slot, cfg, send) {
            Ok(t) => t,
            Err(e) => {
                *slot = None;
                return Err(e);
            }
        };
        match tr.translate(&messages, max_tokens_for(text)) {
            Ok(raw) => return Ok(clean_output(&raw)),
            Err(e) if attempt == 0 && !tr.is_alive() => {
                log::warn!("translator crashed ({e:#}), restarting");
                *slot = None;
            }
            Err(e) => return Err(format!("{e:#}")),
        }
    }
    Err("translator crashed twice".into())
}
