//! Speech recognition on whisper.cpp (Vulkan on Windows/Linux, Metal on macOS).

use std::path::PathBuf;

use anyhow::Context as _;
use sr_core::Lang;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

pub mod filter;

pub const SAMPLE_RATE: usize = 16_000;

#[derive(Clone, Debug)]
pub struct AsrConfig {
    pub model_path: PathBuf,
    pub use_gpu: bool,
    pub threads: usize,
    /// Extra encoder context beyond the phrase length (in positions, 50 = 1 s).
    /// `None`: full 30 s context; more accurate, but several times slower on short phrases.
    pub audio_ctx_margin: Option<i32>,
    /// Flash-attention: on Vulkan it noticeably corrupted text with a shortened context, so it is off by default.
    pub flash_attn: bool,
    /// Response length limit, in tokens per second of audio (protection against looping).
    pub max_tokens_per_sec: Option<f32>,
}

impl AsrConfig {
    pub fn new(model_path: PathBuf, use_gpu: bool, threads: usize) -> Self {
        Self { model_path, use_gpu, threads, audio_ctx_margin: Some(256), flash_attn: false, max_tokens_per_sec: Some(8.0) }
    }
}

/// Interim pass (while speech is ongoing) or final pass (the phrase has ended).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Interim,
    Final,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Mean log-prob of the text tokens, a confidence measure.
    pub avg_logprob: f32,
    pub no_speech: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Text(Transcript),
    /// There was speech, but it couldn't be recognized (or Whisper "hallucinated").
    NotRecognized,
}

pub struct Engine {
    _ctx: WhisperContext,
    state: WhisperState,
    cfg: AsrConfig,
}

/// Routes whisper.cpp/ggml logs into `log` (otherwise they spill into stderr).
pub fn init_logging() {
    whisper_rs::install_logging_hooks();
}

impl Engine {
    pub fn load(cfg: AsrConfig) -> anyhow::Result<Self> {
        let path = cfg.model_path.to_str().context("model path is not UTF-8")?;
        anyhow::ensure!(cfg.model_path.exists(), "model file not found: {}", cfg.model_path.display());
        let mut cp = WhisperContextParameters::default();
        cp.use_gpu(cfg.use_gpu);
        cp.flash_attn(cfg.use_gpu && cfg.flash_attn);
        let ctx = WhisperContext::new_with_params(path, cp).with_context(|| format!("failed to load {}", cfg.model_path.display()))?;
        let state = ctx.create_state().context("failed to create whisper state")?;
        Ok(Self { _ctx: ctx, state, cfg })
    }

    /// The first run on the GPU is several times slower (shader compilation, buffer allocation),
    /// so we do it right after loading, on silence.
    pub fn warmup(&mut self) -> anyhow::Result<()> {
        let silence = vec![0.0f32; SAMPLE_RATE * 2];
        self.detect_language(&silence)?;
        self.transcribe(&silence, Lang::En, Pass::Interim)?;
        Ok(())
    }

    /// Detects the phrase language among the five supported ones. Returns the language and its probability.
    pub fn detect_language(&mut self, audio: &[f32]) -> anyhow::Result<(Lang, f32)> {
        let audio = pad(audio);
        self.state.pcm_to_mel(&audio, self.cfg.threads)?;
        let (_, probs) = self.state.lang_detect(0, self.cfg.threads)?;
        Ok(pick_language(|code| whisper_rs::get_lang_id(code).and_then(|i| probs.get(i as usize).copied()).unwrap_or(0.0)))
    }

    pub fn transcribe(&mut self, audio: &[f32], lang: Lang, pass: Pass) -> anyhow::Result<Outcome> {
        let audio = pad(audio);
        let secs = audio.len() as f32 / SAMPLE_RATE as f32;
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_language(Some(lang.code()));
        p.set_n_threads(self.cfg.threads as i32);
        p.set_translate(false);
        // Each phrase stands alone: context from previous phrases provokes repetitions and hallucinations.
        p.set_no_context(true);
        p.set_single_segment(true);
        p.set_no_timestamps(true);
        p.set_suppress_blank(true);
        p.set_suppress_nst(true);
        p.set_temperature(0.0);
        // In the final pass, allow whisper.cpp to retry at a higher temperature
        // on a "bad" result; not in the interim pass, for speed.
        p.set_temperature_inc(if pass == Pass::Final { 0.2 } else { 0.0 });
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        if let Some(margin) = self.cfg.audio_ctx_margin {
            p.set_audio_ctx(fit_audio_ctx(secs, margin));
        }
        if let Some(tps) = self.cfg.max_tokens_per_sec {
            p.set_max_tokens((secs * tps).ceil() as i32 + 8);
        }
        self.state.full(p, &audio)?;

        let mut text = String::new();
        let mut no_speech: f32 = 0.0;
        let (mut lp_sum, mut lp_n) = (0.0f32, 0u32);
        for seg in self.state.as_iter() {
            text.push_str(&seg.to_str_lossy().unwrap_or_default());
            no_speech = no_speech.max(seg.no_speech_probability());
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else { continue };
                let t = tok.to_str_lossy().unwrap_or_default();
                if t.starts_with("[_") || t.starts_with("<|") {
                    continue;
                }
                lp_sum += tok.token_data().plog;
                lp_n += 1;
            }
        }
        let tr = Transcript { text: text.trim().to_string(), avg_logprob: if lp_n > 0 { lp_sum / lp_n as f32 } else { -10.0 }, no_speech };
        Ok(filter::judge(tr, lang))
    }
}

/// GGML model file for an id from the settings.
pub fn model_file(id: &str) -> Option<&'static str> {
    Some(match id {
        "whisper-large-v3-turbo" => "ggml-large-v3-turbo-q5_0.bin",
        "whisper-medium" => "ggml-medium-q5_0.bin",
        "whisper-small" => "ggml-small-q5_1.bin",
        _ => return None,
    })
}

/// whisper.cpp dislikes input shorter than a second, so pad with silence.
fn pad(audio: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    const MIN: usize = SAMPLE_RATE * 1_100 / 1_000;
    if audio.len() >= MIN {
        audio.into()
    } else {
        let mut v = audio.to_vec();
        v.resize(MIN, 0.0);
        v.into()
    }
}

/// The Whisper encoder is designed for 30 s = 1500 positions (50 per second). For a short phrase
/// compute only the needed part plus a margin; this is several times faster.
pub fn fit_audio_ctx(secs: f32, margin: i32) -> i32 {
    ((secs * 50.0).ceil() as i32 + margin).clamp(256, 1500)
}

/// Picks the most probable language among the supported ones.
pub fn pick_language(prob: impl Fn(&str) -> f32) -> (Lang, f32) {
    Lang::ALL
        .into_iter()
        .map(|l| (l, prob(l.code())))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("Lang::ALL is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_ctx_bounds() {
        assert_eq!(fit_audio_ctx(0.5, 128), 256);
        assert_eq!(fit_audio_ctx(4.0, 128), 328);
        assert_eq!(fit_audio_ctx(30.0, 128), 1500);
        assert_eq!(fit_audio_ctx(60.0, 0), 1500);
    }

    #[test]
    fn language_is_restricted_to_supported() {
        // Japanese would "win", but it is not among the supported ones
        let probs = |c: &str| match c {
            "ja" => 0.9,
            "de" => 0.06,
            "en" => 0.03,
            _ => 0.0,
        };
        assert_eq!(pick_language(probs), (Lang::De, 0.06));
    }
}
