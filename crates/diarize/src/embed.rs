//! Voice embedding: fbank (kaldi-native-fbank) → WeSpeaker ResNet34 (ONNX via tract, pure Rust).

use std::path::Path;

use anyhow::Context as _;
use tract_onnx::prelude::*;

type Plan = std::sync::Arc<TypedRunnableModel>;

pub struct SpeakerEmbedder {
    plan: Plan,
}

/// Minimum audio length for a meaningful embedding.
pub const MIN_SAMPLES: usize = 16_000 / 2;

impl SpeakerEmbedder {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("failed to read {}", path.display()))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan })
    }

    /// 16 kHz mono in [-1, 1] → normalized embedding.
    pub fn embed(&self, audio: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(audio.len() >= MIN_SAMPLES, "audio too short for a speaker embedding");
        let feats = knf_rs::compute_fbank(audio).map_err(|e| anyhow::anyhow!("fbank: {e}"))?;
        let (frames, bins) = feats.dim();
        let data: Vec<f32> = feats.iter().copied().collect();
        let input = tract_ndarray::Array3::from_shape_vec((1, frames, bins), data)?;
        let out = self.plan.run(tvec!(Tensor::from(input).into()))?;
        let mut emb: Vec<f32> = out[0].to_plain_array_view::<f32>()?.iter().copied().collect();
        crate::cluster::normalize(&mut emb);
        Ok(emb)
    }
}
