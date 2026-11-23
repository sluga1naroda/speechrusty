//! Speaker separation (diarization) and mic echo suppression.

pub mod cluster;
pub mod echo;
pub mod embed;

pub use cluster::{Assignment, SpeakerTracker, TrackerConfig, cosine};
pub use echo::{EchoDetector, LoopbackHistory};
pub use embed::SpeakerEmbedder;

/// Voice embedding model file.
pub const EMBEDDING_MODEL_FILE: &str = "voxceleb_resnet34_LM.onnx";

/// Word-level similarity of two phrases (Jaccard), ignoring case and punctuation.
pub fn text_similarity(a: &str, b: &str) -> f32 {
    fn words(s: &str) -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
    }
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    wa.intersection(&wb).count() as f32 / wa.union(&wb).count() as f32
}

#[cfg(test)]
mod tests {
    #[test]
    fn similarity() {
        use super::text_similarity as s;
        assert!(s("Hello everyone, can you hear me?", "hello everyone can you hear me") > 0.99);
        assert!(s("The budget is ready.", "The budget is ready, I'll share it.") >= 0.5);
        assert!(s("Добрый день, коллеги.", "Давайте начнём встречу.") < 0.1);
        assert_eq!(s("", "x"), 0.0);
    }
}
