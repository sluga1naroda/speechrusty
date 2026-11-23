//! Audio for SpeechRusty: capture, resampling to 16 kHz, splitting into phrases.

pub mod capture;
pub mod resample;
pub mod segment;

pub use capture::{Capture, CaptureMsg, SourceKind};
pub use segment::{SegEvent, Segmenter, SegmenterConfig, Vad};

/// Default VAD (earshot, pure Rust, no model files).
pub fn default_vad() -> earshot::Detector {
    earshot::Detector::default()
}

/// Signal level for the meter: RMS → dBFS, mapped from [-60, 0] dB to [0, 1].
pub fn meter_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    let db = 20.0 * rms.max(1e-9).log10();
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter() {
        assert_eq!(meter_level(&[]), 0.0);
        assert_eq!(meter_level(&[0.0; 100]), 0.0);
        assert!((meter_level(&[1.0; 100]) - 1.0).abs() < 1e-6);
        // −30 dB ≈ middle of the scale
        assert!((meter_level(&[0.0316; 100]) - 0.5).abs() < 0.02);
    }
}
