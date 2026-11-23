//! Converts audio to 16 kHz mono, the format VAD and Whisper expect.

use rubato::{FftFixedInOut, Resampler};

pub const TARGET_RATE: u32 = 16_000;

/// Streaming resampler for a mono signal to 16 kHz. Accepts chunks of any length.
pub enum To16k {
    /// Input is already 16 kHz.
    Passthrough,
    Fft { inner: Box<FftFixedInOut<f32>>, pending: Vec<f32> },
}

impl To16k {
    pub fn new(input_rate: u32) -> anyhow::Result<Self> {
        if input_rate == TARGET_RATE {
            return Ok(Self::Passthrough);
        }
        // ~10 ms of input per chunk: a trade-off between latency and overhead.
        let chunk = (input_rate as usize / 100).max(64);
        let inner = FftFixedInOut::<f32>::new(input_rate as usize, TARGET_RATE as usize, chunk, 1)?;
        Ok(Self::Fft { inner: Box::new(inner), pending: Vec::new() })
    }

    /// Appends `input`, resampled to 16 kHz, to `out`. A tail shorter than the resampler's
    /// chunk size is buffered until the next call.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        match self {
            Self::Passthrough => out.extend_from_slice(input),
            Self::Fft { inner, pending } => {
                pending.extend_from_slice(input);
                loop {
                    let need = inner.input_frames_next();
                    if pending.len() < need {
                        break;
                    }
                    match inner.process(&[&pending[..need]], None) {
                        Ok(res) => out.extend_from_slice(&res[0]),
                        Err(e) => log::warn!("resample error: {e}"),
                    }
                    pending.drain(..need);
                }
            }
        }
    }
}

/// Downmixes interleaved channels to mono.
pub fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let k = 1.0 / channels as f32;
    out.extend(interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() * k));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize).map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5).collect()
    }

    #[test]
    fn resamples_48k_and_44k_to_expected_length_and_pitch() {
        for rate in [48_000u32, 44_100] {
            let input = sine(rate, 440.0, 1.0);
            let mut r = To16k::new(rate).unwrap();
            let mut out = Vec::new();
            // in chunks of varying length, as they arrive from a sound card
            for c in input.chunks(777) {
                r.process(c, &mut out);
            }
            let expected = 16_000.0;
            assert!((out.len() as f32 - expected).abs() < 600.0, "rate {rate}: got {} samples", out.len());
            // frequency preserved: count zero crossings in the second half (after the filter delay)
            let tail = &out[out.len() / 2..];
            let crossings = tail.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
            let hz = crossings as f32 / (tail.len() as f32 / 16_000.0);
            assert!((hz - 440.0).abs() < 10.0, "rate {rate}: {hz} Hz");
        }
    }

    #[test]
    fn downmix_stereo() {
        let mut out = Vec::new();
        downmix(&[1.0, 0.0, 0.5, 0.5], 2, &mut out);
        assert_eq!(out, vec![0.5, 0.5]);
    }
}
