//! Echo suppression: other participants from the speakers leak into the mic, and their utterances
//! would be duplicated as "Me". A mic phrase is compared with system audio
//! over the same time span by loudness envelope (10 ms frames), searching for the delay.
//! The envelope is robust to speaker/room frequency response and slight clock drift.

use std::collections::VecDeque;

/// 10 ms @ 16 kHz.
pub const FRAME: usize = 160;
const FRAME_MS: u64 = 10;
/// Keep 60 s of system audio history.
const HISTORY_FRAMES: usize = 6_000;

fn log_rms(frame: &[f32]) -> f32 {
    let e = frame.iter().map(|x| x * x).sum::<f32>() / frame.len().max(1) as f32;
    (e + 1e-10).ln()
}

/// System audio loudness history: frame n covers [n*10, n*10+10) ms of conversation time.
#[derive(Default)]
pub struct LoopbackHistory {
    frames: VecDeque<f32>,
    /// Index of the first frame in `frames`.
    first: u64,
    /// Tail that hasn't filled a whole frame yet.
    pending: Vec<f32>,
    /// Index of the next frame (the time up to which the history is filled).
    next: Option<u64>,
}

impl LoopbackHistory {
    /// Appends audio; `start_ms` is its start time in the conversation (needed for the first chunk
    /// and for realignment after gaps).
    pub fn push(&mut self, start_ms: u64, audio: &[f32]) {
        let start_frame = start_ms / FRAME_MS;
        match self.next {
            None => {
                self.first = start_frame;
                self.next = Some(start_frame);
            }
            // Gap (device reconnect): fill with silence.
            Some(n) if start_frame > n + 1 => {
                for _ in n..start_frame {
                    self.frames.push_back((1e-10f32).ln());
                }
                self.pending.clear();
                self.next = Some(start_frame);
            }
            _ => {}
        }
        self.pending.extend_from_slice(audio);
        while self.pending.len() >= FRAME {
            let f: Vec<f32> = self.pending.drain(..FRAME).collect();
            self.frames.push_back(log_rms(&f));
            *self.next.as_mut().expect("initialized") += 1;
        }
        while self.frames.len() > HISTORY_FRAMES {
            self.frames.pop_front();
            self.first += 1;
        }
    }

    /// Fraction of frames in the interval where system audio is louder than `min_log_rms` (speech/sound present).
    pub fn active_fraction(&self, start_ms: u64, dur_ms: u64, min_log_rms: f32) -> f32 {
        let (a, n) = ((start_ms / FRAME_MS) as i64, (dur_ms / FRAME_MS).max(1) as i64);
        let active = (a..a + n).filter(|&f| self.get(f).is_some_and(|v| v >= min_log_rms)).count();
        active as f32 / n as f32
    }

    fn get(&self, frame: i64) -> Option<f32> {
        if frame < self.first as i64 {
            return None;
        }
        self.frames.get((frame - self.first as i64) as usize).copied()
    }
}

#[derive(Clone, Debug)]
pub struct EchoDetector {
    /// Envelope correlation threshold.
    pub threshold: f32,
    /// Mic delay relative to system audio, ms (can also be negative
    /// due to imprecise stream alignment).
    pub min_lag_ms: i64,
    pub max_lag_ms: i64,
    /// If system audio is quieter than this (mean log-RMS), there can be no echo.
    pub min_loopback_log_rms: f32,
}

impl Default for EchoDetector {
    fn default() -> Self {
        // −50 dBFS ≈ ln(1e-5) in energy
        Self { threshold: 0.6, min_lag_ms: -300, max_lag_ms: 600, min_loopback_log_rms: (1e-5f32).ln() }
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 10 {
        return 0.0;
    }
    let (ma, mb) = (a[..n].iter().sum::<f32>() / n as f32, b[..n].iter().sum::<f32>() / n as f32);
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        cov += x * y;
        va += x * x;
        vb += y * y;
    }
    if va < 1e-9 || vb < 1e-9 { 0.0 } else { cov / (va.sqrt() * vb.sqrt()) }
}

impl EchoDetector {
    /// Highest correlation of a mic phrase with system audio (accounting for delay)
    /// or `None` if system audio was silent at that time / there is no history.
    pub fn correlation(&self, mic: &[f32], mic_start_ms: u64, hist: &LoopbackHistory) -> Option<f32> {
        let env: Vec<f32> = mic.chunks_exact(FRAME).map(log_rms).collect();
        if env.len() < 10 {
            return None;
        }
        let start = (mic_start_ms / FRAME_MS) as i64;
        let mut best: Option<f32> = None;
        let mut lag = self.min_lag_ms / FRAME_MS as i64;
        while lag <= self.max_lag_ms / FRAME_MS as i64 {
            // the mic hears the speakers with a delay: mic frame i ↔ system audio frame i - lag
            let lb: Option<Vec<f32>> = (0..env.len() as i64).map(|i| hist.get(start + i - lag)).collect();
            if let Some(lb) = lb {
                let mean = lb.iter().sum::<f32>() / lb.len() as f32;
                if mean >= self.min_loopback_log_rms {
                    let c = pearson(&env, &lb);
                    best = Some(best.map_or(c, |b: f32| b.max(c)));
                }
            }
            lag += 1;
        }
        best
    }

    pub fn is_echo(&self, mic: &[f32], mic_start_ms: u64, hist: &LoopbackHistory) -> bool {
        self.correlation(mic, mic_start_ms, hist).is_some_and(|c| c >= self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }
    }

    /// "Speech": noise modulated by syllables of random length and loudness.
    fn speechlike(seed: u32, secs: f32) -> Vec<f32> {
        let mut r = Rng(seed);
        let n = (secs * 16_000.0) as usize;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let syl = 1_000 + ((r.next() + 0.5) * 3_000.0) as usize;
            let amp = if r.next() > -0.2 { 0.1 + (r.next() + 0.5) * 0.4 } else { 0.0 };
            for _ in 0..syl {
                out.push(r.next() * amp);
            }
        }
        out.truncate(n);
        out
    }

    fn history_with(audio: &[f32], start_ms: u64) -> LoopbackHistory {
        let mut h = LoopbackHistory::default();
        for (i, c) in audio.chunks(480).enumerate() {
            h.push(start_ms + i as u64 * 30, c);
        }
        h
    }

    #[test]
    fn delayed_quiet_copy_is_echo() {
        let remote = speechlike(1, 6.0);
        let h = history_with(&remote, 0);
        // mic: 1–4 s fragment, 120 ms delay, 5x quieter, plus room noise
        let mut r = Rng(9);
        let delay = 16 * 120;
        let mic: Vec<f32> = (16_000..64_000).map(|i| remote[i - delay] * 0.2 + r.next() * 0.002).collect();
        let d = EchoDetector::default();
        let c = d.correlation(&mic, 1_000, &h).unwrap();
        assert!(c > 0.8, "corr {c}");
        assert!(d.is_echo(&mic, 1_000, &h));
    }

    #[test]
    fn own_speech_is_not_echo() {
        let h = history_with(&speechlike(1, 6.0), 0);
        let mine = speechlike(2, 3.0);
        let d = EchoDetector::default();
        assert!(!d.is_echo(&mine, 1_000, &h), "corr {:?}", d.correlation(&mine, 1_000, &h));
    }

    #[test]
    fn talking_over_remote_is_kept() {
        let remote = speechlike(1, 6.0);
        let h = history_with(&remote, 0);
        let mine = speechlike(3, 3.0);
        // own speech is loud, the other participant's echo is quiet
        let mic: Vec<f32> = (0..48_000).map(|i| mine[i] + remote[16_000 + i] * 0.1).collect();
        let d = EchoDetector::default();
        assert!(!d.is_echo(&mic, 1_000, &h), "corr {:?}", d.correlation(&mic, 1_000, &h));
    }

    #[test]
    fn silent_loopback_or_no_history_is_not_echo() {
        let h = history_with(&vec![0.0; 16_000 * 6], 0);
        let d = EchoDetector::default();
        assert_eq!(d.correlation(&speechlike(2, 2.0), 1_000, &h), None);
        assert_eq!(d.correlation(&speechlike(2, 2.0), 1_000, &LoopbackHistory::default()), None);
    }

    #[test]
    fn active_fraction() {
        let mut h = LoopbackHistory::default();
        h.push(0, &vec![0.0; 16_000]); // 0..1 s silence
        h.push(1_000, &vec![0.3; 16_000]); // 1..2 s sound
        let min = EchoDetector::default().min_loopback_log_rms;
        assert!(h.active_fraction(0, 1_000, min) < 0.01);
        assert!(h.active_fraction(1_000, 1_000, min) > 0.99);
        assert!((h.active_fraction(500, 1_000, min) - 0.5).abs() < 0.02);
        assert_eq!(h.active_fraction(5_000, 1_000, min), 0.0, "no history = not active");
    }

    #[test]
    fn history_gap_and_trim() {
        let mut h = LoopbackHistory::default();
        h.push(0, &[0.1; 1600]); // 0..100 ms
        h.push(1_000, &[0.1; 160]); // gap until 1000 ms
        assert!(h.get(50).unwrap() < -20.0, "gap filled with silence");
        assert!(h.get(100).is_some());
        for i in 0..700 {
            h.push(1_010 + i * 100, &[0.0; 1600]);
        }
        assert!(h.get(0).is_none(), "old frames trimmed");
    }
}
