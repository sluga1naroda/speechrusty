//! Splits a 16 kHz stream into phrases using a voice activity detector (VAD).
//!
//! Same logic as live-captions: a phrase starts after a short burst of confident speech
//! (capturing 250 ms before it), ends after 600 ms of silence, and an overly
//! long phrase is force-split at the quietest point within the last 2 s.

use std::collections::VecDeque;

/// VAD frame size: 16 ms @ 16 kHz.
pub const FRAME: usize = 256;
const MS_PER_FRAME: u64 = 16;

fn frames_for(ms: u64) -> u64 {
    ms.div_ceil(MS_PER_FRAME)
}

pub trait Vad {
    /// Speech probability for the frame, 0..1.
    fn score(&mut self, frame: &[f32]) -> f32;
    fn reset(&mut self);
}

impl Vad for earshot::Detector {
    fn score(&mut self, frame: &[f32]) -> f32 {
        self.predict_f32(frame)
    }
    fn reset(&mut self) {
        earshot::Detector::reset(self);
    }
}

#[derive(Clone, Debug)]
pub struct SegmenterConfig {
    /// Speech start threshold.
    pub on: f32,
    /// Speech continuation threshold (hysteresis: keep the phrase while score ≥ off).
    pub off: f32,
    pub min_speech_ms: u64,
    pub end_silence_ms: u64,
    pub max_segment_ms: u64,
    pub preroll_ms: u64,
    /// Shorter phrases are discarded (clicks, coughs).
    pub min_segment_ms: u64,
    /// How often to emit the phrase's interim audio (for interim recognition).
    pub update_every_ms: u64,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            on: 0.5,
            off: 0.35,
            min_speech_ms: 64,
            end_silence_ms: 600,
            max_segment_ms: 12_000,
            preroll_ms: 250,
            min_segment_ms: 300,
            update_every_ms: 1_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SegEvent {
    /// A phrase started. Time is from the start of the stream, ms.
    Start { id: u64, start_ms: u64 },
    /// Interim snapshot of the whole phrase so far.
    Update { id: u64, audio: Vec<f32> },
    /// The phrase ended. `forced`: cut due to the length limit.
    End { id: u64, start_ms: u64, end_ms: u64, audio: Vec<f32>, forced: bool },
    /// The phrase turned out too short; interim results should be removed.
    Cancel { id: u64 },
}

struct Active {
    id: u64,
    start_frame: u64,
    audio: Vec<f32>,
    /// Score of each frame in the phrase (to find the forced-cut point).
    scores: Vec<f32>,
    silence_frames: u64,
    /// Frames with speech (score ≥ off), to filter out clicks: preroll and tail don't count.
    voiced_frames: u64,
    last_update_len: usize,
}

pub struct Segmenter<V: Vad> {
    vad: V,
    cfg: SegmenterConfig,
    frame_buf: Vec<f32>,
    /// Recent frames before speech start (preroll + start candidates).
    history: VecDeque<(Vec<f32>, f32)>,
    frames: u64,
    /// Consecutive frames above the `on` threshold (while no phrase has started).
    run: u64,
    active: Option<Active>,
    next_id: u64,
    last_score: f32,
}

impl<V: Vad> Segmenter<V> {
    pub fn new(vad: V, cfg: SegmenterConfig) -> Self {
        Self {
            vad,
            cfg,
            frame_buf: Vec::with_capacity(FRAME),
            history: VecDeque::new(),
            frames: 0,
            run: 0,
            active: None,
            next_id: 0,
            last_score: 0.0,
        }
    }

    /// Whether a phrase is currently in progress.
    pub fn in_speech(&self) -> bool {
        self.active.is_some()
    }

    pub fn last_score(&self) -> f32 {
        self.last_score
    }

    /// Feeds 16 kHz mono audio, returns events.
    pub fn push(&mut self, samples: &[f32], events: &mut Vec<SegEvent>) {
        for &s in samples {
            self.frame_buf.push(s);
            if self.frame_buf.len() == FRAME {
                let frame = std::mem::replace(&mut self.frame_buf, Vec::with_capacity(FRAME));
                self.on_frame(frame, events);
            }
        }
    }

    /// Closes the current phrase (e.g. when recording stops).
    pub fn flush(&mut self, events: &mut Vec<SegEvent>) {
        if let Some(a) = self.active.take() {
            self.finish(a, false, events);
        }
        self.history.clear();
        self.run = 0;
        self.vad.reset();
    }

    fn on_frame(&mut self, frame: Vec<f32>, events: &mut Vec<SegEvent>) {
        let score = self.vad.score(&frame);
        self.last_score = score;
        self.frames += 1;

        if self.active.is_none() {
            self.run = if score >= self.cfg.on { self.run + 1 } else { 0 };
            self.history.push_back((frame, score));
            let keep = (frames_for(self.cfg.preroll_ms) + frames_for(self.cfg.min_speech_ms)) as usize;
            while self.history.len() > keep {
                self.history.pop_front();
            }
            if self.run >= frames_for(self.cfg.min_speech_ms) {
                self.next_id += 1;
                let n = self.history.len() as u64;
                let start_frame = self.frames - n;
                let mut audio = Vec::with_capacity(16_000 * 4);
                let mut scores = Vec::new();
                let mut voiced = 0;
                for (f, sc) in self.history.drain(..) {
                    audio.extend_from_slice(&f);
                    voiced += u64::from(sc >= self.cfg.off);
                    scores.push(sc);
                }
                events.push(SegEvent::Start { id: self.next_id, start_ms: start_frame * MS_PER_FRAME });
                self.active = Some(Active { id: self.next_id, start_frame, audio, scores, silence_frames: 0, voiced_frames: voiced, last_update_len: 0 });
                self.run = 0;
            }
            return;
        }

        let end_silence = frames_for(self.cfg.end_silence_ms);
        let max_frames = frames_for(self.cfg.max_segment_ms) as usize;
        let update_samples = (self.cfg.update_every_ms * 16) as usize;

        let a = self.active.as_mut().expect("active");
        a.audio.extend_from_slice(&frame);
        a.scores.push(score);
        if score >= self.cfg.off {
            a.silence_frames = 0;
            a.voiced_frames += 1;
        } else {
            a.silence_frames += 1;
        }

        if a.silence_frames >= end_silence {
            // Trim silence at the end of the phrase, leaving a ~200 ms tail.
            let tail_keep = frames_for(200).min(a.silence_frames);
            let cut = a.scores.len() - (a.silence_frames - tail_keep) as usize;
            a.scores.truncate(cut);
            a.audio.truncate(cut * FRAME);
            let a = self.active.take().expect("active");
            self.finish(a, false, events);
            return;
        }

        if a.scores.len() >= max_frames {
            // Cut at the quietest frame within the last 2 s; the rest continues as a new phrase.
            let window = frames_for(2_000) as usize;
            let from = a.scores.len().saturating_sub(window);
            let (rel, _) = a.scores[from..]
                .iter()
                .enumerate()
                .min_by(|x, y| x.1.total_cmp(y.1))
                .expect("non-empty window");
            let cut = (from + rel + 1).min(a.scores.len());
            let rest_audio = a.audio.split_off(cut * FRAME);
            let rest_scores = a.scores.split_off(cut);
            let rest_voiced = rest_scores.iter().filter(|&&x| x >= self.cfg.off).count() as u64;
            let a = self.active.take().expect("active");
            let cut_frame = a.start_frame + cut as u64;
            self.finish(a, true, events);
            self.next_id += 1;
            events.push(SegEvent::Start { id: self.next_id, start_ms: cut_frame * MS_PER_FRAME });
            self.active = Some(Active {
                id: self.next_id,
                start_frame: cut_frame,
                audio: rest_audio,
                scores: rest_scores,
                silence_frames: 0,
                voiced_frames: rest_voiced,
                last_update_len: 0,
            });
            return;
        }

        // No new words in silence, so no snapshot needed.
        if a.silence_frames == 0 && a.audio.len() - a.last_update_len >= update_samples {
            a.last_update_len = a.audio.len();
            events.push(SegEvent::Update { id: a.id, audio: a.audio.clone() });
        }
    }

    fn finish(&mut self, a: Active, forced: bool, events: &mut Vec<SegEvent>) {
        let dur_ms = a.scores.len() as u64 * MS_PER_FRAME;
        if a.voiced_frames * MS_PER_FRAME < self.cfg.min_segment_ms {
            events.push(SegEvent::Cancel { id: a.id });
            return;
        }
        events.push(SegEvent::End {
            id: a.id,
            start_ms: a.start_frame * MS_PER_FRAME,
            end_ms: a.start_frame * MS_PER_FRAME + dur_ms,
            audio: a.audio,
            forced,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test VAD: "speech" is a frame with amplitude > 0.1.
    struct AmpVad;
    impl Vad for AmpVad {
        fn score(&mut self, f: &[f32]) -> f32 {
            if f.iter().any(|x| x.abs() > 0.1) { 0.9 } else { 0.0 }
        }
        fn reset(&mut self) {}
    }

    fn tone(ms: u64) -> Vec<f32> {
        (0..ms * 16).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect()
    }
    fn silence(ms: u64) -> Vec<f32> {
        vec![0.0; (ms * 16) as usize]
    }

    fn run(input: &[Vec<f32>], cfg: SegmenterConfig) -> Vec<SegEvent> {
        let mut s = Segmenter::new(AmpVad, cfg);
        let mut ev = Vec::new();
        for chunk in input {
            s.push(chunk, &mut ev);
        }
        s.flush(&mut ev);
        ev
    }

    fn ends(ev: &[SegEvent]) -> Vec<(u64, u64, bool)> {
        ev.iter()
            .filter_map(|e| match e {
                SegEvent::End { start_ms, end_ms, forced, .. } => Some((*start_ms, *end_ms, *forced)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn two_phrases_split_by_pause() {
        let ev = run(&[silence(1000), tone(1500), silence(1000), tone(800), silence(1000)], SegmenterConfig::default());
        let e = ends(&ev);
        assert_eq!(e.len(), 2, "{ev:?}");
        // start captures ~250 ms before speech
        assert!(e[0].0 >= 700 && e[0].0 <= 1000, "{e:?}");
        // end ≈ end of tone + ≤200 ms tail
        assert!(e[0].1 >= 2500 && e[0].1 <= 2750, "{e:?}");
        assert!(e[1].0 >= 3200 && e[1].0 <= 3500, "{e:?}");
        assert!(!e[0].2 && !e[1].2);
    }

    #[test]
    fn short_pause_does_not_split() {
        let ev = run(&[tone(1000), silence(300), tone(1000), silence(1000)], SegmenterConfig::default());
        assert_eq!(ends(&ev).len(), 1);
    }

    #[test]
    fn clicks_are_ignored_or_cancelled() {
        // 32 ms click: shorter than min_speech, the phrase never starts
        let ev = run(&[silence(500), tone(32), silence(1000)], SegmenterConfig::default());
        assert!(ev.is_empty(), "{ev:?}");
        // 150 ms: the phrase starts but is cancelled as too short
        let ev = run(&[silence(500), tone(150), silence(1000)], SegmenterConfig { preroll_ms: 0, ..Default::default() });
        assert!(matches!(ev.last(), Some(SegEvent::Cancel { .. })), "{ev:?}");
    }

    #[test]
    fn long_speech_is_force_cut_at_quietest_point() {
        // 10 s of speech, 200 ms "near pause" (shorter than end_silence), another 5 s of speech
        let cfg = SegmenterConfig { max_segment_ms: 12_000, ..Default::default() };
        let ev = run(&[tone(10_000), silence(200), tone(5_000), silence(1000)], cfg);
        let e = ends(&ev);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(e[0].2, "first is forced");
        // the cut is near the quiet spot (10.0–10.2 s from speech start)
        assert!(e[0].1 >= 9_900 && e[0].1 <= 10_300, "{e:?}");
        assert_eq!(e[1].0, e[0].1, "second continues exactly where first ended");
    }

    #[test]
    fn updates_are_periodic_and_grow() {
        let ev = run(&[tone(3_500), silence(1000)], SegmenterConfig::default());
        let sizes: Vec<usize> = ev
            .iter()
            .filter_map(|e| match e {
                SegEvent::Update { audio, .. } => Some(audio.len()),
                _ => None,
            })
            .collect();
        assert_eq!(sizes.len(), 3, "{sizes:?}");
        assert!(sizes.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn flush_closes_open_phrase() {
        let mut s = Segmenter::new(AmpVad, SegmenterConfig::default());
        let mut ev = Vec::new();
        s.push(&tone(1000), &mut ev);
        assert!(s.in_speech());
        s.flush(&mut ev);
        assert!(!s.in_speech());
        assert_eq!(ends(&ev).len(), 1);
    }
}
