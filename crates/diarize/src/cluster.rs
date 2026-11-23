//! Online speaker clustering from voice embeddings.
//!
//! Each new phrase is compared (cosine) with the centroids of known speakers.
//! Similar enough means the same speaker (the centroid is refined), otherwise a new one.
//! Short phrases give a noisy embedding, so they never create a new speaker.

#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// "Same speaker" threshold.
    pub same: f32,
    /// Phrases shorter than this (s) don't create a new speaker; they go to the nearest one.
    pub min_new_secs: f32,
    pub max_speakers: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self { same: 0.55, min_new_secs: 1.5, max_speakers: 8 }
    }
}

struct Centroid {
    /// Sum of normalized embeddings, weighted by phrase duration.
    sum: Vec<f32>,
    weight: f32,
}

impl Centroid {
    fn similarity(&self, e: &[f32]) -> f32 {
        cosine(&self.sum, e)
    }
}

pub struct SpeakerTracker {
    cfg: TrackerConfig,
    centroids: Vec<Centroid>,
    last: Option<usize>,
}

/// Result: speaker number (1-based) and confidence (cosine with the centroid; 1.0 for a new one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assignment {
    pub speaker: u8,
    pub similarity: f32,
    pub is_new: bool,
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

impl SpeakerTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self { cfg, centroids: Vec::new(), last: None }
    }

    /// Last assigned speaker (for the interim text of the next phrase).
    pub fn last(&self) -> Option<u8> {
        self.last.map(|i| i as u8 + 1)
    }

    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    pub fn assign(&mut self, embedding: &[f32], secs: f32) -> Assignment {
        let mut e = embedding.to_vec();
        normalize(&mut e);
        let best = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.similarity(&e)))
            .max_by(|a, b| a.1.total_cmp(&b.1));

        let can_create = secs >= self.cfg.min_new_secs && self.centroids.len() < self.cfg.max_speakers;
        let (idx, sim, is_new) = match best {
            Some((i, s)) if s >= self.cfg.same || !can_create => (i, s, false),
            _ => {
                self.centroids.push(Centroid { sum: vec![0.0; e.len()], weight: 0.0 });
                (self.centroids.len() - 1, 1.0, true)
            }
        };
        // The centroid is refined only with confident phrases, otherwise it "drifts".
        if is_new || sim >= self.cfg.same {
            let c = &mut self.centroids[idx];
            let w = secs.clamp(0.5, 10.0);
            for (s, x) in c.sum.iter_mut().zip(&e) {
                *s += x * w;
            }
            c.weight += w;
        }
        self.last = Some(idx);
        Assignment { speaker: idx as u8 + 1, similarity: sim, is_new }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedding of "voice" `v` with a little noise `k`.
    fn voice(v: usize, k: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; 16];
        e[v] = 1.0;
        e[(v + 1 + k) % 16] += 0.25;
        e
    }

    #[test]
    fn separates_and_tracks_voices() {
        let mut t = SpeakerTracker::new(TrackerConfig::default());
        assert_eq!(t.assign(&voice(0, 0), 3.0).speaker, 1);
        assert_eq!(t.assign(&voice(5, 0), 3.0).speaker, 2);
        assert_eq!(t.assign(&voice(0, 3), 2.0).speaker, 1);
        assert_eq!(t.assign(&voice(5, 7), 2.0).speaker, 2);
        let a = t.assign(&voice(10, 0), 2.0);
        assert!(a.is_new && a.speaker == 3);
        assert_eq!(t.len(), 3);
        assert_eq!(t.last(), Some(3));
    }

    #[test]
    fn short_phrase_never_creates_speaker() {
        let mut t = SpeakerTracker::new(TrackerConfig::default());
        t.assign(&voice(0, 0), 3.0);
        // dissimilar but short: assign to the nearest
        let a = t.assign(&voice(9, 0), 0.8);
        assert_eq!((a.speaker, a.is_new), (1, false));
        assert_eq!(t.len(), 1);
        // the very first short one must still be assigned to someone
        let mut t = SpeakerTracker::new(TrackerConfig::default());
        assert_eq!(t.assign(&voice(0, 0), 0.5).speaker, 1);
    }

    #[test]
    fn speaker_limit() {
        let mut t = SpeakerTracker::new(TrackerConfig { max_speakers: 2, ..Default::default() });
        t.assign(&voice(0, 0), 3.0);
        t.assign(&voice(5, 0), 3.0);
        let a = t.assign(&voice(10, 0), 3.0);
        assert!(!a.is_new && a.speaker <= 2);
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
