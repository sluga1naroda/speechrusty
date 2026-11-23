//! Segmentation of real speech (TTS phrases) with the earshot detector.
use sr_audio::{SegEvent, Segmenter, SegmenterConfig, default_vad};

fn read(name: &str) -> Vec<f32> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    hound::WavReader::open(p).unwrap().into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

/// Deterministic "room noise" at ~ −40 dB.
fn noise(n: usize) -> Vec<f32> {
    let mut x: u32 = 12345;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02
        })
        .collect()
}

#[test]
fn two_tts_phrases_in_noise() {
    let en = read("speech_en.wav");
    let ru = read("speech_ru.wav");
    let gap = 16_000 * 2;
    let mut audio = noise(16_000);
    audio.extend(&en);
    audio.extend(noise(gap));
    audio.extend(&ru);
    audio.extend(noise(16_000));
    // add background noise, also on top of speech
    let n = noise(audio.len());
    for (a, b) in audio.iter_mut().zip(n) {
        *a += b;
    }

    let mut seg = Segmenter::new(default_vad(), SegmenterConfig::default());
    let mut ev = Vec::new();
    for c in audio.chunks(480) {
        seg.push(c, &mut ev);
    }
    seg.flush(&mut ev);

    let ends: Vec<(u64, u64)> = ev.iter().filter_map(|e| match e { SegEvent::End { start_ms, end_ms, .. } => Some((*start_ms, *end_ms)), _ => None }).collect();
    eprintln!("segments: {ends:?}");
    let en_end = 1000 + en.len() as u64 / 16;
    let ru_start = en_end + 2000;
    // Speech must not be lost: every phrase is covered by segments, and no segment lies entirely within a pause.
    assert!(ends.first().unwrap().0 <= 1100, "start of EN missed: {ends:?}");
    assert!(ends.iter().any(|&(s, e)| s <= ru_start + 100 && e > ru_start + 500), "RU not covered: {ends:?}");
    assert!(ends.iter().all(|&(s, e)| !(s > en_end + 300 && e < ru_start - 300)), "segment inside the pause: {ends:?}");
    assert!(ends.iter().all(|&(s, e)| s < e && e - s < 12_100));
}
