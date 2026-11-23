//! cargo run -p sr-diarize --release --example voices -- <model.onnx> <wav16k>...
use std::time::Instant;
use sr_diarize::{SpeakerEmbedder, SpeakerTracker, TrackerConfig, cosine};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    let e = SpeakerEmbedder::load(args[1].as_ref())?;
    println!("load {:?}", t.elapsed());
    let mut embs = Vec::new();
    let mut tracker = SpeakerTracker::new(TrackerConfig::default());
    for f in &args[2..] {
        let a: Vec<f32> = hound::WavReader::open(f)?.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
        let secs = a.len() as f32 / 16000.0;
        let t = Instant::now();
        match e.embed(&a) {
            Ok(v) => {
                let asg = tracker.assign(&v, secs);
                println!("{:>12} {secs:4.1}s embed {:>4}ms -> speaker {} (sim {:.2}{})", name(f), t.elapsed().as_millis(), asg.speaker, asg.similarity, if asg.is_new { ", new" } else { "" });
                embs.push((name(f), v));
            }
            Err(err) => println!("{:>12} {secs:4.1}s: {err}", name(f)),
        }
    }
    print!("{:>12}", "");
    for (n, _) in &embs { print!("{:>9}", &n[..n.len().min(8)]); }
    println!();
    for (n, a) in &embs {
        print!("{n:>12}");
        for (_, b) in &embs { print!("{:>9.2}", cosine(a, b)); }
        println!();
    }
    Ok(())
}
fn name(f: &str) -> String { std::path::Path::new(f).file_stem().unwrap().to_string_lossy().into() }
