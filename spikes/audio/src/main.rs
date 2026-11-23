//! Stage 0 spike: capture system audio (loopback) and microphone via cpal.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn capture(dev: &cpal::Device, label: &str, secs: u64) -> anyhow::Result<()> {
    let cfg = if label == "loopback" { dev.default_output_config()? } else { dev.default_input_config()? };
    let name = dev.description().map(|d| d.to_string()).unwrap_or_default();
    println!("[{label}] device='{name}' fmt={:?} rate={:?} ch={}", cfg.sample_format(), cfg.sample_rate(), cfg.channels());
    let ch = cfg.channels() as usize;
    let buf: Arc<Mutex<Vec<f32>>> = Arc::default();
    let b2 = buf.clone();
    let stream = dev.build_input_stream::<f32, _, _>(
        cfg.config(),
        move |data, _| {
            // downmix to mono
            let mut b = b2.lock().unwrap();
            b.extend(data.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        },
        |e| eprintln!("stream error: {e}"),
        None,
    )?;
    stream.play()?;
    let start = Instant::now();
    let mut last = 0usize;
    while start.elapsed() < Duration::from_secs(secs) {
        std::thread::sleep(Duration::from_millis(500));
        let b = buf.lock().unwrap();
        let chunk = &b[last..];
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32).sqrt();
        println!("[{label}] t={:.1}s samples={} rms={:.4}", start.elapsed().as_secs_f32(), chunk.len(), rms);
        last = b.len();
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();
    let out = host.default_output_device().expect("no output device");
    capture(&out, "loopback", 6)?;
    if let Some(mic) = host.default_input_device() {
        if let Err(e) = capture(&mic, "mic", 2) {
            println!("[mic] error: {e}");
        }
    } else {
        println!("[mic] no input device");
    }
    Ok(())
}
