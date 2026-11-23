//! Audio capture: system output (loopback) and microphone via cpal.
//! Each source lives on its own thread, reconnects by itself when the default
//! device changes or the stream errors, and emits 16 kHz mono.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::resample::{To16k, downmix};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// Everything playing on the system (calls, video…).
    Loopback,
    Microphone,
}

#[derive(Debug)]
pub enum CaptureMsg {
    /// 16 kHz mono.
    Audio(Vec<f32>),
    Opened { device: String, rate: u32, channels: u16 },
    /// Non-fatal problem; the thread keeps trying to reconnect.
    Problem(String),
}

/// Running capture. Stops on drop.
pub struct Capture {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Capture {
    pub fn spawn(kind: SourceKind, out: mpsc::Sender<CaptureMsg>) -> Capture {
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let thread = std::thread::Builder::new()
            .name(format!("sr-capture-{kind:?}"))
            .spawn(move || run(kind, out, s2))
            .expect("spawn capture thread");
        Capture { stop, thread: Some(thread) }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn default_device(host: &cpal::Host, kind: SourceKind) -> Option<cpal::Device> {
    match kind {
        SourceKind::Loopback => host.default_output_device(),
        SourceKind::Microphone => host.default_input_device(),
    }
}

fn device_key(d: &cpal::Device) -> String {
    d.id().map(|i| i.to_string()).unwrap_or_else(|_| d.description().map(|x| x.to_string()).unwrap_or_default())
}

struct Opened {
    _stream: cpal::Stream,
    rx: mpsc::Receiver<Vec<f32>>,
    failed: Arc<AtomicBool>,
    key: String,
    rate: u32,
}

fn open(host: &cpal::Host, kind: SourceKind, out: &mpsc::Sender<CaptureMsg>) -> anyhow::Result<Opened> {
    let dev = default_device(host, kind).ok_or_else(|| anyhow::anyhow!("no default {} device", if kind == SourceKind::Loopback { "output" } else { "input" }))?;
    // For loopback use the output device's format: WASAPI/CoreAudio record exactly that.
    let sc = match kind {
        SourceKind::Loopback => dev.default_output_config()?,
        SourceKind::Microphone => dev.default_input_config()?,
    };
    let ch = sc.channels() as usize;
    let rate = sc.sample_rate();
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(256);
    let failed = Arc::new(AtomicBool::new(false));
    let f2 = failed.clone();
    let on_err = move |e: cpal::Error| {
        log::warn!("audio stream error: {e}");
        // One-off over/underruns are harmless; reconnect only when the stream is lost.
        let s = e.to_string().to_lowercase();
        if !(s.contains("underrun") || s.contains("overrun")) {
            f2.store(true, Ordering::Relaxed);
        }
    };
    let config = sc.config();
    let stream = match sc.sample_format() {
        cpal::SampleFormat::I16 => dev.build_input_stream::<i16, _, _>(
            config,
            move |data, _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                let mut m = Vec::with_capacity(f.len() / ch.max(1));
                downmix(&f, ch, &mut m);
                let _ = tx.try_send(m);
            },
            on_err,
            None,
        )?,
        _ => dev.build_input_stream::<f32, _, _>(
            config,
            move |data, _| {
                let mut m = Vec::with_capacity(data.len() / ch.max(1));
                downmix(data, ch, &mut m);
                // If processing can't keep up, better to drop a chunk than block the audio thread.
                let _ = tx.try_send(m);
            },
            on_err,
            None,
        )?,
    };
    stream.play()?;
    let name = dev.description().map(|d| d.to_string()).unwrap_or_else(|_| "?".into());
    log::info!("{kind:?}: opened '{name}' {rate} Hz × {ch}");
    let _ = out.send(CaptureMsg::Opened { device: name, rate, channels: ch as u16 });
    Ok(Opened { _stream: stream, rx, failed, key: device_key(&dev), rate })
}

fn run(kind: SourceKind, out: mpsc::Sender<CaptureMsg>, stop: Arc<AtomicBool>) {
    let host = cpal::default_host();
    let mut last_problem = String::new();
    while !stop.load(Ordering::Relaxed) {
        let o = match open(&host, kind, &out) {
            Ok(o) => o,
            Err(e) => {
                let msg = e.to_string();
                if msg != last_problem {
                    log::warn!("{kind:?}: {msg}");
                    let _ = out.send(CaptureMsg::Problem(msg.clone()));
                    last_problem = msg;
                }
                sleep_unless(&stop, Duration::from_secs(2));
                continue;
            }
        };
        last_problem.clear();
        let mut rs = match To16k::new(o.rate) {
            Ok(r) => r,
            Err(e) => {
                let _ = out.send(CaptureMsg::Problem(format!("unsupported sample rate {}: {e}", o.rate)));
                sleep_unless(&stop, Duration::from_secs(2));
                continue;
            }
        };
        let mut last_check = Instant::now();
        let mut buf = Vec::new();
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            match o.rx.recv_timeout(Duration::from_millis(50)) {
                Ok(chunk) => {
                    buf.clear();
                    rs.process(&chunk, &mut buf);
                    if !buf.is_empty() && out.send(CaptureMsg::Audio(std::mem::take(&mut buf))).is_err() {
                        return;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if o.failed.load(Ordering::Relaxed) {
                log::info!("{kind:?}: stream failed, reopening");
                break;
            }
            if last_check.elapsed() > Duration::from_secs(2) {
                last_check = Instant::now();
                let now = default_device(&host, kind).map(|d| device_key(&d));
                if now.as_deref() != Some(o.key.as_str()) {
                    log::info!("{kind:?}: default device changed, reopening");
                    break;
                }
            }
        }
        sleep_unless(&stop, Duration::from_millis(300));
    }
}

fn sleep_unless(stop: &AtomicBool, d: Duration) {
    let end = Instant::now() + d;
    while Instant::now() < end && !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(25));
    }
}
