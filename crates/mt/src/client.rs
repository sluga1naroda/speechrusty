//! Client for the `speechrusty-mt` translator process.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail};

use crate::protocol::{ChatMessage, Request, Response};

pub const SIDECAR_NAME: &str = if cfg!(windows) { "speechrusty-mt.exe" } else { "speechrusty-mt" };

/// Looks for the binary next to the app (and one level up, for `examples/`). `SR_MT_SIDECAR` overrides it.
pub fn locate_sidecar() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SR_MT_SIDECAR") {
        return Some(PathBuf::from(p));
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    [dir.join(SIDECAR_NAME), dir.parent()?.join(SIDECAR_NAME)].into_iter().find(|p| p.exists())
}

pub struct LlmTranslator {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<Response>,
    next_id: u64,
    pub devices: Vec<String>,
}

impl LlmTranslator {
    /// Starts the process and loads the model (may take tens of seconds).
    pub fn start(sidecar: &Path, model: &Path, gpu: bool) -> anyhow::Result<Self> {
        anyhow::ensure!(model.exists(), "model file not found: {}", model.display());
        let mut cmd = Command::new(sidecar);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().with_context(|| format!("failed to start {}", sidecar.display()))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("sr-mt-reader".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    match serde_json::from_str::<Response>(&line) {
                        Ok(r) => {
                            if tx.send(r).is_err() {
                                break;
                            }
                        }
                        Err(e) => log::warn!("bad line from translator: {e}: {line}"),
                    }
                }
            })
            .expect("spawn reader");
        let mut me = Self { child, stdin, rx, next_id: 0, devices: Vec::new() };
        me.send(&Request::Load { model: model.to_path_buf(), gpu, ctx: 2048 })?;
        match me.recv(Duration::from_secs(180))? {
            Response::Loaded { ms, devices } => {
                log::info!("translator loaded {} in {ms} ms on {devices:?}", model.display());
                me.devices = devices;
                Ok(me)
            }
            Response::Error { message, .. } => bail!("translator failed to load model: {message}"),
            other => bail!("unexpected response: {other:?}"),
        }
    }

    fn send(&mut self, r: &Request) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(r)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).context("translator process is gone")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv(&mut self, timeout: Duration) -> anyhow::Result<Response> {
        match self.rx.recv_timeout(timeout) {
            Ok(r) => Ok(r),
            Err(mpsc::RecvTimeoutError::Timeout) => bail!("translator timed out"),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let status = self.child.try_wait().ok().flatten();
                bail!("translator process exited ({status:?})")
            }
        }
    }

    /// Translates; returns the raw model response (cleanup is `clean_output`).
    pub fn translate(&mut self, messages: &[ChatMessage], max_tokens: u32) -> anyhow::Result<String> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&Request::Translate { id, messages: messages.to_vec(), max_tokens })?;
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.recv(left)? {
                Response::Result { id: rid, text, .. } if rid == id => return Ok(text),
                Response::Error { id: Some(rid), message } if rid == id => bail!("{message}"),
                Response::Error { id: None, message } => bail!("{message}"),
                other => log::debug!("stale translator response: {other:?}"),
            }
        }
    }

    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for LlmTranslator {
    fn drop(&mut self) {
        let _ = self.send(&Request::Shutdown);
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
    }
}
