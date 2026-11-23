//! Settings (settings.toml) and app data paths.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use sr_core::{Lang, UiLang};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemePref {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowGeom {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub ui_lang: UiLang,
    pub theme: ThemePref,
    /// Window background opacity, 0..1.
    pub opacity: f32,
    pub show_original: bool,
    /// Distinguish participants in system audio (Speaker-1, Speaker-2…).
    pub diarization: bool,
    /// Last selected target language for the conversation. `None` — same as the UI language.
    pub target_lang: Option<Lang>,
    /// `None` — detect automatically.
    pub source_lang: Option<Lang>,
    pub mic_enabled: bool,
    pub asr_model: String,
    pub mt_model: String,
    pub global_hotkey_enabled: bool,
    pub window: Option<WindowGeom>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            ui_lang: UiLang::En,
            theme: ThemePref::System,
            opacity: 0.5,
            show_original: true,
            diarization: true,
            target_lang: None,
            source_lang: None,
            mic_enabled: false,
            asr_model: "whisper-large-v3-turbo".into(),
            mt_model: "llm-4b".into(),
            global_hotkey_enabled: true,
            window: None,
        }
    }
}

impl Settings {
    pub fn target_or_ui(&self) -> Lang {
        self.target_lang.unwrap_or(self.ui_lang.as_lang())
    }

    /// Clamps values to valid ranges (in case the file was edited by hand).
    pub fn sanitized(mut self) -> Self {
        if !self.opacity.is_finite() {
            self.opacity = 0.5;
        }
        self.opacity = self.opacity.clamp(0.0, 1.0);
        if let Some(g) = self.window
            && !(g.w.is_finite() && g.h.is_finite() && g.w >= 200.0 && g.h >= 100.0) {
                self.window = None;
            }
        self
    }
}

/// App directories.
#[derive(Clone, Debug)]
pub struct Paths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl Paths {
    pub fn system() -> anyhow::Result<Self> {
        let d = directories::ProjectDirs::from("", "", "SpeechRusty").context("no home directory")?;
        Ok(Self { config_dir: d.config_dir().to_path_buf(), data_dir: d.data_dir().to_path_buf() })
    }

    pub fn settings_file(&self) -> PathBuf {
        self.config_dir.join("settings.toml")
    }

    pub fn conversations_dir(&self) -> PathBuf {
        self.data_dir.join("conversations")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
}

/// Loads settings. No file → `None` (first launch). Corrupt file → defaults + log.
pub fn load_settings(path: &Path) -> Option<Settings> {
    let text = std::fs::read_to_string(path).ok()?;
    match toml::from_str::<Settings>(&text) {
        Ok(s) => Some(s.sanitized()),
        Err(e) => {
            log::warn!("settings file {path:?} is invalid, using defaults: {e}");
            Some(Settings::default())
        }
    }
}

/// Atomic write: to a temp file, then rename.
pub fn save_settings(path: &Path, s: &Settings) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, toml::to_string_pretty(s)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Conversation files, newest first.
pub fn list_conversations(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .collect();
    v.sort();
    v.reverse();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sr-store-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn roundtrip() {
        let d = tmpdir("rt");
        let f = d.join("settings.toml");
        assert_eq!(load_settings(&f), None);
        let s = Settings {
            ui_lang: UiLang::Ru,
            opacity: 0.3,
            target_lang: Some(Lang::De),
            window: Some(WindowGeom { x: 10.0, y: 20.0, w: 600.0, h: 300.0 }),
            ..Default::default()
        };
        save_settings(&f, &s).unwrap();
        assert_eq!(load_settings(&f), Some(s));
    }

    #[test]
    fn partial_and_invalid_files() {
        let d = tmpdir("partial");
        let f = d.join("settings.toml");
        std::fs::write(&f, "ui_lang = \"de\"\nopacity = 7.0\n").unwrap();
        let s = load_settings(&f).unwrap();
        assert_eq!(s.ui_lang, UiLang::De);
        assert_eq!(s.opacity, 1.0);
        assert!(s.show_original);

        std::fs::write(&f, "this is not toml = = =").unwrap();
        assert_eq!(load_settings(&f), Some(Settings::default()));
    }
}
