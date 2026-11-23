//! Shared SpeechRusty types: languages, speakers, utterances and pipeline events.

use serde::{Deserialize, Serialize};

/// Speech and translation language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    En,
    Ru,
    De,
    Es,
    Fr,
}

impl Lang {
    pub const ALL: [Lang; 5] = [Lang::En, Lang::Ru, Lang::De, Lang::Es, Lang::Fr];

    /// ISO 639-1 code (also what Whisper understands).
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ru => "ru",
            Lang::De => "de",
            Lang::Es => "es",
            Lang::Fr => "fr",
        }
    }

    pub fn from_code(code: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.code() == code)
    }

    /// The language's native name — the same in any interface language.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Ru => "Русский",
            Lang::De => "Deutsch",
            Lang::Es => "Español",
            Lang::Fr => "Français",
        }
    }
}

/// Interface language.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UiLang {
    #[default]
    En,
    De,
    Fr,
    Es,
    Ru,
}

impl UiLang {
    pub const ALL: [UiLang; 5] = [UiLang::En, UiLang::De, UiLang::Fr, UiLang::Es, UiLang::Ru];

    pub fn native_name(self) -> &'static str {
        self.as_lang().native_name()
    }

    /// The corresponding speech language (all 5 UI languages have one).
    pub fn as_lang(self) -> Lang {
        match self {
            UiLang::En => Lang::En,
            UiLang::De => Lang::De,
            UiLang::Fr => Lang::Fr,
            UiLang::Es => Lang::Es,
            UiLang::Ru => Lang::Ru,
        }
    }

    /// UI language from the system locale code (`ru-RU`, `de_DE` …), otherwise English.
    pub fn from_locale(locale: &str) -> UiLang {
        let p = locale.get(..2).unwrap_or("").to_ascii_lowercase();
        match p.as_str() {
            "ru" => UiLang::Ru,
            "de" => UiLang::De,
            "fr" => UiLang::Fr,
            "es" => UiLang::Es,
            _ => UiLang::En,
        }
    }
}

/// Who is speaking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Speaker {
    /// Utterance from the user's microphone.
    Me,
    /// Voice from system audio, numbered from 1.
    Remote(u8),
}

/// Utterance state.
#[derive(Clone, Debug, PartialEq)]
pub enum UtteranceStatus {
    /// Interim text, will still be refined.
    Interim,
    /// Final text with translation (or without, if the language matched the target).
    Final,
    /// There was speech, but it wasn't recognized.
    NotRecognized,
    /// Recognized, but translation failed.
    TranslationFailed(String),
}

/// Utterance. The pipeline sends it several times with the same `id` — the UI upserts it.
#[derive(Clone, Debug, PartialEq)]
pub struct Utterance {
    pub id: u64,
    pub speaker: Speaker,
    /// Start time relative to the start of recording, ms.
    pub start_ms: u64,
    pub src_lang: Option<Lang>,
    pub dst_lang: Lang,
    pub original: String,
    pub translation: Option<String>,
    pub status: UtteranceStatus,
}

/// Where the audio came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AudioSource {
    /// System audio (other participants).
    System,
    /// The user's microphone.
    Mic,
}

/// Settings the pipeline is started with.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    pub target: Lang,
    /// `None` — detect automatically.
    pub source: Option<Lang>,
    pub mic_enabled: bool,
    pub asr_model: String,
    pub mt_model: String,
}

/// Events from the pipeline to the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineEvent {
    Started,
    Stopped,
    Utterance(Utterance),
    /// Signal level 0..1 for the indicator, and whether speech is ongoing right now.
    Level { source: AudioSource, level: f32, speech: bool },
    /// Remove an utterance (the phrase turned out to be noise, but interim text was already shown).
    Retract(u64),
    /// Diarization determined the utterance's speaker. Applied on top of any utterance updates.
    SpeakerAssigned { id: u64, speaker: Speaker },
    /// The recognition model is loading (true) / ready (false).
    AsrLoading(bool),
    /// The translation model is loading (true) / ready (false).
    MtLoading(bool),
    /// Non-fatal error or warning for the UI banner.
    Notice(String),
    /// Fatal error: the pipeline has stopped.
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_codes_roundtrip() {
        for l in Lang::ALL {
            assert_eq!(Lang::from_code(l.code()), Some(l));
        }
        assert_eq!(Lang::from_code("ja"), None);
    }

    #[test]
    fn ui_lang_from_locale() {
        assert_eq!(UiLang::from_locale("ru-RU"), UiLang::Ru);
        assert_eq!(UiLang::from_locale("de_DE"), UiLang::De);
        assert_eq!(UiLang::from_locale("ja-JP"), UiLang::En);
        assert_eq!(UiLang::from_locale(""), UiLang::En);
    }
}
