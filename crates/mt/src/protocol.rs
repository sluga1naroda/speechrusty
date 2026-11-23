//! JSON lines over stdin/stdout between the app and `speechrusty-mt`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Chat message: role = "system" | "user" | "assistant".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Load { model: PathBuf, gpu: bool, ctx: u32 },
    Translate { id: u64, messages: Vec<ChatMessage>, max_tokens: u32 },
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Loaded { ms: u64, devices: Vec<String> },
    Result { id: u64, text: String, ms: u64 },
    Error { id: Option<u64>, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let r = Request::Translate { id: 7, messages: vec![ChatMessage::new("system", "s"), ChatMessage::new("user", "привет\n\"x\"")], max_tokens: 64 };
        let line = serde_json::to_string(&r).unwrap();
        assert!(!line.contains('\n'), "one request = one line");
        assert_eq!(serde_json::from_str::<Request>(&line).unwrap(), r);
        let e: Response = serde_json::from_str(r#"{"type":"error","id":null,"message":"boom"}"#).unwrap();
        assert_eq!(e, Response::Error { id: None, message: "boom".into() });
    }
}
