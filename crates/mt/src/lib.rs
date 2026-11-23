//! Translation: protocol with the `speechrusty-mt` process (llama.cpp), prompts and client.

pub mod client;
pub mod prompt;
pub mod protocol;

pub use protocol::ChatMessage;

pub use client::LlmTranslator;
pub use prompt::{ContextTurn, build_prompt, clean_output, max_tokens_for};
pub use client::locate_sidecar;

/// GGUF model file for an id from the settings.
pub fn model_file(id: &str) -> Option<&'static str> {
    Some(match id {
        "llm-fast" => "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "llm-4b" => "google_gemma-3-4b-it-Q4_K_M.gguf",
        "llm-large" => "google_gemma-3-12b-it-Q4_K_M.gguf",
        _ => return None,
    })
}
