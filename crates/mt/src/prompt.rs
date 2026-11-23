//! Prompt for the LLM translator and response cleanup.

use sr_core::Lang;

use crate::protocol::ChatMessage;

/// A previous translated turn of the conversation, used as context (pronouns, terms, gender).
#[derive(Clone, Debug, PartialEq)]
pub struct ContextTurn {
    pub lang: Lang,
    pub original: String,
    pub translation: String,
}

fn english_name(l: Lang) -> &'static str {
    match l {
        Lang::En => "English",
        Lang::Ru => "Russian",
        Lang::De => "German",
        Lang::Es => "Spanish",
        Lang::Fr => "French",
    }
}

/// Dialogue for the model: instruction, then past turns as "original → translation" pairs
/// (the model sees context and style, but translates only the last message).
/// If the context is in a different language than the current phrase, the pair still helps with terminology.
pub fn build_prompt(src: Lang, dst: Lang, text: &str, context: &[ContextTurn]) -> Vec<ChatMessage> {
    let (s, d) = (english_name(src), english_name(dst));
    let system = format!(
        "You are a professional simultaneous interpreter. Translate each user message from {s} to {d}. \
         Messages are transcripts of live speech, so they may be informal or contain recognition errors — convey the intended meaning naturally. \
         Keep names, numbers and terms, and keep questions as questions. Reply with the {d} translation of the latest message only: no quotes, notes, explanations or the original text."
    );
    let mut msgs = vec![ChatMessage::new("system", system)];
    for t in context {
        msgs.push(ChatMessage::new("user", t.original.clone()));
        msgs.push(ChatMessage::new("assistant", t.translation.clone()));
    }
    msgs.push(ChatMessage::new("user", text));
    msgs
}

/// Response length limit: a translation is rarely more than 2–3 times longer than the original.
pub fn max_tokens_for(text: &str) -> u32 {
    (text.chars().count() as u32).clamp(16, 400) + 32
}

const PREFIXES: &[&str] = &[
    "translation:",
    "here is the translation:",
    "here's the translation:",
    "перевод:",
    "übersetzung:",
    "traduction :",
    "traduction:",
    "traducción:",
];

/// Strips what the LLM adds around the translation: quotes, "Translation:", explanations after a blank line.
pub fn clean_output(raw: &str) -> String {
    let mut s = raw.trim();
    // everything after the first blank line is usually explanation
    if let Some(i) = s.find("\n\n") {
        s = s[..i].trim();
    }
    loop {
        let lower = s.to_lowercase();
        match PREFIXES.iter().find(|p| lower.starts_with(*p)) {
            Some(p) => s = s[p.len()..].trim_start(),
            None => break,
        }
    }
    let pairs = [('"', '"'), ('«', '»'), ('„', '“'), ('“', '”'), ('\'', '\'')];
    for (open, close) in pairs {
        if s.len() >= 2 && s.starts_with(open) && s.ends_with(close) && s[open.len_utf8()..s.len() - close.len_utf8()].find([open, close]).is_none() {
            s = &s[open.len_utf8()..s.len() - close.len_utf8()];
            break;
        }
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_has_context_as_turns() {
        let ctx = [ContextTurn { lang: Lang::En, original: "The budget is ready.".into(), translation: "Бюджет готов.".into() }];
        let m = build_prompt(Lang::De, Lang::Ru, "Ja, gut.", &ctx);
        let roles: Vec<&str> = m.iter().map(|x| x.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert!(m[0].content.contains("from German to Russian"));
        assert_eq!(m[2].content, "Бюджет готов.");
        assert_eq!(m[3].content, "Ja, gut.");
        assert_eq!(build_prompt(Lang::En, Lang::Fr, "x", &[]).len(), 2);
    }

    #[test]
    fn cleaning() {
        assert_eq!(clean_output("  Привет всем!  "), "Привет всем!");
        assert_eq!(clean_output("Перевод: Привет всем!"), "Привет всем!");
        assert_eq!(clean_output("Translation: \"Hello, all.\""), "Hello, all.");
        assert_eq!(clean_output("«Добрый день»"), "Добрый день");
        assert_eq!(clean_output("Guten Tag.\n\nNote: formal greeting."), "Guten Tag.");
        assert_eq!(clean_output("Line one\nline two"), "Line one line two");
        // quotes inside: don't strip the outer ones
        assert_eq!(clean_output("\"A\" and \"B\""), "\"A\" and \"B\"");
        assert_eq!(clean_output(""), "");
    }
}
