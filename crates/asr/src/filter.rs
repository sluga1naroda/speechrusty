//! Decides "recognized / not recognized" and filters out typical Whisper hallucinations.
//!
//! Whisper was trained on subtitles and, on silence/noise, likes to output subtitle credit
//! phrases ("To be continued…", "Thanks for watching!") or looping repetitions.

use sr_core::Lang;

use crate::{Outcome, Transcript};

/// Confidence threshold: below it, the text is treated as garbage.
const MIN_AVG_LOGPROB: f32 = -1.2;

/// Hallucination phrases (matched as substrings, ignoring case and punctuation).
const HALLUCINATIONS: &[&str] = &[
    // ru
    "продолжение следует",
    "субтитры сделал",
    "субтитры создавал",
    "редактор субтитров",
    "корректор",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "dimatorzok",
    // en
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "subtitles by",
    "amara org",
    // de
    "untertitel im auftrag des zdf",
    "untertitel von",
    "vielen dank fürs zuschauen",
    "untertitelung des zdf",
    // fr
    "sous titres réalisés par",
    "sous titres fait par",
    "merci d avoir regardé",
    // es
    "subtítulos realizados por",
    "subtitulado por",
    "gracias por ver",
];

/// Normalization: lowercase, punctuation → space, collapsed whitespace.
fn normalize(s: &str) -> String {
    let lowered: String = s.chars().map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap_or(c) } else { ' ' }).collect();
    lowered.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn is_hallucination(text: &str) -> bool {
    let n = normalize(text);
    if n.is_empty() {
        return true;
    }
    if HALLUCINATIONS.iter().any(|h| n.contains(&normalize(h))) {
        return true;
    }
    has_runaway_repeat(&n)
}

/// Looping: the same word or word pair repeated many times in a row.
fn has_runaway_repeat(n: &str) -> bool {
    let words: Vec<&str> = n.split(' ').collect();
    for span in 1..=3usize {
        if words.len() < span * 5 {
            continue;
        }
        let mut repeats = 1;
        let mut i = span;
        while i + span <= words.len() {
            if words[i..i + span] == words[i - span..i] {
                repeats += 1;
                if repeats >= 5 {
                    return true;
                }
            } else {
                repeats = 1;
            }
            i += span;
        }
    }
    false
}

/// Splits text into sentences, keeping the terminal punctuation.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '…') && !matches!(chars.peek(), Some('.' | '!' | '?' | '…')) {
            let t = cur.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            cur.clear();
        }
    }
    let t = cur.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

/// Removes repetitions that Whisper (especially large-v3/turbo) appends on short
/// phrases: "X. X." → "X.", and drops a truncated tail that starts a repeat ("X? X? X").
pub fn collapse_repeats(text: &str) -> String {
    let sents = sentences(text);
    let mut kept: Vec<String> = Vec::new();
    for (i, s) in sents.iter().enumerate() {
        let ns = normalize(s);
        if let Some(prev) = kept.last() {
            let np = normalize(prev);
            if ns == np {
                continue;
            }
            // last piece without terminal punctuation that is a prefix of the previous sentence
            let is_last = i + 1 == sents.len();
            let unterminated = !s.ends_with(['.', '!', '?', '…']);
            if is_last && unterminated && !ns.is_empty() && np.starts_with(&ns) {
                continue;
            }
        }
        kept.push(s.clone());
    }
    kept.join(" ")
}

pub fn judge(mut tr: Transcript, _lang: Lang) -> Outcome {
    tr.text = collapse_repeats(&tr.text);
    if tr.text.is_empty() || is_hallucination(&tr.text) {
        return Outcome::NotRecognized;
    }
    // Classic whisper rule: "no speech" + low confidence.
    if tr.no_speech > 0.6 && tr.avg_logprob < -0.8 {
        return Outcome::NotRecognized;
    }
    if tr.avg_logprob < MIN_AVG_LOGPROB {
        return Outcome::NotRecognized;
    }
    Outcome::Text(tr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(text: &str, lp: f32, ns: f32) -> Transcript {
        Transcript { text: text.into(), avg_logprob: lp, no_speech: ns }
    }

    #[test]
    fn hallucinations() {
        assert!(is_hallucination("Продолжение следует..."));
        assert!(is_hallucination("  Thanks for watching!  "));
        assert!(is_hallucination("Untertitel im Auftrag des ZDF, 2021"));
        assert!(is_hallucination("Sous-titres réalisés par la communauté d'Amara.org"));
        assert!(is_hallucination("..."));
        assert!(is_hallucination("да да да да да да"));
        assert!(is_hallucination("I'm going I'm going I'm going I'm going I'm going I'm going"));
        assert!(!is_hallucination("Да, давайте продолжим обсуждение бюджета."));
        assert!(!is_hallucination("Thanks, everyone, for joining the call."));
        assert!(!is_hallucination("Ja, ja, das stimmt."));
    }

    #[test]
    fn repeats_are_collapsed() {
        assert_eq!(collapse_repeats("Hello everyone. Hello everyone."), "Hello everyone.");
        assert_eq!(
            collapse_repeats("Can you hear me well? Can you hear me well? Can you hear me well? Can you hear me"),
            "Can you hear me well?"
        );
        assert_eq!(collapse_repeats("Добрый день, коллеги! Добрый день, коллеги!"), "Добрый день, коллеги!");
        // different sentences and ellipses are left alone
        assert_eq!(collapse_repeats("Ja. Ja, das stimmt... Gut."), "Ja. Ja, das stimmt... Gut.");
        assert_eq!(collapse_repeats("Hello. How are"), "Hello. How are");
        assert_eq!(collapse_repeats("no punctuation at all"), "no punctuation at all");
    }

    #[test]
    fn judging() {
        assert!(matches!(judge(tr("Hello there.", -0.3, 0.1), Lang::En), Outcome::Text(_)));
        assert_eq!(judge(tr("", -0.3, 0.1), Lang::En), Outcome::NotRecognized);
        assert_eq!(judge(tr("Hmm.", -0.9, 0.8), Lang::En), Outcome::NotRecognized);
        assert_eq!(judge(tr("blah", -1.6, 0.1), Lang::En), Outcome::NotRecognized);
        // high no_speech but confident text: keep it (quiet but clear speech)
        assert!(matches!(judge(tr("Okay.", -0.2, 0.7), Lang::En), Outcome::Text(_)));
    }
}
