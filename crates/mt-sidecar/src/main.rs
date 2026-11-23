//! `speechrusty-mt`: translation on llama.cpp. Requests are JSON lines on stdin, responses on stdout.
//! llama.cpp logs are disabled: stdout must contain only the protocol.

use std::io::{BufRead, Write};
use std::num::NonZeroU32;
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use sr_mt::protocol::{Request, Response};

struct Loaded {
    // The context borrows the model; the model is leaked for the lifetime of the process (one model per process).
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    n_ctx: u32,
    /// Tokens with CJK ideographs/kana/hangul/emoji: none of our languages need them,
    /// and small multilingual models (Qwen) like to insert Chinese words.
    banned: Vec<LlamaLogitBias>,
}

/// Bytes 0xE3..=0xED are always lead bytes of characters U+3000..U+DFFF (CJK, kana, hangul);
/// 0xF0..=0xF4 are characters outside the BMP (emoji, CJK extensions). Cyrillic and Latin
/// don't need them. Plus fullwidth forms U+FF00..U+FFEF ("？", "，").
fn is_banned_piece(bytes: &[u8]) -> bool {
    bytes.iter().any(|b| (0xE3..=0xED).contains(b) || (0xF0..=0xF4).contains(b))
        || String::from_utf8_lossy(bytes).chars().any(|c| ('\u{FF00}'..='\u{FFEF}').contains(&c))
}

fn banned_tokens(model: &LlamaModel) -> Vec<LlamaLogitBias> {
    let vocab = model.vocab();
    (0..vocab.n_tokens())
        .filter(|&id| is_banned_piece(&vocab.token_to_piece(LlamaToken(id), false, None)))
        .map(|id| LlamaLogitBias::new(LlamaToken(id), f32::NEG_INFINITY))
        .collect()
}

fn reply(r: &Response) {
    let mut out = std::io::stdout().lock();
    let _ = serde_json::to_writer(&mut out, r);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn main() -> anyhow::Result<()> {
    let backend: &'static mut LlamaBackend = Box::leak(Box::new(LlamaBackend::init()?));
    backend.void_logs();
    let backend: &'static LlamaBackend = backend;
    let mut loaded: Option<Loaded> = None;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                reply(&Response::Error { id: None, message: format!("bad request: {e}") });
                continue;
            }
        };
        match req {
            Request::Shutdown => break,
            Request::Load { model, gpu, ctx } => {
                if loaded.is_some() {
                    reply(&Response::Error { id: None, message: "model already loaded; restart the process to switch".into() });
                    continue;
                }
                let t = Instant::now();
                let params = LlamaModelParams::default().with_n_gpu_layers(if gpu { 999 } else { 0 });
                match LlamaModel::load_from_file(backend, &model, &params) {
                    Ok(m) => {
                        let m: &'static LlamaModel = Box::leak(Box::new(m));
                        let cp = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(ctx)).with_n_batch(ctx);
                        match m.new_context(backend, cp) {
                            Ok(c) => {
                                let devices = llama_cpp_2::list_llama_ggml_backend_devices()
                                    .into_iter()
                                    .map(|d| format!("{} ({})", d.description.trim(), d.backend))
                                    .collect();
                                let banned = banned_tokens(m);
                                eprintln!("speechrusty-mt: {} tokens banned (CJK/emoji)", banned.len());
                                let mut l = Loaded { model: m, ctx: c, n_ctx: ctx, banned };
                                // Warm-up: the first run on the GPU is several times slower (shaders, buffers).
                                let warm = [sr_mt::ChatMessage::new("user", "Say OK.")];
                                if let Err(e) = generate(&mut l, &warm, 4) {
                                    eprintln!("speechrusty-mt: warmup failed: {e:#}");
                                }
                                loaded = Some(l);
                                reply(&Response::Loaded { ms: t.elapsed().as_millis() as u64, devices });
                            }
                            Err(e) => reply(&Response::Error { id: None, message: format!("context: {e}") }),
                        }
                    }
                    Err(e) => reply(&Response::Error { id: None, message: format!("load {}: {e}", model.display()) }),
                }
            }
            Request::Translate { id, messages, max_tokens } => {
                let Some(l) = loaded.as_mut() else {
                    reply(&Response::Error { id: Some(id), message: "no model loaded".into() });
                    continue;
                };
                let t = Instant::now();
                match generate(l, &messages, max_tokens) {
                    Ok(text) => reply(&Response::Result { id, text, ms: t.elapsed().as_millis() as u64 }),
                    Err(e) => reply(&Response::Error { id: Some(id), message: format!("{e:#}") }),
                }
            }
        }
    }
    Ok(())
}

fn generate(l: &mut Loaded, messages: &[sr_mt::ChatMessage], max_tokens: u32) -> anyhow::Result<String> {
    let tmpl = l.model.chat_template(None)?;
    let msgs = messages.iter().map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone())).collect::<Result<Vec<_>, _>>()?;
    let prompt = l.model.apply_chat_template(&tmpl, &msgs, true)?;
    let vocab = l.model.vocab();
    // Templates like Gemma's don't insert <bos> themselves; the tokenizer adds it.
    // Without it, Gemma 3 ends the response immediately or outputs garbage.
    let bos_text = vocab.token_to_piece(vocab.bos(), true, None);
    let add_bos = vocab.should_add_bos() && !prompt.as_bytes().starts_with(&bos_text);
    let tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
    if std::env::var_os("SR_MT_DEBUG").is_some() {
        eprintln!("PROMPT {prompt:?} ({} tokens)", tokens.len());
    }
    anyhow::ensure!((tokens.len() as u32) + max_tokens < l.n_ctx, "prompt too long ({} tokens)", tokens.len());

    l.ctx.clear_kv_cache();
    let mut batch = LlamaBatch::new(l.n_ctx as usize, 1);
    let last = tokens.len() as i32 - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i as i32 == last)?;
    }
    l.ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::logit_bias(vocab.n_tokens(), &l.banned), LlamaSampler::greedy()]);
    let mut out: Vec<u8> = Vec::new();
    for pos in (tokens.len() as i32..).take(max_tokens as usize) {
        let tok = sampler.sample(&l.ctx, batch.n_tokens() - 1);
        sampler.accept(tok);
        if vocab.is_eog(tok) {
            break;
        }
        if std::env::var_os("SR_MT_DEBUG").is_some() {
            eprintln!("tok {} {:?}", tok.0, String::from_utf8_lossy(&vocab.token_to_piece(tok, true, None)));
        }
        out.extend(vocab.token_to_piece(tok, false, None));
        batch.clear();
        batch.add(tok, pos, &[0], true)?;
        l.ctx.decode(&mut batch)?;
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

#[cfg(test)]
mod tests {
    use super::is_banned_piece;

    #[test]
    fn banned_pieces() {
        for ok in ["Привет", " hello", "Grüße", "¿Qué?", "«ça»", "…", "€", "—"] {
            assert!(!is_banned_piece(ok.as_bytes()), "{ok}");
        }
        for bad in ["季度", "周四", "下午", "カタ", "한국", "？", "😀"] {
            assert!(is_banned_piece(bad.as_bytes()), "{bad}");
        }
        // UTF-8 "fragment" of a Chinese character (byte-fallback token)
        assert!(is_banned_piece(&[0xE5]));
        assert!(!is_banned_piece(&[0xD0]));
    }
}
