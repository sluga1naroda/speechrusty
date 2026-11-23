//! Spike: llama.cpp on Vulkan. usage: mt <model.gguf> <target-lang> <text>... [--cpu]
use std::num::NonZeroU32;
use std::time::Instant;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let texts: Vec<&String> = args[3..].iter().filter(|a| !a.starts_with("--")).collect();
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    for d in llama_cpp_2::list_llama_ggml_backend_devices() {
        println!("dev {} [{}] {}  mem {:.2} GB", d.name, d.backend, d.description, d.memory_total as f64 / 1e9);
    }
    let t = Instant::now();
    let mp = LlamaModelParams::default().with_n_gpu_layers(if cpu { 0 } else { 999 });
    let model = LlamaModel::load_from_file(&backend, &args[1], &mp)?;
    let mut lctx = model.new_context(&backend, LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048)))?;
    println!("llm loaded in {:?} (gpu={})", t.elapsed(), !cpu);
    for text in texts {
        let t = Instant::now();
        let out = translate(&model, &mut lctx, text, &args[2])?;
        println!("MT {:?}: {text}  =>  {out}", t.elapsed());
    }
    Ok(())
}

fn translate(model: &LlamaModel, ctx: &mut llama_cpp_2::context::LlamaContext, text: &str, target: &str) -> anyhow::Result<String> {
    let tmpl = model.chat_template(None)?;
    let msgs = [
        LlamaChatMessage::new("system".into(), format!("You are a professional interpreter. Translate the user's message into {target}. Output only the translation, nothing else."))?,
        LlamaChatMessage::new("user".into(), text.into())?,
    ];
    let prompt = model.apply_chat_template(&tmpl, &msgs, true)?;
    let toks = model.vocab().tokenize(prompt.as_bytes(), false, true);
    ctx.clear_kv_cache();
    let mut batch = LlamaBatch::new(2048, 1);
    let last = toks.len() as i32 - 1;
    for (i, t) in toks.iter().enumerate() {
        batch.add(*t, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch)?;
    let mut sampler = LlamaSampler::greedy();
    let mut n = toks.len() as i32;
    let mut out = Vec::new();
    let vocab = model.vocab();
    for _ in 0..256 {
        let tok = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(tok);
        if vocab.is_eog(tok) {
            break;
        }
        out.extend(vocab.token_to_piece(tok, false, None));
        batch.clear();
        batch.add(tok, n, &[0], true)?;
        n += 1;
        ctx.decode(&mut batch)?;
    }
    Ok(String::from_utf8_lossy(&out).trim().to_string())
}
