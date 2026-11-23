//! Main window: state, screens and handling of pipeline events.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderState};
use gpui_kit::component::theme::{Theme, ThemeMode};
use gpui_kit::component::{IconName, Sizable as _};
use gpui_kit::*;
use gpui_kit::prelude::FluentBuilder as _;

use sr_core::{AudioSource, Lang, PipelineEvent, Speaker, SessionConfig, UiLang, Utterance, UtteranceStatus};
use sr_i18n::{Key, direction, speaker_label, t};
use sr_pipeline::PipelineHandle;
use sr_store::{Paths, Settings, ThemePref, WindowGeom};

use crate::platform::{self, GlobalHotkey};
use crate::theme::Palette;
use crate::widgets::*;

actions!(speechrusty, [ToggleRecording, Quit]);

pub const KEY_CONTEXT: &str = "SpeechRusty";
const COMPACT_HEIGHT: f32 = 150.0;

pub struct ModelChoice {
    pub id: &'static str,
    pub name: &'static str,
    pub note: Key,
    pub recommended: bool,
}

pub const ASR_MODELS: &[ModelChoice] = &[
    ModelChoice { id: "whisper-large-v3-turbo", name: "Whisper large-v3-turbo", note: Key::AsrTurboNote, recommended: true },
    ModelChoice { id: "whisper-medium", name: "Whisper medium", note: Key::AsrMediumNote, recommended: false },
    ModelChoice { id: "whisper-small", name: "Whisper small", note: Key::AsrSmallNote, recommended: false },
];

pub const MT_MODELS: &[ModelChoice] = &[
    ModelChoice { id: "llm-fast", name: "Qwen2.5 1.5B", note: Key::MtFastNote, recommended: false },
    ModelChoice { id: "llm-4b", name: "Gemma 3 4B", note: Key::MtMidNote, recommended: true },
    ModelChoice { id: "llm-large", name: "Gemma 3 12B", note: Key::MtLargeNote, recommended: false },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Home,
    NewConversation,
    Session,
    History,
    Settings,
}

/// Ongoing conversation.
struct Session {
    title: String,
    cfg: SessionConfig,
    utterances: Vec<Utterance>,
    index: HashMap<u64, usize>,
    started: Instant,
    /// How much has already been recorded in previous segments (between pauses).
    recorded: Duration,
    rec_since: Option<Instant>,
    handle: Option<PipelineHandle>,
    next_id: u64,
    /// Recording run number: each run has its own range of utterance ids — finals from the previous
    /// run may still arrive after the next one has started.
    run: u64,
    /// The recognition model is loading.
    asr_loading: bool,
    /// The translation model is loading.
    mt_loading: bool,
    /// Translation is working in this run (otherwise only the original is shown).
    mt_on: bool,
    /// Level and speech flag: system audio, microphone.
    level_sys: (f32, bool),
    level_mic: (f32, bool),
    /// Warning banner and the moment it appeared (hidden after 8 s).
    notice: Option<(String, Instant)>,
    scroll: ScrollHandle,
    /// New utterances arrived while the user was reading history further up.
    unseen: bool,
    /// Event-receiving tasks of all runs (alive until the run's channel is closed).
    _events: Vec<Task<()>>,
    /// Speakers assigned by diarization (override the provisional speaker in utterances).
    speakers: HashMap<u64, Speaker>,
    /// Names given by the user.
    names: HashMap<Speaker, String>,
    /// Removed utterances (echo, noise): late updates do not resurrect them.
    retracted: std::collections::HashSet<u64>,
    /// Voices of the conversation's participants — persist across recording pauses.
    tracker: std::sync::Arc<std::sync::Mutex<sr_pipeline::SpeakerTracker>>,
}

impl Session {
    fn recording(&self) -> bool {
        self.rec_since.is_some()
    }

    fn elapsed(&self) -> Duration {
        self.recorded + self.rec_since.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn retract(&mut self, id: u64) {
        self.retracted.insert(id);
        if let Some(i) = self.index.remove(&id) {
            self.utterances.remove(i);
            self.index = self.utterances.iter().enumerate().map(|(i, u)| (u.id, i)).collect();
        }
    }

    fn assign_speaker(&mut self, id: u64, speaker: Speaker) {
        self.speakers.insert(id, speaker);
        if let Some(&i) = self.index.get(&id) {
            self.utterances[i].speaker = speaker;
        }
    }

    fn upsert(&mut self, mut u: Utterance) {
        if self.retracted.contains(&u.id) {
            return;
        }
        if let Some(&sp) = self.speakers.get(&u.id) {
            u.speaker = sp;
        }
        self.next_id = self.next_id.max(u.id);
        match self.index.get(&u.id) {
            // A late interim result (interim translation) must not overwrite the final.
            Some(&i) if u.status == UtteranceStatus::Interim && self.utterances[i].status != UtteranceStatus::Interim => {}
            Some(&i) => self.utterances[i] = u,
            None => {
                self.index.insert(u.id, self.utterances.len());
                self.utterances.push(u);
            }
        }
    }
}

pub struct AppView {
    settings: Settings,
    paths: Paths,
    screen: Screen,
    /// Where to return to from settings.
    back_to: Screen,
    focus: FocusHandle,
    opacity: Entity<SliderState>,
    title_input: Entity<InputState>,
    title_input_lang: UiLang,
    /// Speaker renaming: who, and the input field.
    renaming: Option<(Speaker, Entity<InputState>, Subscription)>,
    session: Option<Session>,
    compact: bool,
    full_size: Option<Size<Pixels>>,
    applied_dark: Option<bool>,
    history: Vec<PathBuf>,
    hotkey: Option<GlobalHotkey>,
    hotkey_error: Option<String>,
    /// Demo dialogue instead of real audio (`SR_DEMO=1`).
    demo: bool,
    live_opts: sr_pipeline::LiveOptions,
    save_gen: u64,
    _subs: Vec<Subscription>,
    _ticker: Task<()>,
}

impl AppView {
    pub fn new(settings: Settings, paths: Paths, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let opacity = cx.new(|_| SliderState::new().min(0.0).max(100.0).step(1.0).default_value(settings.opacity * 100.0));
        let title_input = Self::make_title_input(settings.ui_lang, "", window, cx);
        let focus = cx.focus_handle();
        window.focus(&focus, cx);

        let subs = vec![
            cx.subscribe(&opacity, |this, _, ev: &SliderEvent, cx| {
                let (SliderEvent::Change(v) | SliderEvent::Release(v)) = ev;
                this.settings.opacity = (v.end() / 100.0).clamp(0.0, 1.0);
                this.save_soon(cx);
                cx.notify();
            }),
            cx.observe_window_bounds(window, |this, window, cx| {
                if !this.compact {
                    // window_bounds() is in the same coordinates that WindowOptions accepts
                    // (bounds() of a frameless window is offset and caused drift on every launch).
                    let b = window.window_bounds().get_bounds();
                    this.settings.window = Some(WindowGeom {
                        x: f32::from(b.origin.x),
                        y: f32::from(b.origin.y),
                        w: f32::from(b.size.width),
                        h: f32::from(b.size.height),
                    });
                    this.save_soon(cx);
                }
            }),
            cx.observe_window_appearance(window, |_, _, cx| cx.notify()),
        ];

        let (hotkey, hotkey_error) = match GlobalHotkey::new() {
            Ok(mut h) => {
                let err = h.set_enabled(settings.global_hotkey_enabled).err().map(|e| e.to_string());
                (Some(h), err)
            }
            Err(e) => (None, Some(e.to_string())),
        };
        if let Some(e) = &hotkey_error {
            log::warn!("global hotkey unavailable: {e}");
        }

        // 10 times per second: poll the global hotkey and redraw the recording timer.
        let ticker = cx.spawn(async move |this, cx| {
            let mut n: u32 = 0;
            loop {
                cx.background_executor().timer(Duration::from_millis(100)).await;
                n = n.wrapping_add(1);
                let alive = this.update(cx, |this, cx| {
                    if this.hotkey.as_ref().is_some_and(|h| h.poll_pressed()) {
                        this.toggle_recording(cx);
                    }
                    if n.is_multiple_of(5) && this.session.as_ref().is_some_and(|s| s.recording()) {
                        cx.notify();
                    }
                });
                if alive.is_err() {
                    break;
                }
            }
        });

        let history = sr_store::list_conversations(&paths.conversations_dir());
        // SR_DUMP_SEGMENTS=1 — save every phrase as a WAV file (segmentation debugging).
        let live_opts = sr_pipeline::LiveOptions {
            asr: None, // asr and mt are set on every start according to the selected models
            mt: None,
            diarization: None,
            dump_dir: std::env::var_os("SR_DUMP_SEGMENTS").is_some_and(|v| v == "1").then(|| paths.data_dir.join("debug").join("segments")),
        };
        Self {
            settings,
            paths,
            screen: Screen::Home,
            back_to: Screen::Home,
            focus,
            opacity,
            title_input_lang: UiLang::En,
            renaming: None,
            title_input,
            session: None,
            compact: false,
            full_size: None,
            applied_dark: None,
            history,
            hotkey,
            hotkey_error,
            demo: std::env::var_os("SR_DEMO").is_some_and(|v| v == "1"),
            live_opts,
            save_gen: 0,
            _subs: subs,
            _ticker: ticker,
        }
        .with_title_lang()
    }

    fn with_title_lang(mut self) -> Self {
        self.title_input_lang = self.settings.ui_lang;
        self
    }

    fn make_title_input(lang: UiLang, value: &str, window: &mut Window, cx: &mut Context<Self>) -> Entity<InputState> {
        let value = value.to_string();
        cx.new(|cx| InputState::new(window, cx).placeholder(t(lang, Key::TitlePlaceholder)).default_value(value))
    }

    /// Opacity of the backdrops under conversation text: always more visible than the window background.
    fn plate_alpha(&self) -> f32 {
        (self.settings.opacity + 0.35).min(0.92)
    }

    fn l(&self) -> UiLang {
        self.settings.ui_lang
    }

    fn tr(&self, k: Key) -> &'static str {
        t(self.settings.ui_lang, k)
    }

    // ---------- settings ----------

    /// Saves settings 500 ms after the last change.
    fn save_soon(&mut self, cx: &mut Context<Self>) {
        self.save_gen += 1;
        let generation = self.save_gen;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(500)).await;
            let _ = this.update(cx, |this, _| {
                if this.save_gen == generation {
                    this.save_now();
                }
            });
        })
        .detach();
    }

    fn save_now(&self) {
        if let Err(e) = sr_store::save_settings(&self.paths.settings_file(), &self.settings) {
            log::error!("failed to save settings: {e}");
        }
    }

    fn set_hotkey_enabled(&mut self, on: bool, cx: &mut Context<Self>) {
        self.settings.global_hotkey_enabled = on;
        self.hotkey_error = match &mut self.hotkey {
            Some(h) => h.set_enabled(on).err().map(|e| e.to_string()),
            None => self.hotkey_error.take(),
        };
        self.save_soon(cx);
        cx.notify();
    }

    fn is_dark(&self, window: &Window) -> bool {
        match self.settings.theme {
            ThemePref::Dark => true,
            ThemePref::Light => false,
            ThemePref::System => matches!(window.appearance(), WindowAppearance::Dark | WindowAppearance::VibrantDark),
        }
    }

    /// Syncs the gpui-component theme (slider, input field) with ours.
    fn sync_component_theme(&mut self, dark: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.applied_dark == Some(dark) {
            return;
        }
        self.applied_dark = Some(dark);
        Theme::change(if dark { ThemeMode::Dark } else { ThemeMode::Light }, Some(window), cx);
        // Root paints the window with the theme background — we draw the translucent background ourselves.
        let th = Theme::global_mut(cx);
        th.background = transparent_black();
        th.tokens.background = transparent_black().into();
    }

    // ---------- navigation ----------

    fn go(&mut self, screen: Screen, window: &mut Window, cx: &mut Context<Self>) {
        if screen == Screen::Settings && self.screen != Screen::Settings {
            self.back_to = self.screen;
        }
        if screen == Screen::History {
            self.history = sr_store::list_conversations(&self.paths.conversations_dir());
        }
        if screen == Screen::NewConversation && self.title_input_lang != self.settings.ui_lang {
            let v = self.title_input.read(cx).value().to_string();
            self.title_input = Self::make_title_input(self.settings.ui_lang, &v, window, cx);
            self.title_input_lang = self.settings.ui_lang;
        }
        self.screen = screen;
        if screen != Screen::NewConversation {
            window.focus(&self.focus, cx);
        }
        cx.notify();
    }

    fn start_conversation(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let title = self.title_input.read(cx).value().trim().to_string();
        let title = if title.is_empty() { self.tr(Key::Untitled).to_string() } else { title };
        let s = &self.settings;
        let cfg = SessionConfig {
            target: s.target_or_ui(),
            source: s.source_lang,
            mic_enabled: s.mic_enabled,
            asr_model: s.asr_model.clone(),
            mt_model: s.mt_model.clone(),
        };
        self.session = Some(Session {
            title,
            cfg,
            utterances: Vec::new(),
            index: HashMap::new(),
            started: Instant::now(),
            recorded: Duration::ZERO,
            rec_since: None,
            handle: None,
            next_id: 0,
            level_sys: (0.0, false),
            level_mic: (0.0, false),
            run: 0,
            asr_loading: false,
            mt_loading: false,
            mt_on: false,
            notice: None,
            scroll: ScrollHandle::new(),
            unseen: false,
            _events: Vec::new(),
            speakers: HashMap::new(),
            names: HashMap::new(),
            retracted: Default::default(),
            tracker: std::sync::Arc::new(std::sync::Mutex::new(sr_pipeline::SpeakerTracker::new(sr_pipeline::TrackerConfig::default()))),
        });
        self.title_input = Self::make_title_input(self.settings.ui_lang, "", window, cx);
        self.go(Screen::Session, window, cx);
    }

    fn end_conversation(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(mut s) = self.session.take()
            && let Some(mut h) = s.handle.take() {
                h.stop();
            }
        if self.compact {
            self.toggle_compact(window, cx);
        }
        self.go(Screen::Home, window, cx);
    }

    // ---------- speaker names ----------

    fn speaker_name(&self, sp: Speaker) -> String {
        self.session
            .as_ref()
            .and_then(|s| s.names.get(&sp).cloned())
            .unwrap_or_else(|| speaker_label(self.settings.ui_lang, sp))
    }

    fn start_rename(&mut self, sp: Speaker, window: &mut Window, cx: &mut Context<Self>) {
        let current = self.session.as_ref().and_then(|s| s.names.get(&sp).cloned()).unwrap_or_default();
        let placeholder = speaker_label(self.settings.ui_lang, sp);
        let input = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder).default_value(current));
        let sub = cx.subscribe_in(&input, window, |this, _, ev: &InputEvent, window, cx| {
            if matches!(ev, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                this.finish_rename(window, cx);
            }
        });
        input.update(cx, |st, cx| st.focus(window, cx));
        self.renaming = Some((sp, input, sub));
        cx.notify();
    }

    fn finish_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some((sp, input, _)) = self.renaming.take() else { return };
        let name = input.read(cx).value().trim().to_string();
        if let Some(s) = self.session.as_mut() {
            if name.is_empty() {
                s.names.remove(&sp);
            } else {
                s.names.insert(sp, name);
            }
        }
        window.focus(&self.focus, cx);
        cx.notify();
    }

    // ---------- recording ----------

    pub fn toggle_recording(&mut self, cx: &mut Context<Self>) {
        let Some(s) = self.session.as_mut() else { return };
        if let Some(since) = s.rec_since.take() {
            s.recorded += since.elapsed();
            if let Some(mut h) = s.handle.take() {
                h.stop();
            }
            s.level_sys = (0.0, false);
            s.level_mic = (0.0, false);
        } else {
            s.run += 1;
            let id_base = s.run * 1_000_000;
            let (handle, rx) = if self.demo {
                sr_pipeline::start_demo(s.cfg.clone(), id_base, s.started)
            } else {
                let mut opts = self.live_opts.clone();
                match asr_config(&self.paths, &s.cfg.asr_model) {
                    Ok(c) => opts.asr = Some(c),
                    Err(path) => {
                        s.notice = Some((format!("{}: {path}", t(self.settings.ui_lang, Key::ModelNotFound)), Instant::now()));
                        cx.notify();
                        return;
                    }
                }
                // Keep working without a translation model: show the original and warn.
                s.notice = None;
                match mt_config(&self.paths, &s.cfg.mt_model) {
                    Ok(c) => opts.mt = Some(c),
                    Err(what) => {
                        s.notice = Some((format!("{}: {what}", t(self.settings.ui_lang, Key::TranslationUnavailable)), Instant::now()));
                    }
                }
                s.mt_on = opts.mt.is_some();
                if self.settings.diarization {
                    let model = self.paths.models_dir().join(sr_pipeline::EMBEDDING_MODEL_FILE);
                    if model.exists() {
                        opts.diarization = Some(sr_pipeline::DiarizationConfig { model, tracker: s.tracker.clone() });
                    } else {
                        s.notice = Some((format!("{}: {}", t(self.settings.ui_lang, Key::DiarizationUnavailable), model.display()), Instant::now()));
                    }
                }
                sr_pipeline::start_live(s.cfg.clone(), opts, id_base, s.started)
            };
            s.handle = Some(handle);
            s.rec_since = Some(Instant::now());
            s._events.retain(|t| !t.is_ready());
            s._events.push(cx.spawn(async move |this, cx| {
                while let Ok(ev) = rx.recv().await {
                    if this.update(cx, |this, cx| this.on_event(ev, cx)).is_err() {
                        break;
                    }
                }
            }));
        }
        cx.notify();
    }

    fn on_event(&mut self, ev: PipelineEvent, cx: &mut Context<Self>) {
        let Some(s) = self.session.as_mut() else { return };
        match ev {
            PipelineEvent::Started | PipelineEvent::Stopped => {}
            PipelineEvent::Level { source, level, speech } => match source {
                AudioSource::System => s.level_sys = (level, speech),
                AudioSource::Mic => s.level_mic = (level, speech),
            },
            PipelineEvent::Utterance(u) => {
                // Scroll down only if the user was already at the bottom.
                let at_bottom = is_at_bottom(&s.scroll);
                s.upsert(u);
                if at_bottom {
                    s.scroll.scroll_to_bottom();
                } else {
                    s.unseen = true;
                }
            }
            PipelineEvent::Retract(id) => s.retract(id),
            PipelineEvent::SpeakerAssigned { id, speaker } => s.assign_speaker(id, speaker),
            PipelineEvent::AsrLoading(b) => s.asr_loading = b,
            PipelineEvent::MtLoading(b) => s.mt_loading = b,
            PipelineEvent::Notice(m) => s.notice = Some((m, Instant::now())),
            PipelineEvent::Failed(m) => {
                s.notice = Some((m, Instant::now()));
                if let Some(since) = s.rec_since.take() {
                    s.recorded += since.elapsed();
                }
                s.handle = None;
            }
        }
        cx.notify();
    }

    fn on_toggle_action(&mut self, _: &ToggleRecording, _: &mut Window, cx: &mut Context<Self>) {
        self.toggle_recording(cx);
    }

    fn toggle_compact(&mut self, window: &mut Window, _cx: &mut Context<Self>) {
        if self.compact {
            self.compact = false;
            if let Some(sz) = self.full_size.take() {
                window.resize(sz);
            }
        } else {
            let sz = window.bounds().size;
            self.full_size = Some(sz);
            self.compact = true;
            window.resize(size(sz.width.min(px(460.)), px(COMPACT_HEIGHT)));
        }
    }

    pub fn quit(&mut self, cx: &mut App) {
        if let Some(s) = self.session.as_mut()
            && let Some(mut h) = s.handle.take() {
                h.stop();
            }
        self.save_now();
        cx.quit();
    }
}

/// Recognition settings for the selected model. Error — path to the missing file.
/// Translation settings. Error — what exactly was not found (translator process or model file).
fn mt_config(paths: &Paths, model_id: &str) -> Result<sr_pipeline::MtConfig, String> {
    let sidecar = sr_pipeline::locate_sidecar().ok_or_else(|| sidecar_name().to_string())?;
    let model = paths.models_dir().join(sr_pipeline::mt_model_file(model_id).unwrap_or("?"));
    if !model.exists() {
        return Err(model.display().to_string());
    }
    let gpu = !std::env::var_os("SR_MT_CPU").is_some_and(|v| v == "1");
    // Interim text is translated only with the fast model: a large one on a weak GPU
    // can't keep up and steals time from recognition.
    let translate_interim = model_id == "llm-fast";
    Ok(sr_pipeline::MtConfig { sidecar, model, gpu, translate_interim })
}

fn sidecar_name() -> &'static str {
    if cfg!(windows) { "speechrusty-mt.exe" } else { "speechrusty-mt" }
}

fn asr_config(paths: &Paths, model_id: &str) -> Result<sr_pipeline::AsrConfig, String> {
    let file = sr_pipeline::asr_model_file(model_id).unwrap_or("?");
    let path = paths.models_dir().join(file);
    if !path.exists() {
        return Err(path.display().to_string());
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 8);
    // SR_ASR_CPU=1 — force the CPU (choosing by speed benchmark — stage 7).
    let gpu = !std::env::var_os("SR_ASR_CPU").is_some_and(|v| v == "1");
    Ok(sr_pipeline::AsrConfig::new(path, gpu, threads))
}

fn is_at_bottom(h: &ScrollHandle) -> bool {
    let max = h.max_offset().y;
    max <= px(1.) || -h.offset().y >= max - px(24.)
}

fn fmt_clock(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60) } else { format!("{:02}:{:02}", s / 60, s % 60) }
}

// ======================= rendering =======================

impl AppView {
    fn header(&mut self, p: &Palette, cx: &mut Context<Self>) -> Stateful<Div> {
        let mut left = div().flex().items_center().gap_2().flex_1().overflow_hidden();
        match self.screen {
            Screen::Session => {
                let s = self.session.as_ref().expect("session screen without session");
                let rec = s.recording();
                left = left
                    .child(
                        div()
                            .id("rec")
                            .flex()
                            .items_center()
                            .gap_2()
                            .px_2()
                            .py_0p5()
                            .rounded_full()
                            .cursor_pointer()
                            .occlude()
                            .bg(if rec { p.rec.opacity(0.18) } else { p.surface })
                            .border_1()
                            .border_color(if rec { p.rec.opacity(0.6) } else { p.border })
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_recording(cx)))
                            .child({
                                let dot = div().size(px(10.)).rounded_full().bg(if rec { p.rec } else { p.faint });
                                if rec {
                                    dot.with_animation(
                                        "rec-pulse",
                                        Animation::new(Duration::from_millis(1200)).repeat().with_easing(pulsating_between(0.35, 1.0)),
                                        |d, delta| d.opacity(delta),
                                    )
                                    .into_any_element()
                                } else {
                                    dot.into_any_element()
                                }
                            })
                            .child(div().text_xs().text_color(if rec { p.rec } else { p.muted }).child(fmt_clock(s.elapsed()))),
                    )
                    .child(div().text_sm().text_color(p.fg).truncate().child(s.title.clone()));
            }
            screen => {
                if screen != Screen::Home {
                    left = left.child(icon_button("back", IconName::ArrowLeft, p).on_click(cx.listener(|this, _, window, cx| {
                        let to = match this.screen {
                            Screen::Settings => this.back_to,
                            _ if this.session.is_some() => Screen::Session,
                            _ => Screen::Home,
                        };
                        this.go(to, window, cx);
                    })));
                }
                let title = match screen {
                    Screen::Home => "SpeechRusty",
                    Screen::NewConversation => self.tr(Key::NewConversation),
                    Screen::History => self.tr(Key::History),
                    Screen::Settings => self.tr(Key::Settings),
                    Screen::Session => unreachable!(),
                };
                left = left.child(div().text_sm().text_color(p.fg).child(title));
            }
        }

        let mut right = div().flex().items_center().gap_0p5();
        if self.screen == Screen::Session {
            right = right.child(
                icon_button("compact", if self.compact { IconName::Maximize } else { IconName::Minimize }, p)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.toggle_compact(window, cx);
                        cx.notify();
                    })),
            );
        }
        let dark = p.dark;
        right = right
            .child(icon_button("theme", if dark { IconName::Sun } else { IconName::Moon }, p).on_click(cx.listener(move |this, _, _, cx| {
                this.settings.theme = if dark { ThemePref::Light } else { ThemePref::Dark };
                this.save_soon(cx);
                cx.notify();
            })))
            .when(self.screen != Screen::Settings && !self.compact, |d| {
                d.child(icon_button("settings", IconName::Settings, p).on_click(cx.listener(|this, _, window, cx| this.go(Screen::Settings, window, cx))))
            })
            .child(icon_button("close", IconName::Close, p).on_click(cx.listener(|this, _, _, cx| this.quit(cx))));

        let plate = (self.screen == Screen::Session).then(|| p.bg.opacity(self.plate_alpha()));
        div()
            .id("header")
            .when_some(plate, |d, c| d.bg(c).rounded_t_lg())
            .flex_none()
            .flex()
            .items_center()
            .gap_2()
            .h(px(40.))
            .pl_3()
            .pr_2()
            .window_control_area(WindowControlArea::Drag)
            .child(left)
            .child(right)
    }

    fn home(&mut self, p: &Palette, cx: &mut Context<Self>) -> Div {
        div()
            .flex_1()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_3()
            .p_4()
            .child(button("new", self.tr(Key::StartConversation), true, p).w(px(240.)).on_click(cx.listener(|this, _, window, cx| {
                this.go(Screen::NewConversation, window, cx)
            })))
            .when(self.session.is_some(), |d| {
                let title = self.session.as_ref().map(|s| s.title.clone()).unwrap_or_default();
                d.child(button("resume", format!("↩ {title}"), false, p).w(px(240.)).on_click(cx.listener(|this, _, window, cx| {
                    this.go(Screen::Session, window, cx)
                })))
            })
            .child(button("history", self.tr(Key::History), false, p).w(px(240.)).on_click(cx.listener(|this, _, window, cx| {
                this.go(Screen::History, window, cx)
            })))
            .child(div().mt_2().text_xs().text_color(p.faint).child(if self.demo {
                format!("{} · {} {}", self.tr(Key::DemoNote), platform::local_hotkey_label(), self.tr(Key::PressToStart))
            } else {
                format!("{} {}", platform::local_hotkey_label(), self.tr(Key::PressToStart))
            }))
    }

    fn new_conversation(&mut self, p: &Palette, cx: &mut Context<Self>) -> Stateful<Div> {
        let l = self.l();
        let target = self.settings.target_or_ui();
        let source = self.settings.source_lang;

        let target_chips = div().flex().flex_wrap().gap_1p5().children(Lang::ALL.into_iter().map(|lang| {
            chip(SharedString::from(format!("tgt-{}", lang.code())), lang.native_name(), lang == target, p).on_click(cx.listener(
                move |this, _, _, cx| {
                    this.settings.target_lang = Some(lang);
                    this.save_soon(cx);
                    cx.notify();
                },
            ))
        }));
        let source_chips = div()
            .flex()
            .flex_wrap()
            .gap_1p5()
            .child(chip("src-auto", t(l, Key::Auto), source.is_none(), p).on_click(cx.listener(|this, _, _, cx| {
                this.settings.source_lang = None;
                this.save_soon(cx);
                cx.notify();
            })))
            .children(Lang::ALL.into_iter().map(|lang| {
                chip(SharedString::from(format!("src-{}", lang.code())), lang.native_name(), Some(lang) == source, p).on_click(
                    cx.listener(move |this, _, _, cx| {
                        this.settings.source_lang = Some(lang);
                        this.save_soon(cx);
                        cx.notify();
                    }),
                )
            }));

        let models = |list: &'static [ModelChoice], selected: &str, asr: bool, cx: &mut Context<Self>| {
            div().flex().flex_col().gap_1p5().children(list.iter().map(|m| {
                let id = m.id;
                model_option(
                    SharedString::from(format!("model-{id}")),
                    m.name,
                    t(l, m.note),
                    m.recommended.then(|| t(l, Key::Recommended)),
                    selected == id,
                    p,
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    if asr {
                        this.settings.asr_model = id.into();
                    } else {
                        this.settings.mt_model = id.into();
                    }
                    this.save_soon(cx);
                    cx.notify();
                }))
            }))
        };
        let asr = models(ASR_MODELS, &self.settings.asr_model.clone(), true, cx);
        let mt = models(MT_MODELS, &self.settings.mt_model.clone(), false, cx);

        div()
            .id("new-conv")
            .flex_1()
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap_4()
            .px_4()
            .pb_4()
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::Title), p)).child(Input::new(&self.title_input)))
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::TranslateInto), p)).child(target_chips))
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::SpeakersLanguage), p)).child(source_chips))
            .child(toggle("mic", t(l, Key::MicHint), self.settings.mic_enabled, p).on_click(cx.listener(|this, _, _, cx| {
                this.settings.mic_enabled = !this.settings.mic_enabled;
                this.save_soon(cx);
                cx.notify();
            })))
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::SpeechModel), p)).child(asr))
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::TranslationModel), p)).child(mt))
            .child(
                div().flex().justify_end().child(
                    button("start", t(l, Key::Start), true, p)
                        .on_click(cx.listener(|this, _, window, cx| this.start_conversation(window, cx))),
                ),
            )
    }

    fn utterance_row(&self, u: &Utterance, p: &Palette, compact: bool, cx: &mut Context<Self>) -> Div {
        let l = self.l();
        let color = p.speaker(u.speaker);
        let sp = u.speaker;
        // Speaker name: on click — an input field (in this speaker's first utterance on screen).
        let editing = self.renaming.as_ref().filter(|(s, _, _)| *s == sp).map(|(_, input, _)| input.clone());
        let label = match editing {
            Some(input) if !compact => div().w(px(160.)).child(Input::new(&input).small()).into_any_element(),
            _ => div()
                .id(SharedString::from(format!("spk-{}", u.id)))
                .text_color(color)
                .cursor_pointer()
                .hover(|d| d.underline())
                .child(self.speaker_name(sp))
                .on_click(cx.listener(move |this, _, window, cx| this.start_rename(sp, window, cx)))
                .into_any_element(),
        };
        let meta = div()
            .flex()
            .items_center()
            .gap_2()
            .text_xs()
            .child(div().size(px(8.)).rounded_full().bg(color))
            .child(label)
            .child(div().text_color(p.faint).child(direction(u.src_lang, u.dst_lang)))
            .when(!compact, |d| d.child(div().text_color(p.faint).child(fmt_clock(Duration::from_millis(u.start_ms)))));

        let main_text = u.translation.clone().unwrap_or_else(|| u.original.clone());
        let mt_on = self.session.as_ref().is_some_and(|s| s.mt_on);
        // Translation still in progress: the phrase's language isn't the target, and there's no translation yet.
        let pending = mt_on && u.translation.is_none() && u.src_lang.is_some_and(|l| l != u.dst_lang);
        let body = match &u.status {
            UtteranceStatus::Interim => match &u.translation {
                Some(tr) => div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(div().text_color(p.faint).child(format!("{tr} …")))
                    .when(self.settings.show_original && !compact, |d| d.child(div().text_xs().text_color(p.faint).child(u.original.clone()))),
                None => div().text_color(p.faint).child(format!("{} …", u.original)),
            },
            UtteranceStatus::Final => div()
                .flex()
                .flex_col()
                .gap_0p5()
                .child(div().text_color(if pending { p.muted } else { p.fg }).child(main_text))
                .when(pending, |d| d.child(div().text_xs().text_color(p.faint).child(t(l, Key::TranslatingHint))))
                .when(self.settings.show_original && !compact && u.translation.is_some(), |d| {
                    d.child(div().text_xs().text_color(p.muted).child(u.original.clone()))
                }),
            UtteranceStatus::NotRecognized => div().text_color(p.warn).child(format!("⚠ {}", t(l, Key::NotRecognized))),
            UtteranceStatus::TranslationFailed(e) => div()
                .flex()
                .flex_col()
                .gap_0p5()
                .child(div().text_color(p.fg).child(u.original.clone()))
                .child(div().text_xs().text_color(p.warn).child(format!("⚠ {}: {e}", t(l, Key::TranslationFailed)))),
        };
        // Backdrop under text: the background around utterances stays visible, while the text itself stays readable
        // even with strong window transparency (like live captions).
        let plate = p.bg.opacity(self.plate_alpha());
        div()
            .flex()
            .flex_col()
            .gap_0p5()
            .pl_2()
            .pr_2()
            .py_1()
            .rounded_r_md()
            .bg(plate)
            .border_l_2()
            .border_color(color.opacity(0.6))
            .child(meta)
            .child(if compact { body.text_sm().line_clamp(2) } else { body })
    }

    fn session_view(&mut self, p: &Palette, cx: &mut Context<Self>) -> Div {
        let l = self.l();
        let compact = self.compact;
        let s = self.session.as_ref().expect("session");
        let hint = format!("{} {}", platform::local_hotkey_label(), t(l, Key::PressToStart));

        if compact {
            let last: Vec<&Utterance> = s.utterances.iter().rev().take(2).collect();
            return div()
                .flex_1()
                .flex()
                .flex_col()
                .justify_end()
                .gap_1p5()
                .px_3()
                .pb_3()
                .overflow_hidden()
                .when(last.is_empty(), |d| d.child(div().text_sm().text_color(p.faint).child(hint.clone())))
                .children(last.into_iter().rev().map(|u| self.utterance_row(u, p, true, cx)));
        }

        let at_bottom = is_at_bottom(&s.scroll);
        let unseen = s.unseen && !at_bottom;
        let list = div()
            .id("transcript")
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&s.scroll)
            .flex()
            .flex_col()
            .gap_2()
            .px_3()
            .py_2()
            .when(s.utterances.is_empty(), |d| {
                d.child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_1()
                        .py_4()
                        .rounded_md()
                        .bg(p.bg.opacity(self.plate_alpha()))
                        .text_sm()
                        .text_color(p.muted)
                        .child(t(l, Key::WaitingForSpeech))
                        .child(div().text_xs().child(if s.recording() { String::new() } else { hint.clone() })),
                )
            })
            .children(s.utterances.iter().map(|u| self.utterance_row(u, p, false, cx)));

        let (sys, mic) = (s.level_sys, s.level_mic);
        let mic_on = s.cfg.mic_enabled;
        // Indicator: level bar + a dot that lights up while speech is ongoing.
        let meter = move |(v, speech): (f32, bool), c: Hsla, icon: IconName| {
            div()
                .flex()
                .items_center()
                .gap_1()
                .child(gpui_kit::component::Icon::new(icon).size(px(12.)).text_color(if speech { c } else { c.opacity(0.5) }))
                .child(
                    div()
                        .w(px(36.))
                        .h(px(4.))
                        .rounded_full()
                        .bg(c.opacity(0.2))
                        .child(div().h_full().rounded_full().bg(c.opacity(if speech { 1.0 } else { 0.55 })).w(px(36. * v.clamp(0.0, 1.0)))),
                )
        };
        let notice = s
            .notice
            .as_ref()
            .filter(|(_, at)| at.elapsed() < Duration::from_secs(8))
            .map(|(m, _)| m.clone())
            .or_else(|| s.asr_loading.then(|| t(l, Key::LoadingSpeechModel).to_string()))
            .or_else(|| s.mt_loading.then(|| t(l, Key::LoadingTranslationModel).to_string()));

        div()
            .flex_1()
            .flex()
            .flex_col()
            .overflow_hidden()
            .when_some(notice, |d, n| d.child(div().mx_3().mb_1().px_2().py_1().rounded_md().bg(p.warn.opacity(0.15)).text_xs().text_color(p.warn).child(n)))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .overflow_hidden()
                    .child(list)
                    .when(unseen, |d| {
                        d.child(
                            div().absolute().bottom_2().w_full().flex().justify_center().child(
                                button("unseen", format!("↓ {}", t(l, Key::NewLines)), true, p).py_1().text_xs().on_click(cx.listener(
                                    |this, _, _, cx| {
                                        if let Some(s) = this.session.as_mut() {
                                            s.scroll.scroll_to_bottom();
                                            s.unseen = false;
                                        }
                                        cx.notify();
                                    },
                                )),
                            ),
                        )
                    }),
            )
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap_3()
                    .px_3()
                    .pb_3()
                    .pt_2()
                    .bg(p.bg.opacity(self.plate_alpha()))
                    .rounded_b_lg()
                    .text_xs()
                    .text_color(p.muted)
                    .child(meter(sys, p.speaker(sr_core::Speaker::Remote(1)), IconName::Globe))
                    .when(mic_on, |d| d.child(meter(mic, p.accent, IconName::Mic)))
                    .child(format!("{} {:>3}%", t(l, Key::Dimming), (self.settings.opacity * 100.0).round()))
                    .child(div().flex_1().child(Slider::new(&self.opacity)))
                    .child(
                        div()
                            .id("end")
                            .px_2()
                            .py_0p5()
                            .rounded_md()
                            .cursor_pointer()
                            .text_color(p.muted)
                            .hover(|d| d.text_color(p.rec))
                            .child(t(l, Key::EndConversation))
                            .on_click(cx.listener(|this, _, window, cx| this.end_conversation(window, cx))),
                    ),
            )
    }

    fn history_view(&mut self, p: &Palette, cx: &mut Context<Self>) -> Stateful<Div> {
        let l = self.l();
        div()
            .id("history")
            .flex_1()
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap_1p5()
            .px_4()
            .pb_4()
            .when(self.history.is_empty(), |d| d.child(div().py_6().text_sm().text_color(p.faint).child(t(l, Key::HistoryEmpty))))
            .children(self.history.iter().map(|f| {
                let name = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                div().px_3().py_2().rounded_md().bg(p.surface).text_sm().text_color(p.fg).child(name)
            }))
            .child(div().pt_2().child(button("open-dir", t(l, Key::OpenFolder), false, p).on_click(cx.listener(|this, _, _, cx| {
                let dir = this.paths.conversations_dir();
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    log::error!("create {dir:?}: {e}");
                }
                cx.open_with_system(&dir);
            }))))
    }

    fn settings_view(&mut self, p: &Palette, cx: &mut Context<Self>) -> Stateful<Div> {
        let l = self.l();
        let ui_chips = div().flex().flex_wrap().gap_1p5().children(UiLang::ALL.into_iter().map(|lang| {
            chip(SharedString::from(format!("ui-{lang:?}")), lang.native_name(), lang == l, p).on_click(cx.listener(move |this, _, _, cx| {
                this.settings.ui_lang = lang;
                this.save_soon(cx);
                cx.notify();
            }))
        }));
        let theme = self.settings.theme;
        let theme_chips = div().flex().flex_wrap().gap_1p5().children(
            [(ThemePref::System, Key::ThemeSystem), (ThemePref::Light, Key::ThemeLight), (ThemePref::Dark, Key::ThemeDark)].into_iter().map(
                |(pref, key)| {
                    chip(SharedString::from(format!("th-{pref:?}")), t(l, key), pref == theme, p).on_click(cx.listener(move |this, _, _, cx| {
                        this.settings.theme = pref;
                        this.save_soon(cx);
                        cx.notify();
                    }))
                },
            ),
        );
        let hk_label = format!("{} — {}", t(l, Key::GlobalHotkey), platform::global_hotkey_label());
        div()
            .id("settings")
            .flex_1()
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap_4()
            .px_4()
            .pb_4()
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::InterfaceLanguage), p)).child(ui_chips))
            .child(div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::Theme), p)).child(theme_chips))
            .child(
                div().flex().flex_col().gap_1p5().child(section_label(t(l, Key::Dimming), p)).child(
                    div()
                        .flex()
                        .items_center()
                        .gap_3()
                        .text_xs()
                        .text_color(p.muted)
                        .child(div().flex_1().child(Slider::new(&self.opacity)))
                        .child(format!("{:>3}%", (self.settings.opacity * 100.0).round())),
                ),
            )
            .child(toggle("show-orig", t(l, Key::ShowOriginal), self.settings.show_original, p).on_click(cx.listener(|this, _, _, cx| {
                this.settings.show_original = !this.settings.show_original;
                this.save_soon(cx);
                cx.notify();
            })))
            .child(toggle("diarization", t(l, Key::Diarization), self.settings.diarization, p).on_click(cx.listener(|this, _, _, cx| {
                this.settings.diarization = !this.settings.diarization;
                this.save_soon(cx);
                cx.notify();
            })))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(toggle("global-hk", hk_label, self.settings.global_hotkey_enabled, p).on_click(cx.listener(|this, _, _, cx| {
                        let on = !this.settings.global_hotkey_enabled;
                        this.set_hotkey_enabled(on, cx);
                    })))
                    .when_some(self.hotkey_error.clone(), |d, e| d.child(div().pl(px(44.)).text_xs().text_color(p.warn).child(e))),
            )
    }
}

impl Render for AppView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let dark = self.is_dark(window);
        self.sync_component_theme(dark, window, cx);
        let p = if dark { Palette::dark() } else { Palette::light() };
        // The slider controls the background only in the conversation — there you need to see what's under the window.
        // Forms and settings are nearly opaque, otherwise they're unreadable on a busy background.
        let opacity = if self.screen == Screen::Session { self.settings.opacity } else { self.settings.opacity.max(0.92) };

        let header = self.header(&p, cx);
        let body = match self.screen {
            Screen::Home => self.home(&p, cx).into_any_element(),
            Screen::NewConversation => self.new_conversation(&p, cx).into_any_element(),
            Screen::Session => self.session_view(&p, cx).into_any_element(),
            Screen::History => self.history_view(&p, cx).into_any_element(),
            Screen::Settings => self.settings_view(&p, cx).into_any_element(),
        };

        div()
            .key_context(KEY_CONTEXT)
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::on_toggle_action))
            .on_action(cx.listener(|this, _: &Quit, _, cx| this.quit(cx)))
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(p.bg.opacity(opacity))
            .rounded_lg()
            .border_1()
            .border_color(p.border)
            .text_color(p.fg)
            .child(header)
            .child(body)
            .children(corner_marks(p.accent.opacity(0.9)))
    }
}
