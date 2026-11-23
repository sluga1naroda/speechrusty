//! SpeechRusty — a simultaneous translator. Entry point.

mod app;
mod platform;
mod theme;
mod widgets;

use gpui_kit::*;
use sr_core::UiLang;
use sr_store::{Paths, Settings};

use crate::app::{AppView, KEY_CONTEXT, Quit, ToggleRecording};

fn initial_settings(paths: &Paths) -> Settings {
    sr_store::load_settings(&paths.settings_file()).unwrap_or_else(|| {
        // First launch: interface language follows the system locale.
        let locale = sys_locale::get_locale().unwrap_or_default();
        Settings { ui_lang: UiLang::from_locale(&locale), ..Settings::default() }
    })
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,whisper_rs=warn")).init();
    sr_pipeline::init_asr_logging();
    let paths = Paths::system()?;
    let settings = initial_settings(&paths);
    log::info!("config: {:?}, data: {:?}", paths.config_dir, paths.data_dir);

    gpui_kit::application().with_assets(gpui_kit::assets::Assets).run(move |cx| {
        gpui_kit::init(cx);
        cx.bind_keys([
            KeyBinding::new("ctrl-r", ToggleRecording, Some(KEY_CONTEXT)),
            KeyBinding::new("cmd-r", ToggleRecording, Some(KEY_CONTEXT)),
            KeyBinding::new("ctrl-q", Quit, Some(KEY_CONTEXT)),
            KeyBinding::new("cmd-q", Quit, Some(KEY_CONTEXT)),
        ]);

        let bounds = match settings.window {
            Some(g) => Bounds::new(point(px(g.x), px(g.y)), size(px(g.w), px(g.h))),
            None => Bounds::centered(None, size(px(560.), px(420.)), cx),
        };
        let opts = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions { title: Some("SpeechRusty".into()), appears_transparent: true, traffic_light_position: None }),
            kind: platform::window_kind(),
            is_resizable: true,
            app_owns_titlebar_drag: true,
            window_background: WindowBackgroundAppearance::Transparent,
            window_min_size: Some(size(px(300.), px(140.))),
            ..Default::default()
        };
        let res = gpui_kit::open_window(opts, cx, |window, cx| {
            platform::make_topmost(window);
            cx.new(|cx| AppView::new(settings.clone(), paths.clone(), window, cx))
        });
        match res {
            Ok((_, view)) => {
                // Closing the window via the OS (Alt+F4) also saves settings.
                cx.on_window_closed(move |cx, _| view.update(cx, |v, cx| v.quit(cx))).detach();
            }
            Err(e) => {
                log::error!("failed to open window: {e}");
                cx.quit();
            }
        }
        cx.activate(true);
    });
    Ok(())
}
