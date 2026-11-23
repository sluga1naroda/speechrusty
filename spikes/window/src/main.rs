//! Stage 0 spike: overlay window on gpui.
//! Checks: frameless + transparency with adjustable dimming, drag by the header,
//! resize by edges/corners, highlighted corners, always on top, Ctrl/Cmd+R.

use gpui_kit::component::slider::{Slider, SliderEvent, SliderState};
use gpui_kit::*;

actions!(spike, [ToggleRecording, Quit]);

const CORNER: f32 = 14.0;

struct Overlay {
    opacity: f32,
    dark: bool,
    recording: bool,
    slider: Entity<SliderState>,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

impl Overlay {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(100.0)
                .step(1.0)
                .default_value(50.0)
        });
        let sub = cx.subscribe(&slider, |this, _, ev: &SliderEvent, cx| {
            let (SliderEvent::Change(v) | SliderEvent::Release(v)) = ev;
            this.opacity = v.end() / 100.0;
            cx.notify();
        });
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        Self {
            opacity: 0.5,
            dark: true,
            recording: false,
            slider,
            focus,
            _subs: vec![sub],
        }
    }

    fn toggle_recording(&mut self, _: &ToggleRecording, _: &mut Window, cx: &mut Context<Self>) {
        self.recording = !self.recording;
        cx.notify();
    }
}

fn corner(top: bool, left: bool, color: Hsla) -> Div {
    let mut d = div().absolute().size(px(CORNER)).border_color(color);
    d = if top { d.top_0().border_t_2() } else { d.bottom_0().border_b_2() };
    d = if left { d.left_0().border_l_2() } else { d.right_0().border_r_2() };
    d
}

impl Render for Overlay {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (bg, fg, muted) = if self.dark {
            (hsla(230. / 360., 0.15, 0.08, self.opacity), hsla(0., 0., 0.95, 1.), hsla(0., 0., 0.65, 1.))
        } else {
            (hsla(230. / 360., 0.2, 0.97, self.opacity), hsla(0., 0., 0.1, 1.), hsla(0., 0., 0.4, 1.))
        };
        let accent = hsla(239. / 360., 0.84, 0.67, 1.);
        let red = hsla(0., 0.85, 0.55, 1.);

        let rec_dot = div()
            .size(px(12.))
            .rounded_full()
            .bg(if self.recording { red } else { muted.opacity(0.4) });

        let header = div()
            .id("header")
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .h(px(36.))
            .window_control_area(WindowControlArea::Drag)
            .child(rec_dot)
            .child(div().text_sm().text_color(fg).child(if self.recording {
                "Идёт перевод (Ctrl+R — стоп)"
            } else {
                "Ctrl+R — начать"
            }))
            .child(div().flex_1())
            .child(
                div()
                    .id("theme")
                    .px_2()
                    .rounded_md()
                    .text_sm()
                    .text_color(fg)
                    .cursor_pointer()
                    .hover(|s| s.bg(accent.opacity(0.25)))
                    .child(if self.dark { "☀" } else { "☾" })
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.dark = !this.dark;
                        cx.notify();
                    })),
            )
            .child(
                div()
                    .id("close")
                    .px_2()
                    .rounded_md()
                    .text_sm()
                    .text_color(fg)
                    .cursor_pointer()
                    .hover(|s| s.bg(red.opacity(0.4)))
                    .child("✕")
                    .on_click(|_, _, cx| cx.quit()),
            );

        let body = div()
            .flex_1()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .text_color(fg)
            .child(div().text_xs().text_color(accent).child("Спикер-1  en→ru"))
            .child(div().child("Привет, всех слышно?"))
            .child(div().text_xs().text_color(muted).child("Hi, can everyone hear me?"));

        let footer = div()
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .pb_3()
            .text_xs()
            .text_color(muted)
            .child(format!("Затемнение {:>3}%", (self.opacity * 100.).round()))
            .child(div().flex_1().child(Slider::new(&self.slider)));

        div()
            .key_context("Overlay")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::toggle_recording))
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(bg)
            .rounded_lg()
            .border_1()
            .border_color(accent.opacity(0.35))
            .child(header)
            .child(body)
            .child(footer)
            .child(corner(true, true, accent))
            .child(corner(true, false, accent))
            .child(corner(false, true, accent))
            .child(corner(false, false, accent))
    }
}

#[cfg(windows)]
fn make_topmost(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::*;
    if let Ok(h) = HasWindowHandle::window_handle(window) {
        if let RawWindowHandle::Win32(w) = h.as_raw() {
            let hwnd = HWND(w.hwnd.get() as _);
            unsafe {
                let _ = SetWindowPos(hwnd, Some(HWND_TOPMOST), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            }
        }
    }
}
#[cfg(not(windows))]
fn make_topmost(_: &Window) {}

fn main() {
    application().run(|cx| {
        gpui_kit::init(cx);
        {
            // Root paints the window with the theme background; make it transparent and draw the background ourselves.
            let t = gpui_kit::component::theme::Theme::global_mut(cx);
            t.background = transparent_black();
            t.tokens.background = transparent_black().into();
        }
        cx.bind_keys([
            KeyBinding::new("ctrl-r", ToggleRecording, Some("Overlay")),
            KeyBinding::new("cmd-r", ToggleRecording, Some("Overlay")),
            KeyBinding::new("ctrl-q", Quit, None),
        ]);
        cx.on_action(|_: &Quit, cx| cx.quit());

        let bounds = Bounds::centered(None, size(px(520.), px(240.)), cx);
        let opts = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions {
                title: Some("SpeechRusty spike".into()),
                appears_transparent: true,
                traffic_light_position: None,
            }),
            kind: WindowKind::Normal,
            is_resizable: true,
            app_owns_titlebar_drag: true,
            window_background: WindowBackgroundAppearance::Transparent,
            window_min_size: Some(size(px(280.), px(140.))),
            ..Default::default()
        };
        gpui_kit::open_window(opts, cx, |window, cx| {
            make_topmost(window);
            cx.new(|cx| Overlay::new(window, cx))
        })
        .expect("open window");
        cx.activate(true);
    });
}
