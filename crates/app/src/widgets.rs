//! Small widgets in the app palette.

use gpui_kit::component::{Icon, IconName};
use gpui_kit::*;
use gpui_kit::prelude::FluentBuilder as _;

use crate::theme::Palette;

/// Clickable icon in the header.
pub fn icon_button(id: impl Into<ElementId>, icon: IconName, p: &Palette) -> Stateful<Div> {
    let hover = p.surface_hover;
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_center()
        .size(px(26.))
        .rounded_md()
        .cursor_pointer()
        // Buttons sit in the header's drag zone: without occlude, Windows sends the click to HTCAPTION.
        .occlude()
        .text_color(p.muted)
        .hover(move |s| s.bg(hover))
        .child(Icon::new(icon).size(px(15.)))
}

/// Regular button. `primary` — filled with the accent color.
pub fn button(id: impl Into<ElementId>, label: impl Into<SharedString>, primary: bool, p: &Palette) -> Stateful<Div> {
    let (bg, fg, hover) = if primary {
        (p.accent, p.accent_fg, p.accent.opacity(0.85))
    } else {
        (p.surface, p.fg, p.surface_hover)
    };
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_center()
        .gap_2()
        .px_4()
        .py_2()
        .rounded_md()
        .bg(bg)
        .text_color(fg)
        .text_sm()
        .border_1()
        .border_color(if primary { p.accent } else { p.border })
        .cursor_pointer()
        .hover(move |s| s.bg(hover))
        .child(label.into())
}

/// A "chip" for picking one option out of several.
pub fn chip(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool, p: &Palette) -> Stateful<Div> {
    let hover = p.surface_hover;
    let d = div()
        .id(id)
        .px_3()
        .py_1()
        .rounded_full()
        .text_sm()
        .border_1()
        .cursor_pointer();
    let d = if selected {
        d.bg(p.accent).border_color(p.accent).text_color(p.accent_fg)
    } else {
        d.bg(p.surface).border_color(p.border).text_color(p.fg).hover(move |s| s.bg(hover))
    };
    d.child(label.into())
}

/// Form section label.
pub fn section_label(text: impl Into<SharedString>, p: &Palette) -> Div {
    div().text_xs().text_color(p.muted).child(text.into())
}

/// Toggle with a label.
pub fn toggle(id: impl Into<ElementId>, label: impl Into<SharedString>, on: bool, p: &Palette) -> Stateful<Div> {
    let track = div()
        .flex_none()
        .w(px(32.))
        .h(px(18.))
        .rounded_full()
        .p(px(2.))
        .flex()
        .bg(if on { p.accent } else { p.faint.opacity(0.5) })
        .when(on, |d| d.justify_end())
        .child(div().size(px(14.)).rounded_full().bg(p.accent_fg));
    div()
        .id(id)
        .flex()
        .items_center()
        .gap_3()
        .cursor_pointer()
        .text_sm()
        .text_color(p.fg)
        .child(track)
        .child(div().flex_1().child(label.into()))
}

/// Model option in a list: name, comment, "recommended" mark.
pub fn model_option(
    id: impl Into<ElementId>,
    name: &str,
    note: &str,
    recommended: Option<&str>,
    selected: bool,
    p: &Palette,
) -> Stateful<Div> {
    let hover = p.surface_hover;
    div()
        .id(id)
        .flex()
        .flex_col()
        .gap_0p5()
        .px_3()
        .py_2()
        .rounded_md()
        .border_1()
        .cursor_pointer()
        .bg(p.surface)
        .border_color(if selected { p.accent } else { p.border })
        .hover(move |s| s.bg(hover))
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .text_sm()
                .text_color(p.fg)
                .child(div().size(px(8.)).rounded_full().bg(if selected { p.accent } else { p.faint.opacity(0.4) }))
                .child(name.to_string())
                .when_some(recommended, |d, r| d.child(div().text_xs().text_color(p.accent).child(format!("· {r}")))),
        )
        .child(div().pl(px(16.)).text_xs().text_color(p.muted).child(note.to_string()))
}

/// Highlighted L-shaped window corners (a hint that they can be dragged).
pub fn corner_marks(color: Hsla) -> impl Iterator<Item = Div> {
    const S: f32 = 16.0;
    [(true, true), (true, false), (false, true), (false, false)].into_iter().map(move |(top, left)| {
        let d = div().absolute().size(px(S)).border_color(color);
        let d = if top { d.top_0().border_t_2() } else { d.bottom_0().border_b_2() };
        let d = if left { d.left_0().border_l_2() } else { d.right_0().border_r_2() };
        match (top, left) {
            (true, true) => d.rounded_tl_lg(),
            (true, false) => d.rounded_tr_lg(),
            (false, true) => d.rounded_bl_lg(),
            (false, false) => d.rounded_br_lg(),
        }
    })
}
